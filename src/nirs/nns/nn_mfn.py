# src/nirs/nns/nn_mfn

import math
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Filters (g(x; θ))
# -----------------------------
class FourierFilter(nn.Module):
    '''
    g(x) = sin(Ω x + φ), with learnable Ω ∈ R[d, in_dim], φ ∈ R[d].
    Paper Eq. (4).  Fathony et al., ICLR'21.
    '''
    def __init__(self, in_dim: int, d: int, omega_init_std: float = 1.0, phi_init_std: float = 0.0):
        super().__init__()
        self.in_dim, self.d = in_dim, d
        self.Omega = nn.Parameter(torch.randn(d, in_dim) * omega_init_std)
        self.phi   = nn.Parameter(torch.randn(d) * phi_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) -> (B, d)
        return torch.sin(x @ self.Omega.t() + self.phi)


class GaborFilter(nn.Module):
    '''
    g(x) = exp(-γ/2 * ||x - μ||^2) * sin(ω^T x + φ)
    θ = {ω ∈ R[d,in_dim], φ ∈ R[d], μ ∈ R[d,in_dim], γ ∈ R[d] (γ>0 via softplus)}.
    Paper Eq. (8).  Init per paper: μ ~ Uniform(input_range), γ ~ Gamma(α/k, β).
    '''
    def __init__(self,
                 in_dim: int,
                 d: int,
                 input_min: float = -1.0,
                 input_max: float =  1.0,
                 gamma_alpha: float = 2.0,
                 gamma_beta: float  = 1.0,
                 depth_k: int = 4,  # used to scale alpha -> alpha/k
                 omega_init_std: float = 1.0,
                 phi_init_std: float = 0.0):
        super().__init__()
        self.in_dim, self.d = in_dim, d

        # sin part
        self.omega = nn.Parameter(torch.randn(d, in_dim) * omega_init_std)
        self.phi   = nn.Parameter(torch.randn(d) * phi_init_std)

        # Gaussian window params
        mu = (input_min - input_max) * torch.rand(d, in_dim) + input_max  # uniform in [min,max]
        self.mu = nn.Parameter(mu)

        # gamma > 0 via softplus(raw_gamma)
        # sample from Gamma(alpha/k, beta) then invert softplus to seed raw_gamma
        alpha = max(gamma_alpha / max(depth_k, 1), 1e-3)
        beta  = gamma_beta
        with torch.no_grad():
            g0 = torch.distributions.Gamma(alpha, 1.0 / beta).sample((d,))  # rate=1/β ⇒ scale=β
            # inverse softplus for stable init
            raw = g0 + torch.log(-torch.expm1(-g0))
        self.raw_gamma = nn.Parameter(raw)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        sin_part = torch.sin(x @ self.omega.t() + self.phi)          # (B, d)
        # ||x - μ||^2 for every unit j: expand x -> (B,1,D), μ -> (1,d,D)
        diff2 = (x.unsqueeze(1) - self.mu.unsqueeze(0)).pow(2).sum(-1)  # (B, d)
        window = torch.exp(-0.5 * diff2 * self.gamma.unsqueeze(0))       # (B, d)
        return window * sin_part


# -----------------------------
# MFN trunk
# -----------------------------
class MFNTrunk(nn.Module):
    '''
    Implements the MFN recursion:
      z1 = g1(x)
      z_{i+1} = (W_i z_i + b_i) ∘ g_{i+1}(x),  i=1..k-1
    Returns z_k (features for heads).

    Args
    ----
    in_dim: input dimension
    width:  hidden width d_i (constant across layers)
    depth:  number of multiplicative filter layers (k)
    filter_type: 'fourier' or 'gabor'
    '''
    def __init__(self,
                 in_dim: int,
                 width: int,
                 depth: int,
                 filter_type: Literal['fourier', 'gabor'] = 'fourier',
                 **filter_kwargs):
        super().__init__()
        assert depth >= 1, "depth must be ≥ 1"
        self.in_dim, self.width, self.depth = in_dim, width, depth

        # filters g_i
        filters = []
        for i in range(depth):
            if filter_type == 'fourier':
                filters.append(FourierFilter(in_dim, width, **{k:v for k,v in filter_kwargs.items()
                                                               if k in ('omega_init_std','phi_init_std')}))
            elif filter_type == 'gabor':
                # pass depth to scale gamma alpha by 1/k as per paper
                fk = dict(filter_kwargs)
                fk.setdefault('depth_k', depth)
                filters.append(GaborFilter(in_dim, width, **fk))
            else:
                raise ValueError("filter_type must be 'fourier' or 'gabor'")
        self.filters = nn.ModuleList(filters)

        # linear layers W_i for i=1..k-1 mapping width->width
        self.linears = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])

        self._init_weights_divide_by_sqrtk()

    def _init_weights_divide_by_sqrtk(self):
        # Per paper: regardless of base init, divide W by sqrt(k) to keep final freq variance depth-invariant.
        # Biases -> 0.
        with torch.no_grad():
            scale = 1.0 / math.sqrt(max(self.depth, 1))
            for lin in self.linears:
                nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
                lin.weight.mul_(scale)
                nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        z = self.filters[0](x)                         # (B, width)
        for i in range(self.depth - 1):
            z = self.linears[i](z) * self.filters[i+1](x)  # elementwise ∘
        return z                                        # (B, width)


# -----------------------------
# Heads
# -----------------------------
class DistanceHead(nn.Module):
    '''Tiny head for distance; Softplus to keep ≥0 (change if you prefer raw km logits).'''
    def __init__(self, in_dim: int, hidden: Optional[int] = None):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(in_dim, 1)
        self.out_act = nn.Softplus(beta=1.0)  # smooth ReLU

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.net(z))


class ClassHead(nn.Module):
    '''Simple linear (or 2-layer) logits head.'''
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # logits (use BCEWithLogits/CE outside)


# -----------------------------
# Full MFN-NIR (3 heads)
# -----------------------------
class MFN_NIR(nn.Module):
    '''
    Multiplicative Filter Networks NIR with three heads:
      - distance head -> (B,1)
      - c1 head -> (B, n_bits or n_classes)
      - c2 head -> (B, n_bits or n_classes)

    Example:
      model = MFN_NIR(
          in_dim=3, width=256, depth=6,
          filter_type='gabor',        # or 'fourier'
          dist_hidden=128,
          c_bits=32
      )
    '''
    def __init__(self,
                 in_dim: int,
                 width: int,
                 depth: int,
                 filter_type: Literal['fourier', 'gabor'] = 'fourier',
                 c_bits: int = 32,
                 dist_hidden: Optional[int] = 128,
                 cls_hidden: Optional[int] = 128,
                 **filter_kwargs):
        super().__init__()
        self.trunk = MFNTrunk(in_dim, width, depth, filter_type, **filter_kwargs)
        self.head_dist = DistanceHead(width, hidden=dist_hidden)
        self.head_c1   = ClassHead(width, out_dim=c_bits, hidden=cls_hidden)
        self.head_c2   = ClassHead(width, out_dim=c_bits, hidden=cls_hidden)

    def forward(self, x: torch.Tensor):
        '''
        x: (B, in_dim)
        returns:
          dist: (B,1), c1_logits: (B,c_bits), c2_logits: (B,c_bits)
        '''
        z = self.trunk(x)
        dist = self.head_dist(z)
        c1   = self.head_c1(z)
        c2   = self.head_c2(z)
        return dist, c1, c2
