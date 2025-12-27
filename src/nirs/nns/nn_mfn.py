# src/nirs/nns/nn_mfn

import math
from typing import Literal, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nir import ClassHeadConfig


# -----------------------------
# Filters (g(x; θ))
# -----------------------------
class FourierFilter(nn.Module):
    '''
    σ(x,θ_i) = sin(linear(x))
    '''
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 depth: int, 
                 weight_scale: int = 1.0,
                 init_regime:  Optional[Callable] = None,     
        ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth # depth of the total network
        self.linear = nn.Linear(in_dim, out_dim)
        
        init_regime(self.linear, -1, params=(weight_scale, depth, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x))


class GaborFilter(nn.Module):
    '''
    σ(x,θ_i) = exp(-γ_i/2 * ||x - µ_i||^2) * sin(linear(x))
    
    μ ~ Uniform(input_range)
    
    γ ~ Γ(concentration, rate), where:
    concentration = α/depth
    rate = β -> scale = 1/β
    
    (γ>0 via softplus)
    '''
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 depth: int, 
                 weight_scale: int = 1.0,
                 gamma_alpha: float = 6.0,
                 gamma_beta: float  = 1.0,
                 input_min: float = -1.0,
                 input_max: float =  1.0,
                 init_regime:  Optional[Callable] = None,     
        ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = max(depth, 1) # depth of the total network
        self.linear = nn.Linear(in_dim, out_dim)

        # Gaussian window params
        mu = (input_min - input_max) * torch.rand(out_dim, in_dim) + input_max  # uniform in [min,max]
        self.mu = nn.Parameter(mu)

        # gamma > 0 via softplus(raw_gamma)
        # sample from Gamma(alpha/k, beta) then invert softplus to seed raw_gamma
        alpha = max(gamma_alpha / depth, 1e-3)
        beta  = gamma_beta
        with torch.no_grad():
            g0 = torch.distributions.Gamma(alpha, beta).sample((out_dim,)) 
            # inverse softplus for stable init
            raw = g0 + torch.log(-torch.expm1(-g0))
        self.raw_gamma = nn.Parameter(raw)
        
        init_regime(self.linear, -1, params=(weight_scale, depth, torch.sqrt(self.gamma[:, None])))

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        sin_part = torch.sin(self.linear(x))          # (B, d)
        
        # computing the difference squared D 
        # with the expanded formula to avoid memory spikes
        D = (
            (x * x).sum(-1)[..., None]
            + (self.mu * self.mu).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        D = D.clamp_min_(0)
        
        window = torch.exp(-0.5 * D * self.gamma.unsqueeze(0))       # (B, d)
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
                 weight_scale: float = 1.0,
                 linear_init_regime:  Optional[Callable] = None,
                 filter_init_regime:  Optional[Callable] = None,
                 ):
        super().__init__()
        assert depth >= 1, "depth must be ≥ 1"
        self.in_dim, self.width, self.depth = in_dim, width, depth

        # filters g_i
        filters = []
        for _ in range(depth):
            if filter_type == 'fourier':
                filters.append(FourierFilter(in_dim, width, depth, weight_scale, init_regime=filter_init_regime))
            elif filter_type == 'gabor':
                filters.append(GaborFilter(in_dim, width, depth, weight_scale, init_regime=filter_init_regime))
            else:
                raise ValueError("filter_type must be 'fourier' or 'gabor'")
        self.filters = nn.ModuleList(filters)

        # linear layers W_i for i=1..k-1 mapping width->width
        self.linears = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])

        if linear_init_regime is not None:
            for i in range(depth-1):
                linear_init_regime(self.linears[i], i, params=(weight_scale, depth, 1.0))
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        z = self.filters[0](x)                         # (B, width)
        for i in range(self.depth - 1):
            z = self.linears[i](z) * self.filters[i+1](x)  # elementwise ∘
        return z                                        # (B, width)

# -----------------------------
# Full MFN-NIR (3 heads)
# -----------------------------
class MFN_NIR(nn.Module):
    '''
    Multiplicative Filter Networks NIR with three heads:
      - distance head -> (B,1)
      - c1 head -> (B, n_bits or n_classes)
      - c2 head -> (B, n_bits or n_classes)
    '''
    def __init__(self,
                 in_dim: int,
                 width: int,
                 depth: int, # total num of learned linear layers = len(layer_counts) + 1
                 filter_type: Literal['fourier', 'gabor'] = 'fourier',
                 weight_scale: float = 1.0,
                 linear_init_regime:  Optional[Callable] = None,
                 filter_init_regime:  Optional[Callable] = None,
                 class_cfg: ClassHeadConfig = ClassHeadConfig(class_mode="ecoc", n_bits=32)
                 ):
        super().__init__()
        self.trunk = MFNTrunk(
            in_dim,
            width,
            depth,
            filter_type,
            weight_scale=weight_scale,
            linear_init_regime=linear_init_regime,
            filter_init_regime=filter_init_regime)
        
        def make_head(out_dim: int):
            layer = nn.Linear(width, out_dim)
            if linear_init_regime is not None:
                linear_init_regime(layer, -1, params=(weight_scale, depth, 1.0))
            return layer

        if class_cfg.class_mode == "ecoc":
            out_c1 = out_c2 = class_cfg.n_bits
        elif class_cfg.class_mode == "softmax":
            out_c1 = class_cfg.n_classes_c1
            out_c2 = class_cfg.n_classes_c2
        else:
            raise ValueError(f"Unknown class_mode={class_cfg.class_mode}")

        self.dist_head = make_head(1)
        self.c1_head   = make_head(out_c1)
        self.c2_head   = make_head(out_c2)
        self.softplus  = nn.Softplus()

    def forward(self, x: torch.Tensor):
        '''
        x: (B, in_dim)
        returns:
          dist: (B,1), c1_logits: (B,c_bits), c2_logits: (B,c_bits)
        '''
        z = self.trunk(x)
        dist = self.softplus(self.dist_head(z))
        c1   = self.c1_head(z)
        c2   = self.c2_head(z)
        return dist, c1, c2
