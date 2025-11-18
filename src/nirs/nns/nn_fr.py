# src/nirs/nns/nn_fr.py

import math
from typing import Optional, Tuple, Literal, List
import torch
import torch.nn as nn


# -----------------------------
# Fourier Basis 
# -----------------------------
def build_fourier_basis(
    in_dim: int,
    Ffreq: int,                 # F in the paper: frequency span
    Pphases: int,               # P in the paper: number of phases
    sampling_len_scale: float = 1.0,   # range length as multiple of Tmax (∈[0.5,4] works well)
    device=None,
    dtype=None,
) -> torch.Tensor:
    '''
    Construct B ∈ R^{M × in_dim} with M = 2 * Ffreq * Pphases.
      φ_p = 2π * p / P
      ω_low  = {1/F, 2/F, ..., 1}
      ω_high = {1, 2, ..., F}
      z_j ∈ [ -Tmax/2, +Tmax/2 ], Tmax = 2π F  (uniform samples, j = 1..in_dim)

    Returns:
      B: (M, in_dim) fixed (register as buffer).
    '''
    # TODO: change device here
    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.get_default_dtype()

    # sampling grid for the basis along the input dimension
    Tmax = 2.0 * math.pi * Ffreq
    L = sampling_len_scale * Tmax
    z = torch.linspace(-0.5 * L, 0.5 * L, in_dim, device=device, dtype=dtype)  # (in_dim,)

    # phases
    phases = torch.linspace(0.0, 2.0 * math.pi * (Pphases - 1) / Pphases, Pphases, device=device, dtype=dtype)  # (P,)

    # frequencies
    low = torch.arange(1, Ffreq + 1, device=device, dtype=dtype) / Ffreq            # 1/F,...,1
    high = torch.arange(1, Ffreq + 1, device=device, dtype=dtype)                   # 1,...,F

    freqs = torch.cat([low, high], dim=0)  # (2F,)
    # tile over phases → (2F*P,)
    freqs = freqs.repeat_interleave(Pphases)
    phis  = phases.repeat(2 * Ffreq)

    # build B: for each row i, b_i = cos(ω_i * z + φ_i)
    # broadcast: (M,1)*(1,D) + (M,1) → (M,D)
    M = 2 * Ffreq * Pphases
    B = torch.cos(freqs.view(M, 1) * z.view(1, in_dim) + phis.view(M, 1))
    return B


# -----------------------------
# FR Linear layer (train-time W = Λ B)
# -----------------------------
class FRLinear(nn.Module):
    '''
    Fourier Reparameterized Linear:
      y = x @ (B^T) @ (Λ^T) + b
    where B ∈ R^{M×Din} (fixed Fourier bases) and Λ ∈ R^{Dout×M} (learnable).

    After training you may merge to a vanilla Linear by W_eff = Λ @ B.
    '''
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        Ffreq: int = 16,
        Pphases: int = 8,
        sampling_len_scale: float = 1.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Ffreq = Ffreq
        self.Pphases = Pphases
        self.M = 2 * Ffreq * Pphases

        B = build_fourier_basis(
            in_features, Ffreq, Pphases, sampling_len_scale,
            device=device, dtype=dtype
        )  # (M, in_features)
        self.register_buffer("B", B)  # fixed

        # learnable coefficients Λ and bias
        self.Lambda = nn.Parameter(torch.empty(out_features, self.M, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None

        self._init_lambda_uniform_per_basis_row()

    @torch.no_grad()
    def _init_lambda_uniform_per_basis_row(self):
        '''
        Paper Eq.(7): λ_ij ~ U(-bound_j, +bound_j),
          bound_j = sqrt( 6 / ( M * sum_t B[j,t]^2 ) )
        (ReLU case; similar scheme for Sin per supp. We use this safe default.)
        '''
        row_norm_sq = (self.B ** 2).sum(dim=1) + 1e-8      # (M,)
        bounds = torch.sqrt(6.0 / (self.M * row_norm_sq))  # (M,)
        U = torch.rand_like(self.Lambda) * 2.0 - 1.0       # (-1,1)
        self.Lambda.copy_(U * bounds.view(1, self.M))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Din)
        x_proj = x @ self.B.t()            # (B, M) = (B,Din) @ (Din,M)
        y = x_proj @ self.Lambda.t()       # (B, Dout)
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def merged_weight_bias(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''Return W_eff, b suitable for a vanilla Linear.'''
        W_eff = self.Lambda @ self.B       # (Dout, Din)
        b = self.bias.clone() if self.bias is not None else None
        return W_eff, b


# -----------------------------
# FR trunk (stack of FRLinear + activation)
# -----------------------------
class FRTrunk(nn.Module):
    '''
    Shared trunk using FRLinear layers. Keeps activations and architecture unchanged—FR only
    modifies how weights are *trained* (W=ΛB). You can later merge for inference.
    '''
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 256,
        depth: int = 5,
        *,
        act: Literal["relu", "sine"] = "relu",
        Ffreq: int = 16,
        Pphases: int = 8,
        sampling_len_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        assert depth >= 2
        self.depth = depth
        self.act_kind = act

        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (depth - 1)

        for i in range(depth - 1):
            layers.append(FRLinear(
                dims[i], dims[i+1],
                Ffreq=Ffreq, Pphases=Pphases,
                sampling_len_scale=sampling_len_scale,
                bias=bias
            ))
            if act == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif act == "sine":
                # keep it simple; if you use SIREN-scale, handle outside
                layers.append(nn.SiLU())  # smoother than ReLU; swap for custom Sine if you want
            else:
                raise ValueError("act must be 'relu' or 'sine'")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def merge_for_inference(self) -> nn.Sequential:
        '''
        Produce a new nn.Sequential where every FRLinear is replaced by a vanilla nn.Linear
        with W_eff = ΛB, preserving activations. Use this to avoid FR overhead at test-time. 
        '''
        merged: List[nn.Module] = []
        for m in self.net:
            if isinstance(m, FRLinear):
                W, b = m.merged_weight_bias()
                lin = nn.Linear(m.in_features, m.out_features, bias=b is not None)
                lin.weight.copy_(W)
                if b is not None:
                    lin.bias.copy_(b)
                merged.append(lin)
            else:
                merged.append(m)
        return nn.Sequential(*merged)


# -----------------------------
# Simple heads (as in your other NIRs)
# -----------------------------
class TwoLayerHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_mult: float = 1.0, bias: bool = True):
        super().__init__()
        h = max(8, int(hidden_mult * in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, h, bias=bias),
            nn.SiLU(),
            nn.Linear(h, out_dim, bias=True),
        )
        self._init()

    @torch.no_grad()
    def _init(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Full FR NIR (3 heads)
# -----------------------------
class FR_NIR(nn.Module):
    '''
    Fourier Reparameterized Training NIR (Shi et al., CVPR'24) with 3 heads:
      - dist_head: regression (km) -> (B,1)
      - c1_head:  logits -> (B, C1) or (B, n_bits) for ECOC
      - c2_head:  logits -> (B, C2) or (B, n_bits) for ECOC

    Paper-faithful parts:
      * Train-time weight reparameterization W=ΛB with fixed Fourier bases (P phases, 2F freqs).
      * Basis sampling on z ∈ [−Tmax/2, +Tmax/2], Tmax = 2πF.
      * Λ initialization per Eq.(7); compatible with ReLU/Sin backbones.
      * At inference you can merge Λ and B to a standard Linear (no runtime cost increase).
    '''
    def __init__(self,
                 in_dim: int = 3,
                 hidden_dim: int = 256,
                 depth: int = 5,
                 *,
                 act: Literal["relu", "sine"] = "relu",
                 Ffreq: int = 16,
                 Pphases: int = 8,
                 sampling_len_scale: float = 1.0,
                 # heads
                 dist_out_dim: int = 1,
                 c1_out_dim: int = 32,
                 c2_out_dim: int = 32,
                 head_hidden_mult: float = 1.0,
                 bias: bool = True):
        super().__init__()
        self.trunk = FRTrunk(
            in_dim=in_dim, hidden_dim=hidden_dim, depth=depth,
            act=act, Ffreq=Ffreq, Pphases=Pphases,
            sampling_len_scale=sampling_len_scale, bias=bias
        )
        self.dist_head = TwoLayerHead(hidden_dim, dist_out_dim, hidden_mult=head_hidden_mult, bias=bias)
        self.c1_head   = TwoLayerHead(hidden_dim, c1_out_dim,   hidden_mult=head_hidden_mult, bias=bias)
        self.c2_head   = TwoLayerHead(hidden_dim, c2_out_dim,   hidden_mult=head_hidden_mult, bias=bias)

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        feat = self.trunk(x)                 # (B, hidden_dim)
        return {
            "dist":      self.dist_head(feat),
            "c1_logits": self.c1_head(feat),
            "c2_logits": self.c2_head(feat),
            "feat":      feat,
        }

    @torch.no_grad()
    def merged_for_inference(self) -> nn.Sequential:
        '''A fused Sequential (vanilla Linear + activations) for fast test-time eval.'''
        return self.trunk.merge_for_inference()
