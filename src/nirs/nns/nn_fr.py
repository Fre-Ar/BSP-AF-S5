# src/nirs/nns/nn_fr.py

import math
from typing import Optional, Callable, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nir import ClassHeadConfig

# -----------------------------
# FR Linear layer (train-time W = Λ B)
# -----------------------------
class FRLayer(nn.Module):
    '''
    Fourier Reparameterized Linear:
      y = x @ (B^T) @ (Λ^T) + b
    where B ∈ R^{M×Din} (fixed Fourier bases) and Λ ∈ R^{out_dim×M} (learnable).
    '''
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        freq: int = 256,
        phases: int = 8,
        alpha: float = 0.05,
        lambda_denom=1.0
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.freq = freq
        self.phases = phases
        self.alpha = alpha
        self.M = 2 * freq * phases
        
        B = self._generate_basis(in_dim, freq, phases)  # (M, in_dim)
        self.register_buffer("B", B)
        
        # learnable coefficients Λ and bias
        self.Lambda = nn.Parameter(torch.empty(out_dim, self.M))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        self._init_lambda(denom=lambda_denom)

    @torch.no_grad()
    def _generate_basis(self, in_dim, F_num, P_num):
        '''
        Construct B ∈ R^{M × in_dim} with M = 2 * F * P.
        φ_p = 2π * p / P
        ω_low  = {1/F, 2/F, ..., 1}
        ω_high = {1, 2, ..., F}
        z_j ∈ [ -Tmax/2, +Tmax/2 ], Tmax = 2π F  (uniform samples, j = 1..in_dim)

        Returns:
        B: (M, in_dim) fixed (register as buffer).
        '''
        dtype = torch.float32
        # Frequencies: Low freq set {1/F...1} and High freq set {1...F}
        # Note: Paper says "2F different frequencies"
        freqs_high = torch.arange(1, F_num + 1, dtype=dtype)
        freqs_low = torch.arange(1, F_num + 1, dtype=dtype) / F_num
        all_freqs = torch.cat([freqs_low, freqs_high]) # Size: 2F
        
        # Phases: 0 to 2pi * (P-1)/P
        phases = torch.arange(0, P_num, dtype=dtype) * (2 * math.pi / P_num)
        
        # Create grid of (ω, φ) pairs
        # We need M = 2F * P combinations
        # Meshgrid-like expansion
        omega = all_freqs.repeat_interleave(P_num) # Repeat each freq P times
        phi = phases.repeat(2 * F_num)             # Repeat phase sequence 2F times
        
        # Sampling range T_max = 2 * pi * F
        T_max = 2 * math.pi * F_num
        # Uniform sampling points z usually corresponding to input dimension
        # Sample interval [-T_max/2, T_max/2]
        z = torch.linspace(-0.5 * T_max, 0.5 * T_max, steps=in_dim)
        
        # Basis b_ij = cos(ω_i * z_j + φ_i)
        # ω: (M, 1), z: (1, in_dim), φ: (M, 1)
        B = torch.cos(omega.unsqueeze(1) * z.unsqueeze(0) + phi.unsqueeze(1))
        return B * self.alpha # (M, in_dim)
    
    @torch.no_grad()
    def _init_lambda(self, denom=1.0):
        '''
        Paper Eq.(7): λ_ij ~ U(-bound_j, +bound_j),
          bound_j = sqrt( 6 / ( M * sum_t B[j,t]^2 ) )
        '''
        # Calculate L2 norm for each basis vector (dim=1 is in_dim, so we sum over in_dim)
        # B shape is (M, in_dim). We want norm of each row b_i.
        basis_norms = self.B.norm(p=2, dim=1) # Shape (M,)
        
        # We want to init Lambda[:, i] based on basis_norms[i]
        # Standard bound for uniform is sqrt(6 / fan_in)
        # Here effective "fan_in" is related to the basis energy.
        
        # Official code logic adapted to vectorization:
        # bound_i = sqrt(6/M) / norm(b_i) / denom
        
        numerator = math.sqrt(6 / self.M)
        bounds = numerator / (basis_norms * (denom if denom else 1.0))
        
        # Vectorized uniform init is tricky in PyTorch, loop is safer for clarity
        for i in range(self.M):
            b = bounds[i].item()
            nn.init.uniform_(self.Lambda[:, i], -b, b)

    @torch.no_grad()
    def merged_weight_bias(self) -> nn.Linear:
        '''Return W_eff, b suitable for a vanilla Linear.'''
        # Convert to standard nn.Linear for inference
        linear = nn.Linear(self.in_dim, self.out_dim)
        
        linear.weight.data = self.Lambda @ self.B
        linear.bias.data = self.bias.data
        return linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        weight = self.Lambda @ self.B
        y = F.linear(x, weight, self.bias)
        return y

# -----------------------------
# FR trunk (stack of FRLinear)
# -----------------------------
class FRTrunk(nn.Module):
    '''
    Shared trunk using FRLinear layers. Keeps activations and architecture unchanged—FR only
    modifies how weights are *trained* (W=ΛB). You can later merge for inference.
    '''
    def __init__(
        self,
        activation: Type[nn.Module],
        hidden_dim: int = 256,
        n_hidden_layers: int = 4,
        freq: int = 256,
        phases: int = 8,
        alpha: float = 0.05,
        act_params: tuple = (),
        lambda_denom=1.0,
    ):
        super().__init__()
        assert n_hidden_layers >= 1
        self.n_hidden_layers = n_hidden_layers

        layers = []

        for _ in range(n_hidden_layers):
            layers.append(FRLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                freq=freq, phases=phases,
                alpha=alpha,
                lambda_denom=lambda_denom
            ))
            layers.append(activation(*act_params))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def merge_for_inference(self) -> nn.Sequential:
        '''
        Produce a new nn.Sequential where every FRLinear is replaced by a vanilla nn.Linear
        with W_eff = ΛB, preserving activations. Use this to avoid FR overhead at test-time. 
        '''
        merged = []
        for m in self.net:
            if isinstance(m, FRLayer):
                merged.append(m.merged_weight_bias())
            else:
                merged.append(m)
        return nn.Sequential(*merged)

# -----------------------------
# Full FR NIR (3 heads)
# -----------------------------
class FR_NIR(nn.Module):
    '''
    Fourier Reparameterized Training NIR (Shi et al., 2024) with 3 heads:
      - dist_head: regression (km) -> (B,1)
      - c1_head:  logits -> (B, C1) or (B, n_bits) for ECOC
      - c2_head:  logits -> (B, C2) or (B, n_bits) for ECOC

    Paper-faithful parts:
      * Train-time weight reparameterization W=ΛB with fixed Fourier bases (P phases, 2F freqs).
      * Basis sampling on z ∈ [-Tmax/2, +Tmax/2], Tmax = 2πF.
      * Λ initialization per Eq.(7); compatible with ReLU/Sin backbones.
      * At inference you can merge Λ and B to a standard Linear (no runtime cost increase).
    '''
    def __init__(self,
                 activation: Type[nn.Module],
                 init_regime:  Optional[Callable] = None,
                 in_dim: int = 3,
                 hidden_dim: int = 256,
                 depth: int = 5, # total num of learned linear layers = len(layer_counts) + 1
                 freq: int = 256,
                 phases: int = 8,
                 alpha: float = 0.05,
                 params: Optional[tuple] = None, # params tuple (param1, param2, ...) for act function
                 class_cfg: ClassHeadConfig = ClassHeadConfig(class_mode="ecoc", n_bits=32)
                ):
        super().__init__()
        
        # Getting activation module params
        act_params = ()
        if params:
            param1 = params[0]
            act_params = (param1,)
            if len(params) > 1:
                param2 = params[1]
                act_params = (param2,)
            else:
                param2 = param1
        else:
            param1 = 1.0
            param2 = param1

        def make_lin(in_dim: int, out_dim: int, ith: int =-1):
            layer = nn.Linear(in_dim, out_dim)
            if init_regime is not None:
                init_regime(layer, ith, params=(param1))
            return layer
        
        input = make_lin(in_dim, hidden_dim, ith=0)
        if act_params:
            act = activation(param1)
        else:
            act = activation()
        in_modules = [input, act]
        self.input = nn.Sequential(*in_modules)
        
        self.trunk = FRTrunk(
            activation=activation,
            hidden_dim=hidden_dim,
            n_hidden_layers=depth-2, # depth = num of learned layers. Input and Heads are 2, so n_hidden_layers = depth-2
            freq=freq,
            phases=phases,
            alpha=alpha,
            act_params=act_params,
            lambda_denom=param1
        )
        
        if class_cfg.class_mode == "ecoc":
            out_c1 = out_c2 = class_cfg.n_bits
        elif class_cfg.class_mode == "softmax":
            out_c1 = class_cfg.n_classes_c1
            out_c2 = class_cfg.n_classes_c2
        else:
            raise ValueError(f"Unknown class_mode={class_cfg.class_mode}")

        self.dist_head = make_lin(hidden_dim, 1,    )
        self.c1_head   = make_lin(hidden_dim, out_c1)
        self.c2_head   = make_lin(hidden_dim, out_c2)
        self.softplus  = nn.Softplus()
        
    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        z = self.trunk(self.input(x))              # (B, hidden_dim)
        dist = self.softplus(self.dist_head(z))
        c1   = self.c1_head(z)
        c2   = self.c2_head(z)
        return dist, c1, c2

    @torch.no_grad()
    def merged_for_inference(self) -> nn.Sequential:
        '''A fused Sequential (vanilla Linear + activations) for fast test-time eval.'''
        return self.trunk.merge_for_inference()
