# src/nirs/nns/nn_incode.py
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .nn_siren import init_siren_linear
from .nir import ClassHeadConfig
from .fourier_features import RandomGaussianEncoding as RFF

# ---------------------------
# Harmonizer: maps features -> (a,b,c,d)
# Supports per-layer outputs: (B, L, 4)
# ---------------------------
class Harmonizer(nn.Module):
    '''
    Small MLP with SiLU. Neutral output init so (a,b,c,d)=(1,1,0,0) at start.
    If per_layer=True, outputs shape (B, L, 4); else (B, 4).
    '''
    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 5, per_layer: bool = True):
        super().__init__()
        self.per_layer = bool(per_layer)
        self.n_layers = int(n_layers)
        out_dim = 4 * self.n_layers if self.per_layer else 4

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    m.weight.normal_(0.0, 1e-3)
                    m.bias.fill_(0.31)      # hidden bias mild positive
            # last linear → neutral outputs
            last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
            last.bias.zero_()              # => log a=0, log b=0, c=0, d=0

    def forward(self, f: torch.Tensor):
        raw = self.net(f)                                       # (B, out_dim)
        if self.per_layer:
            B = raw.shape[0]
            raw = raw.view(B, self.n_layers, 4)                 # (B, L, 4)
            a = torch.exp(raw[..., 0])
            b = torch.exp(raw[..., 1])
            c = raw[..., 2]
            d = raw[..., 3]
            return a, b, c, d                                   # each (B, L)
        else:
            a = torch.exp(raw[..., 0])
            b = torch.exp(raw[..., 1])
            c = raw[..., 2]
            d = raw[..., 3]
            return (a, b, c, d)                                 # each (B,)

# ---------------------------
# SIREN-like trunk with per-layer modulation
# ---------------------------
class INCODETrunk(nn.Module):
    '''
    h_{l+1} = a_l * sin( b_l * w0_l * (W_l h_l + b_l^lin) + c_l ) + d_l
    If scalars a,b,c,d are (B,1), they broadcast; if per-layer, pass (B, L, 1).
    '''
    def __init__(self,
                 in_dim: int = 3,
                 layer_counts: Tuple[int,...] = (256,)*5,
                 w0_first: float = 30.0,
                 w0_hidden: float = 1.0,
                 bias: bool = True):
        super().__init__()
        depth = len(layer_counts)
        layers = [nn.Linear(in_dim, layer_counts[0], bias=bias)]
        for i in range(1, depth-1):
            layers.append(nn.Linear(layer_counts[i-1], layer_counts[i], bias=bias))
        self.net = nn.ModuleList(layers)
        self.w0s = [w0_first] + [w0_hidden] * (depth - 1)

        # SIREN init per layer
        for i, lin in enumerate(self.net):
            init_siren_linear(lin, layer_counts[i], i, w=self.w0s[i])

    def forward(self, x: torch.Tensor, a, b, c, d) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = x
        B = x.shape[0]
        per_layer = (isinstance(a, torch.Tensor) and a.dim() == 2) or (hasattr(a, 'shape') and len(a.shape) == 2)

        for l, (lin, w0) in enumerate(zip(self.net, self.w0s)):
            z = lin(h)  # (B, H_l)
            if per_layer:
                # use layer-specific scalars: (B, L) -> (B,1) for this layer
                a_l = a[:, l:l+1]
                b_l = b[:, l:l+1]
                c_l = c[:, l:l+1]
                d_l = d[:, l:l+1]
                h = a_l * torch.sin(b_l * w0 * z + c_l) + d_l
            else:
                # broadcast same scalars across layers
                a1 = a.view(B, 1)
                b1 = b.view(B, 1)
                c1 = c.view(B, 1)
                d1 = d.view(B, 1)
                h = a1 * torch.sin(b1 * w0 * z + c1) + d1
        return h  # (B, hidden_dim)

# ---------------------------
# Full NIR (continuous z(x) via RFF)
# ---------------------------
class INCODE_NIR(nn.Module):
    '''
    INCODE variant without codebooks/tiling:
      - x → RFF features φ(x)
      - harmonizer(φ(x)) → (a,b,c,d) per-layer (default)
      - SIREN trunk with per-layer modulation
      - three heads: distance, c1, c2
    '''
    def __init__(self,
                 in_dim: int = 3,
                 layer_counts: Tuple[int,...] = (256,)*5,
                 # SIREN w0
                 w0_first: float = 30.0,
                 w0_hidden: float = 1.0,
                 # x-conditioning features
                 rff_m: int = 64,
                 rff_sigma: float = 1.0,
                 # harmonizer
                 harmonizer_hidden: int = 64,
                 per_layer: bool = True,
                 # heads
                 class_cfg: ClassHeadConfig = ClassHeadConfig(class_mode="ecoc", n_bits=32),
                 head_layers: Tuple[int,...] = (),
                 head_activation: Optional[nn.Module] = None,
                 bias: bool = True):
        super().__init__()
        self.encoder = RFF(in_dim=in_dim, m=rff_m, sigma=rff_sigma)
        self.trunk = INCODETrunk(in_dim=in_dim, layer_counts=layer_counts,
                                 w0_first=w0_first, w0_hidden=w0_hidden, bias=bias)
        self.harmonizer = Harmonizer(in_dim=self.encoder.out_dim,
                                     hidden=harmonizer_hidden,
                                     n_layers=len(layer_counts),
                                     per_layer=per_layer)

        # Heads
        act = head_activation if head_activation is not None else nn.SiLU()
        def init_linear(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

        in_feat = layer_counts[-1]
        def make_head(out_dim: int):
            layers = []
            prev = in_feat
            for hdim in head_layers:
                layers += [nn.Linear(prev, hdim), act]
                prev = hdim
            layers += [nn.Linear(prev, out_dim)]
            seq = nn.Sequential(*layers); seq.apply(init_linear); return seq

        if class_cfg.class_mode == "ecoc":
            out_c1 = out_c2 = class_cfg.n_bits
        elif class_cfg.class_mode == "softmax":
            out_c1 = class_cfg.n_classes_c1; out_c2 = class_cfg.n_classes_c2
        else:
            raise ValueError(f"Unknown class_mode={class_cfg.class_mode}")

        self.dist_head = make_head(1)
        self.c1_head   = make_head(out_c1)
        self.c2_head   = make_head(out_c2)
        self.softplus  = nn.Softplus()

    @staticmethod
    def incode_reg(a, b, c, d, lambdas=(0.1993, 0.0196, 0.0588, 0.0269)) -> torch.Tensor:
        '''
        If per-layer: a,b,c,d are (B,L); otherwise (B,)
        Regularize towards neutral (1,1,0,0).
        '''
        l1, l2, l3, l4 = lambdas
        return (l1 * (a - 1).pow(2) + l2 * (b - 1).pow(2) + l3 * c.pow(2) + l4 * d.pow(2)).mean()

    def forward(self, x: torch.Tensor, return_abcd: bool = False):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # x -> features -> (a,b,c,d)
        f = self.encoder(x)                   # (B, 2m)
        a, b, c, d = self.harmonizer(f)       # per-layer by default
        # trunk
        h = self.trunk(x, a, b, c, d)         # (B, hidden)
        # heads
        dist      = self.softplus(self.dist_head(h))
        c1_logits = self.c1_head(h)
        c2_logits = self.c2_head(h)
        if return_abcd:
            return dist, c1_logits, c2_logits, (a, b, c, d)
        return dist, c1_logits, c2_logits
