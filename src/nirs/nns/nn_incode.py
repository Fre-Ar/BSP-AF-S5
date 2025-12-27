# src/nirs/nns/nn_incode.py
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable

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
    def __init__(self, in_dim: int = 64, hidden_dims: int = [64,32], composer_layers: int = 5, per_layer: bool = True):
        super().__init__()
        self.per_layer = bool(per_layer)
        self.composer_layers = int(composer_layers)
        out_dim = 4 * self.composer_layers if self.per_layer else 4
        
        # Build MLP dynamically based on hidden_dims list
        layers = []
        current_dim = in_dim
        
        # Hidden Layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.SiLU()) 
            current_dim = h_dim
            
        # Final Layer
        layers.append(nn.Linear(current_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    m.weight.normal_(0.0, 1e-3)
                    m.bias.fill_(0.31)      # hidden bias mild positive

    def forward(self, z: torch.Tensor):
        raw = self.net(z)                                       # (B, out_dim)
        
        if self.per_layer:
            B = raw.shape[0]
            raw = raw.view(B, self.composer_layers, 4) # (B, L, 4)

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
                 init_regime:  Optional[Callable] = None,
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
        

        # SIREN init per layer
        self.w0s = [w0_first] + [w0_hidden] * (depth - 1)
        for i, lin in enumerate(self.net):
            init_regime(lin, i, params=(self.w0s[i]))

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
# Full NIR 
# ---------------------------
class INCODE_NIR(nn.Module):
    '''
    INCODE variant:
      - If learn_global_z=False: x -> RFF features -> Harmonizer -> (a,b,c,d)
      - If learn_global_z=True:  Global Latent z -> Harmonizer -> (a,b,c,d)
        - x → RFF features φ(x)
        - harmonizer(φ(x)) → (a,b,c,d)
      - SIREN trunk with per-layer modulation
      - three heads: distance, c1, c2
    '''
    def __init__(self,
                 init_regime:  Optional[Callable] = None,
                 in_dim: int = 3,
                 layer_counts: Tuple[int,...] = (256,)*5,
                 # SIREN w0
                 w0_first: float = 30.0,
                 w0_hidden: float = 1.0,    
                 # x-conditioning features
                 rff_m: int = 32, # RFF with m=32 gives us a harmonizer in_dim = 64
                 rff_sigma: float = 1.0, # TODO: play with this value in [1.0,3.0]
                 # harmonizer
                 harmonizer_hidden_dims: int = [64,32],
                 per_layer: bool = False,
                 # Latent code global z
                 learn_global_z: bool = False,
                 z_dim: int = 64, # Latent dimension 
                 # heads
                 class_cfg: ClassHeadConfig = ClassHeadConfig(class_mode="ecoc", n_bits=32),
                 bias: bool = True):
        super().__init__()
        self.learn_global_z = learn_global_z
        if self.learn_global_z:
            # Global Z Mode: Learn a single latent vector
            self.global_z = nn.Parameter(torch.randn(1, z_dim) * 0.01)
            harmonizer_in_dim = z_dim
            self.encoder = None # RFF not used
        else:
            # RFF Mode: Encode coordinates spatially
            self.encoder = RFF(in_dim=in_dim, m=rff_m, sigma=rff_sigma)
            harmonizer_in_dim = self.encoder.out_dim
            self.global_z = None
        
        self.trunk = INCODETrunk(init_regime=init_regime,in_dim=in_dim, layer_counts=layer_counts,
                                 w0_first=w0_first, w0_hidden=w0_hidden, bias=bias)
        self.harmonizer = Harmonizer(in_dim=harmonizer_in_dim,
                                     hidden_dims=harmonizer_hidden_dims,
                                     composer_layers=len(layer_counts),
                                     per_layer=per_layer)

        def make_head(out_dim: int):
            layer = nn.Linear(layer_counts[-1], out_dim)
            if init_regime is not None:
                init_regime(layer, -1, params=(w0_hidden))
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
        B = x.shape[0]
        # x -> features -> (a,b,c,d)
        # 1. Get input for Harmonizer
        if self.learn_global_z:
            # Broadcast global z to batch size
            # Using repeat() instead of expand() to avoid 0-stride buffer crashes
            f = self.global_z.repeat(B, 1)
            #f = self.global_z.expand(B, -1)
        else:
            # Encode coordinates
            f = self.encoder(x) # (B, 2m) 
        
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
