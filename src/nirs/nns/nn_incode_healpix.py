# src/nirs/nns/nn_incode_healpix.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .nir import ClassHeadConfig
from .nn_siren import init_siren_linear
from nirs.healpix import healpix_vec2pix_nest_batch

# ---------------------------
# Harmonizer (INCODE)
# ---------------------------
class INCODEHarmonizer(nn.Module):
    '''
    Small SiLU MLP that outputs (a,b,c,d) per sample.
    Paper's recs for images: weights ~ N(0, 0.001), bias = 0.31.
    '''
    def __init__(self, z_dim: int = 16, hidden=32, out_dim: int = 4, use_layernorm: bool = False):
        super().__init__()
        dims = [z_dim, hidden]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.SiLU())
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i+1]))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.mlp = nn.Sequential(*layers)
        self._init_weights_normal_001_bias_const_031()

    def _init_weights_normal_001_bias_const_031(self):
        with torch.no_grad():
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    m.weight.normal_(mean=0.0, std=0.001)
                    m.bias.fill_(0.31)

    @staticmethod
    def map_to_activation_params(raw: torch.Tensor):
        # a,b > 0 via exp; c,d unconstrained (phase & bias)
        a = torch.exp(raw[..., 0])
        b = torch.exp(raw[..., 1])
        c = raw[..., 2]
        d = raw[..., 3]
        return a, b, c, d

    def forward(self, z: torch.Tensor):
        raw = self.mlp(z)  # (B,4)
        return self.map_to_activation_params(raw)

# ---------------------------
# INCODE Trunk 
# ---------------------------
class INCODETrunk(nn.Module):
    '''
    SIREN-like trunk modulated by (a,b,c,d) from the harmonizer.
    Returns features (no final linear) to feed multiple heads.
    Layer update:
      h <- a * sin( b * w0_l * (W h + b) + c ) + d
    '''
    def __init__(self,
                 in_dim: int = 3,
                 layer_counts: tuple = (256,)*5,
                 params: tuple = ((30.0,),)*5,
                 bias: bool = True):
        super().__init__()
        depth = len(layer_counts)
        layers = [nn.Linear(in_dim, layer_counts[0], bias=bias)]   # first
        for i in range(1, depth-1):
            layers += [nn.Linear(layer_counts[i-1], layer_counts[i], bias=bias)]
        self.net = nn.ModuleList(layers)
        self.params = params
        
        # init
        for i, lin in enumerate(self.net):
            init_siren_linear(lin, layer_counts[i], i, w=params[i][0])

    def forward(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        '''
        x: (B, Din), a,b,c,d: (B,) or (B,1); returns features (B, hidden_dim)
        '''
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        a, b, c, d = a.view(B, 1), b.view(B, 1), c.view(B, 1), d.view(B, 1)

        h = x
        for i, lin in enumerate(self.net):
            w0 = self.params[i][0]
            z = lin(h)                     # (B, H)
            h = a * torch.sin(b * w0 * z + c) + d
        return h  # features

# ---------------------------
# Full Multi-Head INCODE NIR
# ---------------------------
class INCODE_NIR(nn.Module):
    '''
    Full NIR with shared INCODE trunk and 3 heads:
      - dist_head: regression (km) -> (B,1)  (use MSE or robust loss)
      - c1_head:  classification/logits -> (B, C1) or (B, n_bits) for ECOC
      - c2_head:  classification/logits -> (B, C2) or (B, n_bits) for ECOC

    You can feed a per-sample latent z; if None and learn_global_z=True,
    a single learnable global code is used (expanded to batch).
    '''
    def __init__(self,
                 in_dim: int = 3,
                 layer_counts: tuple = (256,)*5,
                 # INCODE (composer) settings
                 w0: float = 30.0,
                 w_hidden: float = 30.0,
                 # Harmonizer settings
                 z_dim: int = 16,
                 harmonizer_hidden: int = 32,
                 learn_global_z: bool = False,
                 use_layernorm_in_harmonizer: bool = False,
                 nside: int = 64,
                 # Heads
                 class_cfg: ClassHeadConfig = ClassHeadConfig(class_mode="ecoc", n_bits=32),
                 head_layers: tuple = (),
                 head_activation: Optional[nn.Module] = None,
                 bias: bool = True):
        super().__init__()

        # --- Harmonizer ---
        self.harmonizer = INCODEHarmonizer(
            z_dim = z_dim,
            hidden = harmonizer_hidden,
            use_layernorm = use_layernorm_in_harmonizer
        )

        # ---- codebook(s) ----
        self.global_z = None
        self.nside = None
        self.num_tiles = None
        self.z_tile = None
        
        self.learn_global_z = learn_global_z
        if learn_global_z:
            self.global_z = nn.Parameter(torch.zeros(1, z_dim))
        else:
            # ---- tiling / codebook sizes ----
            assert nside >= 1 and (nside & (nside - 1)) == 0, "nside must be a power of two"
            self.nside = int(nside)
            self.num_tiles = 12 * self.nside * self.nside
            # ---- tile embedding ----
            self.z_tile = nn.Embedding(self.num_tiles, z_dim)
            with torch.no_grad():
                self.z_tile.weight.normal_(mean=0.0, std=1e-2)

        # --- Trunk ---
        self.trunk = INCODETrunk(
            in_dim=in_dim,
            layer_counts=layer_counts,
            params=((w0,),)+((w_hidden,),)*(len(layer_counts)-1),
            bias=bias,
        )

        # --- Heads ---
        act = head_activation if head_activation is not None else nn.ReLU(inplace=True)

        def _init_linear(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        head_counts = (layer_counts[-1],) + head_layers

        # Distance head
        dist_layers = []
        for i in range(1, len(head_counts)):
            dist_layers += [nn.Linear(head_counts[i-1], head_counts[i]), act] 
        dist_layers +=  [nn.Linear(head_counts[-1], 1)]    
           
        self.dist_head = nn.Sequential(*dist_layers)
        self.dist_head.apply(_init_linear)
        self.softplus = nn.Softplus()

        
        # Classification heads
        if class_cfg.class_mode == "ecoc":
            assert class_cfg.n_bits is not None and class_cfg.n_bits > 0, "n_bits must be set for ECOC."
            out_c1 = class_cfg.n_bits
            out_c2 = class_cfg.n_bits
        elif class_cfg.class_mode == "softmax":
            assert class_cfg.n_classes_c1 and class_cfg.n_classes_c1 > 1, "n_classes_c1 must be >1 for softmax."
            assert class_cfg.n_classes_c2 and class_cfg.n_classes_c2 > 1, "n_classes_c2 must be >1 for softmax."
            out_c1 = class_cfg.n_classes_c1
            out_c2 = class_cfg.n_classes_c2
        else:  # pragma: no cover
            raise ValueError(f"Unknown class_mode={class_cfg.class_mode}")
        
        c1_layers = []
        for i in range(1, len(head_counts)):
            c1_layers += [nn.Linear(head_counts[i-1], head_counts[i]), act] 
        c1_layers +=  [nn.Linear(head_counts[-1], out_c1)]    
        self.c1_head = nn.Sequential(*c1_layers)
        self.c1_head.apply(_init_linear)

        c2_layers = []
        for i in range(1, len(head_counts)):
            c2_layers += [nn.Linear(head_counts[i-1], head_counts[i]), act] 
        c2_layers +=  [nn.Linear(head_counts[-1], out_c2)]  
        self.c2_head = nn.Sequential(*c2_layers)
        self.c2_head.apply(_init_linear)

    # ---- optional INCODE regularization (anchors a→1, b→1, c→0, d→0) ----
    @staticmethod
    def incode_reg(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
                   lambdas=(0.1993, 0.0196, 0.0588, 0.0269)) -> torch.Tensor:
        l1, l2, l3, l4 = lambdas
        return (l1 * (a - 1).pow(2) + l2 * (b - 1).pow(2) + l3 * c.pow(2) + l4 * d.pow(2)).mean()

    def _get_z(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device = x.device
        if self.learn_global_z:
            return self.global_z.to(device=device, dtype=torch.float32).repeat(B, 1)
        elif self.z_tile is not None:
            with torch.no_grad():
                tile_id = healpix_vec2pix_nest_batch(x, self.nside).to(device=device, dtype=torch.long)
            return self.z_tile(tile_id)
        
        raise ValueError("INCODE_NIR.forward: use z tiling or set learn_global_z=True.")


    def forward(self, x: torch.Tensor, return_abcd: bool = False):
        '''
        Expects:
        - self.nside: int (power of two)
        - self.z_tile: nn.Embedding(num_embeddings=12*nside*nside, embedding_dim=d_tile)
        - (optional) self.z_global: nn.Parameter(1, d_global)
        - (optional) self.z_band: nn.Embedding(num_bands, d_band)
        - self.harmonizer: z -> (a,b,c,d), each (B,)
        - self.trunk, self.dist_head, self.c1_head, self.c2_head
        - self.softplus for nonnegative distance

        Args
        ----
        x : (B,3) unit vectors (or (3,) -> will be unsqueezed)
        band_id : (B,) optional integer r-band ids for per-band codes
        return_abcd : include modulation scalars in output
        '''
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 1) Build latent z code
        z = self._get_z(x)

        # 2) INCODE modulation
        a, b, c, d = self.harmonizer(z)               # each (B,)
        h = self.trunk(x, a, b, c, d)                 # (B, hidden_dim)

        # 3) Heads
        dist      = self.softplus(self.dist_head(h))  # (B,1)  >= 0
        c1_logits = self.c1_head(h)                   # (B, n_c1) or ECOC bits
        c2_logits = self.c2_head(h)                   # (B, n_c2) or ECOC bits

        if return_abcd:
            return dist, c1_logits, c2_logits, (a, b, c, d)
        return dist, c1_logits, c2_logits
