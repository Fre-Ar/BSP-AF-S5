# python/nn_pytorch_tests/nir.py

import math
import torch
import torch.nn as nn
from fourier_features import EncodingBase
from typing import Optional
from dataclasses import dataclass
from typing import Literal

# ===================== NIR CORE =====================


class NIRLayer(nn.Module):
    def __init__(self, activation: nn.Module, in_dim: int, out_dim: int, ith_layer: int, bias=True, complex=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ith_layer = ith_layer
        self.complex = complex
        dtype = torch.cfloat if complex else torch.get_default_dtype()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias, dtype=dtype)
        self.activation = activation

    def forward(self, x):
        if self.complex and (not torch.is_complex(x)):
            x = x.to(torch.cfloat)
        y = self.activation(self.linear(x))
        if self.complex and self.ith_layer < 0:
            y = y.real
        return y

class NIRTrunk(nn.Module):
    """Shared trunk used by all heads."""
    def __init__(self, layer: NIRLayer, encoder: Optional[EncodingBase], in_dim=3, layer_counts: tuple = (256,)*5, params: tuple = ((1.0,),)*5, encoder_params: Optional[tuple] = None):
        super().__init__()
        
        depth = len(layer_counts)
        assert depth >= 2
        assert len(params) == depth
        
        if encoder:
            self.encoder = encoder(in_dim, *encoder_params)
            layers = [self.encoder]
            first_in = self.encoder.out_dim     
        else:
            first_in = in_dim
        
        layers = [layer(first_in, layer_counts[0], params=params[0], ith_layer=0)]
        for i in range(1, depth-1):
            layers += [layer(layer_counts[i-1], layer_counts[i], params=params[i], ith_layer=i)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): 
        return self.net(x)

@dataclass
class ClassHeadConfig:
    """
    Classification head configurables.

    class_mode: 'ecoc' produces bit logits (B, n_bits).
                'softmax' produces class logits (B, n_classes).
    """
    class_mode: Literal["ecoc", "softmax"]
    n_bits: int | None = None
    n_classes_c1: int | None = None
    n_classes_c2: int | None = None


class MultiHeadNIR(nn.Module):
    """
    Shared trunk with three heads:
      - distance (km): regression (>= 0 via Softplus)
      - c1: classification (containing country via ECOC or softmax)
      - c2: classification (adjacent country via ECOC or softmax)
    """
    def __init__(self, layer: NIRLayer,
                 encoder: Optional[EncodingBase] = None,
                 in_dim=3,
                 layer_counts: tuple = (256,)*5,
                 params: tuple = (1.0,),
                 encoder_params: Optional[tuple] = None,
                 class_cfg: ClassHeadConfig = ClassHeadConfig(class_mode="ecoc", n_bits=32),
                 head_layers = (),
                 head_activation: Optional[nn.Module] = None):
        super().__init__()
        # Trunk
        self.class_cfg = class_cfg
        self.trunk = NIRTrunk(layer, encoder, in_dim, layer_counts, params=params, encoder_params=encoder_params)
        
        act = head_activation if head_activation is not None else nn.ReLU(inplace=True)

        def _init_linear(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        head_counts = (layer_counts[-1],) + head_layers

        # Distance head
        dist_layers = []
        for i in range(1, len(head_counts)-1):
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
        for i in range(1, len(head_counts)-1):
            c1_layers += [nn.Linear(head_counts[i-1], head_counts[i]), act] 
        c1_layers +=  [nn.Linear(head_counts[-1], out_c1)]    
        self.c1_head = nn.Sequential(*c1_layers)
        self.c1_head.apply(_init_linear)

        c2_layers = []
        for i in range(1, len(head_counts)-1):
            c2_layers += [nn.Linear(head_counts[i-1], head_counts[i]), act] 
        c2_layers +=  [nn.Linear(head_counts[-1], out_c2)]  
        self.c2_head = nn.Sequential(*c2_layers)
        self.c2_head.apply(_init_linear)

    def forward(self, x):
        h = self.trunk(x)              # (B, hidden)
        dist = self.softplus(self.dist_head(h))  # (B,1) >= 0
        c1_logits = self.c1_head(h)    # (B, n_c1)
        c2_logits = self.c2_head(h)    # (B, n_c2)
        return dist, c1_logits, c2_logits
