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
        dtype = torch.cfloat if complex else torch.get_default_dtype
        self.linear = nn.Linear(in_dim, out_dim, bias=bias, dtype=dtype)
        self.activation = activation

    def forward(self, x):
        if complex and (not torch.is_complex(x)):
            x = x.to(torch.cfloat)
        y = self.activation(self.linear(x))
        if complex and self.ith_layer < 0:
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
                 encoder: Optional[EncodingBase],
                 in_dim=3,
                 layer_counts: tuple = (256,)*5,
                 params: tuple = (1.0,),
                 encoder_params: Optional[tuple] = None,
                 class_cfg: ClassHeadConfig = ClassHeadConfig(class_mode="ecoc", n_bits=32),):
        super().__init__()
        # Trunk
        self.class_cfg = class_cfg
        self.trunk = NIRTrunk(layer, encoder, in_dim, layer_counts, params=params, encoder_params=encoder_params)
        
        # Distance head
        self.dist_head = nn.Linear(layer_counts[-1], 1)
        nn.init.xavier_uniform_(self.dist_head.weight)
        nn.init.zeros_(self.dist_head.bias)
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
        
        self.c1_head   = nn.Linear(layer_counts[-1], out_c1)
        self.c2_head   = nn.Linear(layer_counts[-1], out_c2)
        nn.init.xavier_uniform_(self.c1_head.weight)
        nn.init.zeros_(self.c1_head.bias)
        nn.init.xavier_uniform_(self.c2_head.weight)
        nn.init.zeros_(self.c2_head.bias)

    def forward(self, x):
        h = self.trunk(x)              # (B, hidden)
        dist = self.softplus(self.dist_head(h))  # (B,1) >= 0
        c1_logits = self.c1_head(h)    # (B, n_c1)
        c2_logits = self.c2_head(h)    # (B, n_c2)
        return dist, c1_logits, c2_logits
