# python/nn_pytorch_tests/nir.py

import math
import torch
import torch.nn as nn
from fourier_features import EncodingBase
from typing import Optional

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

class MultiHeadNIR(nn.Module):
    """
    Heads:
      - distance: regression (Softplus to enforce non-negativity)
      - c1: classification (containing country)
      - c2: classification (adjacent country)
    """
    def __init__(self, layer: NIRLayer,
                 encoder: Optional[EncodingBase],
                 in_dim=3,
                 layer_counts: tuple = (256,)*5,
                 params: tuple = (1.0,),
                 encoder_params: Optional[tuple] = None,
                 code_bits=32):
        super().__init__()
        self.trunk = NIRTrunk(layer, encoder, in_dim, layer_counts, params=params, encoder_params=encoder_params)
        self.dist_head = nn.Linear(layer_counts[-1], 1)
        self.c1_head   = nn.Linear(layer_counts[-1], code_bits)
        self.c2_head   = nn.Linear(layer_counts[-1], code_bits)
        nn.init.xavier_uniform_(self.dist_head.weight)
        nn.init.zeros_(self.dist_head.bias)
        nn.init.xavier_uniform_(self.c1_head.weight)
        nn.init.zeros_(self.c1_head.bias)
        nn.init.xavier_uniform_(self.c2_head.weight)
        nn.init.zeros_(self.c2_head.bias)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.trunk(x)              # (B, hidden)
        dist = self.softplus(self.dist_head(h))  # (B,1) >= 0
        c1_logits = self.c1_head(h)    # (B, num_c1)
        c2_logits = self.c2_head(h)    # (B, num_c2)
        return dist, c1_logits, c2_logits
