# python/nn_pytorch_tests/nn_sinc.py

import torch
import torch.nn as nn
from nir import NIRLayer

class Sinc(nn.Module):
    def __init__(self, w=torch.pi): 
        super().__init__()
        self.w = w
        
    def forward(self, x): 
        y = self.w * x
        return torch.where(y.abs() < 1e-7, torch.ones_like(y), torch.sin(y) / y)

class SINCLayer(NIRLayer):
    """
    SINC (Saratchandran et al., 2024):
    Params:
        w (float): scaling factor ω, π by default.

    """
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True):
        self.w = params[0]
        super().__init__(Sinc(self.w), in_dim, out_dim, ith_layer, bias)
        self.register_buffer(f"w{ith_layer}", torch.tensor(float(self.w)))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier/Glorot normal; bias = 0
        nn.init.xavier_normal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)