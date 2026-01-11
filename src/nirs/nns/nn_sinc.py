# src/nirs/nns/nn_sinc.py

import torch
import torch.nn as nn

from .nir import NIRLayer

class Sinc(nn.Module):
    '''σ(x) = sinc(ωx) = sin(ωx)/ωx, ω>0 fixed.'''
    def __init__(self, w=torch.pi): 
        super().__init__()
        self.w = w
        
    def forward(self, x): 
        # torch.sinc(z) = sin(pi * z) / (pi * z)
        # We want sin(y) / y where y = w*x
        # So input to torch.sinc should be (w*x) / pi
        y = self.w * x
        return torch.sinc(y / torch.pi)

class SINCLayer(NIRLayer):
    '''
    SINC (Saratchandran et al., 2024):
    Params:
        w (float): scaling factor ω, π by default.
    '''
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True, is_last=False):
        self.w = params[0]
        super().__init__(Sinc(self.w), in_dim, out_dim, ith_layer, bias, is_last=is_last)
        self.register_buffer(f"w{ith_layer}", torch.tensor(float(self.w)))