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
        y = self.w * x
        # sin(y)/y -> 1 as y -> 0, so:
        return torch.where(y.abs() < 1e-7, torch.ones_like(y), torch.sin(y) / y)

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