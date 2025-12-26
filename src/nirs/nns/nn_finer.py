# src/nirs/nns/nn_finer.py

import torch
import torch.nn as nn

from .nir import NIRLayer

class Finer(nn.Module):
    '''σ(x) = sin( ω(|x|+1)x )'''
    def __init__(self, w=torch.pi): 
        super().__init__()
        self.w = w
        
    def forward(self, x): 
        y = self.w * x
        return torch.where(y.abs() < 1e-7, torch.ones_like(y), torch.sin(y) / y)

class FINERLayer(NIRLayer):
    '''
    FINER (Liu et al., 2024):
    Params:
        w (float): frequency multiplier ω.

    '''
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True, is_last=False):
        self.w = params[0]
        self.bias_k = params[1]
        super().__init__(Finer(self.w), in_dim, out_dim, ith_layer, bias, is_last=is_last)
        self.register_buffer(f"w{ith_layer}", torch.tensor(float(self.w)))
