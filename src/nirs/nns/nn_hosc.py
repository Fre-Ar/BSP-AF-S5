# src/nirs/nns/nn_hosc.py

import torch
import torch.nn as nn

from .nir import NIRLayer

class Hosc(nn.Module):
    '''σ(x) = tanh(β*sin(x)), β>0.'''
    def __init__(self, beta=8.0, adaptive = True): 
        super().__init__()
        self.adaptive = adaptive
        if adaptive:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.beta = beta
        
    def forward(self, x): 
        return torch.tanh(self.beta * torch.sin(x))

class HOSCLayer(NIRLayer):
    '''
    HOSC (Serrano et al., 2024):
    Params:
        beta (float): sharpness (>0), fixed (HOSC baseline).
    Args:
        adaptive (bool): If true, makes beta a learneable parameter.
    '''
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True, is_last=False, adaptive = True):
        self.beta = params[0]
        super().__init__(Hosc(self.beta, adaptive), in_dim, out_dim, ith_layer, bias, is_last=is_last)
        if not adaptive:
            self.register_buffer(f"beta{ith_layer}", torch.tensor(float(self.beta)))