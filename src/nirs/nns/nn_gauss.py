# src/nirs/nns/nn_gauss.py

import torch
import torch.nn as nn

from .nir import NIRLayer

class Gauss(nn.Module):
    '''σ(x) = exp( - |sx|^2 ), s>0 fixed.'''
    def __init__(self, s: float = 7.07): 
        super().__init__()
        self.s = s
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(- (self.s * x) * (self.s * x))
    
class GAUSSLayer(NIRLayer):
    '''
    Gauss Function (Ramasinghe & Lucey, 2022):
    Params:
        s (float): Gaussian window width parameter
    '''
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True, is_last=False):
        # fixed bandwidth hyperparameter 
        self.s = params[0]
        super().__init__(Gauss(self.s), in_dim, out_dim, ith_layer, bias, is_last=is_last)
        self.register_buffer(f"s{ith_layer}", torch.tensor(float(self.s)))

