# src/nirs/nns/nn_gauss.py

import torch
import torch.nn as nn

from .nir import NIRLayer

class Gauss(nn.Module):
    '''φ(z) = exp( - |s0 * z|^2 ), s0>0 fixed.'''
    def __init__(self, s: float = 1.0): 
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
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True):
        # fixed bandwidth hyperparameter (saved in checkpoints but not optimized)
        self.s = params[0]
        super().__init__(Gauss(self.s), in_dim, out_dim, ith_layer, bias)
        self.register_buffer(f"s{ith_layer}", torch.tensor(float(self.s)))
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier/Glorot normal as used in their examples; bias = 0
        nn.init.xavier_normal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)    
