# src/nirs/nns/nn_wire.py

import torch
import torch.nn as nn

from .nir import NIRLayer

class Wire(nn.Module):
    '''σ(x) = exp( -|sx|^2 + iωx )
            = exp( -|sx|^2 ) * exp( iωx )
    ω,s>0 fixed.'''
    def __init__(self, w=10.0, s=20.0): 
        super().__init__()
        self.w = w
        self.s = s
    def forward(self, x): 
        carrier = torch.exp(1j * self.w * x)
        envelope = torch.exp(- (self.s * torch.abs(x))**2)
        return carrier * envelope  # complex output


class WIRELayer(NIRLayer):
    '''
    WIRE (Saragadam et al., 2023):
    Params:
        w (float): carrier frequency ω
        s (float): Gaussian window width parameter
    '''
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True, is_last=False):
        self.w, self.s = params[0], params[1]
        # Fixed hyperparameters as non-learnable buffers
        super().__init__(Wire(self.w, self.s), in_dim, out_dim, ith_layer, bias, complex=True, is_last=is_last)
        self.register_buffer(f"w{ith_layer}", torch.tensor(float(self.w)))
        self.register_buffer(f"s{ith_layer}", torch.tensor(float(self.s)))
