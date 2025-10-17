# python/nn_pytorch_tests/nn_siren.py

import math
import torch
import torch.nn as nn
from nir import NIRLayer, MultiHeadNIR

class Sine(nn.Module):
    def __init__(self, w=1.0): 
        super().__init__()
        self.w = w
    def forward(self, x): 
        return torch.sin(self.w * x)

class SIRENLayer(NIRLayer):
    """
    SIREN (Sitzmann et al., 2020):
    Params:
        w (float): frequency multiplier ω.
    """
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True):
        self.w = params[0]
        self.register_buffer(f"w{ith_layer}", torch.tensor(float(self.w)))
        super().__init__(Sine(self.w), in_dim, out_dim, ith_layer, bias)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.ith_layer == 0:
                bound = 1.0 / self.in_dim
            else:
                bound = math.sqrt(6.0 / self.in_dim) / self.w
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()
                
                
class SIREN(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden=64, depth=4, w0_first=30.0, w0_hidden=1.0):
        super().__init__()
        layers = [SIRENLayer(in_dim, hidden, params=(w0_first,), ith_layer=0)]
        for i in range(1, depth-1):
            layers += [SIRENLayer(hidden, hidden, params=(w0_hidden,), ith_layer=i)]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    def forward(self, x):
        h = self.net(x)
        y = self.out(h)
        return y
