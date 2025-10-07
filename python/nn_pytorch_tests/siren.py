import math
import torch
import torch.nn as nn
from nir import NIRLayer, MultiHeadNIR

# ===================== SIREN CORE =====================


class Sine(nn.Module):
    def __init__(self, w0=1.0): 
        super().__init__()
        self.w0 = w0
    def forward(self, x): 
        return torch.sin(self.w0 * x)

class SIRENLayer(NIRLayer):
    def __init__(self, in_dim: int, out_dim: int, w0: float, ith_layer: int, bias=True):
        super().__init__(Sine(w0), in_dim, out_dim, ith_layer, bias)
        self.w0 = w0
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.ith_layer == 0:
                bound = 1.0 / self.in_dim
            else:
                bound = math.sqrt(6.0 / self.in_dim) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()
