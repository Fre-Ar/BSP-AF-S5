# python/nn_pytorch_tests/nn_finer.py

import torch
import torch.nn as nn
from nir import NIRLayer

class Finer(nn.Module):
    def __init__(self, w=torch.pi): 
        super().__init__()
        self.w = w
        
    def forward(self, x): 
        y = self.w * x
        return torch.where(y.abs() < 1e-7, torch.ones_like(y), torch.sin(y) / y)

class FINERLayer(NIRLayer):
    """
    FINER (Liu et al., 2024):
    Params:
        w (float): frequency multiplier ω.

    """
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True, bias_range_k: float = 5.0):
        self.w = params[0]
        self.register_buffer(f"w{ith_layer}", torch.tensor(float(self.w)))
        self.bias_range_k = float(bias_range_k)
        super().__init__(Finer(self.w), in_dim, out_dim, ith_layer, bias)

        self.reset_parameters()

    def reset_parameters(self):
        # ---- SIREN-style weight init, scaled by 1/ω0 ----
        # W ~ U(-sqrt(6/fan_in)/ω0, +sqrt(6/fan_in)/ω0)
        bound = torch.sqrt(6.0 / self.in_dim) / (float(self.w) if self.w != 0 else 1.0)
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

        # ---- FINER bias init: b ~ U(-k, k) with k > 0 ----
        if self.linear.bias is not None:
            with torch.no_grad():
                self.linear.bias.uniform_(-self.bias_range_k, self.bias_range_k)