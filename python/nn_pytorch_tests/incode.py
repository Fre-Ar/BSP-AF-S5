import math
import torch
import torch.nn as nn
from nir import NIRLayer, MultiHeadNIR

# ===================== SIREN CORE =====================

class ConditionedSine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x, a, b, c, d):
        # x: pre-activation; a,b,c,d: (B,1) broadcasted over features
        # shape-safe broadcast
        while a.dim() < x.dim():  # expand to match layer shape
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
            c = c.unsqueeze(-1)
            d = d.unsqueeze(-1)
        return a * torch.sin(b * self.w0 * x + c) + d


class INCODELayer(NIRLayer):
    def __init__(self, in_dim: int, out_dim: int, w0: float, ith_layer: int, bias=True):
        super().__init__(ConditionedSine(w0), in_dim, out_dim, ith_layer, bias)
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

    def forward(self, x, a, b, c, d):
        # x: (B, N, in_dim) or (B, in_dim). We support (B, N, ·) for grids.
        single_point = (x.dim() == 2)
        if single_point: x = x.unsqueeze(1)

        h = self.layers[0](x)
        h = self.act_first(h, a, b, c, d)
        for lin in self.layers[1:-1]:
            h = lin(h)
            h = self.act_hidden(h, a, b, c, d)
        y = self.layers[-1](h)

        if single_point: y = y.squeeze(1)
        return y