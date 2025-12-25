# src/nirs/nns/nn_relu.py

import math
import torch.nn as nn
from .nir import NIRLayer

class ReLULayer(NIRLayer):
    '''
    Simple ReLU NIRLayer
    '''
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True, is_last=False):
        super().__init__(nn.ReLU(), in_dim, out_dim, ith_layer, bias, is_last=is_last)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear.weight, a=0.0, nonlinearity="relu")
        if self.linear.bias is not None:
            # PyTorch Linear default with kaiming_uniform_: U(-1/sqrt(in_dim), +1/sqrt(in_dim))
            bound = 1.0 / math.sqrt(self.in_dim) if self.in_dim > 0 else 0.0
            nn.init.uniform_(self.linear.bias, -bound, bound)
                