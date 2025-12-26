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

                