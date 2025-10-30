# nn_trident.py
import math
import torch
import torch.nn as nn
from nir import NIRLayer, MultiHeadNIR, ClassHeadConfig
from fourier_features import PositionalEncoding
from nn_gauss import GAUSSLayer as TRIDENTLayer
from typing import Optional



class TridentNIR(MultiHeadNIR):
    """
    TRIDENT network (γ ∘ hidden φ-layers ∘ linear head):
      γ(v) = [v, ..., cos(2π σ^{j/m} v), sin(2π σ^{j/m} v), ...]   [Eq. (1)]
      φ(z) = exp(- s0 |z|^2)                                       [Eq. (2)]
    """
    def __init__(self,
                 in_dim: int = 3,
                 layer_counts: tuple = (256,)*5,
                 params: tuple = (1.0,),
                 s0: float = 5.0,          # Gaussian scale
                 mapping_size: int = 10,   # mapping size (#bands)
                 sigma: float = 2.0,       # frequency parameter σ
                 alpha: float = 2.0 * math.pi,
                 class_cfg: ClassHeadConfig = ClassHeadConfig(class_mode="ecoc", n_bits=32),
                 head_layers: tuple = (),
                 head_activation: Optional[nn.Module] = None,
                 ):
        
        super().__init__(layer = TRIDENTLayer,
                         encoder = PositionalEncoding,
                         in_dim = in_dim,
                         layer_counts = layer_counts,
                         params = ((s0,),)*(len(layer_counts)),
                         encoder_params = (mapping_size, sigma, alpha, True),
                         class_cfg = class_cfg,
                         head_layers = head_layers,
                         head_activation = head_activation)
