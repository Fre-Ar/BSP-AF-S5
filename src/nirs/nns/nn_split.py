# src/nirs/nns/nn_split.py

import torch.nn as nn
from typing import Optional, Callable, List

from .fourier_features import EncodingBase
from .nir import NIRLayer, ClassHeadConfig

# TODO: fix this class
class SplitNIR(nn.Module):
    '''
    3 NIRs in a trench coat:
      - distance (km): regression (>= 0 via Softplus)
      - c1: classification (containing country via ECOC or softmax)
      - c2: classification (adjacent country via ECOC or softmax)
      
    '''
    # TODO: find a way to allow different trunks using different hyper params
    def __init__(
        self, 
        layer: NIRLayer,
        init_regime:  Optional[Callable] = None,
        encoder: Optional[EncodingBase] = None,
        in_dim=3,
        layer_counts: tuple = (256,)*5,
        params: Optional[tuple] = None,
        encoder_params: Optional[tuple] = None,
        class_cfg: ClassHeadConfig = ClassHeadConfig(class_mode="ecoc", n_bits=32)
    ):
        super().__init__()
        # Checks
        depth = len(layer_counts)
        assert depth >= 2
        if not params:
            params = ((),)*depth
        
        self.class_cfg = class_cfg
        dist_params = params[0]
        c1_params = params[1]
        c2_params = params[2]
        
        # Encoding
        dist_layers = []
        c1_layers = []
        c2_layers = []
        
        self.encoder = None
        if encoder is not None:
            self.encoder = encoder(in_dim, *encoder_params)
            first_in = self.encoder.out_dim     
        else:
            first_in = in_dim
    
        # Classification nirs
        self.class_cfg = class_cfg
        if class_cfg.class_mode == "ecoc":
            assert class_cfg.n_bits is not None and class_cfg.n_bits > 0, "n_bits must be set for ECOC."
            out_c1 = class_cfg.n_bits
            out_c2 = class_cfg.n_bits
        elif class_cfg.class_mode == "softmax":
            assert class_cfg.n_classes_c1 and class_cfg.n_classes_c1 > 1, "n_classes_c1 must be >1 for softmax."
            assert class_cfg.n_classes_c2 and class_cfg.n_classes_c2 > 1, "n_classes_c2 must be >1 for softmax."
            out_c1 = class_cfg.n_classes_c1
            out_c2 = class_cfg.n_classes_c2
        else:  # pragma: no cover
            raise ValueError(f"Unknown class_mode={class_cfg.class_mode}")
        
        
        # Fill NIRs' layers
        def fill_layers(layers, out_dim, params_tuple):
            # trunk
            layers += [layer(first_in, layer_counts[0], params=params_tuple[0], ith_layer=0)]
            for i in range(1, depth):
                layers += [layer(layer_counts[i-1], layer_counts[i], params=params_tuple[i], ith_layer=i, is_last=(i==depth-1))]
            # head
            layers +=  [nn.Linear(layer_counts[-1], out_dim)]    
            return layers
        
        self.softplus = nn.Softplus()
        
        dist_layers = fill_layers(dist_layers, 1)
        c1_layers = fill_layers(c1_layers, out_c1)
        c2_layers = fill_layers(c2_layers, out_c2)
        
        self.dist_net = nn.Sequential(*dist_layers)
        self.c1_net = nn.Sequential(*c1_layers)
        self.c2_net = nn.Sequential(*c2_layers)

       
    def forward(self, x):
        z = self.encoder(x) if self.encoder else x
        dist = self.softplus(self.dist_net(z))  # (B,1) >= 0
        c1_logits = self.c1_net(z)    # (B, n_c1)
        c2_logits = self.c2_net(z)    # (B, n_c2)
        return dist, c1_logits, c2_logits
