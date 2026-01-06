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
        
        depth = len(layer_counts)
        assert depth >= 2, "depth must be ≥ 2"
        if not params:
            params = ((),)*depth
        assert len(params) == depth
        
        self.class_cfg = class_cfg
        
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
        def fill_layers(layers, out_dim):
            # trunk
            layers += [layer(first_in, layer_counts[0], params=params[0], ith_layer=0)]
            for i in range(1, depth):
                layers += [layer(layer_counts[i-1], layer_counts[i], params=params[i], ith_layer=i, is_last=(i==depth-1))]
            
            if init_regime is not None:
                for i in range(depth):
                    init_regime(layers[i].linear, i, params[0])

            # head
            layers +=  [nn.Linear(layer_counts[-1], out_dim)]  
            init_regime(layers[-1], -1, params[0])
            
            return layers
        
        dist_layers = fill_layers(dist_layers, 1)
        c1_layers = fill_layers(c1_layers, out_c1)
        c2_layers = fill_layers(c2_layers, out_c2)
        
        self.softplus = nn.Softplus()
        self.dist_net = nn.Sequential(*dist_layers)
        self.c1_net = nn.Sequential(*c1_layers)
        self.c2_net = nn.Sequential(*c2_layers)

       
    def forward(self, x):
        z = self.encoder(x) if self.encoder else x
        dist = self.softplus(self.dist_net(z))  # (B,1) >= 0
        c1_logits = self.c1_net(z)    # (B, n_c1)
        c2_logits = self.c2_net(z)    # (B, n_c2)
        return dist, c1_logits, c2_logits
