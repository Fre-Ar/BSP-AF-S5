# src/nirs/nns/nn_split.py

import torch.nn as nn
from typing import Optional

from .fourier_features import EncodingBase
from .nir import NIRLayer, ClassHeadConfig


class SplitNIR(nn.Module):
    '''
    3 NIRs in a trench coat:
      - distance (km): regression (>= 0 via Softplus)
      - c1: classification (containing country via ECOC or softmax)
      - c2: classification (adjacent country via ECOC or softmax)
    '''
    def __init__(self, layer: NIRLayer,
                 encoder: Optional[EncodingBase] = None,
                 in_dim=3,
                 layer_counts: tuple = (256,)*5,
                 params: tuple = (1.0,),
                 encoder_params: Optional[tuple] = None,
                 class_cfg: ClassHeadConfig = ClassHeadConfig(class_mode="ecoc", n_bits=32)):
        super().__init__()
        # Checks
        depth = len(layer_counts)
        assert depth >= 2
        if not params:
            params = ((),)*depth
        assert len(params) == depth
        
        # Encoding
        dist_layers = []
        c1_layers = []
        c2_layers = []
        if encoder is not None:
            self.encoder = encoder(in_dim, *encoder_params)
            
            dist_layers += [self.encoder]
            c1_layers += [self.encoder]
            c2_layers += [self.encoder]
            
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
        
        # Layer inits (unused for now)
        def _init_linear(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        
        # Fill NIRs' layers
        def fill_layers(layers, out_dim):
            # trunk
            layers += [layer(first_in, layer_counts[0], params=params[0], ith_layer=0)]
            for i in range(1, depth):
                layers += [layer(layer_counts[i-1], layer_counts[i], params=params[i], ith_layer=i, is_last=(i==depth-1))]
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
        dist = self.softplus(self.dist_net(x))  # (B,1) >= 0
        c1_logits = self.c1_net(x)    # (B, n_c1)
        c2_logits = self.c2_net(x)    # (B, n_c2)
        return dist, c1_logits, c2_logits
