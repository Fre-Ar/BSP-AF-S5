import torch
import torch.nn as nn
from nir import NIRLayer

class Hosc(nn.Module):
    def __init__(self, beta=1.0, adaptive = False): 
        super().__init__()
        self.adaptive = adaptive
        if adaptive:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.beta = beta
        
    def forward(self, x): 
        return torch.tanh(self.beta * torch.sin(x))

class HOSCLayer(NIRLayer):
    """
    HOSC (Serrano et al., 2024):
    Params:
        beta (float): sharpness (>0), fixed (HOSC baseline).
    Args:
        adaptive (bool): If true, makes beta a learneable parameter.
    
    Notes:
      - Per the paper, no special 'frequency init' or PE is needed.
      - We use Xavier-normal for weights and zero bias.
    """
    def __init__(self, in_dim: int, out_dim: int, params: tuple, ith_layer: int, bias=True, adaptive = False):
        self.beta = params[0]
        if not adaptive:
            self.register_buffer(f"beta{ith_layer}", torch.tensor(float(self.beta)))
        super().__init__(Hosc(self.beta), in_dim, out_dim, ith_layer, bias, adaptive)

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier/Glorot normal; bias = 0
        nn.init.xavier_normal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)