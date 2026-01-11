import math
import torch
import torch.nn as nn
import numpy as np



# ---------------------------
# SIREN-style initialization
# ---------------------------
@torch.no_grad
def init_siren_linear(m: nn.Module, ith_layer: int, params: tuple) -> None:
    '''Weight/bias init per SIREN (Sitzmann et al., 2020).'''
    w0 = params[0]
    if not isinstance(m, nn.Linear):
        return
    if ith_layer == 0:
        bound = 1.0 / m.in_features
    else:
        bound = np.sqrt(6.0 / m.in_features) / (w0 if w0 != 0 else 1.0)
    m.weight.uniform_(-bound, bound)
    # biases are left at default

# ---------------------------
# Default initialization
# ---------------------------
@torch.no_grad
def init_linear(m: nn.Module, ith_layer: int = 0, params: tuple = ()):
    if not isinstance(m, nn.Linear):
        return
    if m.bias is not None:
        nn.init.zeros_(m.bias)
        
# ---------------------------
# No special initialization
# ---------------------------
@torch.no_grad
def init_none(m: nn.Module, ith_layer: int = 0, params: tuple = ()):
    return

# ---------------------------
# Reset params initialization
# ---------------------------
@torch.no_grad    
def init_reset(m: nn.Module, ith_layer: int = 0, params: tuple = ()):
    if not isinstance(m, nn.Linear):
        return
    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    if m.bias is not None:
        # PyTorch Linear default with kaiming_uniform_: U(-1/sqrt(in_dim), +1/sqrt(in_dim))
        bound = 1.0 / np.sqrt(m.in_features) if m.in_features > 0 else 0.0
        nn.init.uniform_(m.bias, -bound, bound)


# ---------------------------
# FINER-style initialization
# ---------------------------
@torch.no_grad    
def init_finer_linear(m: nn.Module, ith_layer: int=0, params: tuple=()):
    w = params[0]
    k = params[1]
    if not isinstance(m, nn.Linear):
        return
    # ---- SIREN-style weight init ----
    bound = np.sqrt(6.0 / m.in_features) / (w if w != 0 else 1.0)
    m.weight.uniform_(-bound, bound)

    # ---- FINER bias init: b ~ U(-k, k) with k > 0 ----
    m.bias.uniform_(-k, k) 

# ---------------------------
# MFN-style initialization
# ---------------------------
@torch.no_grad
def init_mfn_linear(m: nn.Module, ith_layer: int, params: tuple) -> None:
    if not isinstance(m, nn.Linear):
        return
    weight_scale = params[0]
    
    bound = np.sqrt(weight_scale / m.in_features)
    m.weight.uniform_(-bound, bound)
    # biases are left default

@torch.no_grad
def init_mfn_filter(m: nn.Module, ith_layer: int, params: tuple) -> None:
    '''sqrt_gamma = 1.0 for Fourier Filter, √(γ) for Gabor Filter.'''
    if not isinstance(m, nn.Linear):
        return
    depth = params[1]
    sqrt_gamma = params[2]
    init_scale = m.out_features / np.sqrt(depth)
    m.weight.mul_(init_scale * sqrt_gamma)
    
    bound = torch.pi
    m.bias.uniform_(-bound, bound)
