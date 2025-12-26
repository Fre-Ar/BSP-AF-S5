import math
import torch
import torch.nn as nn



# ---------------------------
# SIREN-style initialization
# ---------------------------
@torch.no_grad
def init_siren_linear(m: nn.Module, ith_layer: int, params: tuple) -> None:
    '''Weight/bias init per SIREN (Sitzmann et al., 2020).'''
    w = params[0]
    if not isinstance(m, nn.Linear):
        return
    if ith_layer == 0:
        bound = 1.0 / m.in_features
    else:
        bound = math.sqrt(6.0 / m.in_features) / (w if w != 0 else 1.0)
    m.weight.uniform_(-bound, bound)
    # biases are left at default

# ---------------------------
# Default initialization
# ---------------------------
@torch.no_grad
def init_linear(m: nn.Module, ith_layer: int = 0, params: tuple = ()):
    if not isinstance(m, nn.Linear):
        return
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

# ---------------------------
# Reset params initialization
# ---------------------------
@torch.no_grad    
def init_reset(m: nn.Module, ith_layer: int = 0, params: tuple = ()):
    if not isinstance(m, nn.Linear):
        return
    nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
    if m.bias is not None:
        # PyTorch Linear default with kaiming_uniform_: U(-1/sqrt(in_dim), +1/sqrt(in_dim))
        bound = 1.0 / math.sqrt(m.in_features) if m.in_features > 0 else 0.0
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
    bound = torch.sqrt(6.0 / m.in_features) / (w if w != 0 else 1.0)
    m.weight.uniform_(-bound, bound)

    # ---- FINER bias init: b ~ U(-k, k) with k > 0 ----
    m.bias.uniform_(-k, k) 
