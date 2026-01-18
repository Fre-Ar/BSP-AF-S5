# src/nirs/nns/fourier_features.py

import math
import abc
import torch
import torch.nn as nn

# -----------------------------
# Utilities
# -----------------------------
def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    '''(B, D) enforced; accepts (D,) and adds batch dim.'''
    return x.unsqueeze(0) if x.dim() == 1 else x

# -----------------------------
# Superclass
# -----------------------------
class EncodingBase(nn.Module, metaclass=abc.ABCMeta):
    '''
    Base class for positional encoders mapping R^{B×D} -> R^{B×D'}.
    Subclasses must implement:
        - out_dim (property)
        - _encode(self, x_2d: Tensor) -> Tensor   # expects (B, D), returns (B, D')
    '''
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)

    @property
    @abc.abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def _encode(self, x_2d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(_ensure_2d(x))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim})"

# -----------------------------
# 1) Basic encoding
# -----------------------------
class BasicEncoding(EncodingBase):
    '''
    Basic Fourier encoding:
        α = 2π
        γ(x) = [cos(α * x), sin(α * x)]
    where s is an optional input scale.

    Args:
        in_dim (int): dimensionality of x
        a (float): 2π by default, scales x (useful if x not in [0,1])
    '''
    def __init__(
        self,
        in_dim: int,
        alpha: float = 2.0 * math.pi,
        sigma: float = 5.0, # sigma and m included for portability with the other 2 encodings
        m: int = 256
    ):
        super().__init__(in_dim)
        self.alpha = alpha

    @property
    def out_dim(self) -> int:
        return 2 * self.in_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        arg = self.alpha * x                    # (B, D)
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)  # (B, 2D)

# ----------------------------------------------------
# 2) Deterministic Positional Encodings
# ----------------------------------------------------
class PositionalEncoding(EncodingBase):
    '''
    Deterministic Fourier features (Tancik et al., 2020):
        α = 2π
        γ(x) = [x, …, cos(α * σ^(j/m)x),sin(α * σ^(j/m)x), …] for j∈{0,…,m-1}

    Args
    ----
    in_dim : int
        Dimensionality of x.
    alpha : float
        Scaling factor α. Use 1.0 for inputs in ~[-1,1]; use α = 2π for inputs in [0,1].
    sigma : float
        Frequency hyperparameter σ.
    m : int
        Embedding size aka Number of *(co)sines*. Total output size is 2mD (+D if include_input=True).
    include_input : bool
        Whether to prepend raw x.
    '''
    def __init__(
        self,
        in_dim: int,
        alpha: float = 2.0 * math.pi,
        sigma: float = 5.0,
        m: int = 256,
        include_input: bool = False,
    ):
        super().__init__(in_dim)
        self.m = m
        self.include_input = include_input

        base = torch.pow(torch.tensor(sigma), 1/m)
        bands = (alpha * (base ** torch.arange(self.m))).float()  # shape (m,)
        self.register_buffer("bands", bands) 
        
    @property
    def out_dim(self) -> int:
        base = self.in_dim * (2 * self.m)
        return base + (self.in_dim if self.include_input else 0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        xb = x.unsqueeze(1) * self.bands.view(1, -1, 1)     # (B, m, D)
        enc = torch.cat([torch.cos(xb), torch.sin(xb)], dim=1)  # (B, 2m, D)
        enc = enc.reshape(x.shape[0], -1)                   # (B, 2mD)
        return torch.cat([x, enc], dim=-1) if self.include_input else enc

# ----------------------------------------------------
# 3) Random Fourier Features (Gaussian projection)
# ----------------------------------------------------
class RandomGaussianEncoding(EncodingBase):
    '''
    Random Fourier Features (Tancik et al., 2020):
        α = 2π
        γ(x) = [cos(α * Bx), sin(α * Bx)] w/ B ∈ R^{m×D} ~ N(0, σ^2) fixed
        
    Args
    ----
    in_dim : int
        Dimensionality of x.
    alpha : float
        Scaling factor α. Use 1.0 for inputs in ~[-1,1]; use α = 2π for inputs in [0,1].
    sigma : float
        Std of Gaussian prior for B (controls frequency spread aka variance of normal distribution).
        Larger σ → higher-frequency features.
    m : int
        Embedding size aka Number of *(co)sines*. Total output size is 2m.
    l2_normalize : bool
        True by default. Whether to scale output by 1/√m to keep variance roughly constant.
    '''
    def __init__(
        self,
        in_dim: int,
        alpha: float = 2.0 * math.pi,
        sigma: float = 10.0,
        m: int = 256,
        l2_normalize: bool = True
    ):
        super().__init__(in_dim)
        self.in_dim = in_dim
        self.m = m
        self.register_buffer("alpha", torch.tensor(float(alpha)))

        B = sigma * torch.randn(self.m, self.in_dim)
        self.register_buffer("B", B)

        scale = (1.0 / math.sqrt(self.m)) if (l2_normalize and self.m > 0) else 1.0
        self.register_buffer("scale", torch.tensor(scale))
        
    @property
    def out_dim(self) -> int:
        return 2 * self.m

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        proj = (x @ self.B.t()) * self.alpha          # (B, m)
        feat = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (B, 2m)
        return self.scale * feat