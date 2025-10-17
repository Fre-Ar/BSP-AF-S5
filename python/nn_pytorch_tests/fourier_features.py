import math
import abc
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

# -----------------------------
# Utilities
# -----------------------------
def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """(B, D) enforced; accepts (D,) and adds batch dim."""
    return x.unsqueeze(0) if x.dim() == 1 else x

# -----------------------------
# Superclass
# -----------------------------
class EncodingBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for positional encoders mapping R^{B×D} -> R^{B×D'}.
    Subclasses must implement:
        - out_dim (property)
        - _encode(self, x_2d: Tensor) -> Tensor   # expects (B, D), returns (B, D')
    """
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
    """
    Basic Fourier encoding:
        γ(x) = [cos(2π s x), sin(2π s x)]
    where s is an optional input scale.

    Args:
        in_dim (int): dimensionality of x
        s (float): scales x (useful if x not in [0,1])
    """
    def __init__(self, in_dim: int, s: float = 1.0):
        super().__init__(in_dim)
        self.two_pi_s = 2.0 * math.pi * s
        self.register_buffer("two_pi_s", torch.tensor(self.two_pi_s))

    @property
    def out_dim(self) -> int:
        return 2 * self.in_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        arg = self.two_pi_s * x                    # (B, D)
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)  # (B, 2D)

# ----------------------------------------------------
# 2) Deterministic Positional Encodings
# ----------------------------------------------------
class PositionalEncoding(EncodingBase):
    """
    Deterministic Fourier features (Tancik et al., 2020):
        γ(x) = concat_{k=0}^{m-1} [ sin(α * ω^(k/m) * x), cos(α * ω^(k/m) * x) ]
    
    Args:
      - w (float): frequency hyperparameter
      - m (int): embedding size
      - α (float): Scaling factor. Use 1.0 for inputs in ~[-1,1]; use α = 2π for inputs in [0,1].
      - include_input (bool): If True, include x in the embedding.
    """
    def __init__(
        self,
        in_dim: int,
        m: int,
        w: float = 2.0,
        alpha: float = 1.0,
        include_input: bool = True,
    ):
        super().__init__(in_dim)
        self.m = m
        self.include_input = include_input

        base = torch.pow(torch.tensor(w), 1/m)
        self.bands = (alpha * (base ** torch.arange(self.m))).float()  # shape (m,)
        self.register_buffer("bands", self.bands)

    @property
    def out_dim(self) -> int:
        base = self.in_dim * (2 * self.m)
        return base + (self.in_dim if self.include_input else 0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        xb = x.unsqueeze(1) * self.bands.view(1, -1, 1)     # (B, m, D)
        enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=1)  # (B, 2m, D)
        enc = enc.reshape(x.shape[0], -1)                   # (B, 2mD)
        return torch.cat([x, enc], dim=-1) if self.include_input else enc

# ----------------------------------------------------
# 3) Random Fourier Features (Gaussian projection)
# ----------------------------------------------------
class RandomGaussianEncoding(EncodingBase):
    """
    Random Fourier Features (Tancik et al., 2020):
        Sample B ∈ R^{m×D},  B_ij ~ N(0, σ^2)
        γ(x) = [ cos( α x B^T), sin( α x B^T) ] / √m  (if l2_normalize=True) 

    Args
    ----
    in_dim : int
    m : int
        Embedding size aka Number of *(co)sines*. Total output is 2*m (+ D if include_input).
    sigma : float
        Std of Gaussian prior for B (controls frequency spread aka variance of normal distribution).
        Larger σ → higher-frequency features.
    include_input : bool
        Whether to concat raw x.

    Notes
    -----
    - Uses 2π by default (natural for inputs normalized to [0,1]).
      If your inputs are ~[-1,1], keeping 2π is still common; feel free to retune σ.
    - Optionally scales output by 1/√m to keep variance roughly constant.
    """
    def __init__(
        self,
        in_dim: int,
        m: int,
        alpha: float = 2.0 * math.pi,
        sigma: float = 1.0,
        learnable_B: bool = False,
        l2_normalize: bool = True,
    ):
        super().__init__(in_dim)
        self.in_dim = in_dim
        self.m = m
        self.alpha = alpha
        self.l2_normalize = bool(l2_normalize)

        B = sigma * torch.randn(self.m, self.in_dim)
        if learnable_B:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        self.scale = (1.0 / math.sqrt(self.m)) if (l2_normalize and self.m > 0) else 1.0
        self.register_buffer("scale", torch.tensor(float(self.scale)))

    @property
    def out_dim(self) -> int:
        return 2 * self.m

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        proj = (x @ self.B.t()) * self.alpha          # (B, m)
        feat = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (B, 2m)
        return self.scale * feat