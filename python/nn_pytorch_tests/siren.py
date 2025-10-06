import math
import torch
import torch.nn as nn

# ===================== SIREN CORE =====================

class Sine(nn.Module):
    def __init__(self, w0=1.0): 
        super().__init__()
        self.w0 = w0
    def forward(self, x): 
        return torch.sin(self.w0 * x)

class SIRENLayer(nn.Module):
    def __init__(self, in_dim, out_dim, w0=1.0, is_first=False, bias=True):
        super().__init__()
        self.in_dim, self.w0, self.is_first = in_dim, w0, is_first
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = Sine(w0)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_dim
            else:
                bound = math.sqrt(6.0 / self.in_dim) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x):
        return self.activation(self.linear(x))

class SIRENTrunk(nn.Module):
    """Shared trunk used by all heads."""
    def __init__(self, in_dim=3, hidden=256, depth=5, w0_first=30.0, w0_hidden=1.0):
        super().__init__()
        assert depth >= 2
        layers = [SIRENLayer(in_dim, hidden, w0=w0_first, is_first=True)]
        for _ in range(depth-2):
            layers += [SIRENLayer(hidden, hidden, w0=w0_hidden, is_first=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): 
        return self.net(x)

class MultiHeadSIREN(nn.Module):
    """
    Heads:
      - distance: regression (Softplus to enforce non-negativity)
      - c1: classification (containing country)
      - c2: classification (adjacent country)
    """
    def __init__(self, in_dim=3, hidden=256, depth=5, w0_first=30.0, w0_hidden=1.0,
                 num_c1=200, num_c2=200):
        super().__init__()
        self.trunk = SIRENTrunk(in_dim, hidden, depth, w0_first, w0_hidden)
        self.dist_head = nn.Linear(hidden, 1)
        self.c1_head   = nn.Linear(hidden, num_c1)
        self.c2_head   = nn.Linear(hidden, num_c2)
        nn.init.xavier_uniform_(self.dist_head.weight)
        nn.init.zeros_(self.dist_head.bias)
        nn.init.xavier_uniform_(self.c1_head.weight)
        nn.init.zeros_(self.c1_head.bias)
        nn.init.xavier_uniform_(self.c2_head.weight)
        nn.init.zeros_(self.c2_head.bias)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.trunk(x)              # (B, hidden)
        dist = self.softplus(self.dist_head(h))  # (B,1) >= 0
        c1_logits = self.c1_head(h)    # (B, num_c1)
        c2_logits = self.c2_head(h)    # (B, num_c2)
        return dist, c1_logits, c2_logits
