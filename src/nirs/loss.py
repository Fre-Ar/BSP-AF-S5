# src/nirs/loss.py

import torch.nn as nn, torch

class UncertaintyWeighting(nn.Module):
    def __init__(self, init_log_vars=(0.0, 0.0, 0.0)):
        super().__init__()
        self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))

    def forward(self, losses):  # losses: [L_dist, L_c1, L_c2]
        total = 0.0
        for i, L in enumerate(losses):
            lv = self.log_vars[i]
            total = total + torch.exp(-lv) * L * 0.5 + 0.5 * lv
        return total
