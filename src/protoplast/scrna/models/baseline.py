import torch
import torch.nn as nn
import torch.nn.functional as F
from protoplast.scrna.train.utils import _to_BD

class BaselinePerturbModel(nn.Module):
    def __init__(self, G, n_cell_lines, n_targets, d_y=64, d_xp=128, hidden_dims=(256, 512)):
        super().__init__()
        self.G = G
        self.y_proj = nn.Linear(n_cell_lines, d_y)
        self.xp_proj = nn.Linear(n_targets, d_xp)

        self.ctrl_net = nn.Sequential(
            nn.Linear(d_y, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], G)
        )
        self.delta_net = nn.Sequential(
            nn.Linear(d_y + d_xp, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], G)
        )

    def forward(self, y, xp):
        """
        y: (B, n_cell_lines) one-hot or dense features
        xp: (B, n_targets)    one-hot (incl. a 'control' class) or dense features
        """
        y  = _to_BD(y).to(dtype=torch.float32)
        xp = _to_BD(xp).to(dtype=torch.float32)

        e_y  = self.y_proj(y)        # (B, d_y)
        e_xp = self.xp_proj(xp)      # (B, d_xp)

        x_ctrl = self.ctrl_net(e_y)  # (B, G)
        cond   = torch.cat([e_y, e_xp], dim=-1)
        delta  = self.delta_net(cond)  # (B, G)
        x_pred = x_ctrl + delta        # (B, G)
        return x_ctrl.contiguous(), delta.contiguous(), x_pred.contiguous()
