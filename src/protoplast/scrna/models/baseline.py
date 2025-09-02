import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselinePerturbModel(nn.Module):
    def __init__(self, G, n_cell_lines, n_targets, d_y=64, d_xp=128, hidden_dims=[256, 512]):
        """
        G: number of genes
        n_cell_lines: number of cell line labels
        n_targets: number of perturbation targets (genes + control)
        """
        super().__init__()
        self.y_proj = nn.Linear(n_cell_lines, d_y)
        self.xp_proj = nn.Linear(n_targets, d_xp)
        
        # Control predictor: just based on cell line
        self.ctrl_net = nn.Sequential(
            nn.Linear(d_y, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], G)
        )
        
        # Δ predictor: based on cell line + perturbation
        self.delta_net = nn.Sequential(
            nn.Linear(d_y + d_xp, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], G)
        )
    
    def forward(self, y, xp):
        """
        y: [B] tensor of cell line indices
        xp: [B] tensor of perturbation indices (0=control)
        """

        if xp.ndim > 2:
            xp = xp.squeeze(1)

        e_y = self.y_proj(y)
        e_xp = self.xp_proj(xp)

        # Control prediction
        x_ctrl = self.ctrl_net(e_y)
        
        # Δ prediction
        cond = torch.cat([e_y, e_xp], dim=-1)
        delta = self.delta_net(cond)
        
        # Perturbed prediction
        x_pred = x_ctrl + delta
        return x_ctrl, delta, x_pred