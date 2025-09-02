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
        self.cell_embed = nn.Embedding(n_cell_lines, d_y)
        self.xp_embed = nn.Embedding(n_targets, d_xp)
        
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
        e_y = self.cell_embed(y)
        e_xp = self.xp_embed(xp)
        
        # Control prediction
        x_ctrl = self.ctrl_net(e_y)
        
        # Δ prediction
        cond = torch.cat([e_y, e_xp], dim=-1)
        delta = self.delta_net(cond)
        
        # Perturbed prediction
        x_pred = x_ctrl + delta
        return x_ctrl, delta, x_pred