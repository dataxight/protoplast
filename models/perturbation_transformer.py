"""
Perturbation Transformer Model for predicting perturbation effects.

This model implements the architecture:
H_x = H_ctrl + H_pert + H_batch
O = H + ft(H)  # where ft is a transformer layer
X_pert_hat = projection(O)
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Any, Dict
from geomloss import SamplesLoss

from .base import PerturbationModel
from .llama_components import LlamaModel
from .nn.utils import get_transformer_backbone
from .baseline import BaselineModel
from .nn.mlp import MLP, MaskNet

class ResidualEncoder(nn.Module):
    """
    Residual encoder.
    """
    
    def __init__(self, d_h: int = 672, n_transformer_layers: int = 12, n_heads: int = 16, d_ff: int = 2048, dropout: float = 0.1, n_genes: int = 18080):
        super().__init__()
        transformer_kwargs = {
            "hidden_size": d_h,
            "num_hidden_layers": n_transformer_layers,
            "num_attention_heads": n_heads,
            "head_dim": d_h // n_heads,
            "intermediate_size": d_ff,
            "dropout": dropout,
            "max_position_embeddings": 1024,  # Max sequence length
            "rms_norm_eps": 1e-6,
        }
        self.transformer, self.model_dim = get_transformer_backbone("llama", transformer_kwargs, input_dim=d_h)

    def forward(self, H):
        cache_position = torch.arange(H.size(1), device=H.device)
        transformer_output = self.transformer(
            inputs_embeds=H, 
            cache_position=cache_position
        )
        return transformer_output.last_hidden_state

class PerturbationTransformer(BaselineModel):
    """
    Transformer-based perturbation prediction model.
    """
    
    def __init__(self, n_transformer_layers: int = 12, n_heads: int = 16, d_ff: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.residual_encoder_transformer = ResidualEncoder(
            d_h=self.d_h,
            n_transformer_layers=n_transformer_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=self.dropout,
            n_genes=self.n_genes,
        )

    def _refine_H(self, H):
        return H + self.residual_encoder_transformer(H)
    
    def forward(self, ctrl_cell_emb, pert_emb, covariates):
        """
        Forward pass implementing H_x = H_ctrl + H_pert + H_batch, then O = H + ft(H).
        """
        return super().forward(ctrl_cell_emb, pert_emb, covariates)