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

from .nn.mlp import MLP, MaskNet


class PerturbationTransformerModel(PerturbationModel):
    """
    Transformer-based perturbation prediction model.
    
    Architecture:
    1. Encode control cells, perturbations, and batch information
    2. Combine encodings: H_x = H_ctrl + H_pert + H_batch
    3. Apply transformer to learn residual effects: O = H + ft(H)
    4. Project to gene space via down-up projection
    """
    
    def __init__(
        self,
        d_h: int = 672,  # Hidden dimension
        n_genes: int = 18080,  # Number of genes (G)
        pert_emb_dim: int = 5120,  # Perturbation embedding dimension
        n_cell_types: int = None,  # Number of cell types (will be set from data)
        n_batches: int = None,  # Number of batches (will be set from data)
        n_transformer_layers: int = 12,
        n_heads: int = 16,
        dropout: float = 0.1,
        d_ff: int = 2048,  # Feed-forward dimension in transformer
        d_x: int = 2260,  # Bottleneck dimension for final projection
        mmd_kernel: str = "energy",
        mmd_blur: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.d_h = d_h
        self.n_genes = n_genes
        self.pert_emb_dim = pert_emb_dim
        self.n_cell_types = n_cell_types
        self.n_batches = n_batches
        self.d_x = d_x
        
        # Save hyperparameters
        # self.save_hyperparameters()
        
        # Perturbation encoder (as specified)
        self.pert_encoder = MLP(pert_emb_dim, d_h, d_h, dropout=dropout, n_layers=4)
        
        # Basal encoder for control cells
        self.basal_encoder = MLP(n_genes, d_h, d_h, dropout=dropout, n_layers=4)
        
        # Batch embedding layer (will be initialized when we know n_batches)
        self.batch_embedding = None
        
        # Cell type embedding layer (will be initialized when we know n_cell_types)
        self.cell_type_embedding = None
        
        # Transformer backbone using utility function
        transformer_kwargs = {
            "hidden_size": d_h,
            "num_hidden_layers": n_transformer_layers,
            "num_attention_heads": n_heads,
            "head_dim": d_h // n_heads,
            "intermediate_size": d_ff,
            "dropout": dropout,
            "max_position_embeddings": 2048,  # Max sequence length
            "rms_norm_eps": 1e-6,
        }
        self.transformer, self.model_dim = get_transformer_backbone("llama", transformer_kwargs)
        
        # Projection layers (as specified)
        self.project_out = MLP(d_h, d_h, n_genes, dropout=dropout, n_layers=4)

        # build mlp for final down-then-up projection
        self.final_down_then_up = MLP(n_genes, d_x, n_genes, dropout=dropout, n_layers=4)
        
        # MMD loss
        self.mmd_loss = SamplesLoss(loss=mmd_kernel, blur=mmd_blur)

    def _sparsity_loss(self, pred, target) -> torch.Tensor:
        """
        Compute sparsity loss for the output.
        """
        # Sparsity preservation
        target_sparse = (target < 0.1).float()  # Near-zero in log1p space
        pred_sparse = (pred < 0.1).float()
    
        sparsity_loss = F.mse_loss(pred_sparse.mean(dim=-1), target_sparse.mean(dim=-1))
        return sparsity_loss.nanmean()
    
    def _initialize_embeddings_if_needed(self, covariates: Dict[str, torch.Tensor]):
        """Initialize embedding layers based on covariate dimensions."""
        if self.batch_embedding is None and "batch_onehot" in covariates:
            batch_dim = covariates["batch_onehot"].shape[-1]
            self.batch_embedding = nn.Linear(batch_dim, self.d_h).to(self.device)
            self.n_batches = batch_dim
            
        if self.cell_type_embedding is None and "cell_type_onehot" in covariates:
            cell_type_dim = covariates["cell_type_onehot"].shape[-1]
            self.cell_type_embedding = nn.Linear(cell_type_dim, self.d_h).to(self.device)
            self.n_cell_types = cell_type_dim
    
    def forward(self, ctrl_cell_emb, pert_emb, covariates):
        """
        Forward pass implementing H_x = H_ctrl + H_pert + H_batch, then O = H + ft(H).
        
        Args:
            ctrl_cell_emb: Control cell embeddings [B, S, G]
            pert_emb: Perturbation embeddings [B, D]
            covariates: Dictionary containing cell_type_onehot [B, S, C] and batch_onehot [B, S, Bx]
            
        Returns:
            Predicted perturbation effects [B, S, G]
        """
        B, S, G = ctrl_cell_emb.shape
        
        # Initialize embeddings if needed
        self._initialize_embeddings_if_needed(covariates)
        
        # Encode control cells: [B, S, G] -> [B, S, d_h]
        H_ctrl = self.basal_encoder(ctrl_cell_emb)  # [B, S, d_h]
        
        # Encode perturbations: [B, D] -> [B, d_h] -> [B, S, d_h]
        H_pert = self.pert_encoder(pert_emb)  # [B, d_h]
        H_pert = H_pert.unsqueeze(1).expand(-1, S, -1)  # [B, S, d_h]
        
        # Encode batch information: [B, S, Bx] -> [B, S, d_h]
        H_batch = torch.zeros_like(H_ctrl)  # Initialize
        if "batch_onehot" in covariates and self.batch_embedding is not None:
            H_batch += self.batch_embedding(covariates["batch_onehot"])
        
        # Encode cell type information: [B, S, C] -> [B, S, d_h]
        if "cell_type_onehot" in covariates and self.cell_type_embedding is not None:
            H_batch += self.cell_type_embedding(covariates["cell_type_onehot"])
        
        # Combine encodings: H_x = H_ctrl + H_pert + H_batch
        H = H_ctrl + H_pert + H_batch  # [B, S, d_h]
        
        # Apply Llama transformer to learn residual effects
        # H is already in the right shape [B, S, d_h] for Llama
        # Create cache_position for the sequence
        cache_position = torch.arange(H.size(1), device=H.device)
        transformer_output = self.transformer(
            inputs_embeds=H, 
            cache_position=cache_position
        )  # [B, S, d_h]
        ft_H = transformer_output.last_hidden_state  # Extract hidden states
        
        # Compute final representation: O = H + ft(H)
        O = H + ft_H  # [B, S, d_h]
        
        # Project to gene space
        gene_proj = self.project_out(O)  # [B, S, G]
        X_pert_hat = self.final_down_then_up(gene_proj)  # [B, S, G]
        # clip X_pert_hat to be non-negative
        X_pert_hat = torch.clamp(X_pert_hat, min=0)
        
        return X_pert_hat
    
    def training_step(self, batch, batch_idx):
        """Training step with MMD loss."""
        # Unpack batch
        pert_cell_emb, ctrl_cell_emb, pert_emb, covariates = self.unpack_batch(batch)
        
        # Forward pass
        X_pert_hat = self.forward(ctrl_cell_emb, pert_emb, covariates)
        
        # Compute MMD loss between predicted and actual perturbed cells
        # Reshape for loss computation: [B, S, G] -> [B*S, G]
        B, S, G = X_pert_hat.shape
        X_pert_hat = X_pert_hat.reshape(-1, S, G).contiguous()
        pert_cell_emb = pert_cell_emb.reshape(-1, S, G).contiguous()
        
        # MMD loss
        dist_loss = self.mmd_loss(X_pert_hat, pert_cell_emb).nanmean()
        sparsity_loss = self._sparsity_loss(X_pert_hat, pert_cell_emb)
        # pool pert_cell_emb and X_pert_hat to [B, G]
        pert_cell_emb = pert_cell_emb.mean(dim=1)
        X_pert_hat = X_pert_hat.mean(dim=1)
        # compute mse loss
        mse_loss = F.mse_loss(X_pert_hat, pert_cell_emb)
        loss = dist_loss + sparsity_loss + mse_loss
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train_dist_loss", dist_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train_sparsity_loss", sparsity_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train_mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Unpack batch
        pert_cell_emb, ctrl_cell_emb, pert_emb, covariates = self.unpack_batch(batch)
        
        # Forward pass
        X_pert_hat = self.forward(ctrl_cell_emb, pert_emb, covariates)
        
        # Compute MMD loss
        B, S, G = X_pert_hat.shape
        X_pert_hat = X_pert_hat.reshape(-1, S, G).contiguous()
        pert_cell_emb = pert_cell_emb.reshape(-1, S, G).contiguous()
        
        dist_loss = self.mmd_loss(X_pert_hat, pert_cell_emb).nanmean()
        sparsity_loss = self._sparsity_loss(X_pert_hat, pert_cell_emb)
        # pool pert_cell_emb and X_pert_hat to [B, G]
        pert_cell_emb = pert_cell_emb.mean(dim=1)
        X_pert_hat = X_pert_hat.mean(dim=1)
        # compute mse loss
        mse_loss = F.mse_loss(X_pert_hat, pert_cell_emb)
        # pearson correlation loss
        val_loss = dist_loss + sparsity_loss + mse_loss
        
        # Log metrics
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val_dist_loss", dist_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val_sparsity_loss", sparsity_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        
        # Additional metrics could be added here (correlation, etc.)
        
        return val_loss
    
    def configure_optimizers(self):
        """Configure optimizer with custom settings for transformer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        if self.lr_scheduler_freq is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'frequency': self.lr_scheduler_freq,
                    'interval': self.lr_scheduler_interval
                }
            }
        
        return optimizer
