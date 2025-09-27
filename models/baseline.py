"""
Baseline perturbation prediction model.

Architecture:
1. Linear projection of batch onehot to d_h, expanded to [B, S, d_h]
2. MLP projection of ctrl_emb [B, S, E] to [B, S, d_h] 
3. MLP projection of pert [B, 5120] to [B, d_h], then expanded to [B, S, d_h]
4. Sum all three: H = H_ctrl + H_batch + H_pert
5. MLP projection from [B, S, d_h] to [B, S, E]
6. Bottleneck layer: up-down-up from E to d_f to G with ReLU activation
7. MSE loss: mean across S, then mean across batches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
import numpy as np

from .base import PerturbationModel, mmd2_rff_batched, energy_distance_batched
from .nn.mlp import MLP


class BaselineModel(PerturbationModel):
    """
    Baseline perturbation prediction model with simple MLP architecture.
    
    Architecture follows the specification:
    - Linear(n_batch, d_h) for batch embedding
    - MLP(E, d_h) for control cell embedding  
    - MLP(5120, d_h) for perturbation embedding
    - Combine: H = H_ctrl + H_batch + H_pert
    - MLP(d_h, E) for projection back to embedding space
    - Bottleneck: E -> d_f -> G with ReLU activation
    """
    
    def __init__(
        self,
        mean_target_map: Dict[str, torch.Tensor],
        mean_target_index_map: Dict[str, Dict[str, int]],
        d_h: int = 512,  # Hidden dimension
        d_f: int = 2048,  # Bottleneck dimension 
        n_genes: int = 18080,  # Number of genes (G)
        embedding_dim: int = 2058,  # Embedding dimension (E)
        pert_emb_dim: int = 5120,  # Perturbation embedding dimension
        n_cell_types: int = None,  # Number of cell types (will be set from data)
        n_batches: int = None,  # Number of batches (will be set from data)
        dropout: float = 0.2,
        hvg_mask: torch.Tensor = None,
        use_nb: bool = True,
        num_mc_samples: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_h = d_h
        self.d_f = d_f
        self.n_genes = n_genes
        self.embedding_dim = embedding_dim
        self.pert_emb_dim = pert_emb_dim
        self.n_cell_types = n_cell_types
        self.n_batches = n_batches
        self.hvg_mask = hvg_mask
        self.use_nb = use_nb
        self.num_mc_samples = num_mc_samples
        self.mean_target_map = mean_target_map
        self.mean_target_index_map = mean_target_index_map

        for key, value in mean_target_map.items():
            mean_target_map[key].to(self.device)

        # Save hyperparameters
        self.save_hyperparameters(ignore=["kwargs"])
        
        # Batch embedding layer (will be initialized when we know n_batches)
        self.batch_embedding = None
        
        # Cell type embedding layer (will be initialized when we know n_cell_types) 
        self.cell_type_embedding = None
        
        # Control cell encoder: MLP(E, d_h)
        self.ctrl_encoder = MLP(
            input_dim=embedding_dim,
            hidden_dim=d_h,
            output_dim=d_h,
            n_layers=3,
            dropout=dropout,
            activation="gelu"
        )
        
        # Perturbation encoder: MLP(5120, d_h)
        self.pert_encoder = MLP(
            input_dim=pert_emb_dim,
            hidden_dim=d_h,
            output_dim=d_h,
            n_layers=3,
            dropout=dropout,
            activation="gelu"
        )
        
        # Projection back to embedding space: MLP(d_h, E)
        self.projection_to_emb = MLP(
            input_dim=d_h,
            hidden_dim=d_h,
            output_dim=embedding_dim,
            n_layers=4,
            dropout=dropout,
            activation="gelu"
        )
        
        # Bottleneck layer: up-down-up from E to d_f to G
        self.bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, d_f),  # Up
            nn.GELU(),
            nn.Linear(d_f, n_genes),  # Down-up to genes
            nn.ReLU()
        )

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
    
    def _initialize_embeddings_from_dimensions(self):
        """Initialize embedding layers from saved dimensions (for checkpoint loading)."""
        if self.batch_embedding is None and self.n_batches is not None:
            self.batch_embedding = nn.Linear(self.n_batches, self.d_h)
            
        if self.cell_type_embedding is None and self.n_cell_types is not None:
            self.cell_type_embedding = nn.Linear(self.n_cell_types, self.d_h)
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle dynamic embedding layers.
        """
        # Initialize embedding layers if they exist in state_dict
        if 'batch_embedding.weight' in state_dict:
            batch_dim = state_dict['batch_embedding.weight'].shape[1]
            if self.batch_embedding is None:
                self.batch_embedding = nn.Linear(batch_dim, self.d_h)
                self.n_batches = batch_dim
                
        if 'cell_type_embedding.weight' in state_dict:
            cell_type_dim = state_dict['cell_type_embedding.weight'].shape[1]
            if self.cell_type_embedding is None:
                self.cell_type_embedding = nn.Linear(cell_type_dim, self.d_h)
                self.n_cell_types = cell_type_dim
        
        # Call parent load_state_dict
        return super().load_state_dict(state_dict, strict=strict)

    def calculate_loss_centroid(self, pred, cell_types: np.ndarray, pert_names: np.ndarray):
        """
        Calculate the distance between the control cell embeddings and the other targets.
        """
        B, S, G = pred.shape
        # mean over S
        pred = pred.mean(dim=1) # [B, G]
        all_target_means = [self.mean_target_map[cell_type] for cell_type in cell_types] # list of shape [n_targets, G]
        all_target_means = torch.stack(all_target_means, dim=0) # [all_targets, G]
        all_targets = [[f"{x}@{cell_type}" for x in self.mean_target_index_map[cell_type].keys()] for cell_type in cell_types] # list of shape [n_targets]
        all_targets = np.concatenate(all_targets)
        pert_addresses = pert_names + "@" + cell_types
        target_mask = torch.tensor(np.isin(all_targets, pert_addresses))
        non_target_mask = ~target_mask
        target_means = all_target_means[target_mask] # [B, G]
        other_target_means = all_target_means[non_target_mask] # [B, G]
        loss_same_target = (1 -F.cosine_similarity(pred, target_means, dim=1)).mean() # scalar, we want to minimize this
        x_expand = pred[:, None, :] # [B, 1, G]
        y_expand = other_target_means[None, :, :] # [1, all_targets, G] 
        loss_other_targets = F.cosine_similarity(x_expand, y_expand, dim=-1).mean(dim=1).mean()

        return loss_same_target, loss_other_targets

    def forward(self, ctrl_cell_emb, pert_emb, covariates):
        """
        Forward pass implementing the baseline architecture.
        
        Args:
            ctrl_cell_emb: Control cell embeddings [B, S, E]
            pert_emb: Perturbation embeddings [B, 5120]
            covariates: Dictionary containing batch_onehot [B, n_batch] and cell_type_onehot
            
        Returns:
            Predicted perturbation effects [B, S, G]
        """
        B, S, E = ctrl_cell_emb.shape
        print(f"B: {B}, S: {S}, E: {E}")
        
        # Initialize embeddings if needed
        self._initialize_embeddings_if_needed(covariates)
        
        # 1. Encode control cells: [B, S, E] -> [B, S, d_h]
        H_ctrl = self.ctrl_encoder(ctrl_cell_emb)  # [B, S, d_h]
        
        # 2. Encode perturbations: [B, 5120] -> [B, d_h] -> [B, S, d_h]
        H_pert = self.pert_encoder(pert_emb)  # [B, d_h]
        H_pert = H_pert.unsqueeze(1).expand(-1, S, -1)  # [B, S, d_h]
        
        # 3. Encode batch information: batch_onehot [B, n_batch] -> [B, d_h] -> [B, S, d_h]
        H_batch = torch.zeros_like(H_ctrl)  # Initialize [B, S, d_h]
        
        if "batch_onehot" in covariates and self.batch_embedding is not None:
            # Note: batch_onehot is [B, S, n_batch] not [B, n_batch] as initially specified
            # We'll handle both cases
            batch_onehot = covariates["batch_onehot"]
            if batch_onehot.dim() == 2:  # [B, n_batch] case
                batch_emb = self.batch_embedding(batch_onehot)  # [B, d_h]
                H_batch += batch_emb.unsqueeze(1).expand(-1, S, -1)  # [B, S, d_h]
            else:  # [B, S, n_batch] case
                H_batch += self.batch_embedding(batch_onehot)  # [B, S, d_h]
        
        # Add cell type information if available
        if "cell_type_onehot" in covariates and self.cell_type_embedding is not None:
            H_batch += self.cell_type_embedding(covariates["cell_type_onehot"])
        
        # 4. Combine encodings: H = H_ctrl + H_batch + H_pert
        H = H_ctrl + H_batch + H_pert  # [B, S, d_h]
        
        # 5. Project back to embedding space: MLP(d_h, E)
        emb_output = self.projection_to_emb(H)  # [B, S, E]
        
        # 6. Bottleneck layer: up-down-up from E to d_f to G with ReLU
        gene_output = self.bottleneck(emb_output)  # [B, S, G]
        
        return gene_output

    def training_step(self, batch, batch_idx):
        """Training step supporting Negative Binomial likelihood on gene counts."""
        pert_emb = batch["pert_emb"]
        ctrl_cell_emb = batch["ctrl_cell_g"]
        pert_cell_data = batch["pert_cell_g"]
        breakpoint()
        ctrl_cell_emb_hvg = ctrl_cell_emb[:, :, self.hvg_mask]
        print(f"ctrl_cell_emb_hvg: {ctrl_cell_emb_hvg.shape}")
        covariates = {
            "cell_type_onehot": batch["cell_type_onehot"],
            "batch_onehot": batch["batch_onehot"],
        }
        
        cell_type = batch["cell_type"]
        pert_names = batch["pert_name"]
        pred = self.forward(ctrl_cell_emb_hvg, pert_emb, covariates)
        B = pred.shape[0]

        loss_same_target, loss_other_targets = self.calculate_loss_centroid(pred, cell_type, pert_names)
        loss_centroid = 0.8 * loss_same_target + 0.2 * loss_other_targets
        energy_distance = energy_distance_batched(pred, pert_cell_data)
        mse_loss = F.mse_loss(pred, pert_cell_data)
        loss = 0.3 * energy_distance + 0.3 * mse_loss + 0.3 * loss_centroid
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train_ed", energy_distance, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train_mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log("train_loss_centroid", loss_centroid, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step supporting Negative Binomial likelihood on gene counts."""
        pert_cell_data = batch["pert_cell_g"]
        ctrl_cell_emb = batch["ctrl_cell_g"]
        
        pert_emb = batch["pert_emb"]
        
        covariates = {
            "cell_type_onehot": batch["cell_type_onehot"],
            "batch_onehot": batch["batch_onehot"],
        }
        
        pred = self.forward(ctrl_cell_emb, pert_emb, covariates)
        B = pred.shape[0]
        cell_type = batch["cell_type"]
        pert_names = batch["pert_name"]
        loss_same_target, loss_other_targets = self.calculate_loss_centroid(pred, cell_type, pert_names)
        loss_centroid = 0.8 * loss_same_target + 0.2 * loss_other_targets
        
        energy_distance = energy_distance_batched(pred, pert_cell_data)
        mse_loss = F.mse_loss(pred, pert_cell_data)
        loss = 0.3 * energy_distance + 0.3 * mse_loss + 0.3 * loss_centroid
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val_ed", energy_distance, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=B)
        self.log("val_loss_centroid", loss_centroid, on_step=False, on_epoch=True, prog_bar=False, batch_size=B)
        return loss