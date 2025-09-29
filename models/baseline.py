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
from typing import Any, Dict, List
import numpy as np

from .base import PerturbationModel, mmd2_rff_batched, energy_distance_batched
from .nn.mlp import MLP
from geomloss import SamplesLoss


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
        mean_target_map: torch.Tensor,
        mean_target_addresses: List[str],
        d_h: int = 512,  # Hidden dimension
        d_f: int = 256,  # Bottleneck dimension 
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
        self.mean_target_addresses = mean_target_addresses

        # mean_target_map.to(self.device)

        # Save hyperparameters
        # self.save_hyperparameters(ignore=["kwargs"])
        
        # Batch embedding layer (will be initialized when we know n_batches)
        self.batch_embedding = None
        
        # Cell type embedding layer (will be initialized when we know n_cell_types) 
        self.cell_type_embedding = None
        
        # Control cell encoder: MLP(E, d_h)
        self.ctrl_encoder = MLP(
            input_dim=embedding_dim,
            hidden_dim=d_h,
            output_dim=d_h,
            n_layers=4,
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

        self.residual_encoder = MLP(
            input_dim=d_h,
            hidden_dim=d_h,
            output_dim=d_h,
            n_layers=2,
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

        self.norm = nn.LayerNorm(d_h)

        self.ot_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)

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

    def calculate_loss_centroid(self, pred, pert_names: np.ndarray):
        """
        Calculate the distance between the control cell embeddings and the other targets.
        """
        B, S, G = pred.shape
        # mean over S
        pred = pred.mean(dim=1) # [B, G]
        pert_addresses = pert_names # + "@" + cell_types
        # print(f"pert_addresses: {pert_addresses.shape}, {pert_addresses}")
        mask = torch.tensor([np.where(np.strings.find(self.mean_target_addresses, s) != -1)[0][0] for s in pert_addresses], device=self.device)
        # print(f"mask: {mask.shape}, {mask}")
        self.mean_target_map = self.mean_target_map.to(self.device)
        target_means = self.mean_target_map[mask, :]
        # print(f"target_means: {target_means.shape}, {target_means}")
        l1_same_target = F.l1_loss(pred, target_means, reduction="none")
        # print(f"l1_same_target: {l1_same_target.shape}, {l1_same_target}")
        distance_same_target = l1_same_target.sum(dim=1)
        # random 100 perturbations in the list of pert_addresses
        random_indices = range(150) # 150 random perturbations
        random_target_means = self.mean_target_map[random_indices, :]
        # print(f"distance_same_target: {distance_same_target}") # [B]
        l1_other_targets = F.l1_loss(pred[:, None, :], random_target_means[None, :, :], reduction="none")
        distance_other_targets = l1_other_targets.sum(dim=-1) # [B, all_targets]
        diff_other_targets = distance_other_targets - distance_same_target[:, None] # [B, all_targets] # we want to maximize this
        sigmoid_diff_other_targets = torch.sigmoid(-diff_other_targets) # set as negative because we want to maximize
        loss_pds = sigmoid_diff_other_targets.mean(dim=1).mean() # scalar
        return loss_pds 

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
        H = H + self.residual_encoder(self.norm(H))
        
        # 5. Project back to embedding space: MLP(d_h, E)
        emb_output = self.projection_to_emb(H)  # [B, S, E]
        
        # 6. Bottleneck layer: up-down-up from E to d_f to G with ReLU
        gene_output = self.bottleneck(emb_output)  # [B, S, G]
        
        return gene_output, emb_output

    def training_step(self, batch, batch_idx):
        """Training step supporting Negative Binomial likelihood on gene counts."""
        pert_emb = batch["pert_emb"]
        ctrl_cell_data = batch["ctrl_cell_g"]
        pert_cell_data = batch["pert_cell_g"]
        ctrl_cell_data_hvg = ctrl_cell_data[:, :, self.hvg_mask]
        pert_cell_data_hvg = pert_cell_data[:, :, self.hvg_mask]
        covariates = {
            "cell_type_onehot": batch["cell_type_onehot"],
            "batch_onehot": batch["batch_onehot"],
        }
        
        pert_names = batch["pert_name"]
        self.mean_target_map = self.mean_target_map.to(self.device)
        pred, pred_emb = self.forward(ctrl_cell_data_hvg, pert_emb, covariates)
        B = pred.shape[0]

        loss_pds = self.calculate_loss_centroid(pred, pert_names)
        mse_loss = 0.2 * F.mse_loss(pred, pert_cell_data) + 0.8 * F.mse_loss(pred_emb, pert_cell_data_hvg)
        loss_ot = self.ot_loss(pred_emb, pert_cell_data_hvg).nanmean()
        loss = 0.4 * mse_loss + 0.4 * loss_pds + 0.2 * loss_ot
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train_loss_ot", loss_ot, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train_mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log("train_loss_pds", loss_pds, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step supporting Negative Binomial likelihood on gene counts."""
        pert_cell_data = batch["pert_cell_g"]
        ctrl_cell_data = batch["ctrl_cell_g"]
        ctrl_cell_data_hvg = ctrl_cell_data[:, :, self.hvg_mask]
        pert_cell_data_hvg = pert_cell_data[:, :, self.hvg_mask]
        
        pert_emb = batch["pert_emb"]
        
        covariates = {
            "cell_type_onehot": batch["cell_type_onehot"],
            "batch_onehot": batch["batch_onehot"],
        }
        
        pred, pred_emb = self.forward(ctrl_cell_data_hvg, pert_emb, covariates)
        B = pred.shape[0]
        pert_names = batch["pert_name"]
        loss_same_target = self.calculate_loss_centroid(pred, pert_names)
        loss_pds = loss_same_target
        
        mse_loss = 0.2 * F.mse_loss(pred, pert_cell_data) + 0.8 * F.mse_loss(pred_emb, pert_cell_data_hvg)
        loss_ot = self.ot_loss(pred_emb, pert_cell_data_hvg).nanmean()
        loss = 0.4 * mse_loss + 0.4 * loss_pds + 0.2 * loss_ot
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val_loss_ot", loss_ot, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=B)
        self.log("val_loss_pds", loss_pds, on_step=False, on_epoch=True, prog_bar=False, batch_size=B)
        return loss
