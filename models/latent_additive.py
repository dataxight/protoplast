"""
Latent Additive Model for perturbation prediction.

Adapted for batch structure:
- pert_cell_emb: [B, S, G] - perturbation cell embeddings  
- ctrl_cell_emb: [B, S, G] - control cell embeddings
- pert_emb: [B, 5120] - perturbation embedding vector
- cell_type_onehot: [B, S, n_cell_types] - cell type one-hot encoding
- batch_onehot: [B, S, n_batches] - batch one-hot encoding
"""


import lightning as L
import torch
import torch.nn.functional as F
from geomloss import SamplesLoss

from ..nn.mlp import MLP, MaskNet
from .base import PerturbationModel


class LatentAdditive(PerturbationModel):
    """
    A latent additive model for predicting perturbation effects.
    
    The model encodes control cells and perturbation information into latent spaces,
    then combines them additively to predict perturbation effects.
    """

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_batches: int,
        pert_emb_dim: int = 5120,
        n_layers: int = 2,
        encoder_width: int = 128,
        latent_dim: int = 32,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        dropout: float | None = None,
        softplus_output: bool = True,
        sparse_additive_mechanism: bool = False,
        inject_covariates_encoder: bool = False,
        inject_covariates_decoder: bool = False,
        pool_ctrl_cells: str = "mean",  # "mean" or "sum" or "max"
        latent_loss_weight: float = 1,  # Weight for latent comparison loss
        datamodule: L.LightningDataModule | None = None,
    ) -> None:
        """
        Constructor for the LatentAdditive class.

        Args:
            n_genes: Number of genes (G dimension)
            n_cell_types: Number of cell types in dataset
            n_batches: Number of batches in dataset  
            pert_emb_dim: Dimension of perturbation embedding (default 5120)
            n_layers: Number of layers in encoder/decoder MLPs
            encoder_width: Width of hidden layers in encoder/decoder
            latent_dim: Dimension of the latent space
            lr: Learning rate
            wd: Weight decay
            lr_scheduler_freq: Learning rate scheduler check frequency
            lr_scheduler_interval: LR scheduler interval ("epoch" or "step")
            lr_scheduler_patience: LR scheduler patience
            lr_scheduler_factor: LR reduction factor
            dropout: Dropout rate (None for no dropout)
            softplus_output: Whether to apply softplus to output
            sparse_additive_mechanism: Whether to use sparse masking
            inject_covariates_encoder: Whether to condition encoder on covariates
            inject_covariates_decoder: Whether to condition decoder on covariates
            pool_ctrl_cells: How to pool control cells ("mean", "sum", "max")
            latent_loss_weight: Weight for latent comparison loss
            datamodule: Lightning datamodule
        """
        super().__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )

        self.save_hyperparameters(ignore=["datamodule"])

        # Set model dimensions
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_batches = n_batches
        self.pert_emb_dim = pert_emb_dim
        self.n_input_features = n_genes  # For compatibility with base class
        self.latent_dim = latent_dim
        self.pool_ctrl_cells = pool_ctrl_cells
        self.latent_loss_weight = latent_loss_weight

        # Calculate covariate dimensions
        n_total_covariates = n_cell_types + n_batches if (inject_covariates_encoder or inject_covariates_decoder) else 0

        # Determine input dimensions for encoders/decoder
        gene_encoder_input_dim = (
            self.n_genes + n_total_covariates
            if inject_covariates_encoder
            else self.n_genes
        )

        decoder_input_dim = (
            latent_dim + n_total_covariates
            if inject_covariates_decoder
            else latent_dim
        )

        # Build networks
        self.gene_encoder = MLP(
            gene_encoder_input_dim, encoder_width, latent_dim, n_layers, dropout
        )

        self.pert_encoder = MLP(
            self.pert_emb_dim, encoder_width, latent_dim, n_layers, dropout
        )
        
        # Add encoder for perturbation cell embeddings to encode true latent space
        self.pert_cell_encoder = MLP(
            gene_encoder_input_dim, encoder_width, latent_dim, n_layers, dropout
        )

        self.decoder = MLP(
            decoder_input_dim, encoder_width, self.n_genes, n_layers, dropout
        )

        # Optional sparse additive mechanism
        if sparse_additive_mechanism:
            self.mask_encoder = MaskNet(
                self.pert_emb_dim, encoder_width, latent_dim, n_layers
            )

        # Store configuration
        self.dropout = dropout
        self.softplus_output = softplus_output
        self.sparse_additive_mechanism = sparse_additive_mechanism
        self.inject_covariates_encoder = inject_covariates_encoder
        self.inject_covariates_decoder = inject_covariates_decoder
        self.sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9, debias=True)

    def _pool_control_cells(self, ctrl_cell_emb: torch.Tensor) -> torch.Tensor:
        """
        Pool control cells across the S dimension.
        
        Args:
            ctrl_cell_emb: Control cell embeddings [B, S, G]
            
        Returns:
            Pooled control embeddings [B, G]
        """
        if self.pool_ctrl_cells == "mean":
            return ctrl_cell_emb.mean(dim=1)
        elif self.pool_ctrl_cells == "sum":
            return ctrl_cell_emb.sum(dim=1)
        elif self.pool_ctrl_cells == "max":
            return ctrl_cell_emb.max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported pooling method: {self.pool_ctrl_cells}")

    def _prepare_covariates(self, covariates: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Prepare covariates by concatenating and pooling.
        
        Args:
            covariates: Dict with cell_type_onehot [B, S, n_cell_types] and batch_onehot [B, S, n_batches]
            
        Returns:
            Concatenated covariates [B, n_cell_types + n_batches]
        """
        cell_type_oh = covariates["cell_type_onehot"]  # [B, S, n_cell_types]
        batch_oh = covariates["batch_onehot"]  # [B, S, n_batches]

        # Pool across S dimension (taking first element since they should be the same for all S)
        cell_type_pooled = cell_type_oh[:, 0, :]  # [B, n_cell_types]
        batch_pooled = batch_oh[:, 0, :]  # [B, n_batches]

        # Concatenate
        return torch.cat([cell_type_pooled, batch_pooled], dim=1)  # [B, n_cell_types + n_batches]

    def batch_samples_loss_loop(self, x, y):
        """
        Calculate SamplesLoss for each batch item independently
        """
        x = x.reshape(-1, x.shape[1], x.shape[-1]).contiguous()
        y = y.reshape(-1, y.shape[1], y.shape[-1]).contiguous()
        return self.sinkhorn(x, y).nanmean()  # Shape: [B]
    
    def compute_latent_loss(self, pred_latent: torch.Tensor, true_latent: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predicted and true latent representations.
        
        Args:
            pred_latent: Predicted latent representation [B, S, latent_dim]
            true_latent: True latent representation [B, S, latent_dim]
            
        Returns:
            MSE loss between latent representations
        """
        return F.mse_loss(pred_latent, true_latent)

    def forward(
        self,
        ctrl_cell_emb: torch.Tensor,
        pert_emb: torch.Tensor,
        covariates: dict[str, torch.Tensor],
        pert_cell_emb: torch.Tensor | None = None,
        return_latents: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass of the latent additive model.
        
        Args:
            ctrl_cell_emb: Control cell embeddings [B, S, G]
            pert_emb: Perturbation embeddings [B, 5120]
            covariates: Dictionary with cell type and batch one-hot encodings
            pert_cell_emb: Perturbation cell embeddings [B, S, G] (optional)
            return_latents: Whether to return latent representations
            
        Returns:
            If return_latents=False: Predicted perturbation effects [B, S, G]
            If return_latents=True: (predicted_expression, latent_dict)
                latent_dict contains:
                - pred_latent: [B, S, latent_dim] - predicted latent (expanded)
                - true_latent: [B, S, latent_dim] - true latent from pert_cell_emb
                - latent_control: [B, latent_dim] - control latent
                - latent_perturbation: [B, latent_dim] - perturbation latent
        """
        batch_size, S, G = ctrl_cell_emb.shape

        # Pool control cells to [B, G]
        ctrl_pooled = self._pool_control_cells(ctrl_cell_emb)

        # Prepare covariates if needed
        if self.inject_covariates_encoder or self.inject_covariates_decoder:
            merged_covariates = self._prepare_covariates(covariates)

        # Prepare input for gene encoder
        if self.inject_covariates_encoder:
            gene_encoder_input = torch.cat([ctrl_pooled, merged_covariates], dim=1)
        else:
            gene_encoder_input = ctrl_pooled

        # Encode control and perturbation to latent space
        latent_control = self.gene_encoder(gene_encoder_input)  # [B, latent_dim]
        latent_perturbation = self.pert_encoder(pert_emb)  # [B, latent_dim]

        # Apply sparse masking if enabled
        if self.sparse_additive_mechanism:
            mask = self.mask_encoder(pert_emb)  # [B, latent_dim]
            latent_perturbation = mask * latent_perturbation

        # Additive combination
        latent_perturbed = latent_control + latent_perturbation  # [B, latent_dim]

        # Encode true perturbation cells to latent space if provided
        true_latent_perturbed = None
        if pert_cell_emb is not None:
            # Reshape pert_cell_emb from [B, S, G] to [B*S, G] for processing
            B, S, G = pert_cell_emb.shape
            pert_reshaped = pert_cell_emb.view(B * S, G)  # [B*S, G]
            
            # Prepare input for perturbation cell encoder
            if self.inject_covariates_encoder:
                # Expand covariates to match [B*S, ...]
                merged_covariates_expanded = merged_covariates.unsqueeze(1).expand(-1, S, -1).reshape(B * S, -1)
                pert_cell_encoder_input = torch.cat([pert_reshaped, merged_covariates_expanded], dim=1)
            else:
                pert_cell_encoder_input = pert_reshaped
                
            # Encode to latent space and reshape back
            true_latent_encoded = self.pert_cell_encoder(pert_cell_encoder_input)  # [B*S, latent_dim]
            true_latent_perturbed = true_latent_encoded.view(B, S, self.latent_dim)  # [B, S, latent_dim]

        # Prepare decoder input
        if self.inject_covariates_decoder:
            decoder_input = torch.cat([latent_perturbed, merged_covariates], dim=1)
        else:
            decoder_input = latent_perturbed

        # Decode to gene space
        predicted_expression = self.decoder(decoder_input)  # [B, G]

        # Apply softplus if configured
        if self.softplus_output:
            predicted_expression = F.softplus(predicted_expression)

        # Expand to match input shape [B, S, G]
        predicted_expression = predicted_expression.unsqueeze(1).expand(-1, S, -1)

        if return_latents:
            # Expand predicted latent to match sequence dimension for comparison
            pred_latent_expanded = latent_perturbed.unsqueeze(1).expand(-1, S, -1)  # [B, S, latent_dim]
            
            latent_dict = {
                "pred_latent": pred_latent_expanded,  # [B, S, latent_dim]
                "true_latent": true_latent_perturbed,  # [B, S, latent_dim] or None
                "latent_control": latent_control,  # [B, latent_dim]
                "latent_perturbation": latent_perturbation,  # [B, latent_dim]
            }
            return predicted_expression, latent_dict

        return predicted_expression

    def analyze_reconstruction_quality(self, target_log1p, pred_recon):
        target_log1p = target_log1p.reshape(-1, target_log1p.shape[-1]).contiguous()
        pred_recon = pred_recon.reshape(-1, pred_recon.shape[-1]).contiguous()
        loss = self.sinkhorn(pred_recon, target_log1p)
    
        # Additional count-specific metrics
        # Convert back to count space for interpretation
        target_counts = torch.expm1(target_log1p)  # Reverse log1p
        pred_counts = torch.expm1(torch.clamp(pred_recon, min=0))  # Ensure non-negative
    
        return {
            'sinkhorn_loss': loss,
            'target_sparsity': (target_counts == 0).float().mean(),
            'pred_sparsity': (pred_counts == 0).float().mean(),
            'magnitude_ratio': pred_counts.mean() / target_counts.mean()
        }
    def training_step(self, batch: dict, batch_idx: int):
        """Training step implementation."""
        pert_cell_emb, ctrl_cell_emb, pert_emb, covariates = self.unpack_batch(batch)

        # Forward pass with latent representations
        predicted_expression, latent_dict = self.forward(
            ctrl_cell_emb, pert_emb, covariates, 
            pert_cell_emb=pert_cell_emb, return_latents=True
        )

        # mse loss between predicted and true expression
        expression_loss = F.mse_loss(predicted_expression, pert_cell_emb)
        
        # Compute latent comparison loss
        latent_loss = 0.0
        if latent_dict["true_latent"] is not None:
            latent_loss = self.compute_latent_loss(
                latent_dict["pred_latent"], latent_dict["true_latent"]
            )
        
        reconstruction_quality = self.analyze_reconstruction_quality(pert_cell_emb, predicted_expression)
        
        # Log losses
        self.log("train_loss", expression_loss + latent_loss + reconstruction_quality["sinkhorn_loss"], prog_bar=True, logger=True, batch_size=pert_cell_emb.size(0))
        self.log("train_expression_loss", expression_loss, logger=True, batch_size=pert_cell_emb.size(0))
        self.log("train_latent_loss", latent_loss, logger=True, batch_size=pert_cell_emb.size(0))
        self.log("train_reconstruction_quality", reconstruction_quality["sinkhorn_loss"], logger=True, batch_size=pert_cell_emb.size(0))
        self.log("train_target_sparsity", reconstruction_quality["target_sparsity"], logger=True, batch_size=pert_cell_emb.size(0))
        self.log("train_pred_sparsity", reconstruction_quality["pred_sparsity"], logger=True, batch_size=pert_cell_emb.size(0))
        self.log("train_magnitude_ratio", reconstruction_quality["magnitude_ratio"], logger=True, batch_size=pert_cell_emb.size(0))
        return latent_loss

    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step implementation."""
        pert_cell_emb, ctrl_cell_emb, pert_emb, covariates = self.unpack_batch(batch)

        # Forward pass with latent representations
        predicted_expression, latent_dict = self.forward(
            ctrl_cell_emb, pert_emb, covariates, 
            pert_cell_emb=pert_cell_emb, return_latents=True
        )

        expression_loss = F.mse_loss(predicted_expression, pert_cell_emb)
        reconstruction_quality = self.analyze_reconstruction_quality(pert_cell_emb, predicted_expression)
        
        # Compute latent comparison loss
        latent_loss = 0.0
        if latent_dict["true_latent"] is not None:
            latent_loss = self.compute_latent_loss(
                latent_dict["pred_latent"], latent_dict["true_latent"]
            )
        
        # Combined loss (same weight as training)
        total_loss = expression_loss + self.latent_loss_weight * latent_loss

        # Log losses
        self.log(
            "val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=pert_cell_emb.size(0),
        )
        self.log(
            "val_expression_loss",
            expression_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=pert_cell_emb.size(0),
        )
        self.log(
            "val_latent_loss",
            latent_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=pert_cell_emb.size(0),
        )
        
        return total_loss

    def predict(self, batch: dict, return_latents: bool = False):
        """
        Prediction method.
        
        Args:
            batch: Input batch dictionary
            return_latents: Whether to return latent representations
            
        Returns:
            Predicted expression or (predicted_expression, latent_dict) if return_latents=True
        """
        pert_cell_emb, ctrl_cell_emb, pert_emb, covariates = self.unpack_batch(batch)
        return self.forward(
            ctrl_cell_emb, pert_emb, covariates, 
            pert_cell_emb=pert_cell_emb if return_latents else None,
            return_latents=return_latents
        )
    
    def get_latent_representations(self, batch: dict):
        """
        Get latent representations for analysis.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Dictionary with latent representations
        """
        pert_cell_emb, ctrl_cell_emb, pert_emb, covariates = self.unpack_batch(batch)
        _, latent_dict = self.forward(
            ctrl_cell_emb, pert_emb, covariates, 
            pert_cell_emb=pert_cell_emb, return_latents=True
        )
        return latent_dict
