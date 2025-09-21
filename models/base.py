"""
Base classes for perturbation models.
"""

from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch


class PerturbationModel(L.LightningModule, ABC):
    """
    Base class for perturbation prediction models.
    """

    def __init__(
        self,
        datamodule: L.LightningDataModule | None = None,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.lr = lr or 1e-3
        self.wd = wd or 0.0
        self.lr_scheduler_freq = lr_scheduler_freq
        self.lr_scheduler_interval = lr_scheduler_interval or "epoch"
        self.lr_scheduler_patience = lr_scheduler_patience or 10
        self.lr_scheduler_factor = lr_scheduler_factor or 0.5

        # Will be set based on data
        self.n_genes = None
        self.n_perts = None
        self.n_input_features = None

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.wd
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
                    'monitor': 'train_loss',
                    'frequency': self.lr_scheduler_freq,
                    'interval': self.lr_scheduler_interval
                }
            }

        return optimizer

    def unpack_batch(self, batch: dict[str, Any]):
        """
        Unpack batch data into components.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Tuple of (pert_cell_emb, ctrl_cell_emb, pert_emb, covariates)
        """
        # Convert sparse tensors to dense for processing
        pert_cell_emb = batch["pert_cell_emb"].to_dense()  # [B, S, G]
        ctrl_cell_emb = batch["ctrl_cell_emb"].to_dense()  # [B, S, G]
        pert_emb = batch["pert_emb"]  # [B, 5120]

        # Covariates: cell_type_onehot [B, S, n_cell_types], batch_onehot [B, S, n_batches]
        covariates = {
            "cell_type_onehot": batch["cell_type_onehot"],
            "batch_onehot": batch["batch_onehot"]
        }

        return pert_cell_emb, ctrl_cell_emb, pert_emb, covariates

    @abstractmethod
    def forward(self, ctrl_cell_emb, pert_emb, covariates):
        """
        Forward pass of the model.
        
        Args:
            ctrl_cell_emb: Control cell embeddings [B, S, G]
            pert_emb: Perturbation embeddings [B, 5120]
            covariates: Dictionary of covariate tensors
            
        Returns:
            Predicted perturbation effects [B, S, G]
        """
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Training step implementation."""
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Validation step implementation."""
        pass

    def predict(self, batch):
        """Prediction method."""
        pert_cell_emb, ctrl_cell_emb, pert_emb, covariates = self.unpack_batch(batch)
        return self.forward(ctrl_cell_emb, pert_emb, covariates)
