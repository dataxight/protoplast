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
        """Configure optimizer and learning rate scheduler with parameter group separation."""
        import math
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR

        # separate param groups so LayerNorm / bias get no weight decay
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n.lower() for nd in ["bias", "norm", "layernorm", "rmsnorm"]):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = AdamW(
            [
                {"params": decay, "weight_decay": 3e-4},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=1e-3,  # peak LR
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # define warmup + cosine decay
        warmup_steps = 2000
        max_steps = 50000  # you can also set dynamically in trainer

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",   # update every step
            "frequency": 1,
        }

        return [optimizer], [scheduler]

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
