"""
Base classes for perturbation models.
"""

from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch
import math
import numpy as np
from typing import Literal, Optional

def mmd2_rff_batched(
    x: torch.Tensor, y: torch.Tensor,
    sigma: float = 1.0,  # single-bandwidth RBF
    num_features: int = 1024,
    reduction: Literal["mean", "sum", "none"] = "mean",
    rng: Optional[torch.Generator] = None,
):
    """
    Random Fourier Feature approximation to MMD^2 (RBF).
    Much cheaper for big S and/or large G.
    """
    B, Sx, G = x.shape
    _,  Sy, Gy = y.shape
    assert G == Gy

    device, dtype = x.device, x.dtype
    if rng is None:
        rng = torch.Generator(device=device)
    # w ~ N(0, I * 1/sigma^2), b ~ U(0, 2π)
    w = torch.randn((G, num_features), device=device, dtype=dtype, generator=rng) / sigma
    b = 2 * math.pi * torch.rand((num_features,), device=device, dtype=dtype, generator=rng)

    def phi(z):
        # z: [B, S, G] -> [B, S, D]
        proj = torch.einsum('bsg,gd->bsd', z, w) + b  # broadcast b
        return (math.sqrt(2.0 / num_features)) * torch.cos(proj)

    phix = phi(x)   # [B, Sx, D]
    phiy = phi(y)   # [B, Sy, D]

    # Empirical mean embeddings
    mux = phix.mean(dim=1)  # [B, D]
    muy = phiy.mean(dim=1)  # [B, D]

    mmd2 = ((mux - muy) ** 2).sum(dim=1)  # [B]

    if reduction == "mean":
        return mmd2.mean()
    elif reduction == "sum":
        return mmd2.sum()
    else:
        return mmd2

def energy_distance_batched(
    x: torch.Tensor,        # [B, Sx, G]
    y: torch.Tensor,        # [B, Sy, G]
    p: float = 2.0,         # norm for torch.cdist; 2.0 = Euclidean
    unbiased: bool = True,  # remove self-pairs in within-cloud terms
    reduction: Literal["mean","sum","none"] = "mean",
) -> torch.Tensor:
    """
    ED = 2*E|X-Y| - E|X-X'| - E|Y-Y'|    (>= 0; 0 iff distributions equal under mild conditions)
    Fully differentiable w.r.t. x, y.
    """
    assert x.dim() == 3 and y.dim() == 3
    B, Sx, Gx = x.shape
    By, Sy, Gy = y.shape
    assert B == By and Gx == Gy

    dxy = torch.cdist(x, y, p=p)        # [B, Sx, Sy]
    dxx = torch.cdist(x, x, p=p)        # [B, Sx, Sx]
    dyy = torch.cdist(y, y, p=p)        # [B, Sy, Sy]

    term_xy = 2.0 * dxy.mean(dim=(1,2))  # [B]

    if unbiased:
        # exclude diagonal self-pairs (which are 0 anyway) in a proper U-statistic normalization
        Sx_denom = max(Sx * (Sx - 1), 1)
        Sy_denom = max(Sy * (Sy - 1), 1)
        sum_xx = dxx.sum(dim=(1,2))  # diag is zero; no need to subtract
        sum_yy = dyy.sum(dim=(1,2))
        term_xx = sum_xx / Sx_denom
        term_yy = sum_yy / Sy_denom
    else:
        term_xx = dxx.mean(dim=(1,2))
        term_yy = dyy.mean(dim=(1,2))

    ed = term_xy - term_xx - term_yy    # [B]

    if reduction == "mean":
        return ed.mean()
    elif reduction == "sum":
        return ed.sum()
    else:
        return ed  # [B]


def loss_fct(pred: torch.Tensor, 
            y: torch.Tensor, 
            perts: np.ndarray, 
            loss_weight: torch.Tensor,
            ctrl: torch.Tensor = None, 
            direction_lambda: float = 1e-3, 
            dict_filter: dict = None, 
            use_mse_loss: bool = False):
    """
    Main MSE Loss function, includes direction loss

    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        perts (list): list of perturbations
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions
        loss_weights_dict (dict): dictionary of loss weights for each perturbation
        use_mse_loss (bool): whether to use MSE loss

    """
    gamma = 2
    mse_p = torch.nn.MSELoss()
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

    for p in set(perts):
        pert_idx = np.where(perts == p)[0] # [N]
        pred_p = pred[pert_idx].mean(dim=1) # [N, G]
        y_p = y[pert_idx].mean(dim=1) # [N, G]
        ctrl_p = ctrl[pert_idx].mean(dim=1) # [N, G]
        weights = loss_weight[pert_idx] # [N, G]
        if not use_mse_loss:
            losses = losses + torch.sum(weights * (pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
        else:
            losses = losses + torch.sum(weights * (pred_p - y_p)**2)/pred_p.shape[0]/pred_p.shape[1]

        ## direction loss
        if not use_mse_loss:
            losses = losses + torch.sum(weights * direction_lambda *
                                (torch.sign(y_p - ctrl_p) -
                                 torch.sign(pred_p - ctrl_p))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
    return losses/(len(set(perts)))

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
        loss_weight_emb = batch["loss_weight_emb"]  # [B, E]
        loss_weight_gene = batch["loss_weight_gene"]  # [B, G]

        # Covariates: cell_type_onehot [B, S, n_cell_types], batch_onehot [B, S, n_batches]
        covariates = {
            "cell_type_onehot": batch["cell_type_onehot"],
            "batch_onehot": batch["batch_onehot"]
        }

        return pert_cell_emb, ctrl_cell_emb, pert_emb, covariates, loss_weight_emb, loss_weight_gene

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
