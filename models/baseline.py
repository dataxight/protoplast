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

from .base import PerturbationModel, mmd2_rff_batched, energy_distance_batched, loss_fct, loss_fct_vectorized
from .nn.mlp import MLP
from .nn.gears_loss import gears_autofocus_direction_loss
from geomloss import SamplesLoss
from .nn.scvi_like import Encoder as ScviEncoder, DecoderSCVI


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
        n_perts: int,
        d_h: int = 672,  # Hidden dimension
        d_f: int = 512,  # Bottleneck dimension 
        n_genes: int = 18080,  # Number of genes (G)
        embedding_dim: int = 2058,  # Embedding dimension (E)
        pert_emb_dim: int = 5120,  # Perturbation embedding dimension
        n_cell_types: int = None,  # Number of cell types (will be set from data)
        n_batches: int = None,  # Number of batches (will be set from data)
        dropout: float = 0.1,
        use_nb: bool = True,
        num_mc_samples: int = 0,
        kl_weight: float = 1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_h = d_h
        self.d_f = d_f
        self.n_genes = n_genes
        self.n_perts = n_perts
        self.embedding_dim = embedding_dim
        self.pert_emb_dim = pert_emb_dim
        self.n_cell_types = n_cell_types
        self.n_batches = n_batches
        self.use_nb = use_nb
        self.num_mc_samples = num_mc_samples
        self.mean_target_map = mean_target_map
        self.mean_target_addresses = mean_target_addresses
        self.dropout = dropout
        self.kl_weight = kl_weight
        self.cls_weight = kwargs.get("cls_weight", 1.0)

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
            n_layers=4,
            dropout=dropout,
            activation="gelu"
        )
        
        # # Projection back to embedding space: MLP(d_h, E)
        # self.projection_to_emb = MLP(
        #     input_dim=d_h,
        #     hidden_dim=d_h,
        #     output_dim=embedding_dim,
        #     n_layers=4,
        #     dropout=dropout,
        #     activation="gelu"
        # )

        self.norm = nn.LayerNorm(d_h)

        self.ot_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)

        # scVI-style latent path: q(z|H) and decoder to counts
        self.z_encoder = ScviEncoder(
            n_input=d_h,
            n_output=d_h,
            n_layers=2,
            n_hidden=d_h,
            dropout_rate=dropout,
        )
        self.z_decoder = DecoderSCVI(
            n_input=d_h,
            n_output=n_genes,
            n_layers=2,
            n_hidden=d_h,
            scale_activation="softmax",
        )
        # fixed library size 1e4 in log-space
        self.register_buffer("fixed_log_library", torch.tensor(np.log(1e4), dtype=torch.float32))

        # CNN-based classifier over latent sequence z (shape: [B, S, d_h])
        # We treat d_h as channels and S as the sequence length for Conv1d.
        # self.cls_cnn = nn.Sequential(
        #     nn.Conv1d(self.d_h, self.d_h, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.Dropout(self.dropout),
        #     nn.Conv1d(self.d_h, max(self.d_h // 2, 1), kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.AdaptiveAvgPool1d(1),  # -> [B, C, 1]
        # )
        # self.cls_head = nn.Linear(max(self.d_h // 2, 1), self.n_perts)
        # self.last_cls_logits = None

    def _refine_H(self, H):
        # baseline path: simple residual MLP
        return H + self.residual_encoder(H)

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
        # Build a dict once (cache it) mapping string -> row index
        if not hasattr(self, "_mean_addr_index"):
            self._mean_addr_index = {k: i for i, k in enumerate(self.mean_target_addresses)}

        idx = torch.tensor([self._mean_addr_index[s] for s in pert_addresses], device=self.device, dtype=torch.long)

        # print(f"mask: {mask.shape}, {mask}")
        self.mean_target_map = self.mean_target_map.to(self.device)
        target_means = self.mean_target_map[idx]
        # print(f"target_means: {target_means.shape}, {target_means}")
        d_same = (pred - target_means).abs().sum(dim=1)  # [B]
        # print(f"l1_same_target: {l1_same_target.shape}, {l1_same_target}")
        # sample K negatives without replacement
        K = min(128, self.mean_target_map.size(0)-1)
        all_idx = torch.arange(self.mean_target_map.size(0), device=pred.device)
        # exclude positives
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask[idx.unique()] = False
        neg_pool = all_idx[mask]
        # neg_sel = neg_pool[torch.randint(0, neg_pool.numel(), (K,))]  # [K]
        neg_sel = neg_pool[:K]  # [K]
        neg_means = self.mean_target_map.to(pred.device)[neg_sel]
        # TODO: mean this across batch
        d_other = (pred[:, None, :] - neg_means[None, :, :]).abs().sum(dim=-1)  # [B,K]
        # margin = 10.0  # tune
        loss = F.relu(d_same[:, None] - d_other).mean()
        return loss

    def l1_loss(self, pred, pert_names: np.ndarray):
        """
        Calculate the L1 loss between the predicted and true values.
        """
        B, S, G = pred.shape
        unique_pert_names, inverse_indices = np.unique(pert_names, return_inverse=True)
        n_unique_perts = len(unique_pert_names)
        pred = pred.mean(dim=1) # [B, G]
        pred_norm = F.normalize(pred, p=2, dim=1)
        cos_similarity = torch.matmul(pred_norm, pred_norm.T)
        cos_dist = 1 - cos_similarity
        cos_dist_sum = cos_dist.sum(dim=1)
        loss = cos_dist_sum.sum() / (B - n_unique_perts)
        return loss

    def _kl_normal_standard(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """KL(q||p) where q=N(mean,var) and p=N(0,I). Returns mean over batch.
        mean/var shaped [BS, D].
        """
        # 0.5 * sum(mu^2 + var - log var - 1)
        kl = 0.5 * (mean.pow(2) + var - var.clamp_min(1e-8).log() - 1.0)
        return kl.sum(dim=-1).mean()

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
        
        # 4. Combine encodings and refine
        H = H_ctrl + H_batch + H_pert  # [B, S, d_h]
        H = self._refine_H(self.norm(H))

        # Infer latent z ~ N(mu, var) using scVI-like encoder
        BS = B * S
        H_flat = H.reshape(BS, self.d_h)
        q_m, q_v, z_flat = self.z_encoder(H_flat)
        # KL against standard normal
        self.last_kl = self._kl_normal_standard(q_m, q_v)
        z = z_flat.reshape(B, S, self.d_h)

        # # Classification logits from latent sequence using CNN
        # z_seq = z.permute(0, 2, 1)  # [B, d_h, S]
        # cls_feat = self.cls_cnn(z_seq)  # [B, C, 1]
        # cls_feat = cls_feat.squeeze(-1)  # [B, C]
        # self.last_cls_logits = self.cls_head(cls_feat)  # [B, n_perts]

        # Project latent to embedding space
        # emb_rate = self.projection_to_emb(z)  # [B, S, E]
        # emb_scale = torch.softmax(emb_rate, dim=-1)
        # emb_out = emb_scale * ctrl_cell_emb.sum(dim=2)[:, :, None]


        # Decode to Gamma-Poisson (NB) params with dropout; fixed library size 1e4
        lib = self.fixed_log_library.expand(BS, 1)
        px_scale, px_r, px_rate, px_dropout = self.z_decoder(
            dispersion="gene-cell", z=z_flat, library=lib
        )
        # Either sample from Gamma-Poisson with dropout or use expectation
        if self.num_mc_samples and self.num_mc_samples > 0:
            DISPERSION_SCALE = 0.5
            theta = F.softplus(px_r) + 1e-4  # inverse dispersion > 0
            theta = theta * DISPERSION_SCALE
            rate = theta / (px_rate.clamp_min(1e-8))
            gamma_dist = torch.distributions.Gamma(concentration=theta, rate=rate)
            lam = gamma_dist.rsample()  # [BS, G]
            counts = torch.poisson(lam)
            dropout_mask = torch.bernoulli(torch.sigmoid(px_dropout)).bool()
            counts = counts.masked_fill(dropout_mask, 0.0)
            out = counts
        else:
            dropout_p = torch.sigmoid(px_dropout)
            out = (1.0 - dropout_p) * px_rate # expectation under ZINB
        gene_output = out.reshape(B, S, self.n_genes)
        gene_output = torch.log1p(gene_output)

        return gene_output

    def training_step(self, batch, batch_idx):
        """Training step supporting Negative Binomial likelihood on gene counts."""
        pert_emb = batch["pert_emb"]
        ctrl_cell_emb = batch["ctrl_cell_emb"]
        pert_cell_emb = batch["pert_cell_emb"]
        pert_cell_data = batch["pert_cell_g"]
        ctrl_cell_data = batch["ctrl_cell_g"]
        pert_onehot = batch["pert_onehot"] # [B, n_perts]
        covariates = {
            "cell_type_onehot": batch["cell_type_onehot"],
            "batch_onehot": batch["batch_onehot"],
        }
        loss_weight_gene = batch["loss_weight_gene"]
        pert_names = batch["pert_name"]
        self.mean_target_map = self.mean_target_map.to(self.device)
        pred = self.forward(ctrl_cell_emb, pert_emb, covariates)
        # loss_centroid = self.calculate_loss_centroid(pred, pert_names)
        # loss_centroid = self.l1_loss(pred, pert_names)

        B, S, G = pred.shape
        loss_all_gene = loss_fct(pred, pert_cell_data, pert_names, loss_weight_gene, ctrl_cell_data, direction_lambda=1e-3)
        kl = getattr(self, "last_kl", torch.tensor(0.0, device=pred.device))
        # Classification loss (targets from provided pert_onehot)
        # logits = self.last_cls_logits  # [B, n_perts]
        # target_cls = pert_onehot.argmax(dim=-1)
        # cls_loss = F.cross_entropy(logits, target_cls)
        # with torch.no_grad():
        #     pred_cls = logits.argmax(dim=-1)
        #     cls_acc = (pred_cls == target_cls).float().mean()
        loss = loss_all_gene + self.kl_weight * kl #+ self.cls_weight * cls_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train_kl", kl, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log("train_loss_gene", loss_all_gene, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        # self.log("train_loss_cls", cls_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        # self.log("train_acc_cls", cls_acc, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step supporting Negative Binomial likelihood on gene counts."""
        ctrl_cell_emb = batch["ctrl_cell_emb"]
        pert_cell_emb = batch["pert_cell_emb"]
        pert_cell_data = batch["pert_cell_g"]
        ctrl_cell_data = batch["ctrl_cell_g"]
        loss_weight_gene = batch["loss_weight_gene"]
        pert_names = batch["pert_name"]
        pert_emb = batch["pert_emb"]
        pert_onehot = batch["pert_onehot"]
        
        covariates = {
            "cell_type_onehot": batch["cell_type_onehot"],
            "batch_onehot": batch["batch_onehot"],
        }

        # no grad
        with torch.no_grad():
            pred = self.forward(ctrl_cell_emb, pert_emb, covariates)
            B, S, G = pred.shape
            # loss_centroid = self.l1_loss(pred, pert_names)
            pert_names = batch["pert_name"]
            loss_all_gene = loss_fct(pred, pert_cell_data, pert_names, loss_weight_gene, ctrl_cell_data, direction_lambda=1e-3)
            kl = getattr(self, "last_kl", torch.tensor(0.0, device=pred.device))
            # Classification loss and accuracy using pert_onehot
            # logits = self.last_cls_logits  # [B, n_perts]
            # target_cls = pert_onehot.argmax(dim=-1)
            # cls_loss = F.cross_entropy(logits, target_cls)
            # pred_cls = logits.argmax(dim=-1)
            # cls_acc = (pred_cls == target_cls).float().mean()
            loss = loss_all_gene + self.kl_weight * kl #+ self.cls_weight * cls_loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val_kl", kl, on_step=False, on_epoch=True, prog_bar=False, batch_size=B)
        self.log("val_loss_gene", loss_all_gene, on_step=False, on_epoch=True, prog_bar=False, batch_size=B)
        # self.log("val_loss_cls", cls_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=B)
        # self.log("val_acc_cls", cls_acc, on_step=False, on_epoch=True, prog_bar=False, batch_size=B)
        return loss
