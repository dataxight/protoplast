import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from scipy.stats import spearmanr, kendalltau
import numpy as np
import torch.nn.functional as F

# ---- small utilities ----
class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.LayerNorm(d_hid),
            nn.Dropout(p),
            nn.Linear(d_hid, d_out),
        )
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        return self.proj(x) + self.net(x)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return GradReverse.apply(x, self.lambd)

# ---- model ----
class CPAVAE(nn.Module):
    def __init__(
        self,
        G: int,
        n_cell_lines: int,
        n_batches: int,
        d_xp: int,
        d_z: int = 64,
        d_y_prior: int = 64,
        d_enc: int = 512,
        d_dec: int = 512,
        d_batch: int = 64,
        delta_hid: int = 128,
        factorized_decoder_rank: int = 512,
        use_dann: bool = True,
        dann_lambda: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.G = G
        self.n_batches = n_batches
        self.use_dann = use_dann

        # --- Prior p(z|y) as small MLPs producing μ_y and logσ²_y
        self.prior_net = nn.Sequential(
            nn.Linear(n_cell_lines, d_y_prior),
            nn.GELU(),
            nn.LayerNorm(d_y_prior),
            nn.Dropout(dropout),
        )
        self.prior_mu = nn.Linear(d_y_prior, d_z)
        self.prior_logvar = nn.Linear(d_y_prior, d_z)

        # --- Encoder q(z|x)
        # (project huge G -> d_enc first for stability)
        self.enc = nn.Sequential(
            nn.Linear(G, d_enc),
            nn.GELU(),
            nn.LayerNorm(d_enc),
            nn.Dropout(dropout),
            ResidualMLP(d_enc, d_enc, d_enc, p=dropout),
        )
        self.enc_mu = nn.Linear(d_enc, d_z)
        self.enc_logvar = nn.Linear(d_enc, d_z)

        # --- Operator: Δz = f([z_ctrl, e_xp])
        self.delta = ResidualMLP(d_z + d_xp, delta_hid, d_z, p=dropout)

        # --- Decoder p(x|z,b) with factorized head + batch offsets
        self.dec_backbone = nn.Sequential(
            nn.Linear(d_z, d_dec),
            nn.GELU(),
            nn.LayerNorm(d_dec),
            nn.Dropout(dropout),
            ResidualMLP(d_dec, d_dec, d_dec, p=dropout),
        )
        # factorized: h_dec (d_dec) -> r -> G
        self.to_rank = nn.Linear(d_dec, factorized_decoder_rank, bias=False)
        self.rank_to_gene = nn.Linear(factorized_decoder_rank, G, bias=True)

        # batch embedding to gene-wise offsets (additive)
        self.batch_proj = nn.Sequential(
            nn.Linear(n_batches, d_batch),
            nn.GELU(),
            nn.LayerNorm(d_batch),
            nn.Linear(d_batch, d_batch)
        )
        self.batch_to_gene = nn.Linear(d_batch, G, bias=False)

        # --- Domain adversarial head on z (to remove batch info)
        if use_dann:
            self.grl = GRL(lambd=dann_lambda)
            self.batch_disc = nn.Sequential(
                nn.Linear(d_z, 128),
                nn.GELU(),
                nn.LayerNorm(128),
                nn.Linear(128, n_batches),
            )

    # --- reparameterization ---
    def _reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --- core blocks ---
    def encode(self, x):
        h = self.enc(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def prior(self, y_onehot):
        h = self.prior_net(y_onehot)
        mu_y = self.prior_mu(h)
        logvar_y = self.prior_logvar(h)
        return mu_y, logvar_y

    def apply_operator(self, z_ctrl, xp_vec):
        dz = self.delta(torch.cat([z_ctrl, xp_vec], dim=-1))
        return z_ctrl + dz, dz

    def decode_base(self, z):
        h = self.dec_backbone(z)
        r = self.to_rank(h)         # [B, r]
        xhat = self.rank_to_gene(r) # [B, G]
        return xhat

    def add_batch_offsets(self, xhat, b_onehot):
        if b_onehot is None:
            return xhat
        batch_h = self.batch_proj(b_onehot)
        offsets = self.batch_to_gene(batch_h)  # [B, G]
        return xhat + offsets

    @torch.no_grad()
    def predict_from_yxp(
        self,
        y_onehot: torch.Tensor,      # [B, n_cell_lines]
        xp_vec: torch.Tensor,        # [B, d_xp]
        b_onehot: torch.Tensor | None = None,  # [B, n_batches] or None
        sample: bool = False,        # False -> use prior mean; True -> sample z_ctrl ~ p(z|y)
    ):
        device = next(self.parameters()).device
        y_onehot = y_onehot.to(device)
        xp_vec = xp_vec.to(device)
        b_onehot = b_onehot.to(device) if b_onehot is not None else None
    
        # prior over control latent from y
        mu_y, logvar_y = self.prior(y_onehot)
        if sample:
            z_ctrl = self._reparam(mu_y, logvar_y)
        else:
            z_ctrl = mu_y
    
        # decode control
        xhat_ctrl = self.decode_base(z_ctrl)
        if b_onehot is not None:
            # if you switched to linear/MLP for batch:
            batch_h = self.batch_proj(b_onehot)
            xhat_ctrl = xhat_ctrl + self.batch_to_gene(batch_h)
    
        # apply operator -> perturbed latent
        z_pert, _ = self.apply_operator(z_ctrl, xp_vec)
    
        # decode perturbed
        xhat_pert = self.decode_base(z_pert)
        if b_onehot is not None:
            batch_h = self.batch_proj(b_onehot)
            xhat_pert = xhat_pert + self.batch_to_gene(batch_h)
    
        return xhat_ctrl, xhat_pert  # (X_pred_control, X_pred)
    
    # --- forward for training ---
    def forward(
        self,
        x: torch.Tensor,            # [B, G]          (observed perturbed; used only for loss)
        y_onehot: torch.Tensor,     # [B, n_cell_lines]
        xp_vec: torch.Tensor,       # [B, d_xp]
        x_ctrl_match: torch.Tensor, # [B, G]          (matched control expression)
        b_onehot: Optional[torch.Tensor] = None          # [B, n_batches] keep optional, default None
    ):
        """
        Paired setting: encode control, predict both control & perturbed.
        """
        # posterior from the matched control (define control latent here)
        mu, logvar = self.encode(x_ctrl_match)
        z_ctrl = self._reparam(mu, logvar)
    
        # y-prior
        mu_y, logvar_y = self.prior(y_onehot)
    
        # operator
        z_pert, delta_z = self.apply_operator(z_ctrl, xp_vec)
    
        # decode
        xhat_ctrl = self.add_batch_offsets(self.decode_base(z_ctrl), b_onehot)
        xhat_pert = self.add_batch_offsets(self.decode_base(z_pert), b_onehot)
    
        out = dict(
            mu=mu, logvar=logvar,
            mu_y=mu_y, logvar_y=logvar_y,
            z_ctrl=z_ctrl, z_pert=z_pert, delta_z=delta_z,
            xhat_ctrl=xhat_ctrl, xhat_pert=xhat_pert,
        )
    
        if self.use_dann and (b_onehot is not None):
            z_grl = self.grl(z_ctrl)
            out["b_logits"] = self.batch_disc(z_grl)
    
        return out

# ---- losses ----

def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    # KL( N(mu_q, var_q) || N(mu_p, var_p) ), per example
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (
        (var_q / var_p)
        + (mu_p - mu_q).pow(2) / var_p
        - 1.0
        + (logvar_p - logvar_q)
    ).sum(dim=-1)
    return kl

def cpa_vae_pair_loss(
    x: torch.Tensor,               # [B, G]          (perturbed observed)
    x_ctrl: torch.Tensor,          # [B, G]          (matched control observed)
    out: dict,
    b_onehot: Optional[torch.Tensor] = None,
    w_kl: float = 1.0,
    w_rec_ctrl: float = 1.0,
    w_rec_pert: float = 1.0,
    w_sparsity: float = 1e-4,
    w_dann: float = 0.1,
):
    """
    Reconstruction:
      - xhat_ctrl vs x_ctrl
      - xhat_pert vs x
    KL:
      - q(z|x_ctrl) || p(z|y)
    Sparsity:
      - L1 on (xhat_pert - xhat_ctrl)
    """
    xhat_ctrl, xhat_pert = out["xhat_ctrl"], out["xhat_pert"]

    rec_ctrl = F.mse_loss(xhat_ctrl, x_ctrl, reduction="none").mean(dim=-1).mean()
    rec_pert = F.mse_loss(xhat_pert, x, reduction="none").mean(dim=-1).mean()
    rec = w_rec_ctrl * rec_ctrl + w_rec_pert * rec_pert

    kl = gaussian_kl(out["mu"], out["logvar"], out["mu_y"], out["logvar_y"]).mean()

    delta_x = (xhat_pert - xhat_ctrl).abs().mean()  # global L1 sparsity on effect
    sparsity = delta_x

    if ("b_logits" in out) and (b_onehot is not None):
        dann = F.cross_entropy(out["b_logits"], b_onehot)
    else:
        dann = torch.tensor(0.0, device=x.device)

    total = rec + w_kl * kl + w_sparsity * sparsity + w_dann * dann
    return {
        "loss": total,
        "rec_ctrl": rec_ctrl.detach(),
        "rec_pert": rec_pert.detach(),
        "kl": kl.detach(),
        "sparsity": sparsity.detach(),
        "dann": dann.detach() if isinstance(dann, torch.Tensor) else torch.tensor(0.0, device=x.device),
    }

def _maybe_subsample_genes(x_true, x_pred, kendall_max_genes=None, rng=None):
    if kendall_max_genes is None or x_true.shape[0] <= kendall_max_genes:
        return x_true, x_pred
    rng = np.random.default_rng(0 if rng is None else rng)
    idx = rng.choice(x_true.shape[0], size=kendall_max_genes, replace=False)
    return x_true[idx], x_pred[idx]

@torch.no_grad()
def evaluate_paired_generate(
    model,
    loader,
    device,
    b_onehot_in_batch: bool = True,  # set True if your loader yields b_onehot as 5th item
    kendall_max_genes: int | None = 2000,
    sample_from_prior: bool = False,  # False: use prior mean (deterministic)
):
    model.eval()
    mse_ctrl_list, mse_pert_list = [], []
    sp_ctrl_list, sp_pert_list = [], []
    kd_ctrl_list, kd_pert_list = [], []
    # Baselines: compare to a random other cell in the same batch
    sp_base_ctrl_list, sp_base_pert_list = [], []
    kd_base_ctrl_list, kd_base_pert_list = [], []

    rng = np.random.default_rng(1234)
    
    for batch in loader:
        x, y_onehot, xp_vec, x_ctrl_match, batch = batch["pert_cell_emb"], batch["cell_type_onehot"], batch["pert_emb"], batch["ctrl_cell_emb"], batch["batch"]
        x = x.squeeze(1)
        x_ctrl_match = x_ctrl_match.squeeze(1)
        x, y_onehot, xp_vec, x_ctrl_match, batch = x.to(device), y_onehot.to(device), xp_vec.to(device), x_ctrl_match.to(device), batch.to(device)

        # ---- generate from (y, xp) ----
        xpred_ctrl, xpred_pert = model.predict_from_yxp(
            y_onehot=y_onehot,
            xp_vec=xp_vec,
            b_onehot=None,
            sample=sample_from_prior,
        )

        # ---- MSE ----
        mse_ctrl = F.mse_loss(xpred_ctrl, x_ctrl_match, reduction="none").mean(dim=-1)
        mse_pert = F.mse_loss(xpred_pert, x, reduction="none").mean(dim=-1)
        mse_ctrl_list.extend(mse_ctrl.detach().cpu().numpy().tolist())
        mse_pert_list.extend(mse_pert.detach().cpu().numpy().tolist())

        # ---- Rank metrics ----
        x_ctrl_np = x_ctrl_match.detach().cpu().numpy()
        xpred_ctrl_np = xpred_ctrl.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        xpred_np = xpred_pert.detach().cpu().numpy()

        B = x_np.shape[0]
        for i in range(B):
            # Spearman
            rho_c, _ = spearmanr(x_ctrl_np[i], xpred_ctrl_np[i]); 
            if not np.isnan(rho_c): sp_ctrl_list.append(rho_c)
            rho_p, _ = spearmanr(x_np[i], xpred_np[i]); 
            if not np.isnan(rho_p): sp_pert_list.append(rho_p)

            # Kendall (subsample for speed)
            xi_c, yi_c = _maybe_subsample_genes(x_ctrl_np[i], xpred_ctrl_np[i], kendall_max_genes)
            tau_c, _ = kendalltau(xi_c, yi_c)
            if not np.isnan(tau_c): kd_ctrl_list.append(tau_c)

            xi_p, yi_p = _maybe_subsample_genes(x_np[i], xpred_np[i], kendall_max_genes)
            tau_p, _ = kendalltau(xi_p, yi_p)
            if not np.isnan(tau_p): kd_pert_list.append(tau_p)

            # -------------------------
            # Baselines: True vs Random other True (same batch)
            # -------------------------
            if B > 1:
                # pick a random j ≠ i
                j_candidates = list(range(B))
                j_candidates.pop(i)
                j = rng.choice(j_candidates)

                # Control baseline
                base_rho_c, _ = spearmanr(x_ctrl_np[i], x_ctrl_np[j])
                if not np.isnan(base_rho_c):
                    sp_base_ctrl_list.append(base_rho_c)

                bxi_c, byi_c = _maybe_subsample_genes(
                    x_ctrl_np[i], x_ctrl_np[j], kendall_max_genes, rng=rng
                )
                base_tau_c, _ = kendalltau(bxi_c, byi_c)
                if not np.isnan(base_tau_c):
                    kd_base_ctrl_list.append(base_tau_c)

                # Perturbed baseline
                base_rho_p, _ = spearmanr(x_np[i], x_np[j])
                if not np.isnan(base_rho_p):
                    sp_base_pert_list.append(base_rho_p)

                bxi_p, byi_p = _maybe_subsample_genes(
                    x_np[i], x_np[j], kendall_max_genes, rng=rng
                )
                base_tau_p, _ = kendalltau(bxi_p, byi_p)
                if not np.isnan(base_tau_p):
                    kd_base_pert_list.append(base_tau_p)

    metrics = {
        "gen/mse_ctrl": float(np.mean(mse_ctrl_list)) if mse_ctrl_list else float("nan"),
        "gen/mse_pert": float(np.mean(mse_pert_list)) if mse_pert_list else float("nan"),
        "gen/spearman_ctrl": float(np.mean(sp_ctrl_list)) if sp_ctrl_list else float("nan"),
        "gen/spearman_pert": float(np.mean(sp_pert_list)) if sp_pert_list else float("nan"),
        "gen/kendall_ctrl": float(np.mean(kd_ctrl_list)) if kd_ctrl_list else float("nan"),
        "gen/kendall_pert": float(np.mean(kd_pert_list)) if kd_pert_list else float("nan"),
         # baselines (true vs random other true in batch)
        "base/spearman_ctrl": float(np.mean(sp_base_ctrl_list)) if sp_base_ctrl_list else float("nan"),
        "base/spearman_pert": float(np.mean(sp_base_pert_list)) if sp_base_pert_list else float("nan"),
        "base/kendall_ctrl": float(np.mean(kd_base_ctrl_list)) if kd_base_ctrl_list else float("nan"),
        "base/kendall_pert": float(np.mean(kd_base_pert_list)) if kd_base_pert_list else float("nan"),
    }
    return metrics