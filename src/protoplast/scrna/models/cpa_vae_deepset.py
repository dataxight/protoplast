import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from geomloss import SamplesLoss
sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9, debias=True)

# ---- utilities ----
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
    def forward(self, x): return self.proj(x) + self.net(x)

class LogCoshLoss(nn.Module):
    def forward(self, y_hat, y):
        e = (y - y_hat) / 3
        return torch.mean(torch.log(torch.cosh(e) + 1e-12))

# ---- permutation-invariant encoder for control set ----
class SetEncoder(nn.Module):
    """
    DeepSets-style posterior q(z | S_ctrl):
      phi: per-cell embedding of x (G -> d_phi)
      pool: mean over K
      rho: pooled -> (mu, logvar)
    """
    def __init__(self, G, d_phi=512, d_rho=512, d_z=64, p=0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(G, d_phi),
            nn.GELU(),
            nn.LayerNorm(d_phi),
            nn.Dropout(p),
            ResidualMLP(d_phi, d_phi, d_phi, p=p),
        )
        self.rho = nn.Sequential(
            nn.Linear(d_phi, d_rho),
            nn.GELU(),
            nn.LayerNorm(d_rho),
            nn.Dropout(p),
        )
        self.mu = nn.Linear(d_rho, d_z)
        self.logvar = nn.Linear(d_rho, d_z)

    def forward(self, x_ctrl_set):  # [B, K, G]
        B, K, G = x_ctrl_set.shape
        x_flat = x_ctrl_set.view(B * K, G)
        h = self.phi(x_flat)                  # [B*K, d_phi]
        h = h.view(B, K, -1).mean(dim=1)      # pooled [B, d_phi]
        h = self.rho(h)                       # [B, d_rho]
        return self.mu(h), self.logvar(h)

# ---- main model ----
class CPAVAE_Simple(nn.Module):
    def __init__(
        self,
        G: int,
        n_cell_lines: int,
        d_xp: int,
        d_z: int = 64,
        d_dec: int = 512,
        delta_hid: int = 128,
        factorized_decoder_rank: int = 512,
        d_y_prior: int = 128,
        dropout: float = 0.1,
        use_y_prior: bool = True,
    ):
        super().__init__()
        self.G = G
        self.use_y_prior = use_y_prior

        # control-set posterior
        self.set_enc = SetEncoder(G, d_phi=512, d_rho=512, d_z=d_z, p=dropout)

        # to encode the try pert cells
        self.enc_cell = nn.Sequential(
            nn.Linear(G, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, d_z)
        )
        
        # optional p(z|y) prior
        if use_y_prior:
            self.prior_net = nn.Sequential(
                nn.Linear(n_cell_lines, d_y_prior),
                nn.GELU(),
                nn.LayerNorm(d_y_prior),
                nn.Dropout(dropout),
            )
            self.prior_mu = nn.Linear(d_y_prior, d_z)
            self.prior_logvar = nn.Linear(d_y_prior, d_z)

        # perturbation operator in latent
        self.delta = ResidualMLP(d_z + d_xp, delta_hid, d_z, p=dropout)

        # decoder p(x|z): factorized head
        self.dec_backbone = nn.Sequential(
            nn.Linear(d_z, d_dec),
            nn.GELU(),
            nn.LayerNorm(d_dec),
            nn.Dropout(dropout),
            ResidualMLP(d_dec, d_dec, d_dec, p=dropout),
        )
        # Make the decoder output μ(z) and logσ²(z) and train with a Gaussian NLL
        self.to_rank = nn.Linear(d_dec, factorized_decoder_rank, bias=False)
        self.rank_to_mu = nn.Linear(factorized_decoder_rank, G, bias=True)
        self.rank_to_logvar = nn.Linear(factorized_decoder_rank, G, bias=True)  # new
        self.min_logvar = -6.0  # floor σ≈0.05
        self.max_logvar =  2.0  # cap  σ≈2.7

    # --- helpers ---
    def decode_params(self, z):
        h = self.dec_backbone(z)
        r = self.to_rank(h)
        mu = self.rank_to_mu(r)
        logvar = self.rank_to_logvar(r).clamp_(self.min_logvar, self.max_logvar)
        return mu, logvar
        
    def decode_sample(self, z, T: float = 1.0, sample: bool = True):
        mu, logvar = self.decode_params(z)
        if sample:
            std = (logvar.mul(0.5).exp()) * T
            eps = torch.randn_like(std)
            return torch.clamp(mu + std * eps, min=0), mu, logvar
            # return mu + std * eps, mu, logvar
        else:
            return mu, mu, logvar
            
    def _reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_backbone(z)
        r = self.to_rank(h)
        return self.rank_to_gene(r)

    def prior(self, y_onehot):
        h = self.prior_net(y_onehot)
        return self.prior_mu(h), self.prior_logvar(h)

    def apply_operator(self, z_ctrl, xp_vec):
        dz = self.delta(torch.cat([z_ctrl, xp_vec], dim=-1))
        return z_ctrl + dz, dz

    @torch.no_grad()
    def predict_from_yxp(
        self,
        y_onehot: torch.Tensor,           # [B, n_cell_lines] (only used if use_y_prior=True and no control set)
        xp_vec: torch.Tensor,             # [B, d_xp]
        x_ctrl_set: Optional[torch.Tensor] = None,  # [B, K, G] control set for (y,b)
        sample: bool = False,
        temperature: float = 1
    ):
        device = next(self.parameters()).device
        y_onehot = y_onehot.to(device)
        xp_vec = xp_vec.to(device)
        if x_ctrl_set is not None: x_ctrl_set = x_ctrl_set.to(device)

        if x_ctrl_set is not None:
            mu_c, lv_c = self.set_enc(x_ctrl_set)
        elif self.use_y_prior:
            mu_c, lv_c = self.prior(y_onehot)
        else:
            raise ValueError("Need control set or y-prior for prediction.")

        z_ctrl = self._reparam(mu_c, lv_c)
        xhat_ctrl, mu_ctrl, logvar_ctrl = self.decode_sample(z_ctrl, sample=sample, T = temperature)
        
        z_pert, _ = self.apply_operator(z_ctrl, xp_vec)
        xhat_pert, mu_pert, logvar_pert = self.decode_sample(z_pert, sample=sample, T = temperature)
        return xhat_ctrl, xhat_pert, mu_pert, logvar_pert

    def forward(
        self,
        x_pert: torch.Tensor,             # [B, G] observed perturbed
        y_onehot: torch.Tensor,  # [B, n_cell_lines] 
        xp_vec: torch.Tensor,             # [B, d_xp]
        x_ctrl_set: torch.Tensor        # [B, K, G] control set for same (y,b)
    ):
        # posterior from control set
        mu_c, lv_c = self.set_enc(x_ctrl_set)
        z_ctrl = self._reparam(mu_c, lv_c)
        xhat_ctrl, mu_ctrl, logvar_ctrl = self.decode_sample(z_ctrl, sample=False)

        # apply operator and decode perturbed
        z_pert, delta_z = self.apply_operator(z_ctrl, xp_vec)
        xhat_pert, mu_pert, logvar_pert = self.decode_sample(z_pert, sample=False)

        z_true = self.enc_cell(x_pert) 

        out = dict(
            z_ctrl=z_ctrl,
            z_true=z_true,
            z_pert=z_pert, delta_z=delta_z,
            xhat_ctrl=xhat_ctrl, xhat_pert=xhat_pert,
            mu_x_ctrl=mu_ctrl, logvar_x_ctrl=logvar_ctrl,
            mu_x_pert=mu_pert, logvar_x_pert=logvar_pert
        )

        # optional KL regularization to p(z|y)
        if self.use_y_prior and (y_onehot is not None):
            mu_y, lv_y = self.prior(y_onehot)
            out["mu_y"], out["logvar_y"] = mu_y, lv_y

        return out

### loss functions ###

def kld_normal(mu, logvar):
    # KL(q||N(0,I))
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=-1).mean()

def kld_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    # KL(N(mu_q,Σ_q) || N(mu_p,Σ_p)) diagonal
    var_q, var_p = logvar_q.exp(), logvar_p.exp()
    term = (var_q / var_p) + (mu_p - mu_q).pow(2) / var_p - 1.0 + (logvar_p - logvar_q)
    return 0.5 * torch.sum(term, dim=-1).mean()

def gaussian_nll(x, mu, logvar):
    return 0.5 * ((x - mu)**2 / logvar.exp() + logvar).mean()

logcosh = LogCoshLoss()

def sinkhorn_latent_batch(z_pred: torch.Tensor,   # [B, d_z] (z_pert)
                          z_true: torch.Tensor,   # [B, d_z]
                          cond_id: torch.Tensor,  # [B]
                          loss_fn: SamplesLoss,
                         max_per_group: int = 8) -> torch.Tensor:
    loss = 0.0
    groups = cond_id.unique()
    for gid in groups:
        idx = (cond_id == gid).nonzero(as_tuple=True)[0]
        # Subsample if too many in group
        if idx.numel() > max_per_group:
            perm = torch.randperm(idx.numel(), device=device)[:max_per_group]
            idx = idx[perm]
        loss = loss + loss_fn(z_pred[idx], z_true[idx])
    return loss / max(len(groups), 1)


def draw_pred_samples(mu, logvar, S=3, T=1.0):
    std = (0.5*logvar).exp() * T
    eps = torch.randn(mu.shape[0], S, mu.shape[1], device=mu.device)
    return (mu.unsqueeze(1) + std.unsqueeze(1)*eps).reshape(-1, mu.shape[1])  # [(B*S), G]

def sample_log1p_from_pred(x_pred, temperature=1.0):
    """
    x_pred: [B, G] predicted log1p-normalized expression (float)
    Returns: [B, G] sampled log1p(counts)
    """
    # step 1–2: invert log1p
    lam = torch.expm1(x_pred)            # expected counts
    lam = lam.clamp(min=1e-8)            # numerical safety
    lam = lam * temperature              # optional dispersion scaling

    # step 3: sample Poisson counts
    counts = torch.poisson(lam)

    # step 4: map back to log1p space
    return torch.log1p(counts)
    
def sinkhorn_gene_batch(mu_x_pert: torch.Tensor,  # [B, G]
                        logvar_x_pert: torch.Tensor,
                        x_pert: torch.Tensor,     # [B, G]
                        cond_id: torch.Tensor,    # [B] ints
                        loss_fn: SamplesLoss) -> torch.Tensor:
    loss = 0.0
    groups = cond_id.unique()
    for gid in groups:
        idx = (cond_id == gid).nonzero(as_tuple=True)[0]
        # point clouds (sizes may differ across conditions; GeomLoss handles that)
        mu_pred = mu_x_pert[idx]          # [n_pred, G]
        logvar_pred = logvar_x_pert[idx]
        X_pred = draw_pred_samples(mu_pred, logvar_pred)
        X_pred = sample_log1p_from_pred(X_pred)
        X_true = x_pert[idx]             # [n_true, G]
        loss = loss + loss_fn(X_pred, X_true)
    return loss / max(len(groups), 1)
    

    
def compute_losses(out, x_pert, x_ctrl_set, pert_ident):
    # recon
    l_pert = gaussian_nll(x_pert,       out["mu_x_pert"], out["logvar_x_pert"])

    loss_sink_latent = sinkhorn_latent_batch(out["z_pert"], out["z_true"], pert_ident, sinkhorn)
    # loss_sink_gene = sinkhorn_gene_batch(out["mu_x_pert"], out["logvar_x_pert"], x_pert, pert_ident, sinkhorn)
    loss_sink = loss_sink_latent #+ loss_sink_gene
    
    return dict(loss=0.8 * loss_sink + 0.2 * l_pert, loss_link = loss_sink, l_pert = l_pert, loss_sink_latent=loss_sink_latent) #, loss_sink_gene=loss_sink_gene)