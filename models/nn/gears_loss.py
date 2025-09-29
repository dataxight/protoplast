import torch
import torch.nn.functional as F

@torch.no_grad()
def _signed_with_threshold(x: torch.Tensor, tau: float = 0.0) -> torch.Tensor:
    """
    Sign with a deadzone around 0: returns -1, 0, or +1 per element.
    tau>0 makes small deltas count as 'no change'.
    """
    if tau > 0:
        mask = x.abs() > tau
        return torch.where(mask, torch.sign(x), torch.zeros_like(x))
    return torch.sign(x)

def gears_autofocus_direction_loss(
    pred: torch.Tensor,   # [B, S, G]  predicted post-perturbation expression
    true: torch.Tensor,   # [B, S, G]  true post-perturbation expression
    ctrl: torch.Tensor,   # [B, S, G]  matched control expression for the same cells
    *,
    gamma: float = 1.0,   # exponent increment for autofocus (paper uses 2+gamma)
    lam: float = 0.1,     # weight λ for the direction loss
    tau: float = 0.0,     # deadzone threshold for sign() (treat |delta|<=tau as 0)
    eps: float = 1e-8,    # numerical stability for the autofocus term
    reduction: str = "mean"
):
    """
    L_total = L_autofocus + λ * L_direction

    L_autofocus = E_{b,s,g} |g - ĝ|^{(2+γ)}
    L_direction = E_{b,s,g} ( sign(g - g^ctrl) - sign(ĝ - g^ctrl) )^2
    """
    assert pred.shape == true.shape == ctrl.shape, "pred/true/ctrl must have same shape [B,S,G]"
    p = pred
    t = true
    c = ctrl

    # ----- Autofocus: raise error to (2 + gamma) with absolute value -----
    # Clamp tiny values so gradients are well-behaved around 0 when gamma<0 (if you ever try)
    err = (t - p).abs().clamp_min(eps)
    L_autofocus_elem = err.pow(2.0 + gamma)  # [B,S,G]

    # ----- Direction-aware term: squared difference between signs -----
    # Compare direction relative to control for true vs predicted
    dt = t - c
    dp = p - c
    s_true = _signed_with_threshold(dt, tau=tau)
    s_pred = _signed_with_threshold(dp, tau=tau)
    L_dir_elem = (s_true - s_pred) ** 2  # [B,S,G]; values in {0,1,4}

    # Reductions (paper averages over perturbations, cells, genes)
    if reduction == "mean":
        L_autofocus = L_autofocus_elem.mean()
        L_direction = L_dir_elem.mean()
    elif reduction == "sum":
        L_autofocus = L_autofocus_elem.sum()
        L_direction = L_dir_elem.sum()
    else:
        # no reduction: return elementwise tensors too
        L_autofocus, L_direction = L_autofocus_elem, L_dir_elem

    L_total = L_autofocus + lam * L_direction
    return L_total, L_autofocus, L_direction
