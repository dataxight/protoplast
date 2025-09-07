import torch
import numpy as np

def _to_BD(t):
    # Coerce to (B, G) and contiguous
    if t.ndim == 3 and t.size(1) == 1: t = t.squeeze(1)
    if t.ndim == 3 and t.size(-1) == 1: t = t.squeeze(-1)
    if t.ndim == 1: t = t.unsqueeze(0)
    if t.ndim != 2: t = t.view(t.size(0), -1)
    return t.contiguous()

def save_checkpoint(model, optimizer, epoch, model_dir):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, f"{model_dir}/epoch={epoch}.pt")

def make_onehot_encoding_map(labels):
    labels = list(labels)  # make sure it's a list
    n = len(labels)
    return {
        label: torch.tensor(np.eye(n, dtype=int)[i])
        for i, label in enumerate(labels)
    }