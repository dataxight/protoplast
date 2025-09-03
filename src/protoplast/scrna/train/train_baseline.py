from tqdm import tqdm
import wandb
from contextlib import nullcontext
from cell_load.data_modules import PerturbationDataModule

import torch.optim as optim
import torch
import torch.nn.functional as F
import os

from protoplast.scrna.models.baseline import BaselinePerturbModel
from protoplast.scrna.train.utils import _to_BD, save_checkpoint

@torch.no_grad()
def evaluate_epoch(model, dataloader, device="cuda"):
    model.eval()
    total, n = 0.0, 0
    sum_ctrl = 0.0
    sum_pert = 0.0
    sum_delta = 0.0

    for batch in dataloader:
        x = batch["pert_cell_emb"].to(device)
        y = batch["cell_type_onehot"].to(device)
        xp = batch["pert_emb"].to(device)
        x_ctrl_match = batch["ctrl_cell_emb"].to(device)

        # sanitize shapes
        x           = _to_BD(x)
        x_ctrl_match= _to_BD(x_ctrl_match)

        x_ctrl_pred, delta_pred, x_pred = model(y, xp)
        x_ctrl_pred= _to_BD(x_ctrl_pred)
        x_pred     = _to_BD(x_pred)
        delta_pred = _to_BD(delta_pred)
        true_delta = _to_BD(x - x_ctrl_match)

        # losses
        loss_ctrl  = F.mse_loss(x_ctrl_pred, x_ctrl_match, reduction="mean")
        loss_pert  = F.mse_loss(x_pred,      x,            reduction="mean")
        loss_delta = F.mse_loss(delta_pred,  true_delta,   reduction="mean")
        loss = loss_ctrl + loss_pert + loss_delta

        bs = x.size(0)
        total += float(loss.item()) * bs
        sum_ctrl  += float(loss_ctrl.item()) * bs
        sum_pert  += float(loss_pert.item()) * bs
        sum_delta += float(loss_delta.item()) * bs
        n += bs

    if n == 0:
        return {"loss": 0.0, "loss_ctrl": 0.0, "loss_pert": 0.0, "loss_delta": 0.0}
    return {
        "loss": total / n,
        "loss_ctrl": sum_ctrl / n,
        "loss_pert": sum_pert / n,
        "loss_delta": sum_delta / n,
    }

def train_baseline_epoch(
    model,
    dataloader,
    optimizer,
    epoch: int,
    device: str = "cuda",
    use_amp: bool = True,
    log_steps: bool = True,
    project_step_offset: int = 0,  # pass (epoch-1)*len(dataloader) if you want global step
):
    model.train()
    total_loss = 0.0
    total_ctrl = 0.0
    total_pert = 0.0
    total_delta= 0.0
    n_samples  = 0

    amp_ctx = torch.cuda.amp.autocast if (use_amp and torch.cuda.is_available()) else nullcontext
    scaler = getattr(train_baseline_epoch, "_scaler", None)
    if scaler is None and use_amp and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        train_baseline_epoch._scaler = scaler

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar, 1):
        x = batch["pert_cell_emb"].to(device)
        y = batch["cell_type_onehot"].to(device)
        xp = batch["pert_emb"].to(device)
        x_ctrl_match = batch["ctrl_cell_emb"].to(device)

        # input shapes
        x           = _to_BD(x)
        x_ctrl_match= _to_BD(x_ctrl_match)

        with amp_ctx():
            x_ctrl_pred, delta_pred, x_pred = model(y, xp)

            x_ctrl_pred= _to_BD(x_ctrl_pred)
            x_pred     = _to_BD(x_pred)
            delta_pred = _to_BD(delta_pred)
            true_delta = _to_BD(x - x_ctrl_match)

            loss_ctrl  = F.mse_loss(x_ctrl_pred, x_ctrl_match, reduction="mean")
            loss_pert  = F.mse_loss(x_pred,      x,            reduction="mean")
            loss_delta = F.mse_loss(delta_pred,  true_delta,   reduction="mean")
            loss = loss_ctrl + loss_pert + loss_delta

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.detach().item()) * bs
        total_ctrl += float(loss_ctrl.detach().item()) * bs
        total_pert += float(loss_pert.detach().item()) * bs
        total_delta+= float(loss_delta.detach().item()) * bs
        n_samples  += bs

        if log_steps:
            wandb.log({
                "train/loss_step": float(loss.detach().item()),
                "train/loss_ctrl_step": float(loss_ctrl.detach().item()),
                "train/loss_pert_step": float(loss_pert.detach().item()),
                "train/loss_delta_step": float(loss_delta.detach().item()),
                "epoch": epoch,
                "step": project_step_offset + step
            })

        pbar.set_postfix({"loss": f"{total_loss / n_samples:.4f}"})

    avg = {
        "loss": total_loss / max(1, n_samples),
        "loss_ctrl": total_ctrl / max(1, n_samples),
        "loss_pert": total_pert / max(1, n_samples),
        "loss_delta": total_delta / max(1, n_samples),
    }
    return avg

def main():
    # Create data module
    max_epoch = 30
    dm = PerturbationDataModule(
        toml_config_path="/home/tphan/Softwares/protoplast/notebooks/vcc-training.toml",
        embed_key=None, 
        num_workers=8,
        batch_col="batch_var",
        pert_col="target_gene",
        cell_type_key="cell_type",
        control_pert="non-targeting",
        use_scplode = True,
        perturbation_features_file="/home/tphan/state/state/competition_support_set/ESM2_pert_features.pt",
        output_space="gene",
        basal_mapping_strategy="random",
        n_basal_samples=1,
        should_yield_control_cells=True,
        batch_size=16,
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Create model
    G = 18080           # genes
    n_cell_lines = 5
    pert_d = 5120   # genes + control

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_epoch = 10
    max_epoch = 30
    model_dir = "baseline-delta-pert-emb"
    last_ck = f"{model_dir}/epoch={start_epoch}.pt"

    model = BaselinePerturbModel(G, n_cell_lines, pert_d).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = dm.train_dataloader()

    if os.path.exists(last_ck):
        ckpt = torch.load(last_ck, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
    else:
        start_epoch = 1

    # init wandb once
    wandb.init(project="vcc-simple", config={
        "lr": 1e-3,
        "epochs": max_epoch,
        "batch_size": train_loader.batch_size,
    })

    best_val = float("inf")
    for epoch in range(1, max_epoch + 1):
        # ---- train ----
        train_metrics = train_baseline_epoch(
            model, train_loader, optimizer,
            epoch=epoch, device=device, use_amp=True,
            log_steps=True, project_step_offset=(epoch-1)*len(train_loader)
        )
        wandb.log({f"train/{k}_epoch": v for k, v in train_metrics.items()} | {"epoch": epoch})

        # ---- validate ----
        val_metrics = evaluate_epoch(model, val_loader, device=device)
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch})

        # ---- test (optional each epoch; or only when best) ----
        test_metrics = evaluate_epoch(model, test_loader, device=device)
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()} | {"epoch": epoch})

        # ---- track best checkpoint by val loss ----
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val": best_val
            }, "best.pt")
            wandb.run.summary["best_val_loss"] = best_val

if __name__ == "__main__":
    main()