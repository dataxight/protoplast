"""
Simple training script that can be run from the project root.
This avoids import issues with relative imports.
"""

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

# Import the data module and model
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule
from models.perturbation_transformer import PerturbationTransformerModel

# Set tensor core precision for better performance
torch.set_float32_matmul_precision('medium')


def main():
    """Main training function."""

    L.seed_everything(42, workers=True)
    # Set up data module
    dm = PerturbationDataModule(
        config_path="configs/data.toml",
        pert_embedding_file="/mnt/hdd2/tan/competition_support_set/ESM2_pert_features.pt",
        batch_size=64,
        group_size_S=16,
        num_workers=8  # Set to 0 to avoid multiprocessing issues
    )
    dm.setup(stage="fit")
    
    # Use known dimensions from the logs to avoid hanging on batch inspection
    # From logs: n_genes: 18080, n_cell_types: 1, n_batches: 49, pert_emb_dim: 5120
    # infer from the first batch
    train_loader = dm.train_dataloader()
    print("len train_loader", len(train_loader))
    print("len val_loader", len(dm.val_dataloader()))
    sample_batch = next(iter(train_loader))
    n_genes = sample_batch["pert_cell_emb"].shape[-1]
    pert_emb_dim = sample_batch["pert_emb"].shape[-1]
    n_cell_types = sample_batch["cell_type_onehot"].shape[-1]
    n_batches = sample_batch["batch_onehot"].shape[-1]
    
    print(f"Data dimensions:")
    print(f"  n_genes: {n_genes}")
    print(f"  pert_emb_dim: {pert_emb_dim}")
    print(f"  n_cell_types: {n_cell_types}")
    print(f"  n_batches: {n_batches}")
    
    # Initialize model with Llama backbone
    model = PerturbationTransformerModel(
        datamodule=dm,
        d_h=672,  # Hidden dimension
        n_genes=n_genes,
        pert_emb_dim=pert_emb_dim,
        n_cell_types=n_cell_types,
        n_batches=n_batches,
        n_transformer_layers=6,
        n_heads=8,
        dropout=0.1,
        d_ff=int(672 * 2.67),  # Llama default intermediate size ratio
        d_x=2260,  # Bottleneck dimension
        lr=1e-4,
        wd=1e-5,
        lr_scheduler_freq=1,
        lr_scheduler_patience=10,
        lr_scheduler_factor=0.5,
        mmd_blur=0.05
    )
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="perturbation-transformer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=15,
        mode="min"
    )
    
    # Set up logger
    logger = CSVLogger(
        save_dir="logs/",
        name="transformer-model"
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,  # Mixed precision only on GPU
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,  # For reproducibility
    )
    
    # Train the model
    trainer.fit(model, dm)
    
    print("Training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
