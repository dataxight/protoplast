"""
Training script for the baseline model.
"""

import torch
import lightning as L
import numpy as np
import logging
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

# Import the data module and baseline model
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule
from models.baseline import BaselineModel
from models.perturbation_transformer import PerturbationTransformer
import pickle


# Set tensor core precision for better performance
torch.set_float32_matmul_precision('medium')


def main():
    """Main training function for baseline model."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train baseline perturbation model")
    parser.add_argument("--mean-target-map", type=str, help="Path to mean target map", required=True)
    parser.add_argument("--mean-target-addresses", type=str, help="Path to mean target addresses", required=True)
    parser.add_argument("--hvg-gene-names", type=str, help="Path to hvg gene names", required=True)
    parser.add_argument("--gene-names", type=str, help="Path to gene names", required=True)
    args = parser.parse_args()
    
    L.seed_everything(42, workers=True)

    gene_names = open(args.gene_names, "r").read().splitlines()
    hvg_gene_names = open(args.hvg_gene_names, "r").read().splitlines()
    hvg_mask = np.isin(gene_names, hvg_gene_names)
    hvg_mask = torch.tensor(hvg_mask)
    
    # Set up data module
    dm = PerturbationDataModule(
        config_path="configs/data.toml",
        pert_embedding_file="/mnt/hdd2/tan/competition_support_set/ESM2_pert_features.pt",
        batch_size=64,
        group_size_S=128,
        num_workers=8
    )
    dm.setup(stage="fit")
    
    mean_target_map, mean_target_addresses = torch.load(args.mean_target_map, map_location="cpu"), pickle.load(open(args.mean_target_addresses, "rb"))
    
    # Infer dimensions from the first batch
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print("len train_loader", len(train_loader))
    print("len val_loader", len(val_loader))
    
    # Check if validation loader is working
    try:
        val_sample = next(iter(val_loader))
        print("Validation loader is working")
    except Exception as e:
        print(f"Validation loader issue: {e}")
    
    sample_batch = next(iter(train_loader))
    
    n_genes = sample_batch["pert_cell_g"].shape[-1]
    pert_emb_dim = sample_batch["pert_emb"].shape[-1]
    n_cell_types = sample_batch["cell_type_onehot"].shape[-1]
    n_batches = sample_batch["batch_onehot"].shape[-1]
    
    print(f"Data dimensions:")
    print(f"  n_genes: {n_genes}")
    print(f"  pert_emb_dim: {pert_emb_dim}")
    print(f"  n_cell_types: {n_cell_types}")
    print(f"  n_batches: {n_batches}")
    
    # Check for existing checkpoint to resume training
    import glob
    import os
    
    # Create baseline model
    model = PerturbationTransformer(
        d_h=512,  # Hidden dimension
        d_f=256,  # Bottleneck dimension
        n_genes=n_genes,
        embedding_dim=len(hvg_gene_names),
        pert_emb_dim=pert_emb_dim,
        n_cell_types=n_cell_types,
        n_batches=n_batches,
        hvg_mask=hvg_mask,
        dropout=0.1,
        mean_target_map=mean_target_map,
        mean_target_addresses=mean_target_addresses,
        lr=1e-3,
        wd=1e-4
    )
    
    # Initialize embeddings based on data dimensions
    model._initialize_embeddings_if_needed({
        "cell_type_onehot": sample_batch["cell_type_onehot"],
        "batch_onehot": sample_batch["batch_onehot"]
    })
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        dirpath="checkpoints/baseline-pds-hvg-gears-transformer/",
        filename="baseline-{epoch:02d}-{val_loss:.4f}"
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )
    
    # Set up logger
    logger = CSVLogger("logs", name="baseline-pds-hvg-gears-transformer")
    
    # Create trainer
    logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)
    trainer = L.Trainer(
        max_epochs=40,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="auto",
        devices=1,
        precision="16-mixed",  # Use mixed precision for efficiency
        gradient_clip_val=1.0,
        log_every_n_steps=2,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        enable_model_summary=True,
        detect_anomaly=True
    )
    
    # Train the model (always start fresh training since we manually loaded weights)
    print("Starting training...")
    trainer.fit(model, dm)
    
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best training loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
