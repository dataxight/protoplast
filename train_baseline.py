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

def load_model(checkpoint_path: str, device: str, mean_target_map, mean_target_addresses):
    """
    Load the baseline model from checkpoint.
    """
    print(f"🔧 Loading baseline model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get hyperparameters from checkpoint
    if 'hyper_parameters' in checkpoint:
        print("Hyperparameters found in checkpoint")
        hparams = checkpoint['hyper_parameters']
    else:
        # Default hyperparameters if not saved
        print("No hyperparameters found in checkpoint, using default hyperparameters")
        hparams = {
            'd_h': 672,
            'd_f': 512,
            'n_perts': 151,
            'n_genes': 18080,
            'embedding_dim': 128,
            'pert_emb_dim': 5120,
            'dropout': 0.1,
            'mean_target_map': mean_target_map,
            'mean_target_addresses': mean_target_addresses
        }
    
    # Create model with hyperparameters
    model = BaselineModel(
        n_perts=hparams.get('n_perts', 151),
        mean_target_map=hparams.get('mean_target_map', {}),
        mean_target_addresses=hparams.get('mean_target_addresses', {}),
        d_h=hparams.get('d_h', 672),
        d_f=hparams.get('d_f', 512),
        n_genes=hparams.get('n_genes', 18080),
        embedding_dim=hparams.get('embedding_dim', 2000),
        pert_emb_dim=hparams.get('pert_emb_dim', 5120),
        dropout=hparams.get('dropout', 0.1)
    )
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Filter out optimizer states and other non-model parameters
        model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() 
                          if k.startswith('model.') or not k.startswith('optimizer')}
        model.load_state_dict(model_state_dict, strict=False)
    
    #model.eval()
    model = model.to(device)
    
    print(f"✅ Baseline model loaded successfully on {device}")
    return model


def main():
    """Main training function for baseline model."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train baseline perturbation model")
    parser.add_argument("--mean-target-map", type=str, help="Path to mean target map", required=False)
    parser.add_argument("--mean-target-addresses", type=str, help="Path to mean target addresses", required=False)
    parser.add_argument("--data-config", type=str, help="Path to data config TOML file", required=True)
    # parser.add_argument("--hvg-gene-names", type=str, help="Path to hvg gene names", required=True)
    #parser.add_argument("--gene-names", type=str, help="Path to gene names", required=True)
    parser.add_argument("--checkpoint-path", type=str, help="Path to the recent checkpoint path", required=False)
    parser.add_argument("--from-epoch", type=int, help="Start from this epoch", required=False)
    parser.add_argument("--max-epoch", type=int, help="Max epoch", required=False, default=20)

    args = parser.parse_args()
   
    from_epoch = args.from_epoch  or 0
    L.seed_everything(42, workers=True)

    #gene_names = open(args.gene_names, "r").read().splitlines()
    # hvg_gene_names = open(args.hvg_gene_names, "r").read().splitlines()
    # hvg_mask = np.isin(gene_names, hvg_gene_names)
    # hvg_mask = torch.tensor(hvg_mask)
    
    # Set up data module
    dm = PerturbationDataModule(
        config_path=args.data_config,
        pert_embedding_file="/mnt/hdd2/tan/competition_support_set/ESM2_pert_features.pt",
        batch_size=64,
        group_size_S=64,
        num_workers=16,
        block_size=256
    )
    dm.setup(stage="fit")
    
    if args.mean_target_map and args.mean_target_addresses:
        mean_target_map, mean_target_addresses = torch.load(args.mean_target_map, map_location="cpu"), pickle.load(open(args.mean_target_addresses, "rb"))
    else:
        mean_target_map, mean_target_addresses = {}, {}
    
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
    
    if "pert_cell_g" in sample_batch:
        n_genes = sample_batch["pert_cell_g"].shape[-1]
    else:
        print("pert_cell_g is not available, assuming using all genes for embedding")
        n_genes = sample_batch["pert_cell_emb"].shape[-1]
    pert_emb_dim = sample_batch["pert_emb"].shape[-1]
    embedding_dim = sample_batch["pert_cell_emb"].shape[-1]
    n_cell_types = sample_batch["cell_type_onehot"].shape[-1]
    n_batches = sample_batch["batch_onehot"].shape[-1]
    n_perts = 151
    
    print(f"Data dimensions:")
    print(f"  n_genes: {n_genes}")
    print(f"  pert_emb_dim: {pert_emb_dim}")
    print(f"  n_cell_types: {n_cell_types}")
    print(f"  n_batches: {n_batches}")
    
    # Create baseline model
    if args.checkpoint_path:
        print("Loading model from checkpoint")
        model = load_model(args.checkpoint_path, "cuda", mean_target_map, mean_target_addresses)
    else:
        print("Create model from scratch")
        model = BaselineModel(
            d_h=672,  # Hidden dimension
            d_f=512,  # Bottleneck dimension
            n_perts=n_perts,
            n_genes=n_genes,
            embedding_dim=embedding_dim,
            pert_emb_dim=pert_emb_dim,
            n_cell_types=n_cell_types,
            n_batches=n_batches,
            dropout=0.1,
            mean_target_map=mean_target_map,
            mean_target_addresses=mean_target_addresses,
            lr=1e-3,
            wd=1e-4,
            cls_weight=1.0,
            recon_weight=0.1,
            kl_weight=1e-3
        )
    
        # Initialize embeddings based on data dimensions
        model._initialize_embeddings_if_needed({
            "cell_type_onehot": sample_batch["cell_type_onehot"],
            "batch_onehot": sample_batch["batch_onehot"]
        })
        print(model)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        dirpath="checkpoints/baseline-scvi-cls-heavy/",
        filename="baseline-{epoch:02d}-{val_loss:.4f}"
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )
    
    # Set up logger
    logger = CSVLogger("logs", name="baseline-scvi-cls-heavy")
    
    # Create trainer
    logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)
    trainer = L.Trainer(
        max_epochs=args.max_epoch,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="auto",
        devices=1,
        precision="16-mixed",  # Use mixed precision for efficiency
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        enable_model_summary=True
    )
    
    # Train the model (always start fresh training since we manually loaded weights)
    print("Starting training...")
    trainer.fit(model, dm)
    
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best training loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
