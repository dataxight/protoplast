"""
Training script for the baseline model.
"""

import torch
import lightning as L
import numpy as np
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

# Import the data module and baseline model
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule
from models.baseline import BaselineModel

# Set tensor core precision for better performance
torch.set_float32_matmul_precision('medium')


def main():
    """Main training function for baseline model."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train baseline perturbation model")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint path to resume from")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh training (ignore checkpoints)")
    args = parser.parse_args()
    
    L.seed_everything(42, workers=True)

    gene_names = open("gene_names.csv", "r").read().splitlines()
    hvg_mask = np.isin(gene_names, open("hvg-2000.txt", "r").read().splitlines())
    hvg_mask = torch.tensor(hvg_mask)
    
    # Set up data module
    dm = PerturbationDataModule(
        config_path="configs/data.toml",
        pert_embedding_file="/mnt/hdd2/tan/competition_support_set/ESM2_pert_features.pt",
        batch_size=16,
        group_size_S=128,
        num_workers=4
    )
    dm.setup(stage="fit")
    
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
    embedding_dim = sample_batch["pert_cell_emb"].shape[-1]
    pert_emb_dim = sample_batch["pert_emb"].shape[-1]
    n_cell_types = sample_batch["cell_type_onehot"].shape[-1]
    n_batches = sample_batch["batch_onehot"].shape[-1]
    
    print(f"Data dimensions:")
    print(f"  n_genes: {n_genes}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  pert_emb_dim: {pert_emb_dim}")
    print(f"  n_cell_types: {n_cell_types}")
    print(f"  n_batches: {n_batches}")
    
    # Check for existing checkpoint to resume training
    import glob
    import os
    
    checkpoint_dir = "checkpoints/baseline/"
    checkpoint_to_load = None
    
    # Handle command line arguments for checkpoint loading
    if args.no_resume:
        print("--no-resume flag set, starting fresh training")
        checkpoint_to_load = None
    elif args.checkpoint:
        # Use specific checkpoint provided
        if os.path.exists(args.checkpoint):
            checkpoint_to_load = args.checkpoint
            print(f"Using specified checkpoint: {checkpoint_to_load}")
        else:
            print(f"Specified checkpoint not found: {args.checkpoint}")
            print("Starting fresh training")
    elif args.resume:
        # Find latest checkpoint automatically
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        if checkpoint_files:
            checkpoint_to_load = max(checkpoint_files, key=os.path.getmtime)
            print(f"Auto-resuming from latest checkpoint: {checkpoint_to_load}")
        else:
            print("No checkpoints found, starting fresh training")
    else:
        # Interactive mode - ask user
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            print(f"Found existing checkpoint: {latest_checkpoint}")
            
            # Ask user if they want to resume from checkpoint
            resume_choice = input("Resume from checkpoint? (y/n): ").lower().strip()
            if resume_choice in ['y', 'yes']:
                checkpoint_to_load = latest_checkpoint
                print(f"Will resume training from: {checkpoint_to_load}")
            else:
                print("Starting fresh training")
    
    # Create baseline model
    model = BaselineModel(
        d_h=512,  # Hidden dimension
        d_f=2048,  # Bottleneck dimension
        n_genes=n_genes,
        embedding_dim=embedding_dim,
        pert_emb_dim=pert_emb_dim,
        n_cell_types=n_cell_types,
        n_batches=n_batches,
        hvg_mask=hvg_mask,
        dropout=0.1,
        lr=1e-3,
        wd=1e-4
    )
    
    # Initialize embeddings based on data dimensions
    model._initialize_embeddings_if_needed({
        "cell_type_onehot": sample_batch["cell_type_onehot"],
        "batch_onehot": sample_batch["batch_onehot"]
    })
    
    # Manually load checkpoint weights if available (avoiding optimizer state issues)
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        print(f"Manually loading model weights from: {checkpoint_to_load}")
        
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            # Filter state dict to match current model
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            
            for key, value in state_dict.items():
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        print(f"⚠️  Shape mismatch for {key}: {model_state_dict[key].shape} vs {value.shape}")
                else:
                    print(f"⚠️  Key not in model: {key}")
            
            # Load the filtered weights
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                print(f"🔧 Missing keys (will be randomly initialized): {len(missing_keys)} keys")
            
            if unexpected_keys:
                print(f"🔧 Unexpected keys (ignored): {len(unexpected_keys)} keys")
            
            print(f"✅ Loaded {len(filtered_state_dict)} parameters from checkpoint")
            print("🚀 Starting training with loaded weights (fresh optimizer state)")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint manually: {e}")
            print("🔄 Starting training from scratch")
            checkpoint_to_load = None  # Clear checkpoint path to avoid trainer issues
    else:
        print("Created new model - starting training from scratch")
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        dirpath="checkpoints/baseline-mmd/",
        filename="baseline-{epoch:02d}-{val_loss:.4f}"
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )
    
    # Set up logger
    logger = CSVLogger("logs", name="baseline")
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="auto",
        devices=1,
        precision="16-mixed",  # Use mixed precision for efficiency
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        enable_progress_bar=True
    )
    
    # Train the model (always start fresh training since we manually loaded weights)
    print("Starting training...")
    trainer.fit(model, dm)
    
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best training loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
