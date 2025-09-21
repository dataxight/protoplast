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
        pert_embedding_file="/ephemeral/vcc/competition_support_set_sorted/ESM2_pert_features.pt",
        batch_size=8,
        group_size_S=256,
        num_workers=4  # Set to 0 to avoid multiprocessing issues
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
    
    # Check for existing checkpoint and detect architecture
    import glob
    import os
    
    checkpoint_dir = "checkpoints/"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    # Architecture parameters
    n_transformer_layers = 6  # Default
    d_ff = int(672 * 2.67)    # Default
    checkpoint_to_load = None
    
    if checkpoint_files:
        # Sort by modification time to get the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        checkpoint_to_load = "checkpoints/perturbation-transformer-epoch=50-train_loss=0.87.ckpt"
        print(f"Found checkpoint: {checkpoint_to_load}")
        
        # Load and analyze checkpoint to detect architecture
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            # Detect architecture
            max_layer = -1
            detected_d_ff = None
            
            for key in state_dict.keys():
                if 'transformer.layers.' in key:
                    layer_num = int(key.split('.')[2])
                    max_layer = max(max_layer, layer_num)
                if 'transformer.layers.0.mlp.gate_proj.weight' in key:
                    detected_d_ff = state_dict[key].shape[0]
            
            if max_layer >= 0:
                n_transformer_layers = max_layer + 1
            if detected_d_ff is not None:
                d_ff = detected_d_ff
                
            print(f"Detected architecture: {n_transformer_layers} layers, d_ff={d_ff}")
        except Exception as e:
            print(f"Could not analyze checkpoint: {e}")
            print("Using default architecture")

    # Initialize model with detected/default architecture
    model = PerturbationTransformerModel(
        datamodule=dm,
        d_h=672,  # Hidden dimension
        n_genes=n_genes,
        pert_emb_dim=pert_emb_dim,
        n_cell_types=n_cell_types,
        n_batches=n_batches,
        n_transformer_layers=n_transformer_layers,
        n_heads=8,
        dropout=0.1,
        d_ff=d_ff,
        d_x=2260,  # Bottleneck dimension
        lr=1e-4,
        wd=1e-5,
        lr_scheduler_freq=1,
        lr_scheduler_patience=10,
        lr_scheduler_factor=0.5,
        mmd_blur=0.05
    )

    model._initialize_embeddings_if_needed({
            "cell_type_onehot": sample_batch["cell_type_onehot"],
            "batch_onehot": sample_batch["batch_onehot"]
        })
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints/",
        filename="perturbation-transformer-{epoch:02d}-{train_loss:.2f}",
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
        precision="bf16-mixed" if torch.cuda.is_available() else 32,  # BFloat16 mixed precision for Flash Attention 2
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,  # For reproducibility
    )
    
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
                    if not key.startswith(('batch_embedding', 'cell_type_embedding')):
                        print(f"⚠️  Key not in model: {key}")
            
            # Load the filtered weights
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                print(f"🔧 Missing keys (will be randomly initialized): {len(missing_keys)} keys")
            
            print(f"✅ Loaded {len(filtered_state_dict)} parameters from checkpoint")
            print("🚀 Starting training with loaded weights (fresh optimizer state)")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint manually: {e}")
            print("🔄 Starting training from scratch")
    
    # Train the model (always start fresh training, no checkpoint resume)
    trainer.fit(model, dm)
    
    print("Training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
