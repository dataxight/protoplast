import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger

# Import your modules
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule
from models.perturbation_transformer import PerturbationTransformerModel

torch.set_float32_matmul_precision('medium')

def main():
    print("=== Starting Debug Training ===")
    
    # Minimal data module setup
    dm = PerturbationDataModule(
        config_path="configs/data.toml",
        pert_embedding_file="/mnt/hdd2/tan/competition_support_set/ESM2_pert_features.pt",
        batch_size=4,  # Smaller batch size
        group_size_S=4,  # Smaller group size
        num_workers=0,
        prefetch_factor=None  # No multiprocessing
    )
    
    print("Setting up data module...")
    dm.setup(stage="fit")
    print("Data module setup complete!")
    
    # Test data loading
    print("Testing data loading...")
    try:
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        
        print("Getting first batch...")
        first_batch = next(iter(train_loader))
        print(f"First batch keys: {first_batch.keys() if isinstance(first_batch, dict) else type(first_batch)}")
        print("Data loading works!")
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # Get dimensions
    n_genes = 18080
    pert_emb_dim = 5120
    n_cell_types = 1
    n_batches = 49
    
    print("Initializing model...")
    model = PerturbationTransformerModel(
        datamodule=dm,
        d_h=64,  # Much smaller for testing
        n_genes=n_genes,
        pert_emb_dim=pert_emb_dim,
        n_cell_types=n_cell_types,
        n_batches=n_batches,
        n_transformer_layers=2,  # Fewer layers
        n_heads=4,  # Fewer heads
        dropout=0.1,
        d_ff=256,  # Smaller FF
        d_x=128,  # Smaller bottleneck
        lr=1e-4,
        wd=1e-5,
        lr_scheduler_freq=1,
        lr_scheduler_patience=10,
        lr_scheduler_factor=0.5,
        mmd_blur=0.1
    )
    print("Model initialized!")
    
    # Minimal trainer
    trainer = L.Trainer(
        max_epochs=2,  # Just 2 epochs for testing
        accelerator="cpu",  # Force CPU to avoid GPU issues
        devices=1,
        logger=CSVLogger("logs/", name="debug"),
        enable_checkpointing=False,  # Disable checkpointing
        enable_model_summary=True,
        log_every_n_steps=1
    )
    
    print("Starting training...")
    try:
        trainer.fit(model, dm)  # Use dm instead of individual dataloaders
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
