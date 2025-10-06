"""
Practical inference example for the baseline perturbation prediction model.
"""

import torch
import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse as sp
from models.baseline import BaselineModel
from models.perturbation_transformer import PerturbationTransformer
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule


class BaselinePredictor:
    """
    A convenient wrapper for the trained baseline model.
    """
    
    def __init__(self, checkpoint_path="checkpoints/baseline/baseline-best.ckpt"):
        """
        Initialize the predictor with a trained checkpoint.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path):
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
                'n_genes': 18080,
                'embedding_dim': 18080,
                'pert_emb_dim': 5120,
                'dropout': 0.1
            }
        
        # Create model with hyperparameters
        model = BaselineModel(
            mean_target_map=hparams.get('mean_target_map', {}),
            mean_target_addresses=hparams.get('mean_target_addresses', {}),
            d_h=hparams.get('d_h', 512),
            d_f=hparams.get('d_f', 256),
            n_genes=hparams.get('n_genes', 18080),
            embedding_dim=hparams.get('embedding_dim', 2000),
            pert_emb_dim=hparams.get('pert_emb_dim', 5120),
            dropout=hparams.get('dropout', 0.2)
        )
        
        # Load state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Filter out optimizer states and other non-model parameters
            model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() 
                              if k.startswith('model.') or not k.startswith('optimizer')}
            model.load_state_dict(model_state_dict, strict=False)
        
        model.eval()
        model = model.to(self.device)
        
        print(f"✅ Baseline model loaded successfully on {self.device}")
        return model
    
    def predict(self, ctrl_cell_emb, pert_emb, covariates):
        """
        Predict perturbation effects using the baseline model.
        
        Args:
            ctrl_cell_emb: Control cell embeddings [B, S, E]
            pert_emb: Perturbation embeddings [B, 5120]
            covariates: Dictionary with 'cell_type_onehot' and 'batch_onehot'
            
        Returns:
            Predicted perturbation effects [B, S, G]
        """
        # Move to device
        if hasattr(ctrl_cell_emb, 'to_dense'):
            ctrl_cell_emb = ctrl_cell_emb.to_dense()
        ctrl_cell_emb = ctrl_cell_emb.to(self.device)
        pert_emb = pert_emb.to(self.device)
        covariates = {k: v.to(self.device) for k, v in covariates.items()}

        with torch.no_grad():
            with torch.autocast(device_type='cuda' if self.device=='cuda' else 'cpu', 
                              dtype=torch.float16, enabled=self.device=='cuda'):
                predictions = self.model(ctrl_cell_emb, pert_emb, covariates)
        
        return predictions
    
    def predict_batch(self, batch_data):
        """
        Predict on a batch of data (similar to training format).
        
        Args:
            batch_data: Dictionary containing:
                - 'ctrl_cell_emb': Control cell embeddings
                - 'pert_emb': Perturbation embeddings  
                - 'cell_type_onehot': Cell type one-hot encodings
                - 'batch_onehot': Batch one-hot encodings
        
        Returns:
            Predicted perturbation effects
        """
        ctrl_cell_emb = batch_data['ctrl_cell_emb']
        pert_emb = batch_data['pert_emb']
        covariates = {
            'cell_type_onehot': batch_data['cell_type_onehot'],
            'batch_onehot': batch_data['batch_onehot']
        }
        
        return self.predict(ctrl_cell_emb, pert_emb, covariates)

def baseline_vcc_inference():
    """
    VCC inference using the baseline model.
    """
    checkpoint_path = "/home/tphan/Softwares/vcc-models/checkpoints/baseline-scvi-sampling/baseline-epoch=45-val_loss=1.7563.ckpt"
    
    # Define our path
    pert_counts_path = "./pert_counts_Validation.csv"
    pert_counts = pd.read_csv(pert_counts_path)
    gene_names = pd.read_csv("./gene_names.csv", header=None)
    gene_names = gene_names[0].tolist()
    
    dm = PerturbationDataModule(
        config_path="configs/data.toml",
        pert_embedding_file="/mnt/hdd2/tan/competition_support_set/ESM2_pert_features.pt",
        batch_size=8,
        group_size_S=256,
        num_workers=4
    )
    dm.setup(stage="fit")
    
    predictor = BaselinePredictor(checkpoint_path)
    adata = ad.read_h5ad("/mnt/hdd2/tan/competition_support_set_sorted/competition_train.h5")
    hvg_mask = np.where(adata.var["highly_variable"])[0]
    control_adata = adata[adata.obs["target_gene"] == "non-targeting"]
    batch_data = control_adata.obs["batch_var"]
    cell_type = "ARC_H1"

    X = None
    pert_names = []
    
    for i, row in enumerate(pert_counts.itertuples()):
        gene = row.target_gene
        print(f"Processing gene {i} / {len(pert_counts)}: {gene}")
        n_cells = row.n_cells
        
        # Randomly select n_cells from control_adata
        control_indices = np.random.choice(range(len(control_adata)), size=n_cells, replace=False)
        X_ctrl = control_adata.X[control_indices]
        X_ctrl = X_ctrl[:, hvg_mask]
        X_ctrl = X_ctrl.toarray()
        X_ctrl = torch.from_numpy(X_ctrl).float()
        X_ctrl = X_ctrl.unsqueeze(0)  # Add batch dimension [1, S, E]
        
        # Get batch and perturbation embeddings
        batch_onehot = [dm.train_ds.get_batch_onehot(batch) for batch in batch_data[control_indices]]
        pert_emb = dm.train_ds._get_pert_embedding(gene).unsqueeze(0)
        
        covariates = {
            "cell_type_onehot": dm.train_ds.get_celltype_onehot(cell_type).unsqueeze(0).expand(1, X_ctrl.shape[1], -1),
            "batch_onehot": torch.stack(batch_onehot).unsqueeze(0)
        }
        
        # Predict perturbation effects
        X_pert_hat = predictor.predict(X_ctrl, pert_emb, covariates)
        
        # Store results
        pert_names += list(np.repeat(gene, X_ctrl.shape[1]))
        
        # Reshape predictions to [S, G] and concatenate
        predictions = X_pert_hat.squeeze(0)  # Remove batch dimension
        # predictions[:, hvg_mask] = emb.squeeze(0)
        
        X = predictions if X is None else torch.cat([X, predictions], dim=0)

    # Add 10000 control cells
    control_indices = np.random.choice(range(len(control_adata)), size=10000, replace=False)
    X_ctrl = control_adata.X[control_indices]
    X_ctrl = X_ctrl.toarray()
    X_ctrl = torch.from_numpy(X_ctrl).float().to("cuda")
    pert_names += list(np.repeat("non-targeting", X_ctrl.shape[0]))
    X = torch.cat([X, X_ctrl], dim=0)
    
    # Convert to numpy and sparse matrix
    X = X.cpu().numpy()
    X = sp.csr_matrix(X)
    pert_names = np.array(pert_names)

    # Save results
    path = "baseline_vcc_inference_hvg_transformer_weight_comp_mse.h5ad"
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "target_gene": pert_names,
            },
            index=np.arange(X.shape[0]).astype(str),
        ),
        var=pd.DataFrame(index=gene_names),
    )
    adata.write_h5ad(path)
    
    print(f"\n🎉 Baseline VCC inference completed successfully!")
    print(f"Results saved to {path}")


def baseline_validation_inference():
    """
    Run inference on validation data using the baseline model.
    """
    # checkpoint_path = "checkpoints/baseline/baseline-best.ckpt"  # Update with actual path
    checkpoint_path = "/home/tphan/Softwares/vcc-models/checkpoints/baseline-scvi-sampling/baseline-epoch=45-val_loss=1.7563.ckpt"
    
    dm = PerturbationDataModule(
        config_path="configs/data.toml",
        pert_embedding_file="/mnt/hdd2/tan/competition_support_set/ESM2_pert_features.pt",
        batch_size=8,
        group_size_S=256,
        num_workers=4
    )
    dm.setup(stage="fit")
    
    # Initialize predictor
    predictor = BaselinePredictor(checkpoint_path)
    
    val_loader = dm.val_dataloader()
    if len(val_loader) == 0:
        print("❌ No validation data available")
        return
    
    X = None
    pert_names = []
    n_batches = len(val_loader)
    
    try:
        for i, batch in enumerate(val_loader):
            if i > n_batches:
                break
            print(f"Processing batch {i+1} / {len(val_loader)}: {batch['pert_name']}")
            
            ctrl_cell_emb = batch["ctrl_cell_emb"]
            pert_emb = batch["pert_emb"]
            covariates = {
                "cell_type_onehot": batch["cell_type_onehot"],
                "batch_onehot": batch["batch_onehot"]
            }
            
            # Store perturbation names
            pert_names += list(np.repeat(batch["pert_name"], ctrl_cell_emb.shape[1]))
            
            # Predict
            # predictions, emb = predictor.predict(ctrl_cell_emb, pert_emb, covariates)  # [B, S, G]
            predictions = predictor.predict(ctrl_cell_emb, pert_emb, covariates)  # [B, S, G]
            
            # Reshape to [B*S, G]
            predictions = predictions.view(-1, predictions.shape[-1])
            
            X = predictions if X is None else torch.cat([X, predictions], dim=0)

        # Convert to numpy
        if "ctrl_cell_g" in batch:
            X_ctrl = batch["ctrl_cell_g"].to_dense()
        else:
            X_ctrl = batch["ctrl_cell_emb"].to_dense()
        X_ctrl = X_ctrl.view(-1, X_ctrl.shape[-1]).numpy().astype(np.float32)
        X = X.cpu().numpy().astype(np.float32)
        X = np.concat([X, X_ctrl])
        X = sp.csr_matrix(X)
        pert_names = np.array(pert_names + ["non-targeting"] * X_ctrl.shape[0])
        print(X.shape)
        print(X_ctrl.shape)
        print(pert_names.shape)

        # Save results
        gene_names = pd.read_csv("./gene_names.csv", header=None)[0].tolist()
        ad.AnnData(
            X=X,
            obs=pd.DataFrame(
                {
                    "target_gene": pert_names,
                },
                index=np.arange(X.shape[0]).astype(str),
            ),
            var=pd.DataFrame(index=gene_names),
        ).write_h5ad("baseline_val_inference.h5ad")
        
        print(f"   Final predictions shape: {X.shape}")
        print(f"   Prediction range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"\n🎉 Baseline validation inference completed successfully!")
        print(f"Results saved to baseline_val_inference.h5ad")
        
    except Exception as e:
        print(f"❌ Error during validation inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with example data first
    # print("Testing baseline inference...")
    # test_baseline_inference()
    
    # If you have validation data available, uncomment:
    baseline_validation_inference()
    
    # For VCC competition inference, uncomment:
    # baseline_vcc_inference()
