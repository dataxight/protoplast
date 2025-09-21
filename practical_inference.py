"""
Practical inference example showing how to use the loaded model
for real perturbation prediction tasks.
"""

import torch
import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse as sp
from models.perturbation_transformer import PerturbationTransformerModel
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule


class PerturbationPredictor:
    """
    A convenient wrapper for the trained perturbation transformer model.
    """
    
    def __init__(self, checkpoint_path="best.ckpt"):
        """
        Initialize the predictor with a trained checkpoint.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path):
        """
        Load the model with correct architecture.
        """
        print(f"🔧 Loading model from {checkpoint_path}")
        
        # Load and analyze checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Detect architecture
        max_layer = -1
        d_ff = None
        
        for key in state_dict.keys():
            if 'transformer.layers.' in key:
                layer_num = int(key.split('.')[2])
                max_layer = max(max_layer, layer_num)
            if 'transformer.layers.0.mlp.gate_proj.weight' in key:
                d_ff = state_dict[key].shape[0]
        
        n_transformer_layers = max_layer + 1
        
        # Create model
        model = PerturbationTransformerModel(
            d_h=672,
            n_genes=18080,
            pert_emb_dim=5120,
            n_transformer_layers=n_transformer_layers,
            n_heads=16,
            dropout=0.1,
            d_ff=d_ff,
            d_x=2260
        )
        
        # Load weights
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if k in model_state_dict and 
                             model_state_dict[k].shape == v.shape}
        
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        model = model.to(self.device)
        
        print(f"✅ Model loaded successfully on {self.device}")
        return model
    
    def predict(self, ctrl_cell_emb, pert_emb, covariates):
        """
        Predict perturbation effects.
        
        Args:
            ctrl_cell_emb: Control cell embeddings [B, S, G]
            pert_emb: Perturbation embeddings [B, D]
            covariates: Dictionary with 'cell_type_onehot' and 'batch_onehot'
            
        Returns:
            Predicted perturbation effects [B, S, G]
        """
        # Move to device
        ctrl_cell_emb = ctrl_cell_emb.to(self.device).to_dense()
        pert_emb = pert_emb.to(self.device)
        covariates = {k: v.to(self.device) for k, v in covariates.items()}

        with torch.no_grad():
            with torch.autocast(device_type='cuda' if self.device=='cuda' else 'cpu', 
                              dtype=torch.bfloat16, enabled=self.device=='cuda'):
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

def create_example_data():
    """
    Create example data for testing.
    """
    batch_size = 2
    seq_length = 20
    n_genes = 18080
    
    # Simulate control cell expression data
    ctrl_cell_emb = torch.randn(batch_size, seq_length, n_genes) * 0.5
    
    # Simulate perturbation embeddings
    pert_emb = torch.randn(batch_size, 5120) * 0.3
    
    # Simulate covariates
    covariates = {
        'cell_type_onehot': torch.zeros(batch_size, seq_length, 2),
        'batch_onehot': torch.zeros(batch_size, seq_length, 98)
    }
    
    # Set some cell types and batches
    covariates['cell_type_onehot'][:, :, 0] = 1.0  # All cells are type 0
    covariates['batch_onehot'][:, :, 0] = 1.0      # All cells are batch 0
    
    return ctrl_cell_emb, pert_emb, covariates


def vcc_inference():
    """
    VCC inference.
    """
    checkpoint_path = "/ephemeral/vcc-models/checkpoints/perturbation-transformer-epoch=39-train_loss=0.79.ckpt"
    # Define our path
    pert_counts_path = "./pert_counts_Validation.csv"
    pert_counts = pd.read_csv(pert_counts_path)
    gene_names = pd.read_csv("./gene_names.csv", header=None)
    gene_names = gene_names[0].tolist()

    dm = PerturbationDataModule(
        config_path="configs/data.toml",
        pert_embedding_file="/ephemeral/vcc/competition_support_set_sorted/ESM2_pert_features.pt",
        batch_size=8,
        group_size_S=256,
        num_workers=4  # Set to 0 to avoid multiprocessing issues
    )
    dm.setup(stage="fit")
    predictor = PerturbationPredictor(checkpoint_path)
    adata = ad.read_h5ad("/ephemeral/vcc/competition_support_set_sorted/competition_train.h5", backed="r")
    control_adata = adata[adata.obs["target_gene"] == "non-targeting"]
    cell_type = "ARC_H1"

    X = None
    pert_names = []
    for i, row in enumerate(pert_counts.itertuples()):
        gene = row.target_gene
        print(f"Processing gene {i} / {len(pert_counts)}: {gene}")
        n_cells = row.n_cells
        # randomly select n_cells from control_adata
        X_ctrl = control_adata.X[np.random.choice(range(len(control_adata)), size=n_cells, replace=False)]
        X_ctrl = X_ctrl.toarray()
        X_ctrl = torch.from_numpy(X_ctrl).float().to("cuda")
        X_ctrl = X_ctrl.unsqueeze(0)
        print(f"X_ctrl shape: {X_ctrl.shape}")
        pert_emb = dm.train_ds._get_pert_embedding(gene).unsqueeze(0)
        print(f"pert_emb shape: {pert_emb.shape}")
        pert_emb = pert_emb.to("cuda")
        covariates = {
            "cell_type_onehot": dm.train_ds.get_celltype_onehot(cell_type).unsqueeze(0).to("cuda"),
            "batch_onehot": torch.zeros(1,n_cells, 98).to("cuda")
        }
        predictions = predictor.predict(X_ctrl, pert_emb, covariates)
        pert_names += list(np.repeat(gene, X_ctrl.shape[1]))
        predictions = predictions.view(-1, predictions.shape[-1])

        X = torch.cat([predictions], dim=0) if X is None else torch.cat([X, predictions], dim=0)

    # add 5000 control cells
    X_ctrl = control_adata.X[np.random.choice(range(len(control_adata)), size=5000, replace=False)]
    X_ctrl = X_ctrl.toarray()
    X_ctrl = torch.from_numpy(X_ctrl).float().to("cuda")
    pert_names += list(np.repeat("non-targeting", X_ctrl.shape[0]))
    X = torch.cat([X, X_ctrl], dim=0)
    # convert X to numpy array
    X = X.cpu().numpy()
    X = sp.csr_matrix(X)
    pert_names = np.array(pert_names)

    ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "target_gene": pert_names,
            },
            index=np.arange(X.shape[0]).astype(str),
        ),
        var=pd.DataFrame(index=gene_names),
        ).write_h5ad("vcc_inference.h5ad")
    
    print(f"\n🎉 VCC inference completed successfully!")

def main():
    """
    Main example showing practical usage.
    """
    print("🧬 Perturbation Transformer Inference Example")
    print("=" * 50)

    checkpoint_path = "/ephemeral/vcc-models/checkpoints/perturbation-transformer-epoch=98-train_loss=0.90.ckpt"

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
    val_loader = dm.val_dataloader() 
    X = None
    pert_names = []
    try:
        # Initialize predictor
        predictor = PerturbationPredictor(checkpoint_path)

        for i, batch in enumerate(val_loader):
            if i > len(val_loader):
                break
            print(f"Processing batch {i} / {len(val_loader)}: {batch['pert_name']}")
            ctrl_cell_emb = batch["ctrl_cell_emb"]
            pert_emb = batch["pert_emb"]
            covariates = {
                "cell_type_onehot": batch["cell_type_onehot"],
                "batch_onehot": torch.zeros(ctrl_cell_emb.shape[0], ctrl_cell_emb.shape[1], 98)
            }
            pert_names += list(np.repeat(batch["pert_name"], ctrl_cell_emb.shape[1]))
            predictions = predictor.predict(ctrl_cell_emb, pert_emb, covariates) # shape: [B, S, G]
            # [B, S, G] -> [B*S, G]
            predictions = predictions.view(-1, predictions.shape[-1])

            X = torch.cat([predictions], dim=0) if X is None else torch.cat([X, predictions], dim=0)

        # convert X to numpy array
        X = X.type(torch.float16).cpu().numpy()
        pert_names = np.array(pert_names)

        ad.AnnData(
            X=X,
            obs=pd.DataFrame(
                {
                    "target_gene": pert_names,
                },
                index=np.arange(X.shape[0]).astype(str),
            )
        ).write_h5ad("val.h5ad")
        
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Prediction range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
        
        print(f"\n🎉 Inference example completed successfully!")
        
        return predictor, predictions
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    vcc_inference()
