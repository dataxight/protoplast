import argparse
import os
from typing import Tuple

import numpy as np
import scanpy as sc
import torch

from models.cell_emb import ExprTransformer, to_dense, make_token_batch

import logging
logging_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_fmt)
logger = logging.getLogger(__name__)
#handler = logging.StreamHandler()
#handler.setFormatter(logging.Formatter(logging_fmt))
#logger.addHandler(handler)

def infer_hparams_from_state(state: dict) -> Tuple[int, int, int, int, int]:
    """Infer minimal hyperparameters from a plain state_dict.

    Returns:
        num_genes, num_classes, d_model, n_layers, topk_tokens
    """
    if "gene_emb.weight" not in state or "head.weight" not in state:
        raise ValueError("State dict does not look like ExprTransformer weights.")

    num_genes, d_model = state["gene_emb.weight"].shape
    num_classes = state["head.weight"].shape[0]

    # Count encoder layers by scanning keys like 'encoder.layers.{i}.'
    layer_indices = set()
    prefix = "encoder.layers."
    for key in state.keys():
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            idx_str = rest.split(".")[0]
            if idx_str.isdigit():
                layer_indices.add(int(idx_str))
    n_layers = (max(layer_indices) + 1) if layer_indices else 4

    # topk_tokens from positional embedding size if present
    if "pos_emb" in state:
        topk_tokens = int(state["pos_emb"].shape[1] - 1)
    else:
        topk_tokens = 256

    return num_genes, num_classes, d_model, n_layers, topk_tokens


def choose_n_heads(d_model: int) -> int:
    """Choose a valid number of heads given d_model.

    We cannot recover it from the state_dict; pick a common divisor.
    """
    for candidate in (16, 12, 8, 6, 4, 3, 2):
        if d_model % candidate == 0:
            return candidate
    return 1


def prepare_adata(adata: sc.AnnData, expected_num_genes: int) -> sc.AnnData:
    """Ensure `adata` has exactly `expected_num_genes` variables.

    - If counts mismatch, select HVGs to match the expected count.
    - Adds `gene_id` if missing.
    """
    if adata.n_vars != expected_num_genes:
        # Prefer using any existing highly_variable flag if available
        if "highly_variable" in adata.var.columns:
            hv_mask = adata.var["highly_variable"].values
            # If too many, trim; if too few or none, recompute
            if hv_mask.sum() >= expected_num_genes:
                hv_idx = np.where(hv_mask)[0][:expected_num_genes]
                adata = adata[:, hv_idx].copy()
            else:
                sc.pp.highly_variable_genes(adata, n_top_genes=expected_num_genes, flavor="seurat_v3")
                adata = adata[:, adata.var["highly_variable"].values].copy()
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=expected_num_genes, flavor="seurat_v3")
            adata = adata[:, adata.var["highly_variable"].values].copy()

    # Ensure size now matches
    if adata.n_vars != expected_num_genes:
        raise ValueError(
            f"Prepared AnnData has {adata.n_vars} genes, expected {expected_num_genes}."
        )

    if "gene_id" not in adata.var.columns:
        adata.var["gene_id"] = np.arange(adata.n_vars)

    return adata


def embed_adata(adata: sc.AnnData, model: ExprTransformer, device: torch.device, batch_size: int, topk_tokens: int) -> np.ndarray:
    model.eval()
    num_cells = adata.n_obs
    d_model = model.head.in_features
    embeddings = np.zeros((num_cells, d_model), dtype=np.float32)

    for start in range(0, num_cells, batch_size):
        end = min(start + batch_size, num_cells)
        X_batch = to_dense(adata[start:end].X)  # torch.FloatTensor on CPU
        # Compute tokens on CPU, then move to device
        tok_ids, tok_vals = make_token_batch(X_batch, k=min(topk_tokens, X_batch.shape[1]))
        tok_ids = tok_ids.to(device)
        tok_vals = tok_vals.to(device)
        with torch.no_grad():
            emb = model.encode(tok_ids, tok_vals).detach().cpu().numpy()
        embeddings[start:end] = emb

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate CLS embeddings for all cells and store in adata.obsm['X_emb']")
    parser.add_argument("--adata", required=True, help="Path to input .h5ad file")
    parser.add_argument("--checkpoint", required=True, help="Path to model state_dict (.pt) file")
    parser.add_argument("--output", default=None, help="Optional path to write output .h5ad; default adds .with_emb suffix")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device(args.device)
        logger.info(f"Using device: {device}")
    

    # Load state dict on CPU first
    logger.info(f"Loading state dict from {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    logger.info(f"Inferring hyperparameters from state dict")
    num_genes, num_classes, d_model, n_layers, topk_tokens = infer_hparams_from_state(state)
    n_heads = choose_n_heads(d_model)
    logger.info(f"Number of genes: {num_genes}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"d_model: {d_model}")
    logger.info(f"n_layers: {n_layers}")
    logger.info(f"n_heads: {n_heads}")
    logger.info(f"topk_tokens: {topk_tokens}")

    # Build model and load weights
    logger.info(f"Creating model")
    model = ExprTransformer(
        num_genes=num_genes,
        num_classes=num_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,
        topk_tokens=topk_tokens,
    )
    logger.info(f"Loading state dict into model")
    model.load_state_dict(state, strict=True)
    model.to(device)

    # Load AnnData and prepare genes to match the checkpoint
    logger.info(f"Loading AnnData from {args.adata}")
    adata = sc.read_h5ad(args.adata)
    logger.info(f"Preparing AnnData")
    adata = prepare_adata(adata, expected_num_genes=num_genes)

    # Generate embeddings
    logger.info(f"Generating embeddings")
    X_emb = embed_adata(
        adata=adata,
        model=model,
        device=device,
        batch_size=args.batch_size,
        topk_tokens=topk_tokens,
    )

    # Store and save
    logger.info(f"Storing embeddings in AnnData")
    adata.obsm["X_emb"] = X_emb

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.adata)
        output_path = f"{base}.with_emb{ext or '.h5ad'}"

    logger.info(f"Saving AnnData to {output_path}")
    adata.write_h5ad(output_path)
    logger.info(f"Wrote embeddings to adata.obsm['X_emb'] and saved to: {output_path}")


if __name__ == "__main__":
    main()


