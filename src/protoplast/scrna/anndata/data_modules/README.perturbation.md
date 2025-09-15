## PerturbationDataModule — features and quick usage

### What it provides
- **Streaming, memory‑efficient loaders** over AnnData `.h5ad` files (one cell type per file), using backed read + HDF5 SWMR.
- **Flexible splitting**: train/val/test at file level with optional per‑target few/zero‑shot overrides.
- **Matched controls**: draws control cells from contiguous control regions of the same cell type.
- **Ready collation**: converts CSR → torch sparse COO and stacks; tensors and strings are batched appropriately.

### Batch schema (per item)
- **pert_cell_emb**: `[S, G]` scipy CSR (perturbed cells)
- **ctrl_cell_emb**: `[S, G]` scipy CSR (matched controls)
- **pert_emb**: `[D]` torch tensor (from `pert_embedding_file`; zero‑vector if missing)
- **pert_name**: `[1]` numpy str
- **cell_type**: `[1]` numpy str
- **cell_type_onehot**: `[S, C]` torch tensor
- **batch_onehot**: `[S, B]` torch tensor
- Optional: **pert_cell_barcode**, **ctrl_cell_barcode** `[S]` numpy arrays when `barcodes=True`

### Minimal usage (explicit files)
```python
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule

files = [
    "/abs/path/celltype_a.h5ad",
    "/abs/path/celltype_b.h5ad",
]

dm = PerturbationDataModule(
    files=files,
    pert_embedding_file="/abs/path/ESM2_pert_features.pt",
    group_size_S=32,
    batch_size=64,
    num_workers=8,
)
dm.setup(stage="fit")
train_loader = dm.train_dataloader()  # uses the module's collate_fn
for batch in train_loader:
    x_pert = batch["pert_cell_emb"]     # [B, S, G] sparse COO
    x_ctrl = batch["ctrl_cell_emb"]     # [B, S, G] sparse COO
    pert_emb = batch["pert_emb"]        # [B, D_pert]
    # ...
    break
```

### Config‑based setup (few/zero‑shot)
```python
dm = PerturbationDataModule(
    config_path="/abs/path/pert-dataconfig.toml",
    pert_embedding_file="/abs/path/ESM2_pert_features.pt",
    barcodes=True,
)
dm.setup(stage="fit")
```

Example TOML (see `notebooks/pert-dataconfig.toml`):
```toml
[datasets]
replogle_h1 = "/path/{competition_train,k562_gwps,rpe1,jurkat,k562,hepg2}.h5"

[training]
replogle_h1 = "train"

[zeroshot]
# e.g. "dataset.celltype" = "val" | "test"

[fewshot."replogle_h1.ARC_H1"]
val = ["TMSB4X", "PRCP", "TADA1", "HIRA", "IGF2R", "NCK2", "MED13", "MED12", "STAT1"]
```

### Key options
- **pert_embedding_file**: `torch.load`‑able dict `pert_id -> embedding`.
- **group_size_S**: cells per target group (samples with replacement if short).
- **block_size**: lower bound for consolidated regions when building indices.
- **cell_type_label/target_label/control_label/batch_label**: `obs` column names.
- **barcodes**: include cell barcodes when `True`.
- **num_workers/prefetch_factor/batch_size**: DataLoader performance knobs.

### Notes
- Each `.h5ad` should contain a single cell type and be pre‑sorted by target (and typically by batch); control cells are contiguous and labeled by `control_label` in `obs[target_label]`.
- Collation converts CSR to sparse COO and stacks to `[B, S, G]`; densify only for debugging.
- Missing perturbation embeddings are created as zero vectors of the correct size.

### How splitting works
- `build_indices(...)` scans each file, skips control targets, and groups cells by `target_label`.
- For each target, it computes how many dataset items to emit (`max(1, floor(n_cells / S))`).
- Splits can be assigned at file level with optional per-target overrides.
- Control cells are sampled from the file’s control region for the same cell type.