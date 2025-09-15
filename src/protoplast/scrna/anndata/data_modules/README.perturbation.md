## Perturbation scRNA-seq DataModule (AnnData-backed)

Developer overview of `protoplast.scrna.anndata.data_modules.perturbation`.

### What this provides
- **`PerturbationDataset`**: Streaming, memory-efficient `IterableDataset` over AnnData `.h5ad` files for perturbation scRNA-seq.
- **`PerturbationDataModule`**: PyTorch Lightning DataModule that builds train/val/test indices with few-shot/zero-shot target control and provides ready-to-use DataLoaders.

### Data assumptions
- Each `.h5ad` contains a single cell type and is pre-sorted by `target_label` then `batch_label`.
- Controls are labeled by `control_label` in `obs[target_label]` and are contiguous per file; they are sampled to match perturbation group size.
- Expression matrix is read lazily via AnnData backed mode and `anndata._core.sparse_dataset.sparse_dataset`.

### Sample schema (yielded by dataset)
Each iteration yields a dict per target group of size `S` (=`group_size_S`):
- `pert_cell_emb`: scipy CSR of shape `[S, G]` (perturbed cells)
- `ctrl_cell_emb`: scipy CSR of shape `[S, G]` (matched controls)
- `pert_emb`: torch tensor `[D_pert]` (embedding for the target, loaded from `pert_embedding_file`)
- `pert_name`: numpy array `[1]` with target name (string)
- `cell_type`: numpy array `[1]` with cell type (string)
- `cell_type_onehot`: torch tensor `[S, C]`
- `batch_onehot`: torch tensor `[S, B]`
- `pert_cell_barcode` (optional): numpy array `[S]`
- `ctrl_cell_barcode` (optional): numpy array `[S]`

The provided `collate_fn` converts scipy CSR to torch COO and stacks across the batch:
- `pert_cell_emb`/`ctrl_cell_emb`: `torch.sparse_coo_tensor` stacked to shape `[batch, S, G]`
- Other tensors are stacked with `torch.stack`; string arrays are concatenated.

### Quickstart
```python
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule

# Minimal: provide explicit files
files = [
    "/abs/path/dataset_a_celltype_x.h5ad",
    "/abs/path/dataset_b_celltype_y.h5ad",
]

dm = PerturbationDataModule(
    files=files,
    pert_embedding_file="/abs/path/ESM2_pert_features.pt",
    batch_size=64,
    group_size_S=32,
    num_workers=8,
)

dm.setup(stage="fit")
train_loader = dm.train_dataloader()  # uses the custom collate_fn

for batch in train_loader:
    x_pert = batch["pert_cell_emb"]     # [B, S, G] sparse COO
    x_ctrl = batch["ctrl_cell_emb"]     # [B, S, G] sparse COO
    pert_emb = batch["pert_emb"]        # [B, D_pert]
    # ...
    break
```

Config-based initialization (few-shot/zero-shot):
```python
from protoplast.scrna.anndata.data_modules.perturbation import PerturbationDataModule

dm = PerturbationDataModule(
    config_path="/abs/path/dataset_config.toml",
    pert_embedding_file="/abs/path/ESM2_pert_features.pt",
    barcodes=True,
)

dm.setup(stage="fit")
```
See `protoplast.scrna.anndata.data_modules.utils.parse_dataset_config` for the TOML schema. The config can define:
- **dataset files** (with brace expansion),
- **file-level splits** (train/val/test),
- **target-level overrides** per `(dataset, cell_type)` for few-/zero-shot,
- **dataset_opts** (labels, `n_basal_samples` → `group_size_S`),
- **loader** (e.g., `num_workers`, `prefetch_factor`).

### Key parameters
- **files/config_path**: Provide either an explicit list of `.h5ad` files or a config file.
- **pert_embedding_file**: Path to `torch.load`-able dict mapping `pert_id → embedding`.
- **cell_type_label/target_label/control_label/batch_label**: `obs` column names.
- **group_size_S**: Number of cells per target group (with replacement if needed).
- **block_size**: Minimum contiguous region size when consolidating target ranges per file.
- **barcodes**: Include `obs.index` in outputs when `True`.
- **num_workers/prefetch_factor/batch_size**: DataLoader performance knobs.

### How splitting works
- `build_indices(...)` scans each file, skips control targets, and groups cells by `target_label`.
- For each target, it computes how many dataset items to emit (`max(1, floor(n_cells / S))`).
- Splits can be assigned at file level with optional per-target overrides.
- Control cells are sampled from the file’s control region for the same cell type.

### Notes and tips
- Ensure `.h5ad` are pre-sorted by target then batch; controls contiguous within file.
- Using backed AnnData + SWMR HDF5 enables multi-worker safe reads.
- The sparse COO batches are not directly dense; densify only for debugging.
- If an embedding for a target is missing, a zero vector of the correct size is created on-the-fly.

### Local smoke test
You can run the module’s main block for a quick check (prints regions and a few samples):
```bash
python -m protoplast.scrna.anndata.data_modules.perturbation | cat
```

