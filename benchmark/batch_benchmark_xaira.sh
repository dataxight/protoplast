
. .env

# Xaira smallest file
uv run python benchmark/benchmark.py --class protoplast --label sample --batch-size 1024 "$XAIRA_ROOT_DIR/HCT116_filtered_dual_guide_cells.h5ad"
uv run python benchmark/benchmark.py --class scdataset --label sample --batch-size 1024 "$XAIRA_ROOT_DIR/HCT116_filtered_dual_guide_cells.h5ad"
uv run python benchmark/benchmark.py --class scvi2 --label sample --batch-size 1024 "$XAIRA_ROOT_DIR/HCT116_filtered_dual_guide_cells.h5ad"
uv run python benchmark/benchmark.py --class scvi --label sample --batch-size 1024 "$XAIRA_ROOT_DIR/HCT116_filtered_dual_guide_cells.h5ad"
uv run python benchmark/benchmark.py --class annloader --label sample --batch-size 1024 "$XAIRA_ROOT_DIR/HCT116_filtered_dual_guide_cells.h5ad"
uv run python benchmark/benchmark.py --class anndata --label sample --batch-size 1024 "$XAIRA_ROOT_DIR/HCT116_filtered_dual_guide_cells.h5ad"

# Xaira all files
uv run python benchmark/benchmark.py --class protoplast --label sample --batch-size 1024 "$XAIRA_ROOT_DIR/*.h5ad"
uv run python benchmark/benchmark.py --class scdataset --label sample --batch-size 1024 "$XAIRA_ROOT_DIR/*.h5ad"
uv run python benchmark/benchmark.py --class annloader --label sample --batch-size 1024 "$XAIRA_ROOT_DIR/*.h5ad"
