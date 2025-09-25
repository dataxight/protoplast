. .env

# Tahoe plate3
uv run python benchmark/benchmark.py --class protoplast --label cell_line --batch-size 1024 "$TAHOE_ROOT_DIR/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
uv run python benchmark/benchmark.py --class scdataset --label cell_line --batch-size 1024 "$TAHOE_ROOT_DIR/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
uv run python benchmark/benchmark.py --class scvi2 --label cell_line --batch-size 1024 "$TAHOE_ROOT_DIR/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
uv run python benchmark/benchmark.py --class scvi --label cell_line --batch-size 1024 "$TAHOE_ROOT_DIR/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
uv run python benchmark/benchmark.py --class annloader --label cell_line --batch-size 1024 "$TAHOE_ROOT_DIR/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
uv run python benchmark/benchmark.py --class anndata --label cell_line --batch-size 1024 "$TAHOE_ROOT_DIR/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"

# Tahoe all plates
uv run python benchmark/benchmark.py --class protoplast --label cell_line --batch-size 1024 "$TAHOE_ROOT_DIR/*.h5ad"
uv run python benchmark/benchmark.py --class scdataset --label cell_line --batch-size 1024 "$TAHOE_ROOT_DIR/*.h5ad"
uv run python benchmark/benchmark.py --class annloader --label cell_line --batch-size 1024 "$TAHOE_ROOT_DIR/*.h5ad"
