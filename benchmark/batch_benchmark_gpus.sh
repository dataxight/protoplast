. .env

uv run python benchmark/benchmark.py --class protoplast --label cell_line --batch-size 1024 --gpus 1 --workers 36 "$TAHOE_ROOT_DIR/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
uv run python benchmark/benchmark.py --class protoplast --label cell_line --batch-size 1024 --gpus 2 --workers 18 "$TAHOE_ROOT_DIR/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
uv run python benchmark/benchmark.py --class protoplast --label cell_line --batch-size 1024 --gpus 3 --workers 12 "$TAHOE_ROOT_DIR/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
uv run python benchmark/benchmark.py --class protoplast --label cell_line --batch-size 1024 --gpus 4 --workers 9 "$TAHOE_ROOT_DIR/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"

uv run python benchmark/benchmark.py --class protoplast --label cell_line --batch-size 1024 --gpus 1 --workers 36 "$TAHOE_ROOT_DIR/*.h5ad"
uv run python benchmark/benchmark.py --class protoplast --label cell_line --batch-size 1024 --gpus 2 --workers 18 "$TAHOE_ROOT_DIR/*.h5ad"
uv run python benchmark/benchmark.py --class protoplast --label cell_line --batch-size 1024 --gpus 3 --workers 12 "$TAHOE_ROOT_DIR/*.h5ad"
uv run python benchmark/benchmark.py --class protoplast --label cell_line --batch-size 1024 --gpus 4 --workers 9 "$TAHOE_ROOT_DIR/*.h5ad"
