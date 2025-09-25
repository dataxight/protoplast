. .env

# VCC competition training set
uv run python benchmark/benchmark.py --class protoplast --label batch_var --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"
uv run python benchmark/benchmark.py --class scdataset --label batch_var --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"
uv run python benchmark/benchmark.py --class scvi2 --label batch_var --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"
uv run python benchmark/benchmark.py --class scvi --label batch_var --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"
uv run python benchmark/benchmark.py --class annloader --label batch_var --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"
uv run python benchmark/benchmark.py --class anndata --label batch_var --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"

# VCC all files
uv run python benchmark/benchmark.py --class protoplast --label batch_var --batch-size 1024 "$VCC_ROOT_DIR/*.h5"
uv run python benchmark/benchmark.py --class scdataset --label batch_var --batch-size 1024 "$VCC_ROOT_DIR/*.h5"
uv run python benchmark/benchmark.py --class annloader --label batch_var --batch-size 1024 "$VCC_ROOT_DIR/*.h5"
