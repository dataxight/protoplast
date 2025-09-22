. .env

# VCC competition training set
uv run python benchmark/benchmark.py --class protoplast --label sample --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"
uv run python benchmark/benchmark.py --class scdataset --label sample --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"
uv run python benchmark/benchmark.py --class scvi --label sample --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"
uv run python benchmark/benchmark.py --class annloader --label sample --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"
uv run python benchmark/benchmark.py --class anndata --label sample --batch-size 1024 "$VCC_ROOT_DIR/competition_train.h5"

# VCC all files
uv run python benchmark/benchmark.py --class protoplast --label sample --batch-size 1024 "$VCC_ROOT_DIR/*.h5"
uv run python benchmark/benchmark.py --class scdataset --label sample --batch-size 1024 "$VCC_ROOT_DIR/*.h5"
uv run python benchmark/benchmark.py --class annloader --label sample --batch-size 1024 "$VCC_ROOT_DIR/*.h5"
