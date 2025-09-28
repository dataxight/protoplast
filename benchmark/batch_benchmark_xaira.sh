#   Copyright 2025 DataXight, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


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
