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
