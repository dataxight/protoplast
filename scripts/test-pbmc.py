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


import os
import time

# Apply daft patches for per-CPU workers
import anndata
import daft

from protoplast.patches.anndata_read_h5ad_backed import apply_read_h5ad_backed_patch
from protoplast.patches.anndata_remote import apply_file_backing_patch
from protoplast.patches.daft_flotilla import apply_flotilla_patches

os.environ["MAX_WORKERS"] = "2"
apply_flotilla_patches()
apply_file_backing_patch()
apply_read_h5ad_backed_patch()

daft.context.set_execution_config(native_parquet_writer=False)
file_path = "s3://anyscale-ap-data/test_medium.h5ad"


def test_read_h5ad():
    start = time.time()
    ad = anndata.read_h5ad(file_path, backed="r")
    print(ad.X.shape)
    print(ad.X[1, :])
    print(f"Time took: {time.time() - start}")


if __name__ == "__main__":
    test_read_h5ad()
