import time

import daft

from protoplast.genomics.anndata import read_h5ad
from protoplast.genomics.anndata import AnnDataSink

# Apply daft patches for per-CPU workers
from protoplast.patches.daft_flotilla import apply_flotilla_patches
import os
import pyarrow as pa

os.environ["MAX_WORKERS"] = "2"
apply_flotilla_patches()
daft.context.set_runner_ray()
#daft.context.set_runner_native(1)
daft.context.set_execution_config(native_parquet_writer=False)
file_path = "/Users/tanphan/Downloads/pbmc_seurat_v4.h5ad"

def test_read_h5ad():
    df = read_h5ad(file_path, batch_size=25000, preview_size=0)
    start = time.time()
    df.write_parquet("pbmc-full-b25000-w10-pa_debug")
    print(f"Time took: {time.time() - start}")

if __name__ == "__main__":
    test_read_h5ad()
