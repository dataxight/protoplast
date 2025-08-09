import time

import daft

from protoplast.genomics.anndata import read_h5ad
from protoplast.genomics.anndata import AnnDataSink

# Apply daft patches for per-CPU workers
from protoplast.patches.daft_flotilla import apply_flotilla_patches
import os

#os.environ["MAX_WORKERS"] = "2"
apply_flotilla_patches()

def main():
    daft.context.set_runner_ray()
    #daft.context.set_execution_config(scan_tasks_min_size_bytes=100000, min_cpu_per_task=1)
    df = read_h5ad("/Users/tanphan/Downloads/pbmc_seurat_v4.h5ad", batch_size=5000, preview_size=0)
    start = time.time()
    sink = AnnDataSink("pbmc-sink-w10-b5000")
    df.write_sink(sink)
    print(f"Time took: {time.time() - start}")

if __name__ == "__main__":
    main()
