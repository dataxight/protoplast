import time

import daft

from protoplast.genomics.anndata import read_h5ad

# Apply daft patches for per-CPU workers
from protoplast.patches.daft_flotilla import apply_flotilla_patches

apply_flotilla_patches()

def main():
    daft.context.set_runner_ray()
    #daft.context.set_execution_config(scan_tasks_min_size_bytes=100000, min_cpu_per_task=1)
    df = read_h5ad("/Users/tanphan/Downloads/pbmc_seurat_v4.h5ad", batch_size=5000, preview_size=0)
    start = time.time()
    #df.count().show()
    df.write_parquet("pbmc-full")
    print(f"Time took: {time.time() - start}")

if __name__ == "__main__":
    main()
