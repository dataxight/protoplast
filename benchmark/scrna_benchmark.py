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
import sys

from scd import train as scl_train

from protoplast.scrna.anndata.lightning_models import LinearClassifier, NullClassifier
from protoplast.scrna.anndata.torch_dataloader import DistributedCellLineAnnDataset as Dcl
from protoplast.scrna.anndata.torch_dataloader import cell_line_metadata_cb
from protoplast.scrna.anndata.trainer import RayTrainRunner

if __name__ == "__main__":
    tahoe_dir = sys.argv[1]
    mode = sys.argv[2]
    thread_per_worker = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    num_workers = int(sys.argv[5])
    if os.path.isfile(tahoe_dir):
        paths = [tahoe_dir]
    else:
        paths = os.listdir(tahoe_dir)

    paths = [os.path.join(tahoe_dir, p) for p in paths if p.endswith(".h5ad")]
    if mode == "null":
        print("Running null model")
        trainer = RayTrainRunner(
            NullClassifier,  # replace with your own model
            Dcl,  # replace with your own Dataset
            ["num_classes"],  # change according to what you need for your model
            cell_line_metadata_cb,  # include data you need for your dataset
            runtime_env_config={
                "env_vars": {
                    "OMP_NUM_THREADS": str(thread_per_worker),
                    "MKL_NUM_THREADS": str(thread_per_worker),
                    "NUMEXPR_NUM_THREADS": str(thread_per_worker),
                    "OPENBLAS_NUM_THREADS": str(thread_per_worker),
                    "TORCH_NUM_THREADS": str(thread_per_worker),
                }
            },
        )
        trainer.train(
            paths,
            thread_per_worker=thread_per_worker,
            batch_size=batch_size,
            num_workers=num_workers,
            is_gpu=False,
            test_size=0.0,
            val_size=0.2,
            is_shuffled=False,
        )
    elif mode == "pl":
        print("Running protoplast linear model")
        trainer = RayTrainRunner(
            LinearClassifier,  # replace with your own model
            Dcl,  # replace with your own Dataset
            ["num_genes", "num_classes"],  # change according to what you need for your model
            cell_line_metadata_cb,  # include data you need for your dataset
            max_open_files=12,
        )
        trainer.train(
            paths,
            thread_per_worker=thread_per_worker,
            batch_size=batch_size,
            num_workers=num_workers,
            test_size=0.0,
            val_size=0.2,
            drop_last=False,
        )
    elif mode == "scl":
        print("Running with scDataset streaming")
        scl_train(paths, batch_size=batch_size, num_workers=num_workers, fetch_factor=16)
    elif mode == "scls":
        print("Running with scDataset block shuffling")
        scl_train(paths, batch_size=batch_size, num_workers=num_workers, fetch_factor=16, is_shuffle=True, block_size=4)
