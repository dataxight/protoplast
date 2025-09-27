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


import argparse

from protoplast.scrna.anndata.lightning_models import LinearClassifier
from protoplast.scrna.anndata.torch_dataloader import DistributedCellLineAnnDataset as Dcl
from protoplast.scrna.anndata.torch_dataloader import cell_line_metadata_cb
from protoplast.scrna.anndata.trainer import RayTrainRunner

"""
Think of this as a template consult the documentation
on how to modify this code for another model

here you can write your own split data algorithm or use the default by looking at ann_split_data
You can create your own model by extending BaseAnnDataLightningModule
Create your own Dataset to feed the correct data to your model by extending
DistributedAnnDataset

This library is design to be very flexible consult the documentation for more details or how
use it to fit your training situation

"""


def parse_list(s):
    if "," in s:
        return s.split(",")
    else:
        return [s]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a linear classifier on cell line data.")
    parser.add_argument("--file_paths", required=True, type=parse_list, help="Comma-separated list of integers")
    # if reading from local disk without raid configuration should set to zero otherwise increase this
    # number for faster processing need to do more experimentation
    parser.add_argument(
        "--thread_per_worker", default=1, type=int, help="Amount of thread per ray worker for data loading"
    )
    # recommended to be around 1000-2000 for maximum speed this also depends on storage type need to experiment
    # however we can set a warning if batch size is too large for GPU or CPU
    parser.add_argument("--batch_size", default=1000, type=int, help="Dataloader batch size")
    parser.add_argument(
        "--test_size",
        default=None,
        type=float,
        help="How big is the test data as a fraction of the whole data per plate or offsets",
    )
    parser.add_argument(
        "--val_size",
        default=None,
        type=float,
        help="How big is the validation data as a fraction of the whole data per plpate or offsets",
    )
    parser.add_argument("--ckpt_path", default=None, type=str, help="Path to the checkpoint you want to resume from")
    args = parser.parse_args()
    trainer = RayTrainRunner(
        LinearClassifier,  # replace with your own model
        Dcl,  # replace with your own Dataset
        ["num_genes", "num_classes"],  # change according to what you need for your model
        cell_line_metadata_cb,  # include data you need for your dataset
    )
    trainer.train(
        args.file_paths,
        args.batch_size,
        args.test_size,
        args.val_size,
        thread_per_worker=args.thread_per_worker,
        ckpt_path=args.ckpt_path,
        is_shuffled=False,
        result_storage_path="tmp",
    )
