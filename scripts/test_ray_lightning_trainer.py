import argparse

import ray
import ray.train
import ray.train.torch

from protoplast.scrna.anndata.lightning_models import LinearClassifier
from protoplast.scrna.anndata.torch_dataloader import DistrbutedCellLineAnnDataset as Dcl
from protoplast.scrna.anndata.torch_dataloader import ann_split_data, cell_line_metadata_cb
from protoplast.scrna.anndata.trainer import new_trainer

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
        "--thread_per_worker", required=True, type=int, help="Amount of thread per ray worker for data loading"
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
    args = parser.parse_args()
    ray.init()
    use_gpu = False
    resources = ray.cluster_resources()
    if resources.get("GPU", 0) > 0:
        use_gpu = True
    else:
        raise Exception("Only support with GPU is available only")
    indices = ann_split_data(args.file_paths, args.batch_size, args.test_size, metadata_cb=cell_line_metadata_cb)
    print("Finish spliting the data starting distributed training")
    train_config = {
        "batch_size": args.batch_size,
        "test_size": args.test_size,
        "indices": indices,
    }
    scaling_config = ray.train.ScalingConfig(
        num_workers=int(resources.get("GPU")), use_gpu=use_gpu, resources_per_worker={"CPU": args.thread_per_worker}
    )
    my_train_func = new_trainer(LinearClassifier, Dcl, ["num_genes", "num_classes"])
    par_trainer = ray.train.torch.TorchTrainer(
        my_train_func, scaling_config=scaling_config, train_loop_config=train_config
    )
    print("Spawning Ray worker and initiating distributed training")
    result = par_trainer.fit()
    print(result.metrics)
