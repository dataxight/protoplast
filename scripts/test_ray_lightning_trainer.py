from protoplast.scrna.anndata.lightning_models import LinearClassifier
import argparse
from protoplast.scrna.anndata.torch_dataloader import DistrbutedCellLineAnnDataset as Dcl, ann_split_data, cell_line_metadata_cb
import lightning.pytorch as pl
import ray
import os
from torch.utils.data import DataLoader
import json

def parse_list(s):
    if "," in s:
        return s.split(",")
    else:
        return [s]

def anndata_train_func(config):
    ctx = ray.train.get_context()
    if ctx:
        rank = ctx.get_world_rank()
    else:
        rank = 0
    share_path = config.get("share_path")
    indices_path = indices_path = os.path.join(share_path, "indices.json")
    num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
    print(f"=========Starting the training on {rank} with num threads: {num_threads}=========")
    train_ds, test_ds = Dcl.create_distributed_ds(indices_path), Dcl.create_distributed_ds(indices_path, is_test=True)
    loader_config = dict(
        batch_size=None,
        num_workers=num_threads,
        prefetch_factor=4
    )
    train_dl, test_dl = DataLoader(train_ds,**loader_config), DataLoader(test_ds, **loader_config) 
    model = LinearClassifier(num_genes=train_ds.num_genes, num_classes=train_ds.num_classes)
    profile_path = os.path.join(share_path, "profiling", f"rank_{rank}")
    os.makedirs(profile_path, exist_ok=True)
    
    trainer = pl.Trainer(
            max_epochs=1,
            devices="auto",
            accelerator="auto",
            strategy=ray.train.lightning.RayDDPStrategy(),
            plugins=[ray.train.lightning.RayLightningEnvironment()],
            callbacks=[ray.train.lightning.RayTrainReportCallback()],
            enable_checkpointing=False,
            # profiler=profiler
            
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a linear classifier on cell line data.")
    parser.add_argument("--file_paths", required=True, type=parse_list, help='Comma-separated list of integers')
    # if reading from local disk without raid configuration should set to zero otherwise increase this number for faster processing need to do more experimentation
    parser.add_argument("--thread_per_worker", required=True, type=int, help="Amount of thread per ray worker for data loading")
    # recommended to be around 1000-2000 for maximum speed this also depends on storage type need to experiment
    # however we can set a warning if batch size is too large for GPU or CPU
    parser.add_argument("--batch_size",  type=int, help="Dataloader batch size")
    parser.add_argument("--test_size", default=None, type=float, help="How big is the test data as a fraction of the whole data per plate or offsets")
    parser.add_argument("--share_path", required=True, type=str, help="Share path where all node have access to this will store the supporting files required for each worker")
    args = parser.parse_args()
    ray.init()
    use_gpu = False
    resources = ray.cluster_resources()
    if resources.get("GPU", 0) > 0:
        use_gpu = True
    else:
        raise Exception("Only support with GPU is available only")
    train_indices, test_indices, metadata = ann_split_data(args.file_paths, args.batch_size, args.test_size, metadata_cb=cell_line_metadata_cb)
    indices = dict(
        files=args.file_paths,
        train_indices=train_indices,
        test_indices=test_indices,
        metadata=metadata
    )
    indices_path = os.path.join(args.share_path, "indices.json")
    with open(indices_path, 'w') as f:
        json.dump(indices, f)
    print("Finish spliting the data saving to: ", indices_path)
    train_config = {
            "batch_size": args.batch_size,
            "mode": args.mode,
            "test_size": args.test_size,
            "share_path": args.share_path,
    }
    scaling_config = ray.train.ScalingConfig(num_workers=int(resources.get("GPU")), use_gpu=use_gpu, resources_per_worker={"CPU": args.thread_per_worker})
    par_trainer = ray.train.torch.TorchTrainer(
        anndata_train_func,
        scaling_config=scaling_config,
        train_loop_config=train_config
    )
    print("Spawning Ray worker and initiating distributed training")
    result = par_trainer.fit()
    print(result.metrics)