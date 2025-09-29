# protoplast

![PyPI version](https://img.shields.io/pypi/v/protoplast.svg)
[![Documentation Status](https://readthedocs.org/projects/protoplast/badge/?version=latest)](https://protoplast.readthedocs.io/en/latest/?version=latest)

Early developer preview of PROTOplast targets acceleration of ML model training workflows

-   PyPI package: https://pypi.org/project/protoplast/

## Features

* Stream directly from remote/cloud storage (via runtime patching of anndata to use fsspec)
* Accelerated training of your ML models (14.5minutes on an A100 instance with 4 GPUs). Scale to multi-node clusters with zero code changes (with native Ray integration)
* Drop-in replacement of your custom ML training (by subclassing Lightning's LightningModule)


## Getting started

It's easy to get started with PROTOplast

```python
from protoplast import RayTrainRunner, DistributedCellLineAnnDataset, LinearClassifier
import glob

files = glob.glob("/data/tahoe100/*.h5ad")

trainer = RayTrainRunner(
   LinearClassifier,  # replace with your own model
   DistributedCellLineAnnDataset,  # replace with your own Dataset
   ["num_genes", "num_classes"],  # change according to what you need for your model
)
trainer.train(file_paths=files)
```

Additional tutorials are available at https://protoplast.dataxight.com/tutorials

Full documentation at https://protoplast.dataxight.com
