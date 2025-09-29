# Usage

## Quick Start
Get up and running with minimal setup. Learn how to install dependencies, prepare your data, and run your first example in just a few commands.

```python
from protoplast import RayTrainRunner, DistributedCellLineAnnDataset, LinearClassifier
import glob

trainer = RayTrainRunner(
   LinearClassifier,  # replace with your own model
   DistributedCellLineAnnDataset,  # replace with your own Dataset
   ["num_genes", "num_classes"],  # change according to what you need for your model
)

file_paths = glob.glob("/data/tahoe100/*.h5ad")
trainer.train(file_paths)
```

## Common Use Cases

### Train a Pertubation Model

Step through the process of training a model to predict cellular responses to genetic or chemical perturbations using your own single-cell datasets.

[Learn more](https://github.com/dataxight/protoplast-ml-example/blob/main/notebooks/perturbation_examples.ipynb)

### Use custom models

Integrate external machine learning models into the workflow. Learn how to plug in your own architectures or pretrained models for flexible experimentation.

[Learn more](https://github.com/dataxight/protoplast-ml-example/blob/main/notebooks/classification_examples.ipynb)

## More Examples

Please visit [protoplast-ml-example](https://github.com/dataxight/protoplast-ml-example) to get the latest examples of PROTOplast.