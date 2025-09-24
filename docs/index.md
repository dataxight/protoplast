# Welcome to PROTOplast's documentation!

## Contents

- [Readme](readme.md)
- [Installation](installation.md)
- [Usage](usages/index.md)
    - [Single Cell Training](usages/scrna.md)
- [API](apis/index.md)
    - [Single Cell](apis/scrna.md)
- [Contributing](contributing.md)
- [History](history.md)


# Introduction

This RayTrainRunner is designed to train single cell RNA data where the amount of data exceeds the memory in the machine. By default the amount of workers is set equal to the amount of GPU in the Ray cluster. This can be overridden by num_workers. Each worker will use `threads_per_worker` CPU to fetch the data from the storage. In the future we will set this value automatically based on the storage type and the amount of CPU cores in the node. Please note that Ray will reserve 1 thread for its own processing so only `threads_per_worker - 1` will be available for your process.

This trainer assumes an active running Ray cluster if there is no active Ray cluster it will spawn a local cluster (Refer to `ray.init()`).  The user can also specify the address manually in case the head node is in another machine. Check the api for more information.