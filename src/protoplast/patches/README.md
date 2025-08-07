# Daft Performance Patches

This directory contains monkey patches for the [Daft](https://github.com/Eventual-Inc/Daft) library to improve performance and resource utilization.

## Available Patches

### Flotilla Per-CPU Workers (`daft_flotilla.py`)

**Problem**: By default, Daft's `start_ray_workers` function creates one worker per node, with each worker using all CPUs on that node. This can lead to suboptimal resource utilization and reduced parallelism.

**Solution**: This patch modifies the function to create one worker per CPU core, providing better parallelization and resource distribution.

**Benefits**:
- Better CPU utilization (one worker per core vs one worker per node)
- Improved parallelism for CPU-bound tasks
- More granular resource allocation
- Better memory distribution across workers

## Usage

### Method 1: Manual Application (Recommended)

```python
# Import daft normally
import daft

# Apply patches after import
from protoplast.patches.daft_flotilla import apply_flotilla_patches
apply_flotilla_patches()

# Now use daft normally - it will create one worker per CPU
daft.context.set_runner_ray()
df = daft.read_parquet("data.parquet")
df.count().show()
```

### Method 2: Context Manager

```python
from protoplast.patches.auto_patch import patched_daft

with patched_daft():
    import daft
    # Your daft code here with patches applied
    df = daft.read_parquet("data.parquet")
    result = df.compute()
```

## Technical Details

### What the Patch Changes

**Original behavior**:
- One `RaySwordfishWorker` per node
- Each worker uses all CPUs on the node
- GPUs allocated to single worker per node
- Memory allocated to single worker per node

**Patched behavior**:
- One `RaySwordfishWorker` per CPU core
- Each worker uses exactly 1 CPU
- GPUs allocated only to first worker on each node (to avoid conflicts)
- Memory distributed evenly across workers on each node
- Worker IDs in format: `{NodeID}_cpu_{cpu_index}`

### Resource Allocation

For a node with 8 CPUs, 2 GPUs, and 32GB memory:

**Before patch**:
- 1 worker with 8 CPUs, 2 GPUs, 32GB memory

**After patch**:
- 8 workers, each with:
  - 1 CPU
  - First worker gets 2 GPUs, others get 0 GPUs
  - 4GB memory each (32GB ÷ 8 workers)

## Rollback

To revert to original Daft behavior:

```python
from protoplast.patches.daft_flotilla import rollback_flotilla_patches
rollback_flotilla_patches()
```

## Examples

See `examples/daft_with_cpu_workers.py` for complete usage examples.