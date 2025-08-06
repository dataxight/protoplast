# AnnData Bloom Filter

This module provides bloom filter functionality for efficiently checking membership of (cell_index, gene_index) pairs in single-cell sparse expression matrices stored in h5ad format.

## Features

- **Memory-efficient**: Bloom filters use much less memory than storing all pairs explicitly
- **Fast queries**: O(k) time complexity for membership queries, where k is the number of hash functions
- **No false negatives**: If the filter says a pair doesn't exist, it definitely doesn't exist
- **Tunable false positive rate**: Configure the trade-off between memory usage and accuracy
- **H5ad integration**: Direct support for reading from h5ad files with CSR sparse matrix format

## Usage

### Basic Usage

```python
from protoplast.genomics.anndata import (
    build_bloom_filter_from_h5ad,
    check_pairs_in_h5ad
)

# Build a bloom filter from an h5ad file
bloom_filter = build_bloom_filter_from_h5ad("data.h5ad", false_positive_rate=0.01)

# Check if specific pairs exist
pairs_to_check = [(0, 100), (5, 250), (10, 500)]
results = check_pairs_in_h5ad("data.h5ad", pairs_to_check, use_bloom_filter=True)

# Check individual pairs
exists = bloom_filter.contains(cell_index=0, gene_index=100)
```

### Advanced Usage

```python
from protoplast.genomics.anndata import CellGeneBloomFilter

# Create a custom bloom filter
bf = CellGeneBloomFilter(size=100000, num_hashes=5)

# Add pairs manually
bf.add(0, 100)
bf.add(5, 250)

# Check membership
if bf.contains(0, 100):
    print("Pair (0, 100) might exist")

# Calculate optimal parameters
optimal_size = CellGeneBloomFilter.optimal_size(num_items=10000, false_positive_rate=0.01)
optimal_hashes = CellGeneBloomFilter.optimal_num_hashes(size=optimal_size, num_items=10000)
```

## H5AD File Format

The bloom filter expects h5ad files with the standard CSR (Compressed Sparse Row) format:

- `X/indptr`: Array of indices pointing to the start of each cell's data
- `X/indices`: Array of gene indices for non-zero values
- `X/data`: Array of non-zero expression values

This is the standard format used by AnnData and scanpy.

## Performance Characteristics

| Items | FP Rate | Filter Size | Hash Functions | Memory Usage |
|-------|---------|-------------|----------------|--------------|
| 1,000 | 0.01    | ~10KB       | 6              | Very Low     |
| 10,000| 0.01    | ~95KB       | 6              | Low          |
| 100,000| 0.01   | ~950KB      | 6              | Medium       |

## Use Cases

1. **Pre-filtering**: Quickly eliminate pairs that definitely don't exist before expensive operations
2. **Memory constraints**: When you can't afford to store all pairs in memory
3. **Distributed computing**: Send compact bloom filters instead of large pair sets
4. **Interactive analysis**: Fast membership queries for exploratory data analysis

## Example

See `examples/bloom_filter_example.py` for a complete demonstration of the functionality.

## API Reference

### CellGeneBloomFilter

Main bloom filter class for (cell_index, gene_index) pairs.

**Methods:**
- `add(cell_index, gene_index)`: Add a pair to the filter
- `contains(cell_index, gene_index)`: Check if a pair might be in the filter
- `false_positive_rate(num_items)`: Calculate theoretical false positive rate
- `optimal_size(num_items, false_positive_rate)`: Calculate optimal filter size
- `optimal_num_hashes(size, num_items)`: Calculate optimal number of hash functions

### Functions

- `build_bloom_filter_from_h5ad(h5ad_path, false_positive_rate=0.01, max_size=None)`: Build filter from h5ad file
- `check_pairs_in_h5ad(h5ad_path, pairs, use_bloom_filter=True, false_positive_rate=0.01)`: Check pairs in h5ad file