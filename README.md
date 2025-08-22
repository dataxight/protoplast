# protoplast

![PyPI version](https://img.shields.io/pypi/v/protoplast.svg)
[![Documentation Status](https://readthedocs.org/projects/protoplast/badge/?version=latest)](https://protoplast.readthedocs.io/en/latest/?version=latest)

A collection of drivers for Protoform, the data engineering framework from DataXight

-   PyPI package: https://pypi.org/project/protoplast/

## Features

### Genomics Data Processing

**VCF File Support:**
- Read VCF (Variant Call Format) files with lazy evaluation using Daft DataFrames
- Column projection and row filtering for efficient data processing
- Sample-specific data extraction and INFO field selection
- Standard genomics operations (chromosome filtering, variant analysis)

**AnnData/h5ad Support:**
- Read single-cell genomics data from h5ad files
- Column selection, row filtering, and count operations

## Usage

### VCF Files

```python
from protoplast.genomics.vcf import read_vcf
from daft.expressions import col

# Read a VCF file
df = read_vcf("variants.vcf")

# Basic operations
result = df.collect()  # Load data into memory
print(f"Found {len(result)} variants")

# Column selection
selected = df.select(col("chrom"), col("pos"), col("ref"), col("alt"))

# Filter by chromosome
chr1_variants = df.where(col("chrom") == "chr1")

# Read specific samples only
df_samples = read_vcf("variants.vcf", samples=["SAMPLE1", "SAMPLE2"])

# Extract specific INFO fields
df_info = read_vcf("variants.vcf", info_fields=["DP", "AF"])
```

### AnnData/h5ad Files

```python
from protoplast.scrna.anndata import read_h5ad
from daft.expressions import col

# Read h5ad file with batch processing
df = read_h5ad("data.h5ad", batch_size=1000)

# Select specific genes
genes_of_interest = df.select(col("gene_1"), col("gene_3"))

# Filter cells based on gene expression
high_expression = df.where(col("gene_1") > 0.5)

# Combine filtering and selection
filtered_genes = df.where(col("gene_1") > 0).select(col("gene_1"), col("gene_3"))

# Count operations
total_cells = df.count()

# Limit number of rows for quick inspection
sample_data = df.limit(100)
```