"""
Bloom filter implementation for sparse matrix cell-gene pairs.

This module provides functionality to build and query bloom filters for
(cell_index, gene_index) pairs from h5ad files containing single-cell sparse expression matrices.
"""

import hashlib
import struct
from typing import Tuple, Union
import numpy as np
import h5py


class CellGeneBloomFilter:
    """
    A Bloom filter for checking membership of (cell_index, gene_index) pairs.
    
    This implementation uses a bit array to efficiently check whether a
    (cell_index, gene_index) pair exists in the sparse matrix without
    storing all pairs explicitly.
    
    Attributes:
        size (int): Size of the bit array
        num_hashes (int): Number of hash functions to use
        bit_array (np.ndarray): The bit array storing filter state
    """
    
    def __init__(self, size: int, num_hashes: int = 3):
        """
        Initialize the bloom filter.
        
        Args:
            size: Size of the bit array (should be a power of 2 for best performance)
            num_hashes: Number of hash functions to use (default: 3)
        """
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = np.zeros(size, dtype=bool)
    
    def _hash_pair(self, cell_index: int, gene_index: int, seed: int) -> int:
        """
        Hash a (cell_index, gene_index) pair using a given seed.
        
        Args:
            cell_index: Index of the cell
            gene_index: Index of the gene  
            seed: Seed for the hash function
            
        Returns:
            Hash value modulo the filter size
        """
        # Combine cell_index and gene_index into a single bytes object
        data = struct.pack('QQ', cell_index, gene_index)
        
        # Create hash with seed
        hasher = hashlib.sha256()
        hasher.update(struct.pack('I', seed))
        hasher.update(data)
        
        # Convert first 8 bytes of hash to int and take modulo
        hash_bytes = hasher.digest()[:8]
        hash_int = struct.unpack('Q', hash_bytes)[0]
        return hash_int % self.size
    
    def add(self, cell_index: int, gene_index: int) -> None:
        """
        Add a (cell_index, gene_index) pair to the bloom filter.
        
        Args:
            cell_index: Index of the cell
            gene_index: Index of the gene
        """
        for i in range(self.num_hashes):
            hash_val = self._hash_pair(cell_index, gene_index, i)
            self.bit_array[hash_val] = True
    
    def contains(self, cell_index: int, gene_index: int) -> bool:
        """
        Check if a (cell_index, gene_index) pair might be in the filter.
        
        Args:
            cell_index: Index of the cell
            gene_index: Index of the gene
            
        Returns:
            True if the pair might be in the filter (could be false positive),
            False if the pair is definitely not in the filter
        """
        for i in range(self.num_hashes):
            hash_val = self._hash_pair(cell_index, gene_index, i)
            if not self.bit_array[hash_val]:
                return False
        return True
    
    def false_positive_rate(self, num_items: int) -> float:
        """
        Calculate the theoretical false positive rate.
        
        Args:
            num_items: Number of items added to the filter
            
        Returns:
            Theoretical false positive rate
        """
        if num_items == 0:
            return 0.0
        
        # Formula: (1 - e^(-k*n/m))^k
        # where k = num_hashes, n = num_items, m = size
        exp_term = np.exp(-self.num_hashes * num_items / self.size)
        return (1 - exp_term) ** self.num_hashes
    
    @classmethod
    def optimal_size(cls, num_items: int, false_positive_rate: float) -> int:
        """
        Calculate optimal filter size for given parameters.
        
        Args:
            num_items: Expected number of items to insert
            false_positive_rate: Desired false positive rate
            
        Returns:
            Optimal filter size
        """
        if num_items == 0 or false_positive_rate <= 0 or false_positive_rate >= 1:
            raise ValueError("Invalid parameters for optimal size calculation")
        
        # Formula: m = -(n * ln(p)) / (ln(2)^2)
        # where n = num_items, p = false_positive_rate, m = size
        optimal_size = -(num_items * np.log(false_positive_rate)) / (np.log(2) ** 2)
        return int(optimal_size)
    
    @classmethod
    def optimal_num_hashes(cls, size: int, num_items: int) -> int:
        """
        Calculate optimal number of hash functions.
        
        Args:
            size: Size of the bit array
            num_items: Expected number of items to insert
            
        Returns:
            Optimal number of hash functions
        """
        if num_items == 0 or size == 0:
            return 1
        
        # Formula: k = (m/n) * ln(2)
        # where m = size, n = num_items, k = num_hashes
        optimal_k = (size / num_items) * np.log(2)
        return max(1, int(optimal_k))


def build_bloom_filter_from_h5ad(
    h5ad_path: str,
    false_positive_rate: float = 0.01,
    max_size: int = None
) -> CellGeneBloomFilter:
    """
    Build a bloom filter from h5ad file sparse matrix data.
    
    This function reads the sparse matrix data from an h5ad file (specifically
    X/indptr, X/indices, and X/data) and builds a bloom filter containing
    all (cell_index, gene_index) pairs that have non-zero values.
    
    Args:
        h5ad_path: Path to the h5ad file
        false_positive_rate: Desired false positive rate (default: 0.01)
        max_size: Maximum size for the bloom filter (optional)
        
    Returns:
        Initialized bloom filter containing all non-zero pairs
        
    Raises:
        FileNotFoundError: If h5ad file doesn't exist
        KeyError: If required datasets are not found in h5ad file
    """
    with h5py.File(h5ad_path, 'r') as f:
        # Check if X group exists and has required datasets
        if 'X' not in f:
            raise KeyError("X group not found in h5ad file")
        
        x_group = f['X']
        required_datasets = ['indptr', 'indices', 'data']
        for dataset in required_datasets:
            if dataset not in x_group:
                raise KeyError(f"X/{dataset} not found in h5ad file")
        
        # Read sparse matrix data
        indptr = x_group['indptr'][:]
        indices = x_group['indices'][:]
        data = x_group['data'][:]
        
        # Get matrix dimensions from attributes or calculate
        if 'shape' in x_group.attrs:
            n_cells, n_genes = x_group.attrs['shape']
        else:
            n_cells = len(indptr) - 1
            n_genes = np.max(indices) + 1 if len(indices) > 0 else 0
        
        # Count non-zero entries
        num_nonzero = len(data)
        
        # Handle empty matrix case
        if num_nonzero == 0:
            filter_size = 1000  # Default size for empty matrix
            num_hashes = 3  # Default number of hashes
        else:
            # Calculate optimal filter parameters
            if max_size is not None:
                filter_size = min(max_size, CellGeneBloomFilter.optimal_size(num_nonzero, false_positive_rate))
            else:
                filter_size = CellGeneBloomFilter.optimal_size(num_nonzero, false_positive_rate)
            
            num_hashes = CellGeneBloomFilter.optimal_num_hashes(filter_size, num_nonzero)
        
        # Create bloom filter
        bloom_filter = CellGeneBloomFilter(filter_size, num_hashes)
        
        # Add all non-zero (cell_index, gene_index) pairs
        data_idx = 0
        for cell_idx in range(n_cells):
            start_idx = indptr[cell_idx]
            end_idx = indptr[cell_idx + 1]
            
            for gene_idx in indices[start_idx:end_idx]:
                bloom_filter.add(cell_idx, gene_idx)
                data_idx += 1
        
        return bloom_filter


def check_pairs_in_h5ad(
    h5ad_path: str,
    pairs: list[Tuple[int, int]],
    use_bloom_filter: bool = True,
    false_positive_rate: float = 0.01
) -> list[bool]:
    """
    Check if (cell_index, gene_index) pairs exist in h5ad sparse matrix.
    
    Args:
        h5ad_path: Path to the h5ad file
        pairs: List of (cell_index, gene_index) pairs to check
        use_bloom_filter: Whether to use bloom filter for checking (default: True)
        false_positive_rate: False positive rate for bloom filter (default: 0.01)
        
    Returns:
        List of boolean values indicating membership for each pair
    """
    if use_bloom_filter:
        # Build bloom filter and check pairs
        bloom_filter = build_bloom_filter_from_h5ad(h5ad_path, false_positive_rate)
        return [bloom_filter.contains(cell_idx, gene_idx) for cell_idx, gene_idx in pairs]
    else:
        # Direct check without bloom filter (exact results)
        with h5py.File(h5ad_path, 'r') as f:
            x_group = f['X']
            indptr = x_group['indptr'][:]
            indices = x_group['indices'][:]
            
            results = []
            for cell_idx, gene_idx in pairs:
                if cell_idx >= len(indptr) - 1:
                    results.append(False)
                    continue
                
                start_idx = indptr[cell_idx]
                end_idx = indptr[cell_idx + 1]
                cell_genes = indices[start_idx:end_idx]
                
                results.append(gene_idx in cell_genes)
            
            return results