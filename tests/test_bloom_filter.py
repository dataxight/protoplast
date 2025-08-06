"""
Unit tests for the bloom filter functionality.
"""

import tempfile
import pathlib
import numpy as np
import pandas as pd
import pytest
import h5py
from scipy.sparse import csr_matrix
import anndata as ad

from protoplast.genomics.anndata.bloom_filter import (
    CellGeneBloomFilter,
    build_bloom_filter_from_h5ad,
    check_pairs_in_h5ad
)


class TestCellGeneBloomFilter:
    """Test cases for CellGeneBloomFilter class."""
    
    def test_initialization(self):
        """Test bloom filter initialization."""
        bf = CellGeneBloomFilter(size=1000, num_hashes=3)
        assert bf.size == 1000
        assert bf.num_hashes == 3
        assert len(bf.bit_array) == 1000
        assert not np.any(bf.bit_array)  # All bits should be False initially
    
    def test_add_and_contains(self):
        """Test adding pairs and checking membership."""
        bf = CellGeneBloomFilter(size=1000, num_hashes=3)
        
        # Add some pairs
        pairs_to_add = [(0, 1), (5, 10), (100, 200)]
        for cell_idx, gene_idx in pairs_to_add:
            bf.add(cell_idx, gene_idx)
        
        # Check that added pairs are found
        for cell_idx, gene_idx in pairs_to_add:
            assert bf.contains(cell_idx, gene_idx)
        
        # Test some pairs that weren't added (might have false positives)
        # We can't guarantee these will be False due to false positives,
        # but we can test that the method works
        result = bf.contains(999, 999)
        assert isinstance(result, bool)
    
    def test_hash_pair_deterministic(self):
        """Test that hash function is deterministic."""
        bf = CellGeneBloomFilter(size=1000, num_hashes=3)
        
        # Same inputs should produce same hash
        hash1 = bf._hash_pair(10, 20, 0)
        hash2 = bf._hash_pair(10, 20, 0)
        assert hash1 == hash2
        
        # Different seeds should produce different hashes
        hash3 = bf._hash_pair(10, 20, 1)
        assert hash1 != hash3
        
        # Different inputs should produce different hashes (usually)
        hash4 = bf._hash_pair(11, 20, 0)
        assert hash1 != hash4
    
    def test_false_positive_rate_calculation(self):
        """Test false positive rate calculation."""
        bf = CellGeneBloomFilter(size=1000, num_hashes=3)
        
        # Empty filter should have 0% false positive rate
        assert bf.false_positive_rate(0) == 0.0
        
        # Non-empty filter should have positive false positive rate
        fpr = bf.false_positive_rate(100)
        assert 0 < fpr < 1
    
    def test_optimal_size_calculation(self):
        """Test optimal size calculation."""
        # Valid parameters
        size = CellGeneBloomFilter.optimal_size(1000, 0.01)
        assert size > 0
        assert isinstance(size, int)
        
        # Invalid parameters should raise ValueError
        with pytest.raises(ValueError):
            CellGeneBloomFilter.optimal_size(0, 0.01)
        
        with pytest.raises(ValueError):
            CellGeneBloomFilter.optimal_size(1000, 0)
        
        with pytest.raises(ValueError):
            CellGeneBloomFilter.optimal_size(1000, 1.0)
    
    def test_optimal_num_hashes(self):
        """Test optimal number of hashes calculation."""
        # Valid parameters
        num_hashes = CellGeneBloomFilter.optimal_num_hashes(1000, 100)
        assert num_hashes >= 1
        assert isinstance(num_hashes, int)
        
        # Edge cases
        assert CellGeneBloomFilter.optimal_num_hashes(0, 100) == 1
        assert CellGeneBloomFilter.optimal_num_hashes(1000, 0) == 1


@pytest.fixture
def test_h5ad_file():
    """Create a test h5ad file with known sparse matrix data."""
    # Create test data similar to the existing test fixture
    n_obs = 4
    n_vars = 5
    
    # Create sparse matrix with known non-zero positions
    # Dense representation:
    # [[1, 0, 2, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 3, 0, 4, 0],
    #  [5, 0, 0, 0, 0]]
    indptr = np.array([0, 2, 2, 4, 5])
    indices = np.array([0, 2, 1, 3, 0])
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    
    X = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
    
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
        adata.write_h5ad(tmp.name)
        yield tmp.name, {
            'n_obs': n_obs,
            'n_vars': n_vars,
            'nonzero_pairs': [(0, 0), (0, 2), (2, 1), (2, 3), (3, 0)],
            'zero_pairs': [(0, 1), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 2), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)]
        }
    
    # Cleanup happens automatically with NamedTemporaryFile


class TestH5adIntegration:
    """Test cases for h5ad file integration."""
    
    def test_build_bloom_filter_from_h5ad(self, test_h5ad_file):
        """Test building bloom filter from h5ad file."""
        h5ad_path, expected_data = test_h5ad_file
        
        bf = build_bloom_filter_from_h5ad(h5ad_path, false_positive_rate=0.01)
        
        # Check that filter was created
        assert isinstance(bf, CellGeneBloomFilter)
        assert bf.size > 0
        assert bf.num_hashes > 0
        
        # Check that all known non-zero pairs are found
        for cell_idx, gene_idx in expected_data['nonzero_pairs']:
            assert bf.contains(cell_idx, gene_idx), f"Failed to find pair ({cell_idx}, {gene_idx})"
    
    def test_build_bloom_filter_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            build_bloom_filter_from_h5ad("nonexistent_file.h5ad")
    
    def test_build_bloom_filter_invalid_h5ad(self):
        """Test error handling for invalid h5ad file."""
        # Create a temporary file that's not a valid h5ad
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
            with h5py.File(tmp.name, 'w') as f:
                # Create file without X group
                f.create_dataset('dummy', data=[1, 2, 3])
            
            with pytest.raises(KeyError, match="X group not found"):
                build_bloom_filter_from_h5ad(tmp.name)
    
    def test_build_bloom_filter_missing_datasets(self):
        """Test error handling for missing required datasets."""
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
            with h5py.File(tmp.name, 'w') as f:
                x_group = f.create_group('X')
                # Only create some of the required datasets
                x_group.create_dataset('indptr', data=[0, 1, 2])
                # Missing 'indices' and 'data'
            
            with pytest.raises(KeyError, match="X/indices not found"):
                build_bloom_filter_from_h5ad(tmp.name)
    
    def test_check_pairs_in_h5ad_with_bloom_filter(self, test_h5ad_file):
        """Test checking pairs using bloom filter."""
        h5ad_path, expected_data = test_h5ad_file
        
        # Test with known non-zero pairs
        results = check_pairs_in_h5ad(
            h5ad_path,
            expected_data['nonzero_pairs'],
            use_bloom_filter=True
        )
        
        # All non-zero pairs should be found (no false negatives)
        assert all(results), "Bloom filter should find all non-zero pairs"
        
        # Test with some zero pairs (might have false positives)
        zero_results = check_pairs_in_h5ad(
            h5ad_path,
            expected_data['zero_pairs'][:3],  # Test first 3 zero pairs
            use_bloom_filter=True
        )
        
        # Results should be all boolean values
        assert all(isinstance(r, bool) for r in zero_results)
    
    def test_check_pairs_in_h5ad_without_bloom_filter(self, test_h5ad_file):
        """Test checking pairs without bloom filter (exact method)."""
        h5ad_path, expected_data = test_h5ad_file
        
        # Test with known non-zero pairs
        results = check_pairs_in_h5ad(
            h5ad_path,
            expected_data['nonzero_pairs'],
            use_bloom_filter=False
        )
        
        # All non-zero pairs should be found
        assert all(results), "Exact method should find all non-zero pairs"
        
        # Test with known zero pairs
        zero_results = check_pairs_in_h5ad(
            h5ad_path,
            expected_data['zero_pairs'][:3],  # Test first 3 zero pairs
            use_bloom_filter=False
        )
        
        # All zero pairs should return False
        assert not any(zero_results), "Exact method should not find zero pairs"
    
    def test_check_pairs_out_of_bounds(self, test_h5ad_file):
        """Test checking pairs with out-of-bounds indices."""
        h5ad_path, expected_data = test_h5ad_file
        
        # Test with cell index out of bounds
        out_of_bounds_pairs = [
            (expected_data['n_obs'], 0),  # Cell index too large
            (0, expected_data['n_vars']),  # Gene index too large
            (expected_data['n_obs'] + 10, expected_data['n_vars'] + 10)  # Both too large
        ]
        
        # Using exact method (should handle gracefully)
        results = check_pairs_in_h5ad(
            h5ad_path,
            out_of_bounds_pairs,
            use_bloom_filter=False
        )
        
        # All out-of-bounds pairs should return False
        assert not any(results), "Out-of-bounds pairs should return False"
    
    def test_bloom_filter_consistency(self, test_h5ad_file):
        """Test that bloom filter results are consistent across calls."""
        h5ad_path, expected_data = test_h5ad_file
        
        test_pairs = expected_data['nonzero_pairs'][:3]
        
        # Build filter twice and check same pairs
        results1 = check_pairs_in_h5ad(h5ad_path, test_pairs, use_bloom_filter=True)
        results2 = check_pairs_in_h5ad(h5ad_path, test_pairs, use_bloom_filter=True)
        
        # Results should be identical
        assert results1 == results2, "Bloom filter should give consistent results"
    
    def test_bloom_filter_max_size_parameter(self, test_h5ad_file):
        """Test bloom filter with max size parameter."""
        h5ad_path, expected_data = test_h5ad_file
        
        # Build filter with small max size
        bf = build_bloom_filter_from_h5ad(h5ad_path, max_size=100)
        
        # Filter should respect max size
        assert bf.size <= 100
        
        # Should still find the non-zero pairs (though with higher false positive rate)
        for cell_idx, gene_idx in expected_data['nonzero_pairs']:
            assert bf.contains(cell_idx, gene_idx)


class TestPerformanceAndEdgeCases:
    """Test cases for performance and edge cases."""
    
    def test_empty_matrix(self):
        """Test handling of empty sparse matrix."""
        # Create empty h5ad file
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
            # Create minimal AnnData with empty sparse matrix
            X = csr_matrix((0, 0), dtype=np.float32)
            obs = pd.DataFrame(index=[])
            var = pd.DataFrame(index=[])
            adata = ad.AnnData(X=X, obs=obs, var=var)
            adata.write_h5ad(tmp.name)
            
            # Should handle empty matrix gracefully
            bf = build_bloom_filter_from_h5ad(tmp.name)
            assert isinstance(bf, CellGeneBloomFilter)
            
            # Any query should return False
            assert not bf.contains(0, 0)
    
    def test_single_cell_single_gene(self):
        """Test with minimal 1x1 matrix."""
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
            # Create 1x1 matrix with single non-zero value
            X = csr_matrix(np.array([[1.0]]), dtype=np.float32)
            obs = pd.DataFrame(index=['cell_0'])
            var = pd.DataFrame(index=['gene_0'])
            adata = ad.AnnData(X=X, obs=obs, var=var)
            adata.write_h5ad(tmp.name)
            
            bf = build_bloom_filter_from_h5ad(tmp.name)
            
            # Should find the single pair
            assert bf.contains(0, 0)
    
    def test_large_indices(self):
        """Test with large cell and gene indices."""
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
            # Create sparse matrix with large dimensions but few non-zeros
            n_obs, n_vars = 1000, 2000
            indices = np.array([0, 999, 1500])  # Large gene indices
            indptr = np.array([0, 2, 2, 3, 3] + [3] * (n_obs - 4))  # Most cells empty
            data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            
            X = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
            obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
            var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
            adata = ad.AnnData(X=X, obs=obs, var=var)
            adata.write_h5ad(tmp.name)
            
            bf = build_bloom_filter_from_h5ad(tmp.name)
            
            # Should find the non-zero pairs
            expected_pairs = [(0, 0), (0, 999), (2, 1500)]
            for cell_idx, gene_idx in expected_pairs:
                assert bf.contains(cell_idx, gene_idx)