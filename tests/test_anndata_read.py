#   Copyright 2025 DataXight, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


"""
Tests for reading h5ad files.
"""

import os
import pathlib

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from daft.dataframe import DataFrame
from daft.expressions import col
from scipy.sparse import csr_matrix

from protoplast.scrna.anndata import read_h5ad


@pytest.fixture(scope="function")
def test_h5ad_file(tmpdir: pathlib.Path) -> str:
    # Create a small AnnData object with sparse data
    # dense matrix:
    # [[1, 0, 2, 0, 0],
    #  [0, 0, 0, 0, 0],
    #  [0, 3, 0, 4, 0],
    #  [5, 0, 0, 0, 0]]
    n_obs = 4
    n_vars = 5

    indptr = np.array([0, 2, 2, 4, 5])
    indices = np.array([0, 2, 1, 3, 0])
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    X = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    filepath = tmpdir / "test.h5ad"
    adata.write_h5ad(filepath)
    return str(filepath)


def test_read_h5ad_single_file(test_h5ad_file: str):
    """Reads a single h5ad file and checks basic properties"""
    df = read_h5ad(test_h5ad_file, batch_size=2)
    assert isinstance(df, DataFrame)

    # H5ADScanOperator in scan.py hardcodes schema to first 10 genes
    pd_df = df.to_pandas()
    assert len(pd_df) == 4
    assert list(pd_df.columns) == [f"gene_{i}" for i in range(5)]

    expected_data = {
        "gene_0": [1.0, 0.0, 0.0, 5.0],
        "gene_1": [0.0, 0.0, 3.0, 0.0],
        "gene_2": [2.0, 0.0, 0.0, 0.0],
        "gene_3": [0.0, 0.0, 4.0, 0.0],
        "gene_4": [0.0, 0.0, 0.0, 0.0],
    }
    expected_df = pd.DataFrame(expected_data).astype(np.float32)
    pd.testing.assert_frame_equal(pd_df, expected_df)


def test_read_h5ad_select_columns(test_h5ad_file: str):
    """Tests selecting a subset of columns"""
    df = read_h5ad(test_h5ad_file).select(col("gene_1"), col("gene_3"))

    pd_df = df.to_pandas()
    assert len(pd_df) == 4
    assert list(pd_df.columns) == ["gene_1", "gene_3"]

    expected_data = {
        "gene_1": [0.0, 0.0, 3.0, 0.0],
        "gene_3": [0.0, 0.0, 4.0, 0.0],
    }
    expected_df = pd.DataFrame(expected_data).astype(np.float32)
    pd.testing.assert_frame_equal(pd_df, expected_df)


def test_read_h5ad_with_limit(test_h5ad_file: str):
    """Tests limiting the number of rows"""
    df = read_h5ad(test_h5ad_file).limit(2)

    pd_df = df.to_pandas()
    assert len(pd_df) == 2

    expected_data = {
        "gene_0": [1.0, 0.0],
        "gene_1": [0.0, 0.0],
        "gene_2": [2.0, 0.0],
        "gene_3": [0.0, 0.0],
        "gene_4": [0.0, 0.0],
    }
    expected_df = pd.DataFrame(expected_data).astype(np.float32)
    pd.testing.assert_frame_equal(pd_df, expected_df)


def test_read_h5ad_with_filter_only(test_h5ad_file: str):
    """Tests filtering rows based on a condition"""
    df = read_h5ad(test_h5ad_file).filter(col("gene_1") > 0)

    pd_df = df.to_pandas()
    assert len(pd_df) == 1

    expected_data = {
        "gene_0": [0.0],
        "gene_1": [3.0],
        "gene_2": [0.0],
        "gene_3": [4.0],
        "gene_4": [0.0],
    }
    # Resetting index because .where() can lead to a filtered index
    pd.testing.assert_frame_equal(pd_df.reset_index(drop=True), pd.DataFrame(expected_data).astype(np.float32))


def test_read_h5ad_with_filter_and_select(test_h5ad_file: str):
    """Tests filtering and selecting at the same time"""
    df = read_h5ad(test_h5ad_file).where(col("gene_1") > 0).select(col("gene_1"), col("gene_3"))
    pd_df = df.to_pandas()
    assert len(pd_df) == 1
    assert sorted(list(pd_df.columns)) == ["gene_1", "gene_3"]


def test_count_data(test_h5ad_file: str):
    """Tests reading count data"""
    df = read_h5ad(test_h5ad_file, batch_size=2)
    count = df.count().to_pandas()["count"][0]
    assert count == 4


@pytest.fixture(scope="function")
def test_h5ad_file_custom_var_dataset(tmpdir: pathlib.Path) -> str:
    """Create an h5ad file with variable names stored in a custom location"""
    # Create a small AnnData object with sparse data
    n_obs = 4
    n_vars = 5

    indptr = np.array([0, 2, 2, 4, 5])
    indices = np.array([0, 2, 1, 3, 0])
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    X = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"custom_gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    filepath = tmpdir / "test_custom_var.h5ad"
    adata.write_h5ad(filepath)

    # Now modify the h5ad file to store variable names in a custom location
    with h5py.File(filepath, "r+") as f:
        # Copy the original variable names to a custom location
        original_var_names = f["var/_index"][:]

        # Create a custom group and dataset
        if "custom_var" not in f:
            custom_group = f.create_group("custom_var")
        else:
            custom_group = f["custom_var"]

        if "gene_names" in custom_group:
            del custom_group["gene_names"]

        custom_group.create_dataset("gene_names", data=original_var_names)

    return str(filepath)


def test_read_h5ad_custom_var_dataset(test_h5ad_file_custom_var_dataset: str):
    """Tests reading h5ad file with custom var dataset location"""
    # Read using the custom var dataset location
    df = read_h5ad(test_h5ad_file_custom_var_dataset, var_h5dataset="custom_var/gene_names")
    assert isinstance(df, DataFrame)

    pd_df = df.to_pandas()
    assert len(pd_df) == 4
    assert list(pd_df.columns) == [f"custom_gene_{i}" for i in range(5)]

    expected_data = {
        "custom_gene_0": [1.0, 0.0, 0.0, 5.0],
        "custom_gene_1": [0.0, 0.0, 3.0, 0.0],
        "custom_gene_2": [2.0, 0.0, 0.0, 0.0],
        "custom_gene_3": [0.0, 0.0, 4.0, 0.0],
        "custom_gene_4": [0.0, 0.0, 0.0, 0.0],
    }
    expected_df = pd.DataFrame(expected_data).astype(np.float32)
    pd.testing.assert_frame_equal(pd_df, expected_df)


@pytest.mark.skipif(os.environ.get("AWS_ACCESS_KEY") is None, reason="test requires AWS credentials")
def test_anndata_read_h5ad_remote():
    """Tests reading a remote h5ad file"""
    df = read_h5ad("s3://anyscale-ap-data/test_small.h5ad", batch_size=1000, preview_size=0)
    assert isinstance(df, DataFrame)

    pd_df = df.to_pandas()
    assert len(pd_df) == 100
    assert len(pd_df.columns) == 50
