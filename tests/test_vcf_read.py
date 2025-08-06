import os

import pytest
from daft.dataframe import DataFrame
from daft.expressions import col

from protoplast.genomics.vcf import read_vcf


def test_read_vcf_basic():
    """Test basic VCF reading functionality"""
    # Use the working test file
    vcf_path = "data/test-simple.vcf"

    # Skip if file doesn't exist
    if not os.path.exists(vcf_path):
        pytest.skip(f"Test VCF file {vcf_path} not found")

    # Test basic reading
    df = read_vcf(vcf_path)
    assert isinstance(df, DataFrame)

    # Check that we can collect the data
    result = df.collect()
    assert len(result) == 3  # We know our test file has 3 rows

    # Check for expected columns (standard VCF columns)
    schema = df.schema()
    column_names = [field.name for field in schema]

    # VCF files should have these basic columns
    expected_columns = ["chrom", "pos", "ref", "alt"]  # oxbow uses lowercase
    for col_name in expected_columns:
        assert col_name in column_names, f"Expected column {col_name} not found in {column_names}"


def test_read_vcf_with_column_selection():
    """Test VCF reading with column projection"""
    vcf_path = "data/test-simple.vcf"

    if not os.path.exists(vcf_path):
        pytest.skip(f"Test VCF file {vcf_path} not found")

    # Test reading with specific columns
    df = read_vcf(vcf_path)

    # Select only specific columns
    selected_df = df.select(col("chrom"), col("pos"), col("ref"))
    selected_df.collect()

    # Check that only selected columns are present
    schema = selected_df.schema()
    column_names = [field.name for field in schema]
    expected_columns = ["chrom", "pos", "ref"]

    assert len(column_names) == len(expected_columns)
    for col_name in expected_columns:
        assert col_name in column_names


def test_read_vcf_with_filtering():
    """Test VCF reading with row filtering"""
    vcf_path = "data/test-simple.vcf"

    if not os.path.exists(vcf_path):
        pytest.skip(f"Test VCF file {vcf_path} not found")

    # Test reading with filtering
    df = read_vcf(vcf_path)

    # Filter for chromosome chr1
    filtered_df = df.where(col("chrom") == "chr1")
    result = filtered_df.collect()

    # Check that filtering worked
    assert len(result) == 2  # chr1 has 2 variants in our test file
    for row in result.to_pydict()["chrom"]:
        assert row == "chr1", f"Expected chromosome chr1, got {row}"


def test_read_vcf_with_specific_samples():
    """Test VCF reading with specific samples"""
    vcf_path = "data/test-simple.vcf"

    if not os.path.exists(vcf_path):
        pytest.skip(f"Test VCF file {vcf_path} not found")

    # Test reading with specific samples
    df = read_vcf(vcf_path, samples=["SAMPLE1"])
    result = df.collect()

    # Check that we got data and only the specified sample column
    assert len(result) == 3
    schema = df.schema()
    column_names = [field.name for field in schema]
    assert "SAMPLE1" in column_names
    assert "SAMPLE2" not in column_names


def test_read_vcf_with_info_fields():
    """Test VCF reading with specific INFO fields"""
    vcf_path = "data/test-simple.vcf"

    if not os.path.exists(vcf_path):
        pytest.skip(f"Test VCF file {vcf_path} not found")

    # Test reading with specific INFO fields
    df = read_vcf(vcf_path, info_fields=["DP", "AF"])
    result = df.collect()

    # Check that we got data
    assert len(result) == 3


def test_read_vcf_file_not_found():
    """Test that appropriate error is raised for non-existent file"""
    with pytest.raises(FileNotFoundError):
        read_vcf("non_existent_file.vcf")
