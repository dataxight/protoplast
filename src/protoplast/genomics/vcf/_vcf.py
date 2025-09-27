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


# ruff: noqa: I002
# isort: dont-add-import: from __future__ import annotations


from daft.daft import ScanOperatorHandle
from daft.dataframe import DataFrame
from daft.logical.builder import LogicalPlanBuilder

from .scan import VCFScanOperator


def read_vcf(
    path: str,
    info_fields: list[str] | None = None,
    samples: list[str] | None = None,
    genotype_fields: list[str] | None = None,
    regions: list[str] | None = None,
    batch_size: int = 10000,
) -> DataFrame:
    """Create a DataFrame from a VCF (Variant Call Format) file.

    Args:
        path (str): Path to the VCF file.
        info_fields (list[str], optional): List of INFO fields to include.
            If None, all INFO fields are included.
        samples (list[str], optional): List of sample IDs to include.
            If None, all samples are included.
        genotype_fields (list[str], optional): List of genotype FORMAT fields to include.
            If None, all genotype fields are included.
        regions (list[str], optional): List of genomic regions to query (e.g., ["chr1:1000-2000"]).
            If None, the entire file is read.
        batch_size (int, optional): Number of records to read per batch. Defaults to 131072.

    Returns:
        DataFrame: A DataFrame with the data from the VCF file.

    Examples:
        Reading a VCF file with all data:
        >>> df = read_vcf("variants.vcf")
        >>> df.show()

        Reading specific INFO fields:
        >>> df = read_vcf("variants.vcf", info_fields=["DP", "AF"])
        >>> df.show()

        Reading specific samples:
        >>> df = read_vcf("variants.vcf", samples=["NA00001", "NA00002"])
        >>> df.show()

        Reading a specific genomic region:
        >>> df = read_vcf("variants.vcf", regions=["chr1:1000-2000"])
        >>> df.show()
    """
    vcf_operator = VCFScanOperator(
        path=path,
        info_fields=info_fields,
        samples=samples,
        genotype_fields=genotype_fields,
        regions=regions,
        batch_size=batch_size,
    )
    handle = ScanOperatorHandle.from_python_scan_operator(vcf_operator)
    builder = LogicalPlanBuilder.from_tabular_scan(scan_operator=handle)
    return DataFrame(builder)
