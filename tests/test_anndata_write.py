from __future__ import annotations

import pathlib

import daft
import numpy as np
import pyarrow as pa
import pytest
from protoplast.genomics.anndata import AnnDataSink


@pytest.fixture(scope="function")
def daft_df():
    return daft.from_pydict(
        {
            "obs_names": pa.array(["cell1", "cell2", "cell3"], type=pa.string()),
            "arr": pa.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], type=pa.list_(pa.int64(), 3)),
            "bar": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            "baz": pa.array([True, False, True], type=pa.bool_()),
        }
    )


def test_anndata_write_sink(daft_df: daft.DataFrame, tmp_path: pathlib.Path):
    path = tmp_path / "test_output"
    sink = AnnDataSink(str(path))
    daft_df.write_sink(sink)

    assert path.exists()
    files = list(path.glob("*.parquet"))
    assert len(files) > 0

    tables = [pa.parquet.read_table(f) for f in files]
    result_table = pa.concat_tables(tables)
    result_df = daft.from_arrow(result_table)

    # Sort by "obs_names" to ensure the order is the same
    daft_df = daft_df.sort(daft_df["obs_names"])
    result_df = result_df.sort(result_df["obs_names"])

    assert daft_df.column_names == result_df.column_names

    # Check that the data is the same
    for col_name in daft_df.column_names:
        # Pydantic arrays can't be compared directly
        if col_name == "arr":
            np.testing.assert_array_equal(
                daft_df.to_pydict()[col_name],
                result_df.to_pydict()[col_name],
            )
        else:
            assert daft_df.to_pydict()[col_name] == result_df.to_pydict()[col_name]
