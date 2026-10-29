from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    TARGET_COLUMN,
    discover_csv_files,
    iter_clean_chunks,
    standardize_columns,
    standardize_column_name,
    write_clean_sample,
)


def test_standardize_columns_disambiguates_duplicate_names():
    names, _ = standardize_columns([" Fwd Header Length", "Fwd Header Length", "Label"])
    assert names == ["fwd_header_length", "fwd_header_length__2", "label"]
    assert standardize_column_name("Flow Bytes/s") == "flow_bytes_s"


def test_clean_chunks_normalize_invalid_values_and_target(tmp_path: Path):
    source = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            " Destination Port": [80, 443, 22],
            "Flow Bytes/s": ["Infinity", "-Infinity", "NaN"],
            "Label": [" BENIGN", "Attack", "Attack"],
        }
    ).to_csv(source, index=False)
    chunk = next(iter_clean_chunks(source, chunk_size=2))
    assert list(chunk.columns) == ["destination_port", "flow_bytes_s", TARGET_COLUMN]
    assert chunk["flow_bytes_s"].isna().all()
    assert chunk[TARGET_COLUMN].tolist() == ["BENIGN", "Attack"]


def test_discover_csv_files_fails_for_missing_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_csv_files(tmp_path / "missing")


def test_clean_sample_is_bounded_and_does_not_change_source(tmp_path: Path):
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    source = source_dir / "one.csv"
    pd.DataFrame({"Feature": [1, 2, 3], "Label": ["BENIGN", "Attack", "Attack"]}).to_csv(source, index=False)
    original = source.read_bytes()
    metadata = write_clean_sample(source_dir, tmp_path / "processed" / "sample.csv", rows_per_file=2)
    assert metadata["rows_written"] == 2
    assert source.read_bytes() == original

