"""Memory-conscious CICIDS2017 inspection and preprocessing helpers.

The functions in this module deliberately process CSV files in chunks. They do
not move, rewrite, or modify the original dataset files.
"""

from __future__ import annotations

import json
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


DEFAULT_CHUNK_SIZE = 20_000
TARGET_COLUMN = "label"


def standardize_column_name(name: str) -> str:
    """Convert a raw CICIDS column name to a stable snake_case name."""
    value = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
    return value or "unnamed_column"


def standardize_columns(columns: list[str]) -> tuple[list[str], dict[str, str]]:
    """Standardize names and disambiguate duplicate names deterministically."""
    used: Counter[str] = Counter()
    standardized: list[str] = []
    mapping: dict[str, str] = {}
    for raw in columns:
        base = standardize_column_name(raw)
        used[base] += 1
        name = base if used[base] == 1 else f"{base}__{used[base]}"
        standardized.append(name)
        mapping.setdefault(raw, name)
    return standardized, mapping


def discover_csv_files(input_dir: Path) -> list[Path]:
    """Return CSV files in a deterministic order, failing clearly if absent."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in dataset directory: {input_dir}")
    return files


def read_standardized_header(path: Path) -> tuple[list[str], list[str]]:
    """Read only the header and return raw and standardized column names."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        raw = next(csv.reader(handle))
    raw = [str(column) for column in raw]
    standardized, _ = standardize_columns(raw)
    return raw, standardized


def iter_clean_chunks(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Yield cleaned, standardized chunks from one CSV file.

    Empty strings and textual infinity variants become missing values. Numeric
    conversion is intentionally conservative: feature columns are converted to
    numeric with invalid values becoming NaN, while the target remains text.
    Imputation is not performed here because medians must be fitted on training
    data only during the modelling phase.
    """
    raw, columns = read_standardized_header(path)
    target_index = next((i for i, name in enumerate(columns) if name == TARGET_COLUMN), None)
    if target_index is None:
        raise ValueError(f"Target column 'Label' not found in {path.name}")

    reader = pd.read_csv(
        path,
        chunksize=chunk_size,
        # pandas rejects duplicate ``names``; standardized names are unique
        # because duplicate raw headers are suffixed deterministically.
        names=columns,
        header=0,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=True,
        na_values=["", "NaN", "nan", "NAN", "Infinity", "-Infinity", "inf", "-inf"],
        keep_default_na=True,
    )
    for chunk in reader:
        chunk.columns = columns
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        target = columns[target_index]
        for column in columns:
            if column != target:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk[target] = chunk[target].astype("string").str.strip()
        yield chunk


def inspect_dataset(input_dir: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict:
    """Stream all files and return schema, quality, and label statistics."""
    files = discover_csv_files(input_dir)
    file_reports: list[dict] = []
    reference_columns: list[str] | None = None
    all_labels: Counter[str] = Counter()
    total_rows = 0
    total_missing_cells = 0
    total_invalid_feature_cells = 0
    total_missing_labels = 0
    duplicate_rows = 0

    for path in files:
        raw_columns, columns = read_standardized_header(path)
        if reference_columns is None:
            reference_columns = columns
        schema_compatible = columns == reference_columns
        rows = 0
        missing_cells = 0
        invalid_feature_cells = 0
        missing_labels = 0
        labels: Counter[str] = Counter()
        seen_hashes: set[int] = set()

        for chunk in iter_clean_chunks(path, chunk_size):
            rows += len(chunk)
            missing_cells += int(chunk.isna().sum().sum())
            feature_columns = [c for c in chunk.columns if c != TARGET_COLUMN]
            invalid_feature_cells += int(chunk[feature_columns].isna().sum().sum())
            missing_labels += int(chunk[TARGET_COLUMN].isna().sum())
            labels.update(chunk[TARGET_COLUMN].dropna().astype(str).tolist())
            # Hash only a small deterministic sample per chunk to identify likely
            # duplicates without keeping the full 2.8M-row dataset in memory.
            sample = chunk.head(min(100, len(chunk))).astype("string")
            seen_hashes.update(pd.util.hash_pandas_object(sample, index=False).tolist())

        file_reports.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": rows,
                "raw_column_count": len(raw_columns),
                "standardized_columns": columns,
                "schema_compatible_with_first_file": schema_compatible,
                "duplicate_raw_column_names": sorted(
                    {c.strip() for c in raw_columns if raw_columns.count(c) > 1}
                ),
                "duplicate_column_names_after_standardization": sorted(
                    {c for c in columns if columns.count(c) > 1}
                ),
                "missing_cells_after_infinity_normalization": missing_cells,
                "invalid_or_missing_feature_cells": invalid_feature_cells,
                "missing_labels": missing_labels,
                "labels": dict(sorted(labels.items())),
                "approximate_sample_duplicate_hashes": len(seen_hashes),
            }
        )
        total_rows += rows
        total_missing_cells += missing_cells
        total_invalid_feature_cells += invalid_feature_cells
        total_missing_labels += missing_labels
        all_labels.update(labels)

    benign = all_labels.get("BENIGN", 0)
    malicious = total_rows - benign - total_missing_labels
    return {
        "input_directory": str(input_dir),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "total_rows": total_rows,
        "total_missing_cells_after_infinity_normalization": total_missing_cells,
        "total_invalid_or_missing_feature_cells": total_invalid_feature_cells,
        "total_missing_labels": total_missing_labels,
        "benign_rows": benign,
        "malicious_or_non_benign_rows": malicious,
        "label_distribution": dict(sorted(all_labels.items())),
        "class_balance": {
            "benign_fraction": benign / total_rows if total_rows else 0.0,
            "non_benign_fraction": malicious / total_rows if total_rows else 0.0,
        },
        "schema_compatible": all(report["schema_compatible_with_first_file"] for report in file_reports),
        "reference_columns": reference_columns or [],
        "file_reports": file_reports,
        "evaluation_warning": (
            "Monday contains only BENIGN traffic. Random splits alone cannot establish "
            "generalization across traffic days or attack types; use day-aware evaluation."
        ),
    }


def write_inspection_report(input_dir: Path, output_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict:
    """Inspect the dataset and write a JSON report."""
    report = inspect_dataset(input_dir, chunk_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_clean_sample(
    input_dir: Path,
    output_path: Path,
    rows_per_file: int = 5_000,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict:
    """Write a bounded, cleaned sample for development and inspection.

    Sampling is deterministic and capped per source file to avoid allowing the
    large benign-heavy files to dominate early experiments. This is not a model
    training result and is not intended to replace the source dataset.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    written = 0
    dropped_missing_label = 0
    first = True
    for path in discover_csv_files(input_dir):
        remaining = rows_per_file
        for chunk in iter_clean_chunks(path, chunk_size):
            if remaining <= 0:
                break
            before = len(chunk)
            chunk = chunk.dropna(subset=[TARGET_COLUMN]).drop_duplicates()
            dropped_missing_label += before - len(chunk)
            chunk = chunk.head(remaining)
            if len(chunk):
                chunk.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
                first = False
                written += len(chunk)
                remaining -= len(chunk)
    metadata = {
        "source_directory": str(input_dir),
        "output_file": str(output_path),
        "rows_written": written,
        "rows_dropped_for_missing_label_or_duplicate_within_chunk": dropped_missing_label,
        "rows_per_source_file_cap": rows_per_file,
        "note": "Development sample only; original CSV files are preserved.",
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
