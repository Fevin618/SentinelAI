"""Create a small, dashboard-compatible sample from the Wednesday CICIDS2017 CSV.

The source is read in chunks and is never modified. Sampling is bounded per
label so the script remains suitable for an 8 GB RAM Windows laptop.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SOURCE = Path("data/MachineLearningCSV/Wednesday-workingHours.pcap_ISCX.csv")
DEFAULT_OUTPUT = Path("data/processed/wednesday_demo_sample.csv")
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_SAMPLES_PER_LABEL = 500
DEFAULT_PER_CHUNK_PER_LABEL = 100
RANDOM_STATE = 42


def _find_label_column(columns: Iterable[object]) -> str:
    """Return the normalized Label column name or raise a useful error."""
    normalized = [str(column).strip() for column in columns]
    for column in normalized:
        if column.lower() == "label":
            return column
    raise ValueError(
        "Could not find the target label column. Expected a column named "
        "'Label' after whitespace normalization."
    )


def _read_unique_header(source: Path) -> list[str]:
    """Read the raw header and disambiguate duplicate names before pandas sees it."""
    with source.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        raw = next(csv.reader(handle), [])
    if not raw:
        raise ValueError(f"Dataset has no readable header: {source.resolve()}")
    used: dict[str, int] = {}
    unique: list[str] = []
    for value in raw:
        name = str(value).strip()
        used[name] = used.get(name, 0) + 1
        unique.append(name if used[name] == 1 else f"{name}__{used[name]}")
    return unique


def sample_dataset(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    samples_per_label: int = DEFAULT_SAMPLES_PER_LABEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    per_chunk_per_label: int = DEFAULT_PER_CHUNK_PER_LABEL,
) -> dict:
    """Create and validate a bounded sample, returning its metadata."""
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source.resolve()}")
    if source.stat().st_size == 0:
        raise ValueError(f"Dataset is empty: {source.resolve()}")
    if samples_per_label <= 0 or chunk_size <= 0 or per_chunk_per_label <= 0:
        raise ValueError("Sampling limits and chunk size must be positive integers")

    normalized_columns = _read_unique_header(source)
    label_column = _find_label_column(normalized_columns)
    sampled_chunks: list[pd.DataFrame] = []
    missing_label_rows = 0
    chunk_number = 0

    print("Reading dataset in chunks...")
    for chunk in pd.read_csv(
        source,
        chunksize=chunk_size,
        names=normalized_columns,
        header=0,
        low_memory=True,
    ):
        chunk_number += 1
        chunk.columns = normalized_columns

        missing_label_rows += int(chunk[label_column].isna().sum())
        chunk = chunk.dropna(subset=[label_column])
        if chunk.empty:
            continue
        for _, group in chunk.groupby(label_column, sort=False, dropna=False):
            take = min(per_chunk_per_label, len(group))
            sampled_chunks.append(
                group.sample(n=take, random_state=RANDOM_STATE + chunk_number)
            )

    if not sampled_chunks:
        raise ValueError("The dataset contains no rows with a usable label")

    bounded = pd.concat(sampled_chunks, ignore_index=True)
    selected: list[pd.DataFrame] = []
    for _, group in bounded.groupby(label_column, sort=False):
        selected.append(
            group.sample(
                n=min(samples_per_label, len(group)),
                random_state=RANDOM_STATE,
            )
        )
    sampled = pd.concat(selected, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output, index=False)
    size_bytes = output.stat().st_size
    size_mb = size_bytes / 1_000_000
    if size_bytes >= 500 * 1_000_000:
        raise ValueError(
            f"Generated sample is {size_mb:.2f} MB, which is not below the 500 MB upload limit"
        )

    return {
        "source": str(source),
        "output": str(output),
        "records": int(len(sampled)),
        "columns": list(sampled.columns),
        "label_column": label_column,
        "label_distribution": {
            str(label): int(count)
            for label, count in sampled[label_column].value_counts(dropna=False).items()
        },
        "size_bytes": int(size_bytes),
        "size_mb": round(size_mb, 2),
        "missing_label_rows_skipped": missing_label_rows,
        "chunk_size": chunk_size,
        "samples_per_label": samples_per_label,
    }


def validate_dashboard_compatibility(output: Path) -> dict:
    """Validate the generated file with the dashboard's actual upload helpers."""
    from dashboard.dashboard_utils import (
        load_compatible_model,
        read_uploaded_csv,
        validate_and_align_features,
    )

    class LocalUpload:
        def __init__(self, path: Path):
            self._data = path.read_bytes()

        def getvalue(self) -> bytes:
            return self._data

    frame, _ = read_uploaded_csv(LocalUpload(output))
    features, expected = validate_and_align_features(frame)
    model = load_compatible_model(Path("models/comparison/logistic_regression_balanced.joblib"))
    predictions = model.predict(features)
    return {
        "rows_read": int(len(frame)),
        "feature_count": len(features.columns),
        "expected_feature_count": len(expected),
        "has_label": "label" in frame.columns,
        "columns_aligned": list(features.columns) == expected,
        "predictions_available": len(predictions) == len(features),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-label", type=int, default=DEFAULT_SAMPLES_PER_LABEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    metadata = sample_dataset(
        source=args.input,
        output=args.output,
        samples_per_label=args.samples_per_label,
        chunk_size=args.chunk_size,
    )
    compatibility = validate_dashboard_compatibility(args.output)
    if (
        not compatibility["has_label"]
        or not compatibility["columns_aligned"]
        or not compatibility["predictions_available"]
    ):
        raise ValueError("Generated sample failed dashboard feature or label validation")

    print("\nSample created successfully!")
    print("File:", args.output.resolve())
    print("Records:", metadata["records"])
    print("Size (MB):", metadata["size_mb"])
    print("Columns:", len(metadata["columns"]))
    print("\nLabel distribution:")
    for label, count in metadata["label_distribution"].items():
        print(f"{label}: {count}")
    print("\nDashboard compatibility:")
    print(compatibility)


if __name__ == "__main__":
    main()
