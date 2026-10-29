"""CLI for creating a bounded cleaned development sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_pipeline import write_clean_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/MachineLearningCSV"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/clean_sample.csv"))
    parser.add_argument("--rows-per-file", type=int, default=5_000)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()
    metadata = write_clean_sample(args.input, args.output, args.rows_per_file, args.chunk_size)
    print(f"Wrote {metadata['rows_written']:,} cleaned development rows to {args.output}.")
    print("No model was trained; imputation/scaling remain training-only operations for Phase 3.")


if __name__ == "__main__":
    main()
