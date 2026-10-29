"""CLI for streaming CICIDS2017 inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_pipeline import write_inspection_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/MachineLearningCSV"))
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_inspection.json"))
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()
    report = write_inspection_report(args.input, args.output, args.chunk_size)
    # CICIDS2017 contains a Web Attack label whose encoding may not be
    # representable by the Windows console code page.
    try:
        import sys

        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    print(f"Inspected {report['file_count']} files and {report['total_rows']:,} rows.")
    print(f"Labels: {report['label_distribution']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
