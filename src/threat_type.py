"""Train and evaluate the separate multiclass attack-type model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_pipeline import DEFAULT_CHUNK_SIZE, TARGET_COLUMN, discover_csv_files, iter_clean_chunks
from .train import DEFAULT_PER_CLASS_PER_FILE, RANDOM_STATE, _bounded_class_sample


DEFAULT_MIN_CLASS_COUNT = 100
THREAT_TARGET_COLUMN = "threat_type"
BENIGN_LABEL = "BENIGN"


def count_attack_labels(input_dir: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Counter[str]:
    """Count the real non-benign labels by streaming the source files."""
    counts: Counter[str] = Counter()
    for path in discover_csv_files(input_dir):
        for chunk in iter_clean_chunks(path, chunk_size):
            labels = chunk[TARGET_COLUMN].dropna().astype(str)
            counts.update(labels[labels != BENIGN_LABEL].tolist())
    return counts


def supported_attack_labels(label_counts: dict[str, int] | Counter[str], min_class_count: int = DEFAULT_MIN_CLASS_COUNT) -> set[str]:
    """Select only real attack labels with enough examples for evaluation."""
    return {label for label, count in label_counts.items() if label != BENIGN_LABEL and count >= min_class_count}


def load_attack_sample(
    input_dir: Path,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
    per_class_per_file: int = DEFAULT_PER_CLASS_PER_FILE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Load a bounded attack-only sample after selecting real supported labels."""
    if min_class_count <= 0 or per_class_per_file <= 0 or chunk_size <= 0:
        raise ValueError("Sampling limits must be positive integers")
    label_counts = count_attack_labels(input_dir, chunk_size)
    supported = supported_attack_labels(label_counts, min_class_count)
    if len(supported) < 2:
        raise ValueError("Fewer than two sufficiently represented attack labels were found")

    rng = np.random.default_rng(random_state)
    rows: list[pd.DataFrame] = []
    for path in discover_csv_files(input_dir):
        sampled = {label: pd.DataFrame() for label in supported}
        for chunk in iter_clean_chunks(path, chunk_size):
            chunk = chunk.dropna(subset=[TARGET_COLUMN]).drop_duplicates()
            chunk = chunk.loc[chunk[TARGET_COLUMN].isin(supported)].copy()
            if chunk.empty:
                continue
            chunk[THREAT_TARGET_COLUMN] = chunk[TARGET_COLUMN].astype(str)
            chunk["source_file"] = path.name
            for label in supported:
                sampled[label] = _bounded_class_sample(
                    sampled[label],
                    chunk.loc[chunk[THREAT_TARGET_COLUMN] == label],
                    per_class_per_file,
                    rng,
                )
        rows.extend(part for part in sampled.values() if not part.empty)

    sample = pd.concat(rows, ignore_index=True).drop_duplicates().reset_index(drop=True)
    sample = sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sample_counts = {label: int(count) for label, count in sample[THREAT_TARGET_COLUMN].value_counts().items()}
    return sample, dict(sorted(label_counts.items())), sample_counts


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return only the original numeric model features."""
    return [column for column in frame.columns if column not in {TARGET_COLUMN, THREAT_TARGET_COLUMN, "source_file"}]


def make_threat_type_model(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create a separate leakage-safe multiclass Random Forest pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=160,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_and_evaluate(
    input_dir: Path,
    model_path: Path,
    report_path: Path,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
    per_class_per_file: int = DEFAULT_PER_CLASS_PER_FILE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Fit only on the training split and save held-out multiclass metrics."""
    sample, all_attack_counts, sample_counts = load_attack_sample(
        input_dir, min_class_count, per_class_per_file, chunk_size, random_state
    )
    columns = feature_columns(sample)
    train_frame, test_frame = train_test_split(
        sample,
        test_size=0.25,
        random_state=random_state,
        stratify=sample[THREAT_TARGET_COLUMN],
    )
    model = make_threat_type_model(random_state)
    model.fit(train_frame[columns], train_frame[THREAT_TARGET_COLUMN])
    predictions = model.predict(test_frame[columns])
    labels = list(model.named_steps["classifier"].classes_)
    report_metrics = classification_report(
        test_frame[THREAT_TARGET_COLUMN], predictions, labels=labels, target_names=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(test_frame[THREAT_TARGET_COLUMN], predictions, labels=labels)
    metrics = {
        "per_class": {
            label: {
                "precision": float(report_metrics[label]["precision"]),
                "recall": float(report_metrics[label]["recall"]),
                "f1": float(report_metrics[label]["f1-score"]),
                "support": int(report_metrics[label]["support"]),
            }
            for label in labels
        },
        "macro_avg": {key: float(report_metrics["macro avg"][key]) for key in ("precision", "recall", "f1-score")},
        "weighted_avg": {key: float(report_metrics["weighted avg"][key]) for key in ("precision", "recall", "f1-score")},
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_labels": labels,
        "test_rows": int(len(test_frame)),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report = {
        "task": "attack-type classification for binary-model malicious records",
        "model": "RandomForestClassifier",
        "random_state": random_state,
        "supported_labels": labels,
        "minimum_class_count": min_class_count,
        "all_attack_label_counts": all_attack_counts,
        "sample_class_distribution": sample_counts,
        "sampling": {
            "per_class_per_file": per_class_per_file,
            "chunk_size": chunk_size,
            "source_directory": str(input_dir),
            "note": "Bounded deterministic stream sample; duplicate rows were removed before splitting.",
        },
        "evaluation": {
            "test_size": 0.25,
            "metrics": metrics,
            "warning": (
                "The stratified holdout is disjoint after duplicate removal, but CICIDS2017 attack "
                "categories are concentrated in particular capture files. Results may not generalize "
                "to unseen traffic days or unsupported rare labels."
            ),
        },
        "unsupported_labels": {label: count for label, count in all_attack_counts.items() if label not in labels},
        "artifacts": {"model": str(model_path)},
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/MachineLearningCSV"))
    parser.add_argument("--model-output", type=Path, default=Path("models/random_forest_threat_type.joblib"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/threat_type_evaluation.json"))
    parser.add_argument("--min-class-count", type=int, default=DEFAULT_MIN_CLASS_COUNT)
    parser.add_argument("--per-class-per-file", type=int, default=DEFAULT_PER_CLASS_PER_FILE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    report = train_and_evaluate(args.input, args.model_output, args.report_output, args.min_class_count, args.per_class_per_file, args.chunk_size, args.random_state)
    print(f"Supported attack labels: {', '.join(report['supported_labels'])}")
    print(f"Held-out rows: {report['evaluation']['metrics']['test_rows']:,}")
    print(f"Macro F1: {report['evaluation']['metrics']['macro_avg']['f1-score']:.4f}")
    print(f"Model: {args.model_output}")


if __name__ == "__main__":
    main()
