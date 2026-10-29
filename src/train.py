"""Train and evaluate a resource-conscious CICIDS2017 binary baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_pipeline import DEFAULT_CHUNK_SIZE, TARGET_COLUMN, discover_csv_files, iter_clean_chunks


RANDOM_STATE = 42
DEFAULT_PER_CLASS_PER_FILE = 3_000
DAY_HOLDOUT_FILE = "Wednesday-workingHours.pcap_ISCX.csv"


def _bounded_class_sample(
    current: pd.DataFrame,
    incoming: pd.DataFrame,
    limit: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Keep a deterministic bounded sample without retaining all rows."""
    if incoming.empty:
        return current
    combined = pd.concat([current, incoming], ignore_index=True)
    if len(combined) <= limit:
        return combined
    indices = rng.choice(len(combined), size=limit, replace=False)
    return combined.iloc[np.sort(indices)].reset_index(drop=True)


def load_balanced_sample(
    input_dir: Path,
    per_class_per_file: int = DEFAULT_PER_CLASS_PER_FILE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Stream real CSVs and retain at most N rows per binary class per file."""
    rng = np.random.default_rng(random_state)
    rows: list[pd.DataFrame] = []
    for path in discover_csv_files(input_dir):
        sampled = {0: pd.DataFrame(), 1: pd.DataFrame()}
        for chunk in iter_clean_chunks(path, chunk_size):
            chunk = chunk.dropna(subset=[TARGET_COLUMN]).drop_duplicates()
            if chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["binary_target"] = (chunk[TARGET_COLUMN] != "BENIGN").astype("int8")
            chunk["source_file"] = path.name
            for target in (0, 1):
                part = chunk.loc[chunk["binary_target"] == target]
                sampled[target] = _bounded_class_sample(sampled[target], part, per_class_per_file, rng)
        rows.extend(part for part in sampled.values() if not part.empty)
    if not rows:
        raise ValueError("No usable labelled records were found in the dataset")
    sample = pd.concat(rows, ignore_index=True)
    return sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model features, excluding labels and provenance columns."""
    excluded = {TARGET_COLUMN, "binary_target", "source_file"}
    return [column for column in frame.columns if column not in excluded]


def make_random_forest(
    random_state: int = RANDOM_STATE,
    class_weight: str | dict[int, float] | None = "balanced_subsample",
) -> Pipeline:
    """Create a leakage-safe preprocessing/model pipeline."""
    return Pipeline(
        steps=[
            # Median values and missingness indicators are learned from training
            # rows only when Pipeline.fit is called.
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight=class_weight,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_logistic_regression(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create a scaled, class-weighted logistic regression baseline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=300,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def calculate_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Calculate required binary metrics, including false-positive rate."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "class_labels": {"0": "BENIGN", "1": "MALICIOUS"},
        "positive_class": {"value": 1, "label": "MALICIOUS"},
    }


def class_distribution(values: pd.Series | np.ndarray) -> dict[str, int]:
    counts = pd.Series(values).value_counts().to_dict()
    return {str(key): int(counts.get(key, 0)) for key in [0, 1]}


def fit_and_evaluate(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    model: Pipeline | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit only on train_frame and evaluate on test_frame."""
    columns = feature_columns(train_frame)
    if columns != feature_columns(test_frame):
        raise ValueError("Training and test feature columns are not aligned")
    model = model or make_random_forest(random_state)
    model.fit(train_frame[columns], train_frame["binary_target"])
    predictions = model.predict(test_frame[columns])
    metrics = calculate_metrics(test_frame["binary_target"], predictions)
    metrics["train_class_distribution"] = class_distribution(train_frame["binary_target"])
    metrics["test_class_distribution"] = class_distribution(test_frame["binary_target"])
    metrics["feature_count"] = len(columns)
    return model, metrics


def train_baseline(
    input_dir: Path,
    model_path: Path,
    report_path: Path,
    per_class_per_file: int = DEFAULT_PER_CLASS_PER_FILE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train the random-split baseline and a Wednesday day-aware evaluation."""
    sample = load_balanced_sample(input_dir, per_class_per_file, chunk_size, random_state)
    train_frame, test_frame = train_test_split(
        sample,
        test_size=0.25,
        random_state=random_state,
        stratify=sample["binary_target"],
    )
    model, random_metrics = fit_and_evaluate(train_frame, test_frame, random_state)

    holdout = sample.loc[sample["source_file"] == DAY_HOLDOUT_FILE]
    day_train = sample.loc[sample["source_file"] != DAY_HOLDOUT_FILE]
    if holdout.empty or holdout["binary_target"].nunique() < 2 or day_train["binary_target"].nunique() < 2:
        day_metrics: dict[str, Any] = {"status": "not_evaluable", "reason": "holdout lacks both classes"}
    else:
        _, day_metrics = fit_and_evaluate(day_train, holdout, random_state)
        day_metrics["status"] = "evaluated"
        day_metrics["holdout_file"] = DAY_HOLDOUT_FILE

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report = {
        "task": "binary benign versus malicious traffic classification",
        "model": "RandomForestClassifier",
        "random_state": random_state,
        "sample_rows": int(len(sample)),
        "sample_class_distribution": class_distribution(sample["binary_target"]),
        "sampling": {
            "per_class_per_file": per_class_per_file,
            "chunk_size": chunk_size,
            "source_directory": str(input_dir),
            "note": "Bounded deterministic stream sample from the actual CICIDS2017 CSVs.",
        },
        "random_split": {
            "test_size": 0.25,
            "metrics": random_metrics,
        },
        "day_aware_evaluation": {
            "metrics": day_metrics,
            "warning": (
                "This is a held-out Wednesday traffic-day experiment, not proof of universal "
                "generalization. Monday has no malicious rows and cannot measure attack recall."
            ),
        },
        "artifacts": {"model": str(model_path)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/MachineLearningCSV"))
    parser.add_argument("--model-output", type=Path, default=Path("models/random_forest_binary.joblib"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/baseline_evaluation.json"))
    parser.add_argument("--per-class-per-file", type=int, default=DEFAULT_PER_CLASS_PER_FILE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    report = train_baseline(
        args.input,
        args.model_output,
        args.report_output,
        args.per_class_per_file,
        args.chunk_size,
        args.random_state,
    )
    metrics = report["random_split"]["metrics"]
    print(f"Trained on {report['sample_rows']:,} sampled rows.")
    print(f"Random-split metrics: accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, FPR={metrics['false_positive_rate']:.4f}")
    print(f"Model: {args.model_output}")
    print(f"Report: {args.report_output}")


if __name__ == "__main__":
    main()
