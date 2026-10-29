"""Controlled comparison of binary SentinelAI baseline models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .train import (
    DAY_HOLDOUT_FILE,
    RANDOM_STATE,
    calculate_metrics,
    class_distribution,
    feature_columns,
    fit_and_evaluate,
    load_balanced_sample,
    make_logistic_regression,
    make_random_forest,
)


def attack_category_metrics(frame: pd.DataFrame, predictions: pd.Series) -> dict[str, dict[str, Any]]:
    """Report binary malicious recall separately for every attack label."""
    result: dict[str, dict[str, Any]] = {}
    for label, group in frame.loc[frame["label"] != "BENIGN"].groupby("label", dropna=False):
        result[str(label)] = {
            "support": int(len(group)),
            "detected_as_malicious": int(predictions.loc[group.index].sum()),
            "recall": float(predictions.loc[group.index].mean()) if len(group) else None,
        }
    return result


def evaluate_model(
    name: str,
    factory: Callable[[int], Pipeline],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    random_state: int,
) -> tuple[Pipeline, dict[str, Any]]:
    model, metrics = fit_and_evaluate(train_frame, test_frame, random_state, factory(random_state))
    return model, {"model": name, "metrics": metrics}


def compare_models(
    input_dir: Path,
    report_path: Path,
    models_dir: Path,
    per_class_per_file: int = 3_000,
    chunk_size: int = 20_000,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    sample = load_balanced_sample(input_dir, per_class_per_file, chunk_size, random_state)
    random_train, random_test = train_test_split(
        sample,
        test_size=0.25,
        random_state=random_state,
        stratify=sample["binary_target"],
    )
    factories: dict[str, Callable[[int], Pipeline]] = {
        "random_forest_current_balanced_subsample": lambda seed: make_random_forest(seed, "balanced_subsample"),
        "random_forest_unweighted": lambda seed: make_random_forest(seed, None),
        "logistic_regression_balanced": make_logistic_regression,
    }
    random_results: dict[str, Any] = {}
    day_results: dict[str, Any] = {}
    models_dir.mkdir(parents=True, exist_ok=True)
    day_test = sample.loc[sample["source_file"] == DAY_HOLDOUT_FILE]
    day_train = sample.loc[sample["source_file"] != DAY_HOLDOUT_FILE]

    for name, factory in factories.items():
        model, result = evaluate_model(name, factory, random_train, random_test, random_state)
        random_results[name] = result["metrics"]
        joblib.dump(model, models_dir / f"{name}.joblib")

        day_model, day_result = evaluate_model(name, factory, day_train, day_test, random_state)
        day_predictions = pd.Series(day_model.predict(day_test[feature_columns(day_test)]), index=day_test.index)
        day_result["metrics"]["per_attack_category"] = attack_category_metrics(day_test, day_predictions)
        day_results[name] = day_result["metrics"]

    report = {
        "task": "controlled binary model comparison",
        "label_encoding": {"0": "BENIGN", "1": "MALICIOUS", "matrix_order": "[[TN, FP], [FN, TP]]"},
        "sample_rows": int(len(sample)),
        "sample_class_distribution": class_distribution(sample["binary_target"]),
        "random_split": {
            "test_size": 0.25,
            "train_class_distribution": class_distribution(random_train["binary_target"]),
            "test_class_distribution": class_distribution(random_test["binary_target"]),
            "models": random_results,
        },
        "day_aware_split": {
            "holdout_file": DAY_HOLDOUT_FILE,
            "train_class_distribution": class_distribution(day_train["binary_target"]),
            "test_class_distribution": class_distribution(day_test["binary_target"]),
            "models": day_results,
            "warning": "The test sets were not used for model selection. Monday is benign-only and cannot measure malicious recall.",
        },
        "interpretation": {
            "selection_policy": "No model was selected using the reported test results; all models were evaluated on the same untouched splits.",
            "tradeoff": "Higher malicious recall generally requires accepting more false positives. In an IDS, this tradeoff should be chosen with an operational threshold and validation data, not by accuracy alone.",
            "leakage_checks": [
                "Each model is a Pipeline, so imputation/scaling are fitted inside model.fit on training rows only.",
                "Random split is stratified but can contain related traffic patterns across sets.",
                "Wednesday holdout separates an entire source day from training to probe distribution shift.",
            ],
        },
        "artifacts_directory": str(models_dir),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/MachineLearningCSV"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/model_comparison.json"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/comparison"))
    parser.add_argument("--per-class-per-file", type=int, default=3_000)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    report = compare_models(args.input, args.report_output, args.models_dir, args.per_class_per_file, args.chunk_size, args.random_state)
    print(f"Compared 3 models on {report['sample_rows']:,} sampled rows.")
    for name, result in report["day_aware_split"]["models"].items():
        print(f"{name}: day-aware recall={result['recall']:.4f}, F1={result['f1']:.4f}, FPR={result['false_positive_rate']:.4f}")
    print(f"Report: {args.report_output}")


if __name__ == "__main__":
    main()
