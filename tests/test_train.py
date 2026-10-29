from pathlib import Path

import numpy as np
import pandas as pd

import pytest

from src.train import calculate_metrics, fit_and_evaluate, load_balanced_sample
from src.threat_type import supported_attack_labels


def test_rare_attack_labels_are_excluded_from_supported_multiclass_set():
    labels = supported_attack_labels({"BENIGN": 1000, "Bot": 100, "Heartbleed": 11, "Infiltration": 36}, min_class_count=100)
    assert labels == {"Bot"}


def _small_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for target in [0, 1]:
        for i in range(12):
            rows.append(
                {
                    "feature_a": float(target * 10 + i),
                    "feature_b": np.nan if i == 0 else float(i),
                    "label": "BENIGN" if target == 0 else "Attack",
                    "binary_target": target,
                    "source_file": "synthetic.csv",
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)


def test_metrics_include_confusion_matrix_and_false_positive_rate():
    metrics = calculate_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["false_positive_rate"] == 0.5


def test_fit_and_evaluate_handles_nan_without_training_leakage():
    frame = _small_frame()
    train = pd.concat(
        [frame.loc[frame["binary_target"] == 0].head(9), frame.loc[frame["binary_target"] == 1].head(9)]
    )
    test = frame.drop(train.index)
    model, metrics = fit_and_evaluate(train, test)
    assert metrics["feature_count"] == 2
    assert metrics["train_class_distribution"] == {"0": 9, "1": 9}
    assert metrics["test_class_distribution"] == {"0": 3, "1": 3}
    assert hasattr(model, "predict")


def test_stream_sampler_uses_files_and_is_bounded(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "Feature": list(range(10)),
            "Label": ["BENIGN"] * 7 + ["Attack"] * 3,
        }
    ).to_csv(raw / "one.csv", index=False)
    sampled = load_balanced_sample(raw, per_class_per_file=2, chunk_size=3)
    assert len(sampled) == 4
    assert sampled["binary_target"].value_counts().to_dict() == {0: 2, 1: 2}


def test_feature_alignment_is_checked():
    frame = _small_frame()
    with pytest.raises(ValueError, match="not aligned"):
        fit_and_evaluate(frame, frame.rename(columns={"feature_a": "different_feature"}))
