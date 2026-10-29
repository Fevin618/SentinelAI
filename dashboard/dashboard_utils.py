"""Pure helpers used by the SentinelAI dashboard and its tests."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline import TARGET_COLUMN, standardize_columns


EXPECTED_FEATURES = [
    "destination_port", "flow_duration", "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets", "fwd_packet_length_max",
    "fwd_packet_length_min", "fwd_packet_length_mean", "fwd_packet_length_std",
    "bwd_packet_length_max", "bwd_packet_length_min", "bwd_packet_length_mean",
    "bwd_packet_length_std", "flow_bytes_s", "flow_packets_s", "flow_iat_mean",
    "flow_iat_std", "flow_iat_max", "flow_iat_min", "fwd_iat_total", "fwd_iat_mean",
    "fwd_iat_std", "fwd_iat_max", "fwd_iat_min", "bwd_iat_total", "bwd_iat_mean",
    "bwd_iat_std", "bwd_iat_max", "bwd_iat_min", "fwd_psh_flags", "bwd_psh_flags",
    "fwd_urg_flags", "bwd_urg_flags", "fwd_header_length", "bwd_header_length",
    "fwd_packets_s", "bwd_packets_s", "min_packet_length", "max_packet_length",
    "packet_length_mean", "packet_length_std", "packet_length_variance", "fin_flag_count",
    "syn_flag_count", "rst_flag_count", "psh_flag_count", "ack_flag_count", "urg_flag_count",
    "cwe_flag_count", "ece_flag_count", "down_up_ratio", "average_packet_size",
    "avg_fwd_segment_size", "avg_bwd_segment_size", "fwd_header_length__2",
    "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_avg_bulk_rate", "bwd_avg_bytes_bulk",
    "bwd_avg_packets_bulk", "bwd_avg_bulk_rate", "subflow_fwd_packets", "subflow_fwd_bytes",
    "subflow_bwd_packets", "subflow_bwd_bytes", "init_win_bytes_forward", "init_win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "active_mean", "active_std", "active_max",
    "active_min", "idle_mean", "idle_std", "idle_max", "idle_min",
]
ROW_NUMBER_COLUMN = "original_row_number"
THREAT_TYPE_UNCERTAIN = "Malicious — threat type uncertain"


MODEL_PATHS = {
    "Logistic Regression (balanced)": Path("models/comparison/logistic_regression_balanced.joblib"),
    "Random Forest (current weighted)": Path("models/random_forest_binary.joblib"),
    "Random Forest (unweighted)": Path("models/comparison/random_forest_unweighted.joblib"),
}


def display_label(label: str) -> str:
    """Make CICIDS2017 label names readable.

    The raw files store "Web Attack – Brute Force" with a broken dash that reads
    as "Web Attack � Brute Force"; show a plain dash instead.
    """
    return str(label).replace("�", "-").replace("  ", " ").strip()


def load_compatible_model(path: Path) -> Any:
    """Load a saved pipeline across the minor sklearn versions in this workspace.

    The persisted artifacts were created with sklearn 1.7.2, while the project
    .venv currently provides 1.9.x. The imputer attribute was renamed between
    those versions; restoring the equivalent fitted attribute keeps the saved
    transformation semantics unchanged. This does not retrain or rewrite the
    model artifact.
    """
    import joblib

    model = joblib.load(path)
    for step in getattr(model, "named_steps", {}).values():
        if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
            step._fill_dtype = step._fit_dtype
        if hasattr(step, "n_jobs"):
            # Keep inference compatible with restricted environments that deny
            # joblib worker creation. This changes runtime parallelism only;
            # the saved model artifact and its predictions remain unchanged.
            step.n_jobs = 1
    return model


def read_uploaded_csv(uploaded_file: Any, max_rows: int = 100_000) -> tuple[pd.DataFrame, list[str]]:
    """Read and standardize an uploaded CSV without changing model expectations."""
    stream = uploaded_file
    if not all(hasattr(stream, attribute) for attribute in ("seek", "tell", "readline")):
        raw_bytes = stream.getvalue() if hasattr(stream, "getvalue") else stream.read()
        stream = io.BytesIO(raw_bytes)
    stream.seek(0, 2)
    file_size = stream.tell()
    stream.seek(0)
    if file_size > 500 * 1024 * 1024:
        raise ValueError("Upload is larger than the 500 MB local safety limit")
    raw_header = next(csv.reader([stream.readline().decode("utf-8", errors="replace")]))
    standardized, _ = standardize_columns([str(value) for value in raw_header])
    # CSVs exported by our own preprocessing step already contain the
    # disambiguated name; normalize that spelling back to the training schema.
    standardized = [
        "fwd_header_length__2" if column == "fwd_header_length_2" else column
        for column in standardized
    ]
    stream.seek(0)
    chunks = pd.read_csv(
        stream,
        names=standardized,
        header=0,
        chunksize=max_rows,
        on_bad_lines="error",
        na_values=["", "NaN", "nan", "Infinity", "-Infinity", "inf", "-inf"],
        keep_default_na=True,
    )
    frame = next(chunks, pd.DataFrame(columns=standardized))
    frame = frame.replace([np.inf, -np.inf], np.nan)
    for column in standardized:
        if column != TARGET_COLUMN:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if TARGET_COLUMN in frame.columns:
        frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("string").str.strip()
    frame.insert(0, ROW_NUMBER_COLUMN, np.arange(2, len(frame) + 2, dtype="int64"))
    return frame, standardized


def validate_and_align_features(frame: pd.DataFrame, expected: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Validate an upload and return exactly the feature columns expected by models."""
    expected = expected or EXPECTED_FEATURES
    actual = [column for column in frame.columns if column not in {TARGET_COLUMN, ROW_NUMBER_COLUMN}]
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing columns: {missing}")
        if unexpected:
            problems.append(f"unexpected columns: {unexpected}")
        raise ValueError("CSV schema does not match the trained pipeline (" + "; ".join(problems) + ")")
    aligned = frame.loc[:, expected].copy()
    if aligned.empty:
        raise ValueError("CSV contains no data rows")
    return aligned, expected


def prediction_table(model: Any, features: pd.DataFrame, row_numbers: pd.Series | None = None) -> pd.DataFrame:
    """Predict with a saved complete pipeline and add transparent risk estimates."""
    predictions = np.asarray(model.predict(features)).astype(int)
    if hasattr(model, "predict_proba"):
        malicious_score = np.asarray(model.predict_proba(features))[:, 1]
    else:
        malicious_score = np.full(len(features), np.nan)
    result = pd.DataFrame(
        {
            "prediction": np.where(predictions == 1, "MALICIOUS", "BENIGN"),
            "malicious_model_score": malicious_score,
        }
    )
    result["risk_estimate"] = np.where(result["prediction"] == "MALICIOUS", "HIGH", "LOW")
    result["risk_note"] = "Heuristic estimate; model score is not attack probability or security risk."
    if row_numbers is not None:
        result.insert(0, ROW_NUMBER_COLUMN, row_numbers.to_numpy())
    return result


def threat_type_table(
    model: Any,
    features: pd.DataFrame,
    malicious_mask: pd.Series,
    uncertainty_threshold: float = 0.60,
) -> tuple[pd.Series, pd.Series]:
    """Predict supported threat types only for binary-model malicious rows."""
    malicious_mask = pd.Series(malicious_mask, index=features.index).fillna(False).astype(bool)
    threat_types = pd.Series("", index=features.index, dtype="string")
    threat_scores = pd.Series(np.nan, index=features.index, dtype="float64")
    threat_types.loc[~malicious_mask] = "No threat detected by model."
    if not malicious_mask.any():
        return threat_types, threat_scores
    predictions = np.asarray(model.predict(features.loc[malicious_mask])).astype(str)
    probabilities = np.asarray(model.predict_proba(features.loc[malicious_mask]))
    scores = probabilities.max(axis=1)
    supported = scores >= uncertainty_threshold
    threat_types.loc[malicious_mask] = np.where(supported, predictions, THREAT_TYPE_UNCERTAIN)
    threat_scores.loc[malicious_mask] = scores
    return threat_types, threat_scores


def export_csv(frame: pd.DataFrame) -> bytes:
    """Serialize a dashboard table for a CSV download without an index column."""
    return frame.to_csv(index=False).encode("utf-8")


def shap_explanation(
    model: Any,
    features: pd.DataFrame,
    max_rows: int = 200,
    selected_row: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return bounded global and selected-record SHAP importance tables."""
    import shap

    sample = features.head(max_rows)
    selected_position = min(max(selected_row, 0), len(sample) - 1)
    imputer = model.named_steps["imputer"]
    transformed = imputer.transform(sample)
    if "scaler" in model.named_steps:
        transformed_for_explainer = model.named_steps["scaler"].transform(transformed)
    else:
        transformed_for_explainer = transformed
    classifier = model.named_steps["classifier"]
    names = list(imputer.get_feature_names_out(sample.columns))
    if classifier.__class__.__name__ == "RandomForestClassifier":
        explainer = shap.TreeExplainer(classifier)
    else:
        explainer = shap.LinearExplainer(classifier, transformed_for_explainer)
    values = explainer.shap_values(transformed_for_explainer)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    global_importance = pd.DataFrame(
        {"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False).head(20)
    local = pd.DataFrame({"feature": names, "shap_value": values[selected_position]}).sort_values(
        "shap_value", key=lambda series: series.abs(), ascending=False
    ).head(20)
    return global_importance, local


def initialize_history(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction TEXT NOT NULL,
                attack_category TEXT,
                model_used TEXT NOT NULL,
                risk_classification TEXT NOT NULL,
                malicious_model_score REAL
            )"""
        )


def save_predictions(db_path: Path, predictions: pd.DataFrame, model_name: str) -> None:
    initialize_history(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    has_threat_type = "predicted_threat_type" in predictions.columns
    rows = [
        (
            timestamp,
            row.prediction,
            # Attack type is only meaningful for records flagged as malicious.
            row.predicted_threat_type if has_threat_type and row.prediction == "MALICIOUS" else None,
            model_name,
            row.risk_estimate,
            None if pd.isna(row.malicious_model_score) else float(row.malicious_model_score),
        )
        for row in predictions.itertuples()
    ]
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            "INSERT INTO predictions(timestamp,prediction,attack_category,model_used,risk_classification,malicious_model_score) VALUES (?,?,?,?,?,?)",
            rows,
        )


def read_history(db_path: Path, limit: int = 500) -> pd.DataFrame:
    initialize_history(db_path)
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT timestamp,prediction,attack_category,model_used,risk_classification,malicious_model_score FROM predictions ORDER BY id DESC LIMIT ?",
            connection,
            params=(limit,),
        )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
