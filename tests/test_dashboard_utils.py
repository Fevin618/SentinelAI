import io
import sqlite3

import numpy as np
import pandas as pd
import pytest

from dashboard.dashboard_utils import (
    EXPECTED_FEATURES,
    ROW_NUMBER_COLUMN,
    THREAT_TYPE_UNCERTAIN,
    display_label,
    initialize_history,
    load_compatible_model,
    prediction_table,
    read_history,
    read_uploaded_csv,
    export_csv,
    save_predictions,
    threat_type_table,
    validate_and_align_features,
)


def _csv_bytes() -> bytes:
    frame = pd.DataFrame({column: [1.0] for column in EXPECTED_FEATURES})
    return frame.to_csv(index=False).encode()


def test_uploaded_csv_is_aligned_to_exact_feature_schema():
    frame, columns = read_uploaded_csv(io.BytesIO(_csv_bytes()))
    aligned, expected = validate_and_align_features(frame)
    assert columns == EXPECTED_FEATURES
    assert list(aligned.columns) == expected


def test_ten_row_upload_preserves_original_csv_row_numbers_and_excludes_label():
    frame = pd.DataFrame({column: range(10) for column in EXPECTED_FEATURES})
    frame["Label"] = ["BENIGN"] * 5 + ["DoS Hulk"] * 5
    uploaded, columns = read_uploaded_csv(io.BytesIO(frame.to_csv(index=False).encode()))
    aligned, _ = validate_and_align_features(uploaded)
    assert len(aligned) == 10
    assert uploaded[ROW_NUMBER_COLUMN].tolist() == list(range(2, 12))
    assert "label" not in columns[:-1]
    assert ROW_NUMBER_COLUMN not in aligned.columns


def test_multiclass_predictions_only_label_malicious_rows_and_mark_uncertain():
    class ThreatModel:
        classes_ = np.array(["Bot", "PortScan"])

        def predict(self, features):
            return np.array(["Bot"] * len(features))

        def predict_proba(self, features):
            return np.tile([[0.8, 0.2]], (len(features), 1))

    features = pd.DataFrame({"feature": [1.0, 2.0]})
    malicious = pd.Series([True, False])
    labels, scores = threat_type_table(ThreatModel(), features, malicious, uncertainty_threshold=0.9)
    assert labels.tolist() == [THREAT_TYPE_UNCERTAIN, "No threat detected by model."]
    assert scores.iloc[0] == 0.8


def test_display_label_fixes_broken_web_attack_dash():
    assert display_label("Web Attack � Brute Force") == "Web Attack - Brute Force"
    assert display_label("DDoS") == "DDoS"


def test_csv_export_has_no_dataframe_index():
    exported = export_csv(pd.DataFrame({"row": [2], "alert_status": ["Potential threat"]}))
    assert exported.decode().splitlines() == ["row,alert_status", "2,Potential threat"]


def test_invalid_upload_reports_missing_and_unexpected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        validate_and_align_features(pd.DataFrame({"wrong": [1]}))


def test_prediction_history_round_trip(tmp_path):
    db = tmp_path / "history.db"
    initialize_history(db)
    predictions = prediction_table(type("Model", (), {"predict": lambda self, x: [1] * len(x), "predict_proba": lambda self, x: [[0.1, 0.9]] * len(x)})(), pd.DataFrame({"x": [1]}))
    save_predictions(db, predictions, "test-model")
    history = read_history(db)
    assert len(history) == 1
    assert history.iloc[0]["prediction"] == "MALICIOUS"


def test_prediction_history_saves_attack_type_for_malicious_rows_only(tmp_path):
    db = tmp_path / "history.db"
    predictions = pd.DataFrame(
        {
            "prediction": ["MALICIOUS", "BENIGN"],
            "malicious_model_score": [0.9, 0.1],
            "risk_estimate": ["HIGH", "LOW"],
            "predicted_threat_type": ["DDoS", "No threat detected by model."],
        }
    )
    save_predictions(db, predictions, "test-model")
    history = read_history(db).iloc[::-1].reset_index(drop=True)
    assert history.loc[0, "attack_category"] == "DDoS"
    assert history.loc[1, "attack_category"] is None


def test_saved_model_compatibility_patch_allows_prediction():
    model = load_compatible_model(__import__("pathlib").Path("models/comparison/logistic_regression_balanced.joblib"))
    frame, _ = read_uploaded_csv(io.BytesIO(_csv_bytes()))
    aligned, _ = validate_and_align_features(frame)
    assert len(model.predict(aligned)) == 1
