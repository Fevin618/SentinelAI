# SentinelAI

SentinelAI is a local, explainable network-traffic classification prototype for the CICIDS2017 MachineLearningCSV dataset. It classifies traffic records as benign or malicious, compares baseline models, explains predictions, and stores local prediction history in SQLite.

## Abstract

The project demonstrates an end-to-end machine-learning workflow for network intrusion analysis under an 8 GB RAM constraint. It uses chunked CSV processing, training-only preprocessing, reproducible experiments, day-aware evaluation, model comparison, SHAP explanations, and a Streamlit interface. It is an academic prototype and is not live network monitoring or production-grade protection.

## Architecture

```text
CICIDS2017 CSV files
        |
        v
Chunked loader -> schema normalization -> infinity/NaN handling
        |
        +--> inspection report and bounded sample
        |
        +--> train-only imputer/scaler -> RF or Logistic Regression pipeline
                                      |
                                      +--> metrics and confusion matrices
                                      +--> SHAP/model explanations
                                      +--> Streamlit upload prediction
                                                  |
                                                  v
                                           SQLite prediction history
```

Important files:

- [`src/data_pipeline.py`](src/data_pipeline.py): streaming inspection and cleaning helpers.
- [`src/train.py`](src/train.py): Random Forest baseline training and evaluation.
- [`src/compare_models.py`](src/compare_models.py): controlled RF/Logistic Regression comparison.
- [`dashboard/app.py`](dashboard/app.py): local Streamlit dashboard.
- [`dashboard/dashboard_utils.py`](dashboard/dashboard_utils.py): upload validation, prediction, SHAP, and SQLite helpers.
- [`src/threat_type.py`](src/threat_type.py): separate, held-out multiclass attack-type training and evaluation.
- [`reports/dataset_inspection.json`](reports/dataset_inspection.json): real dataset inspection.
- [`reports/baseline_evaluation.json`](reports/baseline_evaluation.json): preserved original baseline result.
- [`reports/model_comparison.json`](reports/model_comparison.json): model comparison and Wednesday holdout results.

## Dataset

The project uses the existing local CICIDS2017 files under `data/MachineLearningCSV/`. The inspection found 8 CSV files, 2,830,743 records, 78 traffic features, and a `Label` target. The full dataset is approximately 80.3% benign and 19.7% non-benign. The source files are not modified or committed.

The raw headers contain a duplicated `Fwd Header Length`; the pipeline preserves both columns using deterministic names. `Infinity`, `-Infinity`, `NaN`, empty cells, and invalid numeric values become missing values. Imputation is fitted only inside each model pipeline on training records.

## Methodology

To stay within laptop memory limits, experiments use a deterministic stream sample capped at 3,000 benign and 3,000 malicious rows per source file. This produced 40,160 rows for the reported comparison. The sample is more attack-balanced than the full dataset; this is documented and should not be interpreted as the natural deployment distribution.

The binary target is:

- `0`: `BENIGN`
- `1`: every non-`BENIGN` attack label

The confusion matrix order is `[[TN, FP], [FN, TP]]`. The random split is stratified. The day-aware experiment holds out `Wednesday-workingHours.pcap_ISCX.csv` entirely. Monday contains only benign traffic and cannot measure malicious recall.

## Algorithms

- Random Forest with `class_weight="balanced_subsample"` — the preserved current baseline.
- Unweighted Random Forest — comparison point for class weighting.
- Logistic Regression with class balancing, median imputation, missingness indicators, and standard scaling.

All saved models are complete scikit-learn pipelines. Uploaded CSV data is validated against the exact standardized 78-feature schema before prediction.

## Evaluation results

Random split:

| Model | Accuracy | Precision | Recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|
| Current weighted RF | 0.9961 | 0.9980 | 0.9923 | 0.9952 | 0.0013 |
| Unweighted RF | 0.9958 | 0.9983 | 0.9913 | 0.9948 | 0.0012 |
| Balanced Logistic Regression | 0.9103 | 0.8501 | 0.9433 | 0.8943 | 0.1120 |

Wednesday holdout:

| Model | Accuracy | Precision | Recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|
| Current weighted RF | 0.6000 | 1.0000 | 0.2000 | 0.3333 | 0.0000 |
| Unweighted RF | 0.6102 | 1.0000 | 0.2203 | 0.3611 | 0.0000 |
| Balanced Logistic Regression | 0.6342 | 0.8449 | 0.3287 | 0.4732 | 0.0603 |

The Logistic Regression candidate detects more unseen Wednesday attacks but produces more false positives. The dashboard displays this trade-off rather than optimizing for accuracy alone.

Per-attack recall is shown with support counts. On the Wednesday sample, Logistic Regression recall was 31.19% for DoS GoldenEye with support 2,799, 55.33% for DoS Hulk with support 197, and 100% for Heartbleed with support 4. The Heartbleed result is not conclusive because its support is only four examples.

## Installation

Use the configured local Anaconda Python environment:

```powershell
C:/Users/Admin/Anaconda/python.exe -m pip install -r requirements.txt
```

No paid API, cloud service, or payment details are required.

The saved comparison artifacts were trained under scikit-learn 1.7.2. The current `.venv` may report an `InconsistentVersionWarning` if it uses a newer scikit-learn release; the dashboard applies a narrow compatibility repair for the fitted imputer attribute and does not retrain or rewrite the saved models. Matching the training scikit-learn version is still preferable for a future productionized artifact.

## Run tests

```powershell
C:/Users/Admin/Anaconda/python.exe -m pytest -q tests
```

## Launch the dashboard

From the project root:

```powershell
C:/Users/Admin/Anaconda/python.exe -m streamlit run dashboard/app.py
```

The dashboard provides Overview, Dataset Explorer, Traffic Prediction, Model Evaluation, Explainable AI, and Prediction History sections. It stores only prediction metadata in `data/sentinelai_history.db` and does not store uploaded feature rows.

The Threat Detection & Alerts page preserves the binary BENIGN/MALICIOUS detector and adds the separately trained attack-type model. It preserves original CSV row numbers, excludes the optional `Label` column from inference features, exports potential-threat alerts, and marks unsupported or low-confidence type predictions as uncertain. The multiclass artifact is `models/random_forest_threat_type.joblib`, with held-out metrics in `reports/threat_type_evaluation.json`. Train or refresh it with:

```powershell
C:/Users/Admin/Anaconda/python.exe -m src.threat_type
```

The default training job uses a bounded stream sample, a minimum class count of 100, duplicate removal before splitting, and a 25% stratified holdout. Rare labels are reported as unsupported rather than grouped or guessed.

## Limitations

- The application analyzes uploaded records; it does not monitor a live network.
- A binary model cannot reliably name a previously unseen attack category. Category distributions shown by the dashboard are observed dataset labels, while uploaded predictions are benign/malicious only.
- Random splits can leak related traffic patterns across train and test sets. The Wednesday result demonstrates distribution shift and is the more cautious generalization signal.
- Monday is benign-only, so it cannot measure attack recall.
- SHAP explanations describe model behavior and are not causal proof or a security verdict.
- The reported sample is bounded and class-capped for memory safety, so its distribution differs from the full dataset.
- Heuristic risk labels are estimates; model confidence is not attack probability or security risk.
- The project has not been validated for operational alert volumes, encrypted traffic, adversarial inputs, or real-time deployment.

## Future improvements

Future work could add a leakage-aware group split, a properly validated multiclass model, threshold selection on a validation set, calibrated alert scores, more diverse days, and stronger upload auditing. These should be evaluated before making operational claims.
