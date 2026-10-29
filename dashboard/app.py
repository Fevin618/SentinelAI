"""SentinelAI dashboard: a local, explainable network-traffic classifier."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.dashboard_utils import (
    EXPECTED_FEATURES,
    MODEL_PATHS,
    ROW_NUMBER_COLUMN,
    THREAT_TYPE_UNCERTAIN,
    display_label,
    export_csv,
    load_compatible_model,
    load_json,
    prediction_table,
    read_history,
    read_uploaded_csv,
    save_predictions,
    shap_explanation,
    threat_type_table,
    validate_and_align_features,
)


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "reports/model_comparison.json"
INSPECTION_PATH = ROOT / "reports/dataset_inspection.json"
DB_PATH = ROOT / "data/sentinelai_history.db"
THREAT_MODEL_PATH = ROOT / "models/random_forest_threat_type.joblib"
DEMO_SAMPLE_PATH = ROOT / "data/processed/wednesday_demo_sample.csv"
LOGO_PATH = Path(__file__).resolve().parent / "assets/logo.svg"

NAVY = "#07111f"
PANEL_ALT = "#10243b"
TEXT = "#e8eef7"
MUTED = "#91a4ba"
TEAL = "#2dd4bf"
RED = "#ef4444"
AMBER = "#f59e0b"
GRID = "#20334a"
SEVERITY_COLORS = {"Critical": RED, "High": AMBER, "Medium": "#eab308"}

MODEL_COMPARISON_KEYS = {
    "Logistic Regression (balanced)": "logistic_regression_balanced",
    "Random Forest (current weighted)": "random_forest_current_balanced_subsample",
    "Random Forest (unweighted)": "random_forest_unweighted",
}
DEFAULT_MODEL = "Random Forest (current weighted)"
# Shorter names that fit in the sidebar.
SHORT_MODEL_NAMES = {
    "Random Forest (current weighted)": "Random Forest",
    "Random Forest (unweighted)": "Random Forest (no weights)",
    "Logistic Regression (balanced)": "Logistic Regression",
}

# Plain-language metric names so the numbers are easy to explain.
METRIC_NAMES = {
    "recall": "Attacks caught",
    "false_positive_rate": "False alarms",
    "precision": "Alert precision",
    "accuracy": "Accuracy",
    "f1": "F1 score",
}
METRIC_HELP = {
    "recall": "Share of real attacks the model flagged (recall). Higher is better.",
    "false_positive_rate": "Share of normal traffic wrongly flagged as an attack (false-positive rate). Lower is better.",
    "precision": "When the model raises an alert, how often it is a real attack.",
    "accuracy": "Share of all records classified correctly.",
    "f1": "Balance of precision and recall.",
}
MIN_RECORDS_TO_LEARN = 100  # matches DEFAULT_MIN_CLASS_COUNT in src/threat_type.py
SPLIT_UNSEEN = "Unseen day (Wednesday)"
SPLIT_FAMILIAR = "Familiar traffic (random split)"
SPLIT_KEYS = {SPLIT_UNSEEN: "day_aware_split", SPLIT_FAMILIAR: "random_split"}

# Friendly column names for result tables.
RESULT_COLUMNS = {
    ROW_NUMBER_COLUMN: "Row",
    "ground_truth": "True label",
    "prediction": "Prediction",
    "malicious_model_score": "Attack score",
    "predicted_threat_type": "Likely attack type",
    "threat_type_score": "Type confidence",
    "severity": "Severity",
}

# Filled in by main() so pages can link to each other.
PAGES: dict[str, st.Page] = {}


st.set_page_config(page_title="SentinelAI", page_icon=":material/shield:", layout="wide")


# --- Data loading -----------------------------------------------------------

@st.cache_data
def load_reports() -> tuple[dict, dict]:
    return load_json(INSPECTION_PATH), load_json(COMPARISON_PATH)


@st.cache_resource
def load_model(model_path: str):
    return load_compatible_model(ROOT / model_path)


@st.cache_resource
def load_threat_type_model():
    if not THREAT_MODEL_PATH.exists():
        return None
    return load_compatible_model(THREAT_MODEL_PATH)


def model_for(model_name: str):
    return load_model(str(MODEL_PATHS[model_name]).replace("\\", "/"))


# SHAP on a Random Forest is slow, so explain a small sample and remember results.
SHAP_SAMPLE_ROWS = 50


@st.cache_data(max_entries=50, show_spinner=False)
def explain_records(model_name: str, sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_importance, local = shap_explanation(model_for(model_name), sample, max_rows=len(sample), selected_row=0)
    global_importance["feature"] = global_importance["feature"].map(readable_feature)
    local["feature"] = local["feature"].map(readable_feature)
    return global_importance, local


def readable_feature(name: str) -> str:
    """'fwd_iat_total' -> 'Fwd iat total'; imputer flags become '... (missing)'."""
    missing = name.startswith("missingindicator_")
    text = name.removeprefix("missingindicator_").replace("__2", " (2)").replace("_", " ").capitalize()
    return f"{text} (missing)" if missing else text


# --- Small display helpers ----------------------------------------------------

def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def plot_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "Inter, Segoe UI, sans-serif"},
        margin={"l": 8, "r": 8, "t": 16, "b": 8},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": MUTED}},
        hoverlabel={"bgcolor": PANEL_ALT, "font": {"color": TEXT}},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def donut_chart(normal: int, attack: int, center_text: str) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["Normal", "Attack"],
            values=[normal, attack],
            hole=0.72,
            sort=False,
            marker={"colors": [TEAL, RED], "line": {"color": NAVY, "width": 3}},
            textinfo="percent",
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        )
    )
    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.05},
        annotations=[{"text": center_text, "showarrow": False, "font": {"size": 15, "color": TEXT}}],
    )
    return plot_layout(fig, 280)


def bar_chart(frame: pd.DataFrame, value: str, label: str, color: str = RED, height: int = 300) -> go.Figure:
    """Horizontal bar chart, largest bar on top, with values printed on the bars."""
    fig = px.bar(frame.sort_values(value), x=value, y=label, orientation="h", text=value, color_discrete_sequence=[color])
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
    # Extra room on the right so the value labels are never cut off.
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        xaxis={"showticklabels": False, "showgrid": False, "zeroline": False, "range": [0, frame[value].max() * 1.25]},
        yaxis={"showgrid": False},
    )
    return plot_layout(fig, height)


def readable_day(file_name: str) -> str:
    """'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' -> 'Friday · Afternoon · DDos'."""
    parts = file_name.replace(".pcap_ISCX.csv", "").split("-")
    return " · ".join(part for part in parts if part.lower() != "workinghours")


def results_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pick and rename result columns for display."""
    columns = [column for column in RESULT_COLUMNS if column in predictions.columns]
    table = predictions[columns].rename(columns=RESULT_COLUMNS)
    table["Prediction"] = table["Prediction"].map({"MALICIOUS": "Attack", "BENIGN": "Normal"})
    table["Likely attack type"] = table["Likely attack type"].map(display_label)
    if "True label" in table:
        table["True label"] = table["True label"].map(display_label)
    return table


RESULT_COLUMN_CONFIG = {
    "Attack score": st.column_config.ProgressColumn("Attack score", min_value=0, max_value=1, format="%.2f", help="Model score for 'attack'. Not a calibrated probability."),
    "Type confidence": st.column_config.ProgressColumn("Type confidence", min_value=0, max_value=1, format="%.2f", help="How sure the attack-type model is about the type."),
}


# --- Analysis -----------------------------------------------------------------

def severity_level(binary_score: float, threat_score: float, prediction: str, high_threshold: float, critical_threshold: float) -> str:
    """Transparent severity rules; scores are confidence signals, not probabilities."""
    if prediction != "MALICIOUS":
        return "None"
    if threat_score >= critical_threshold and binary_score >= 0.90:
        return "Critical"
    if threat_score >= high_threshold or binary_score >= 0.80:
        return "High"
    return "Medium"


def run_analysis(csv_file, source_name: str, model_name: str) -> dict:
    """Validate the CSV, run both models, and return everything the pages need."""
    frame, _ = read_uploaded_csv(csv_file)
    features, _ = validate_and_align_features(frame)
    predictions = prediction_table(model_for(model_name), features, frame[ROW_NUMBER_COLUMN])
    malicious_mask = predictions["prediction"].eq("MALICIOUS")

    threat_model = load_threat_type_model()
    if threat_model is None:
        threat_types = pd.Series(THREAT_TYPE_UNCERTAIN, index=predictions.index, dtype="string")
        threat_types.loc[~malicious_mask] = "No threat detected by model."
        threat_scores = pd.Series(np.nan, index=predictions.index, dtype="float64")
    else:
        threat_types, threat_scores = threat_type_table(threat_model, features, malicious_mask, st.session_state["threat_threshold"])
    predictions["predicted_threat_type"] = threat_types.to_numpy()
    predictions["threat_type_score"] = threat_scores.to_numpy()
    predictions["severity"] = [
        severity_level(
            binary_score,
            threat_score if pd.notna(threat_score) else 0.0,
            prediction,
            st.session_state["high_threshold"],
            st.session_state["critical_threshold"],
        )
        for binary_score, threat_score, prediction in zip(
            predictions["malicious_model_score"], predictions["threat_type_score"], predictions["prediction"]
        )
    ]
    if "label" in frame.columns:
        predictions.insert(1, "ground_truth", frame["label"].astype("string").fillna("MISSING").to_numpy())
    return {
        "predictions": predictions,
        "features": features,
        "model": model_name,
        "source": source_name,
        "time": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
    }


# --- Pages ----------------------------------------------------------------------

def overview_page() -> None:
    inspection, comparison = load_reports()
    model_name = st.session_state["selected_model"]
    labels = inspection["label_distribution"]
    attack_counts = {display_label(label): count for label, count in labels.items() if label != "BENIGN"}

    st.title("Security overview")
    st.caption("SentinelAI sorts network-traffic records into normal or attack, names the likely attack type, and explains why.")

    with st.container(horizontal=True):
        st.metric("Traffic records", f"{inspection['total_rows'] / 1e6:.2f} M", border=True, help=f"{inspection['total_rows']:,} records in CICIDS2017.")
        st.metric("Share that are attacks", pct(inspection["class_balance"]["non_benign_fraction"]), border=True)
        st.metric("Attack types", len(attack_counts), border=True)
        st.metric("Features per record", len(EXPECTED_FEATURES), border=True, help="Measurements per network flow, such as packet sizes and timings.")

    st.subheader("How well does it catch attacks?")
    st.caption(f"Model: **{SHORT_MODEL_NAMES[model_name]}** (change it in the sidebar)")
    key = MODEL_COMPARISON_KEYS[model_name]
    familiar = comparison["random_split"]["models"][key]
    unseen = comparison["day_aware_split"]["models"][key]
    for column, (title, note, metrics) in zip(
        st.columns(2, border=True),
        [
            ("Familiar traffic", "Tested on a random 25% of the same days it learned from", familiar),
            ("Unseen day", "Tested on Wednesday, a day it never saw during training", unseen),
        ],
    ):
        with column:
            st.markdown(f"**{title}**")
            st.caption(note)
            with st.container(horizontal=True):
                for metric in ("recall", "false_positive_rate", "accuracy"):
                    st.metric(METRIC_NAMES[metric], pct(metrics[metric]), help=METRIC_HELP[metric])
    st.info(
        f"On familiar traffic the model catches **{pct(familiar['recall'])}** of attacks, but on an unseen day only "
        f"**{pct(unseen['recall'])}**. The unseen-day result is the honest estimate of real-world performance.",
        icon=":material/lightbulb:",
    )

    left, right = st.columns([1, 1.5], border=True)
    with left:
        st.markdown("**Normal vs attack traffic**")
        show_chart(donut_chart(inspection["benign_rows"], inspection["malicious_or_non_benign_rows"], "2.83 M<br>records"))
    with right:
        st.markdown("**Most common attack types**")
        top = pd.DataFrame(list(attack_counts.items()), columns=["Attack type", "Records"]).nlargest(6, "Records")
        show_chart(bar_chart(top, "Records", "Attack type", height=280))
        st.caption(f"Top 6 of {len(attack_counts)} attack types. All of them are in the Dataset explorer.")

    st.page_link(PAGES["analyze"], label="Try it: analyze a traffic sample", icon=":material/arrow_forward:")


def analyze_page() -> None:
    st.title("Analyze traffic")
    st.caption("Run the saved models on traffic records. Each record is marked normal or attack; attacks get a likely type and a severity.")

    with st.container(border=True):
        source = st.segmented_control("Data source", ["Demo sample", "Upload a CSV"], default="Demo sample", key="analysis_source") or "Demo sample"
        uploaded = None
        if source == "Upload a CSV":
            uploaded = st.file_uploader(
                "CICIDS2017-style CSV (max 500 MB)",
                type=["csv"],
                help="Needs the same 78 feature columns as the training data. A Label column is optional and is only used for comparison.",
            )
        else:
            st.caption(f"`{DEMO_SAMPLE_PATH.name}`: real, labelled Wednesday records, including attacks the model never trained on.")
        with st.expander("Alert settings", icon=":material/tune:"):
            st.slider("Attack-type confidence needed", 0.50, 0.95, step=0.05, key="threat_threshold", help="Below this, the attack type is shown as uncertain.")
            st.slider("High severity from", 0.50, 0.95, step=0.05, key="high_threshold")
            st.slider("Critical severity from", 0.50, 0.99, step=0.05, key="critical_threshold")
            st.caption("Scores are model confidence, not calibrated probabilities.")
        run = st.button(
            f"Run analysis with {SHORT_MODEL_NAMES[st.session_state['selected_model']]}",
            type="primary",
            icon=":material/play_arrow:",
            disabled=source == "Upload a CSV" and uploaded is None,
        )

    if run:
        try:
            with st.spinner("Checking the file and running the models…"):
                if source == "Demo sample":
                    csv_file, name = io.BytesIO(DEMO_SAMPLE_PATH.read_bytes()), DEMO_SAMPLE_PATH.name
                else:
                    csv_file, name = uploaded, uploaded.name
                st.session_state["analysis"] = run_analysis(csv_file, name, st.session_state["selected_model"])
        except Exception as error:
            st.error(f"Could not analyze this file: {error}", icon=":material/error:")
            return

    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.info("Choose a data source and press **Run analysis**.", icon=":material/info:")
        return
    show_results(analysis)


def show_results(analysis: dict) -> None:
    predictions = analysis["predictions"]
    total = len(predictions)
    is_attack = predictions["prediction"].eq("MALICIOUS")
    attacks = int(is_attack.sum())

    st.subheader("Results")
    st.caption(f"{analysis['source']} · {SHORT_MODEL_NAMES[analysis['model']]} · {analysis['time']}")
    with st.container(horizontal=True):
        st.metric("Records checked", f"{total:,}", border=True)
        st.metric("Flagged as attack", f"{attacks:,} ({attacks / total:.0%})", border=True)
        st.metric("Critical alerts", f"{int(predictions['severity'].eq('Critical').sum()):,}", border=True)
        if "ground_truth" in predictions:
            agrees = (predictions["ground_truth"] != "BENIGN") == is_attack
            st.metric("Agrees with true label", pct(agrees.mean()), border=True, help="Only available because the file includes a Label column.")

    table = results_table(predictions)
    alerts_tab, summary_tab, all_tab = st.tabs(["Alerts", "Summary", "All records"])
    with alerts_tab:
        if attacks == 0:
            st.success("No attacks detected in this data.", icon=":material/check_circle:")
        else:
            chosen = st.pills("Severity", ["Critical", "High", "Medium"], selection_mode="multi", default=["Critical", "High", "Medium"])
            alerts = table.loc[is_attack.to_numpy() & predictions["severity"].isin(chosen).to_numpy()]
            alert_columns = [c for c in ["Row", "Likely attack type", "Type confidence", "Severity", "True label"] if c in alerts]
            st.dataframe(alerts[alert_columns], hide_index=True, column_config=RESULT_COLUMN_CONFIG, height=320)
            with st.container(horizontal=True):
                st.download_button("Download alerts", export_csv(alerts[alert_columns]), "sentinelai_alerts.csv", "text/csv", icon=":material/download:")
                st.page_link(PAGES["explain"], label="Explain why a record was flagged", icon=":material/psychology:")
    with summary_tab:
        left, right = st.columns([1, 1.5])
        with left:
            show_chart(donut_chart(total - attacks, attacks, f"{total:,}<br>records"))
        with right:
            if attacks:
                types = table.loc[is_attack.to_numpy(), "Likely attack type"].value_counts().rename_axis("Attack type").reset_index(name="Records")
                show_chart(bar_chart(types, "Records", "Attack type", height=280))
    with all_tab:
        st.dataframe(table, hide_index=True, column_config=RESULT_COLUMN_CONFIG, height=360)
        st.download_button("Download all results", export_csv(table), "sentinelai_predictions.csv", "text/csv", icon=":material/download:")

    if st.button("Save results to history", icon=":material/save:"):
        save_predictions(DB_PATH, predictions, analysis["model"])
        st.toast(f"Saved {total:,} results to history.", icon=":material/check:")


def explain_page() -> None:
    st.title("Explain a prediction")
    st.caption("Pick a record to see which traffic features pushed the model toward 'attack' or 'normal' (SHAP values).")
    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.info("Run an analysis first, then come back here to explain any record.", icon=":material/info:")
        st.page_link(PAGES["analyze"], label="Go to Analyze traffic", icon=":material/arrow_forward:")
        return

    predictions, features = analysis["predictions"], analysis["features"]
    is_attack = predictions["prediction"].eq("MALICIOUS").to_numpy()
    show = st.segmented_control("Show records the model called", ["Attacks", "Normal"], default="Attacks" if is_attack.any() else "Normal", key="explain_filter") or "Attacks"
    positions = [i for i in range(len(predictions)) if is_attack[i] == (show == "Attacks")]
    if not positions:
        st.info("No records of this kind. Pick another option above.")
        return
    st.caption(f"{len(positions):,} of {len(predictions):,} records")

    def describe(position: int) -> str:
        row = predictions.iloc[position]
        verdict = display_label(row["predicted_threat_type"]) if row["prediction"] == "MALICIOUS" else "Normal"
        return f"Row {row[ROW_NUMBER_COLUMN]} · {verdict}"

    # A separate key per filter, so switching filters starts at that list's first record.
    position = st.selectbox("Record", positions, format_func=describe, key=f"explain_record_{show}")
    row = predictions.iloc[position]
    with st.container(horizontal=True):
        st.metric("Prediction", "Attack" if row["prediction"] == "MALICIOUS" else "Normal", border=True)
        st.metric("Attack score", f"{row['malicious_model_score']:.2f}", border=True)
        if "ground_truth" in predictions:
            st.metric("True label", display_label(row["ground_truth"]), border=True)

    # Put the chosen record first so it is always inside the small SHAP sample.
    sample = pd.concat([features.iloc[[position]], features.drop(index=features.index[position]).head(SHAP_SAMPLE_ROWS - 1)])
    try:
        with st.spinner("Calculating explanation…"):
            global_importance, local = explain_records(analysis["model"], sample)
    except Exception as error:
        st.error(f"Explanation unavailable: {error}")
        return

    with st.container(border=True):
        st.markdown("**Why the model decided this**")
        local = local.head(10).copy()
        local["Effect"] = np.where(local["shap_value"] > 0, "Toward attack", "Toward normal")
        fig = px.bar(local.sort_values("shap_value"), x="shap_value", y="feature", orientation="h", color="Effect", color_discrete_map={"Toward attack": RED, "Toward normal": TEAL})
        fig.update_layout(xaxis_title="Push on the decision", yaxis_title="", legend_title="")
        show_chart(plot_layout(fig, 360))
        st.caption("Red bars pushed the model toward 'attack', teal bars toward 'normal'. Longer bars mattered more.")
    with st.expander("Which features matter most overall?", icon=":material/bar_chart:"):
        fig = px.bar(global_importance.head(10).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h", color_discrete_sequence=[TEAL])
        fig.update_layout(xaxis_title="Average influence", yaxis_title="")
        show_chart(plot_layout(fig, 340))
        st.caption(f"Averaged over {len(sample):,} records from this analysis.")
    st.caption("SHAP explains the model's behaviour. It is not proof of an attack.")


def performance_page() -> None:
    _, comparison = load_reports()
    st.title("Model performance")
    st.caption("Results from the saved evaluation reports. Nothing is retrained here.")

    split = st.segmented_control("Test", [SPLIT_UNSEEN, SPLIT_FAMILIAR], default=SPLIT_UNSEEN, key="performance_split") or SPLIT_UNSEEN
    st.caption(
        "Wednesday was held out completely, so these attacks are new to the model. This is the realistic test."
        if split == SPLIT_UNSEEN
        else "A random 25% of the sampled records. Similar traffic appears in training, so scores look higher than reality."
    )
    models = comparison[SPLIT_KEYS[split]]["models"]
    names = {value: SHORT_MODEL_NAMES[label] for label, value in MODEL_COMPARISON_KEYS.items()}

    rows = [{"Model": names[key], **{METRIC_NAMES[m]: metrics[m] * 100 for m in METRIC_NAMES}} for key, metrics in models.items()]
    score_columns = {
        name: st.column_config.ProgressColumn(name, min_value=0, max_value=100, format="%.1f%%", help=METRIC_HELP[metric])
        for metric, name in METRIC_NAMES.items()
    }
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config=score_columns)

    left, right = st.columns(2, border=True)
    with left:
        st.markdown("**The trade-off: catching attacks vs false alarms**")
        tradeoff = pd.DataFrame(rows).melt(id_vars="Model", value_vars=["Attacks caught", "False alarms"], var_name="Measure", value_name="Percent")
        fig = px.bar(tradeoff, x="Percent", y="Model", color="Measure", orientation="h", barmode="group", text_auto=".1f", color_discrete_map={"Attacks caught": TEAL, "False alarms": RED})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="", yaxis_title="", xaxis_range=[0, 110], legend_title="", legend={"orientation": "h", "y": -0.15})
        show_chart(plot_layout(fig, 300))
    with right:
        model_name = st.session_state["selected_model"]
        st.markdown(f"**Where {SHORT_MODEL_NAMES[model_name]} gets it right and wrong**")
        # The confusion matrix, written out as four plain counts.
        (tn, fp), (fn, tp) = models[MODEL_COMPARISON_KEYS[model_name]]["confusion_matrix"]
        top_left, top_right = st.columns(2)
        top_left.metric("Attacks caught", f"{tp:,}", border=True, help="Real attacks flagged as attack (true positives).")
        top_right.metric("Missed attacks", f"{fn:,}", border=True, help="Real attacks the model called normal (false negatives).")
        bottom_left, bottom_right = st.columns(2)
        bottom_left.metric("Correctly passed", f"{tn:,}", border=True, help="Normal traffic called normal (true negatives).")
        bottom_right.metric("False alarms", f"{fp:,}", border=True, help="Normal traffic flagged as attack (false positives).")

    with st.expander("Attacks caught per attack type (unseen day)", icon=":material/list:"):
        category_rows = [
            {"Model": names[key], "Attack type": display_label(category), "Records": values["support"], "Caught": values["detected_as_malicious"], "Caught %": values["recall"] * 100}
            for key, metrics in comparison["day_aware_split"]["models"].items()
            for category, values in metrics["per_attack_category"].items()
        ]
        st.dataframe(
            pd.DataFrame(category_rows),
            hide_index=True,
            column_config={"Caught %": st.column_config.ProgressColumn("Caught %", min_value=0, max_value=100, format="%.1f%%")},
        )
        st.caption("Heartbleed has only 4 records, so its result is not conclusive.")


def history_page() -> None:
    st.title("Prediction history")
    st.caption("Results you saved from Analyze traffic. Only the outcome is stored, never the traffic data itself.")
    history = read_history(DB_PATH, 500)
    if history.empty:
        st.info("Nothing saved yet. Run an analysis and press **Save results to history**.", icon=":material/info:")
        st.page_link(PAGES["analyze"], label="Go to Analyze traffic", icon=":material/arrow_forward:")
        return

    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, format="ISO8601")
    history["prediction"] = history["prediction"].map({"MALICIOUS": "Attack", "BENIGN": "Normal"})
    history["attack_category"] = history["attack_category"].map(lambda value: display_label(value) if isinstance(value, str) else "-")
    with st.container(horizontal=True):
        st.metric("Saved results (latest 500)", f"{len(history):,}", border=True)
        st.metric("Flagged as attack", f"{int(history['prediction'].eq('Attack').sum()):,}", border=True)
        st.metric("Last saved", history["timestamp"].max().strftime("%d %b %Y, %H:%M"), border=True)

    chosen = st.pills("Show", ["Attack", "Normal"], selection_mode="multi", default=["Attack", "Normal"])
    st.dataframe(
        history.loc[history["prediction"].isin(chosen)],
        hide_index=True,
        height=460,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Saved at", format="D MMM YYYY, HH:mm"),
            "prediction": "Prediction",
            "attack_category": "Attack type",
            "model_used": "Model",
            "risk_classification": "Risk",
            "malicious_model_score": st.column_config.ProgressColumn("Attack score", min_value=0, max_value=1, format="%.2f"),
        },
    )


def dataset_page() -> None:
    inspection, _ = load_reports()
    st.title("Dataset explorer")
    st.caption("The CICIDS2017 files the models learned from. They are only read, never changed.")

    with st.container(horizontal=True):
        st.metric("Files (one per capture session)", inspection["file_count"], border=True)
        st.metric("Records", f"{inspection['total_rows']:,}", border=True)
        st.metric("Missing or invalid cells", f"{inspection['total_invalid_or_missing_feature_cells']:,}", border=True, help="Includes 'Infinity' values, which are treated as missing.")
        st.metric("Same columns in every file", "Yes" if inspection["schema_compatible"] else "No", border=True)
    st.caption(":material/cleaning_services: 'Infinity' and invalid numbers become missing values, filled in using training data only.")

    st.subheader("What each file contains")
    days = []
    for report in inspection["file_reports"]:
        attacks = {display_label(label): count for label, count in report["labels"].items() if label != "BENIGN"}
        days.append(
            {
                "Capture session": readable_day(report["file"]),
                "Records": report["rows"],
                "Attack records": sum(attacks.values()),
                "Attack types": ", ".join(attacks) or "None (normal traffic only)",
                "Size (MB)": round(report["bytes"] / 1e6, 1),
            }
        )
    st.dataframe(
        pd.DataFrame(days),
        hide_index=True,
        column_config={
            "Records": st.column_config.NumberColumn(format="localized"),
            "Attack records": st.column_config.NumberColumn(format="localized"),
        },
    )

    st.subheader("Attack types")
    st.caption(f"{inspection['malicious_or_non_benign_rows']:,} attack records in total, alongside {inspection['benign_rows']:,} normal ones.")
    attacks = pd.DataFrame(
        [(display_label(label), count) for label, count in inspection["label_distribution"].items() if label != "BENIGN"],
        columns=["Attack type", "Records"],
    ).sort_values("Records", ascending=False)
    attacks["Share of attacks"] = attacks["Records"] / attacks["Records"].sum() * 100
    # Same rule as src/threat_type.py: labels need enough examples to be learned.
    attacks["Attack-type model"] = np.where(attacks["Records"] >= MIN_RECORDS_TO_LEARN, "Can name it", "Too rare to learn")
    st.dataframe(
        attacks,
        hide_index=True,
        height=(len(attacks) + 1) * 35 + 3,  # tall enough to show every row without scrolling
        column_config={
            "Records": st.column_config.NumberColumn(format="localized"),
            "Share of attacks": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Attack-type model": st.column_config.TextColumn(help=f"Types with fewer than {MIN_RECORDS_TO_LEARN} records are too rare to learn, so they are reported as 'uncertain'."),
        },
    )


# --- App shell --------------------------------------------------------------------

def main() -> None:
    defaults = {"selected_model": DEFAULT_MODEL, "threat_threshold": 0.60, "high_threshold": 0.70, "critical_threshold": 0.85}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    PAGES.update(
        overview=st.Page(overview_page, title="Overview", icon=":material/dashboard:", default=True),
        analyze=st.Page(analyze_page, title="Analyze traffic", icon=":material/radar:", url_path="analyze"),
        explain=st.Page(explain_page, title="Explain a prediction", icon=":material/psychology:", url_path="explain"),
        performance=st.Page(performance_page, title="Model performance", icon=":material/monitoring:", url_path="performance"),
        history=st.Page(history_page, title="Prediction history", icon=":material/history:", url_path="history"),
        dataset=st.Page(dataset_page, title="Dataset explorer", icon=":material/database:", url_path="dataset"),
    )
    navigation = st.navigation(
        {
            "": [PAGES["overview"]],
            "Detect": [PAGES["analyze"], PAGES["explain"]],
            "Review": [PAGES["performance"], PAGES["history"]],
            "Data": [PAGES["dataset"]],
        }
    )

    st.logo(str(LOGO_PATH), size="large")
    with st.sidebar:
        st.selectbox(
            "Active model",
            list(MODEL_PATHS),
            key="selected_model",
            format_func=lambda name: SHORT_MODEL_NAMES.get(name, name),
            help="Used when analyzing traffic and shown on the overview.",
        )
        st.caption(":material/lock: Runs locally. No data leaves this computer.")

    navigation.run()


if __name__ == "__main__":
    main()
