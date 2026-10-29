from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "model_comparison.json"
OUTPUT_PATH = ROOT / "output" / "pdf" / "sentinelai_viva_questions_answers.pdf"

NAVY = colors.HexColor("#102A43")
TEAL = colors.HexColor("#0F766E")
BLUE = colors.HexColor("#1D4ED8")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#52606D")
PALE_BLUE = colors.HexColor("#EAF2FF")
PALE_TEAL = colors.HexColor("#E7F7F3")
PALE_AMBER = colors.HexColor("#FFF7E6")
PALE_RED = colors.HexColor("#FFF0F0")
GRID = colors.HexColor("#CBD5E1")


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def q_and_a(question: str, answer: str, styles: dict[str, ParagraphStyle]) -> list[object]:
    return [
        p(f"<b>Q:</b> {escape(question)}", styles["question"]),
        p(f"<b>A:</b> {answer}", styles["answer"]),
        Spacer(1, 3 * mm),
    ]


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=25, leading=29, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom", parent=base["Normal"], fontName="Helvetica",
            fontSize=11, leading=15, textColor=MUTED, spaceAfter=6 * mm,
        ),
        "h1": ParagraphStyle(
            "H1Custom", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=17, leading=21, textColor=NAVY, spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
        "h2": ParagraphStyle(
            "H2Custom", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=TEAL, spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.3, leading=13.2, textColor=INK, spaceAfter=2 * mm,
        ),
        "small": ParagraphStyle(
            "SmallCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.8, leading=10.5, textColor=MUTED,
        ),
        "question": ParagraphStyle(
            "QuestionCustom", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=10.2, leading=13.5, textColor=NAVY, spaceBefore=1 * mm,
            spaceAfter=1 * mm,
        ),
        "answer": ParagraphStyle(
            "AnswerCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.2, leading=13, textColor=INK, leftIndent=5 * mm,
            spaceAfter=1 * mm,
        ),
        "callout": ParagraphStyle(
            "CalloutCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10, leading=14, textColor=NAVY,
        ),
        "metric": ParagraphStyle(
            "MetricCustom", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=15, leading=18, textColor=TEAL, alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabelCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.5, leading=9.5, textColor=MUTED, alignment=TA_CENTER,
        ),
        "mono": ParagraphStyle(
            "MonoCustom", parent=base["Code"], fontName="Courier",
            fontSize=8.6, leading=12, textColor=INK,
        ),
    }


def box(content: object, background: colors.Color, border: colors.Color = GRID) -> Table:
    table = Table([[content]], colWidths=[166 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.7, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
    ]))
    return table


def styled_table(data: list[list[object]], widths: list[float], header_rows: int = 1,
                 background: colors.Color = colors.white, font_size: float = 8.2) -> Table:
    table = Table(data, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
        ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2 * mm),
    ]
    for row in range(header_rows, len(data)):
        commands.append(("BACKGROUND", (0, row), (-1, row), background if row % 2 else colors.white))
    table.setStyle(TableStyle(commands))
    return table


def on_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(1.2)
    canvas.line(doc.leftMargin, height - 15 * mm, width - doc.rightMargin, height - 15 * mm)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(TEAL)
    canvas.drawString(doc.leftMargin, height - 11.5 * mm, "SENTINELAI | VIVA PREPARATION")
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - doc.rightMargin, 10 * mm, f"Page {doc.page}")
    canvas.drawString(doc.leftMargin, 10 * mm, "Based on the project's model comparison report")
    canvas.restoreState()


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    styles = make_styles()

    sample_rows = report["sample_rows"]
    class_dist = report["sample_class_distribution"]
    random_split = report["random_split"]
    random_models = random_split["models"]
    rf = random_models["random_forest_current_balanced_subsample"]
    unweighted_rf = random_models["random_forest_unweighted"]
    logistic = random_models["logistic_regression_balanced"]
    day_models = report["day_aware_split"]["models"]
    day_rf = day_models["random_forest_current_balanced_subsample"]
    day_logistic = day_models["logistic_regression_balanced"]

    test_rows = sum(random_split["test_class_distribution"].values())
    train_rows = sum(random_split["train_class_distribution"].values())
    tn, fp, fn, tp = logistic["confusion_matrix"][0][0], logistic["confusion_matrix"][0][1], logistic["confusion_matrix"][1][0], logistic["confusion_matrix"][1][1]

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        rightMargin=22 * mm, leftMargin=22 * mm,
        topMargin=23 * mm, bottomMargin=17 * mm,
        title="SentinelAI Viva Questions and Answers",
        author="SentinelAI",
    )

    story: list[object] = []

    # Cover and quick facts.
    story.append(Spacer(1, 4 * mm))
    story.append(p("SentinelAI", styles["title"]))
    story.append(p("Viva Questions and Simple Answers", styles["subtitle"]))
    story.append(box(p(
        "<b>One sentence to remember:</b> SentinelAI is a binary network-intrusion classifier. "
        "It takes 78 numeric network-flow features and predicts whether a flow is BENIGN or MALICIOUS.",
        styles["callout"],
    ), PALE_TEAL, TEAL))
    story.append(Spacer(1, 6 * mm))
    story.append(p("Key facts from the report", styles["h1"]))
    metric_data = [[
        p(str(sample_rows), styles["metric"]),
        p(str(test_rows), styles["metric"]),
        p(fmt(logistic["accuracy"], 4), styles["metric"]),
        p(fmt(rf["accuracy"], 4), styles["metric"]),
    ], [
        p("sample rows", styles["metric_label"]),
        p("random test rows", styles["metric_label"]),
        p("logistic accuracy", styles["metric_label"]),
        p("weighted RF accuracy", styles["metric_label"]),
    ]]
    metrics_table = Table(metric_data, colWidths=[41.5 * mm] * 4)
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 5 * mm))

    class_table = styled_table([
        ["Binary value", "Meaning", "Rows in sampled data"],
        ["0", "BENIGN / normal traffic", str(class_dist["0"])],
        ["1", "MALICIOUS / any non-BENIGN attack", str(class_dist["1"])],
    ], [35 * mm, 86 * mm, 45 * mm], background=PALE_TEAL)
    story.append(class_table)
    story.append(Spacer(1, 5 * mm))
    story.append(p("30-second answer", styles["h1"]))
    story.append(box(p(
        "The project reads CICIDS2017 flow records. Each row is one network flow and each of the 78 input columns is a numeric traffic feature. "
        "The Label column is kept as the answer: BENIGN becomes 0, and every other attack label becomes 1. "
        f"The report compares models on a {sample_rows:,}-row sample and evaluates a 25% test split. "
        f"The {fmt(logistic['accuracy'], 4)} result belongs to balanced Logistic Regression; the current weighted Random Forest reports {fmt(rf['accuracy'], 4)} on the same random split.",
        styles["callout"],
    ), PALE_AMBER, colors.HexColor("#D97706")))

    story.append(PageBreak())

    # Dataset and labeling questions.
    story.append(p("Part 1 - Dataset and labels", styles["h1"]))
    story.extend(q_and_a(
        "What does one row represent?",
        "One row represents a bidirectional network flow summarized from packets. It is not necessarily one packet. The model uses the row as one example to classify.",
        styles,
    ))
    story.extend(q_and_a(
        "What do the columns represent?",
        "There are 78 input features, such as destination port, flow duration, packet counts, packet lengths, rates, TCP flags, and active or idle times. The final Label column is the target and is not used as an input feature.",
        styles,
    ))
    story.extend(q_and_a(
        "How are the labels encoded in this project?",
        "The code creates <font name='Courier'>binary_target = 0</font> when Label is BENIGN and <font name='Courier'>binary_target = 1</font> for every other label. Therefore DDoS is treated as MALICIOUS in the binary task.",
        styles,
    ))
    story.extend(q_and_a(
        "Why can a file with DDoS in its name contain BENIGN rows?",
        "The filename identifies a capture session, not the label of every row. The capture contains normal background traffic as well as the attack traffic. The row-level Label is the value used for training.",
        styles,
    ))
    story.extend(q_and_a(
        "How does the program read the raw CSV?",
        "It treats commas as separators, reads line 1 as column names, converts feature values to numbers, keeps the label as text, and disambiguates the duplicated Fwd Header Length column. The model never reads the editor screenshot; it receives a numeric table.",
        styles,
    ))
    story.append(box(p(
        "<b>Important:</b> If asked whether Label is a feature, answer: No. Label is the target. Passing Label into the model would leak the correct answer and make the evaluation invalid.",
        styles["callout"],
    ), PALE_RED, colors.HexColor("#DC2626")))

    story.append(Spacer(1, 5 * mm))
    story.append(p("How a row becomes a model input", styles["h2"]))
    flow_table = styled_table([
        ["CSV content", "Project meaning"],
        ["54865, 3, 2, 0, 12, ... , BENIGN", "One row with 78 feature values and one target label"],
        ["X = [54865, 3, 2, 0, 12, ...]", "Features sent to the model"],
        ["y = BENIGN -> 0", "Expected answer used for training or evaluation"],
    ], [72 * mm, 94 * mm], background=PALE_BLUE)
    story.append(flow_table)

    story.append(PageBreak())

    # Training and metrics.
    story.append(p("Part 2 - Training and accuracy", styles["h1"]))
    story.extend(q_and_a(
        "How many rows were used for the reported comparison?",
        f"The report used a deterministic sample of {sample_rows:,} rows: {class_dist['0']:,} BENIGN and {class_dist['1']:,} MALICIOUS. The sample is not the entire raw dataset.",
        styles,
    ))
    story.extend(q_and_a(
        "How was the data split?",
        f"The random split uses 75% for training ({train_rows:,} rows) and 25% for testing ({test_rows:,} rows). It is stratified, so both classes are represented in both parts. The random state is 42, so the split is repeatable.",
        styles,
    ))
    story.extend(q_and_a(
        "Which models were compared?",
        "The report compares a weighted Random Forest, an unweighted Random Forest, and balanced Logistic Regression. The weighted model uses class weighting to reduce the effect of class imbalance.",
        styles,
    ))
    story.extend(q_and_a(
        "What does accuracy mean?",
        "Accuracy is the fraction of test rows classified correctly: (true positives + true negatives) divided by all test rows. It is a batch metric, not a property of a single row or a single CSV file.",
        styles,
    ))

    model_table = styled_table([
        ["Model", "Accuracy", "Precision", "Recall", "F1", "FPR"],
        ["Weighted Random Forest", fmt(rf["accuracy"]), fmt(rf["precision"]), fmt(rf["recall"]), fmt(rf["f1"]), fmt(rf["false_positive_rate"])],
        ["Unweighted Random Forest", fmt(unweighted_rf["accuracy"]), fmt(unweighted_rf["precision"]), fmt(unweighted_rf["recall"]), fmt(unweighted_rf["f1"]), fmt(unweighted_rf["false_positive_rate"])],
        ["Balanced Logistic Regression", fmt(logistic["accuracy"]), fmt(logistic["precision"]), fmt(logistic["recall"]), fmt(logistic["f1"]), fmt(logistic["false_positive_rate"])],
    ], [57 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 21 * mm], background=PALE_TEAL, font_size=7.8)
    story.append(model_table)
    story.append(Spacer(1, 3 * mm))
    story.append(p(
        "The value 0.9103 belongs to Balanced Logistic Regression. The current weighted Random Forest has 0.9961 on this same random split. Do not say that 0.9103 is the accuracy of the Friday DDoS CSV by itself.",
        styles["small"],
    ))

    story.append(Spacer(1, 4 * mm))
    story.append(p("How the 0.9103 value is calculated", styles["h2"]))
    story.append(p(
        f"For Logistic Regression, the test set contains {test_rows:,} rows. The report's confusion matrix is shown on the next page. "
        f"The correct predictions are {tn:,} normal rows plus {tp:,} malicious rows, for {tn + tp:,} correct predictions.",
        styles["body"],
    ))
    story.append(box(p(
        f"Accuracy = ({tn:,} + {tp:,}) / {test_rows:,} = {(tn + tp) / test_rows:.4f} = {logistic['accuracy']:.4f}",
        styles["callout"],
    ), PALE_BLUE, BLUE))

    story.append(PageBreak())

    # Confusion matrix.
    story.append(p("Part 3 - Confusion matrix", styles["h1"]))
    story.append(p(
        "Always read the table using both directions: the rows are the actual labels from the dataset, and the columns are the model's predictions.",
        styles["body"],
    ))
    cm_data = [
        ["Actual / Predicted", "BENIGN", "MALICIOUS"],
        ["BENIGN", str(tn), str(fp)],
        ["MALICIOUS", str(fn), str(tp)],
    ]
    cm_table = Table(cm_data, colWidths=[60 * mm, 50 * mm, 50 * mm], rowHeights=[13 * mm, 18 * mm, 18 * mm])
    cm_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1), PALE_BLUE),
        ("BACKGROUND", (1, 1), (1, 1), PALE_TEAL),
        ("BACKGROUND", (2, 2), (2, 2), PALE_TEAL),
        ("BACKGROUND", (2, 1), (2, 1), PALE_RED),
        ("BACKGROUND", (1, 2), (1, 2), PALE_RED),
        ("GRID", (0, 0), (-1, -1), 0.8, GRID),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(cm_table)
    story.append(Spacer(1, 5 * mm))

    cm_explanations = [
        ["5,328", "True Negative", "Actual BENIGN and predicted BENIGN. Correct normal result."],
        ["672", "False Positive", "Actual BENIGN but predicted MALICIOUS. A false alarm."],
        ["229", "False Negative", "Actual MALICIOUS but predicted BENIGN. A missed attack."],
        ["3,811", "True Positive", "Actual MALICIOUS and predicted MALICIOUS. Correct detection."],
    ]
    explanation_data = [["Count", "Name", "Plain meaning"]] + cm_explanations
    story.append(styled_table(explanation_data, [25 * mm, 38 * mm, 97 * mm], background=PALE_TEAL, font_size=8.2))
    story.append(Spacer(1, 5 * mm))
    story.append(p("Useful follow-up answers", styles["h2"]))
    story.extend(q_and_a(
        "What is a false positive?",
        "The model raises an alert for traffic that is actually BENIGN. Here, there are 672 false alarms for the Logistic Regression test set.",
        styles,
    ))
    story.extend(q_and_a(
        "What is a false negative?",
        "The model says traffic is BENIGN even though it is MALICIOUS. Here, 229 malicious flows were missed. In security, false negatives are especially important.",
        styles,
    ))
    story.extend(q_and_a(
        "Why do we need precision, recall, F1, and false-positive rate too?",
        "Accuracy alone hides the type of mistake. Recall measures how many attacks were found, precision measures how many alerts were truly attacks, F1 balances precision and recall, and FPR measures the fraction of normal traffic incorrectly alerted.",
        styles,
    ))

    story.append(PageBreak())

    # Generalization and limitations.
    story.append(p("Part 4 - Difficult questions and honest limitations", styles["h1"]))
    story.extend(q_and_a(
        "Why is the random-split accuracy so high?",
        "A random split can place very similar flows, generated by the same capture period or attack tool, in both training and test sets. The model may therefore see patterns close to the test traffic during training.",
        styles,
    ))
    story.extend(q_and_a(
        "What happens on a more realistic day-aware test?",
        f"The report holds out the complete Wednesday traffic file. On that distribution shift, the weighted Random Forest accuracy is {fmt(day_rf['accuracy'])} and Logistic Regression accuracy is {fmt(day_logistic['accuracy'])}. This is much lower than the random-split result.",
        styles,
    ))
    story.extend(q_and_a(
        "Does the result prove the model is ready for production?",
        "No. It proves performance on the tested benchmark split. A production claim would require time-based validation, traffic from another environment, monitoring, and a chosen trade-off between missed attacks and false alarms.",
        styles,
    ))
    story.extend(q_and_a(
        "Is CICIDS2017 a perfect dataset?",
        "No. It is useful as a benchmark but has known issues involving flow construction, attack timing, labeling, TCP flag counts, and a duplicated Fwd Header Length feature. It is better to call the labels benchmark ground truth or imperfect labels, not perfect real-world truth.",
        styles,
    ))
    story.extend(q_and_a(
        "What improvements would you propose?",
        "Use a corrected or independently validated dataset, split by time or source environment, remove duplicate features, report recall and false-positive rate beside accuracy, test on unseen traffic, and monitor drift after deployment.",
        styles,
    ))
    story.append(box(p(
        "<b>Best honest conclusion:</b> The project demonstrates that the models can classify this benchmark data, but the gap between random-split and day-aware results warns us not to treat 0.9961 or 0.9103 as universal real-world accuracy.",
        styles["callout"],
    ), PALE_AMBER, colors.HexColor("#D97706")))

    story.append(PageBreak())

    # Final memorization page.
    story.append(p("Part 5 - Memorize this answer", styles["h1"]))
    story.append(box(p(
        "<b>Short viva answer:</b> The system performs binary network-flow classification. Each record has 78 numeric traffic features and a Label column. The Label is converted to 0 for BENIGN and 1 for every non-BENIGN attack. The report evaluates models on a deterministic 40,160-row sample using a 75/25 stratified split. The 0.9103 value is balanced Logistic Regression accuracy on 10,040 test rows, calculated as 9,139 correct predictions divided by 10,040. The weighted Random Forest gives 0.9961 on the same random split, but the day-aware holdout is much lower, so accuracy should not be presented as universal.",
        styles["callout"],
    ), PALE_TEAL, TEAL))
    story.append(Spacer(1, 6 * mm))
    story.append(p("Five numbers to remember", styles["h2"]))
    remember_table = styled_table([
        ["Number", "Meaning"],
        ["78", "Input traffic features"],
        ["40,160", "Rows in the deterministic comparison sample"],
        ["10,040", "Rows in the random test set"],
        ["0.9103", "Balanced Logistic Regression random-split accuracy"],
        ["0.9961", "Current weighted Random Forest random-split accuracy"],
        ["0.6000", "Current weighted Random Forest Wednesday holdout accuracy"],
    ], [35 * mm, 131 * mm], background=PALE_BLUE)
    story.append(remember_table)
    story.append(Spacer(1, 6 * mm))
    story.append(p("Sources used", styles["h2"]))
    sources = [
        "Local report: reports/model_comparison.json",
        "Local project description: README.md",
        "CIC-IDS2017 official documentation: https://www.unb.ca/cic/datasets/ids-2017.html",
        "CICIDS2017 quality analysis: https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html",
    ]
    story.append(p("<br/>".join(f"- {escape(item)}" for item in sources), styles["small"]))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
