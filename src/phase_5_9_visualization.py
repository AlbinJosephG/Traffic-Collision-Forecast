
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = ROOT / "processed" / "phase_5_8_test_evaluation"

TRAINING_SUMMARY = (
    ROOT
    / "processed"
    / "fixed_rate_training_history"
    / "fixed_rate_training_summary.csv"
)

OUTPUT_DIR = ROOT / "reports" / "phase_5_9_visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [500, 1000, 1500]

plt.rcParams.update({
    "figure.figsize": (9, 6),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

def load_results():

    summary_path = EVAL_DIR / "test_evaluation_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Test summary not found: {summary_path}"
        )

    summary = pd.read_csv(summary_path)

    required_columns = [
        "horizon_ms",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
    ]

    missing = [
        column for column in required_columns
        if column not in summary.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in test summary: {missing}"
        )

    predictions = {}

    for horizon in HORIZONS:

        path = EVAL_DIR / f"predictions_{horizon}ms.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {path}"
            )

        df = pd.read_csv(path)

        required = [
            "actual_label",
            "predicted_label",
            "collision_probability",
        ]

        if not all(column in df.columns for column in required):
            raise ValueError(
                f"Required prediction columns missing in {path}"
            )

        if not np.isfinite(
            df["collision_probability"].to_numpy()
        ).all():
            raise ValueError(
                f"Invalid probabilities in {path}"
            )

        predictions[horizon] = df

    print("Test summary and prediction files loaded.")

    return summary, predictions


# ============================================================
# LOAD VALIDATION AP FROM PHASE 5.7
# ============================================================

def load_validation_results():

    if not TRAINING_SUMMARY.exists():
        print(
            "\nWARNING: Phase 5.7 training summary not found."
        )
        print("Validation AP comparison will be skipped.")
        return None

    df = pd.read_csv(TRAINING_SUMMARY)

    if not {
        "horizon_ms",
        "best_validation_ap",
    }.issubset(df.columns):

        print(
            "\nWARNING: Required validation AP columns "
            "were not found. Comparison will be skipped."
        )
        return None

    return df[
        ["horizon_ms", "best_validation_ap"]
    ]


# ============================================================
# SAVE FIGURES
# ============================================================

def save_figure(filename):

    path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


# ============================================================
# CHART 1: PERFORMANCE METRIC COMPARISON
# ============================================================

def plot_performance_comparison(summary):

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]

    x = np.arange(len(HORIZONS))
    width = 0.19

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (metric, label) in enumerate(zip(metrics, labels)):

        values = []

        for horizon in HORIZONS:
            row = summary[
                summary["horizon_ms"] == horizon
            ]

            if row.empty:
                raise ValueError(
                    f"Missing horizon {horizon} in summary."
                )

            values.append(float(row.iloc[0][metric]))

        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=label,
        )

    ax.set_title("LSTM Test Performance Comparison")
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{h} ms" for h in HORIZONS]
    )
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure("01_performance_comparison.png")


# ============================================================
# CHART 2: ROC CURVES
# ============================================================

def plot_roc_curves(predictions):

    fig, ax = plt.subplots(figsize=(8, 7))

    for horizon in HORIZONS:

        df = predictions[horizon]

        y_true = df["actual_label"].to_numpy()
        probabilities = df[
            "collision_probability"
        ].to_numpy()

        fpr, tpr, _ = roc_curve(
            y_true,
            probabilities,
        )

        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{horizon} ms (AUC = {roc_auc:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_title("ROC Curves - LSTM Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    save_figure("02_roc_curves.png")


# ============================================================
# CHART 3: PRECISION-RECALL CURVES
# ============================================================

def plot_precision_recall_curves(predictions):

    fig, ax = plt.subplots(figsize=(8, 7))

    for horizon in HORIZONS:

        df = predictions[horizon]

        y_true = df["actual_label"].to_numpy()
        probabilities = df[
            "collision_probability"
        ].to_numpy()

        precision, recall, _ = precision_recall_curve(
            y_true,
            probabilities,
        )

        ap = average_precision_score(
            y_true,
            probabilities,
        )

        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{horizon} ms (AP = {ap:.3f})",
        )

    ax.set_title("Precision-Recall Curves - LSTM Models")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    save_figure("03_precision_recall_curves.png")


# ============================================================
# CHART 4: CONFUSION MATRIX HEATMAPS
# ============================================================

def plot_confusion_matrices(predictions):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    for ax, horizon in zip(axes, HORIZONS):

        df = predictions[horizon]

        y_true = df["actual_label"].to_numpy()
        y_pred = df["predicted_label"].to_numpy()

        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

        matrix = np.array([
            [tn, fp],
            [fn, tp],
        ])

        image = ax.imshow(matrix)

        ax.set_title(f"{horizon} ms")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])

        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    fontsize=14,
                )

    fig.suptitle("Confusion Matrices - LSTM Models")

    save_figure("04_confusion_matrices.png")


# ============================================================
# CHART 5: VALIDATION AP VS TEST AP
# ============================================================

def plot_validation_vs_test(
    summary,
    validation_summary,
):

    if validation_summary is None:
        return

    merged = summary.merge(
        validation_summary,
        on="horizon_ms",
        how="inner",
    )

    if len(merged) != len(HORIZONS):
        print(
            "WARNING: Not all horizons matched. "
            "Skipping validation/test comparison."
        )
        return

    x = np.arange(len(HORIZONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        x - width / 2,
        merged["best_validation_ap"],
        width,
        label="Best Validation AP",
    )

    ax.bar(
        x + width / 2,
        merged["average_precision"],
        width,
        label="Test AP",
    )

    ax.set_title("Validation vs Test Average Precision")
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Average Precision")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{h} ms" for h in merged["horizon_ms"]]
    )
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure("05_validation_vs_test_ap.png")


# ============================================================
# CHART 6: ROC-AUC AND AVERAGE PRECISION
# ============================================================

def plot_auc_comparison(summary):

    x = np.arange(len(HORIZONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        x - width / 2,
        summary["roc_auc"],
        width,
        label="ROC-AUC",
    )

    ax.bar(
        x + width / 2,
        summary["average_precision"],
        width,
        label="Average Precision",
    )

    ax.set_title("ROC-AUC vs Average Precision")
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{h} ms" for h in summary["horizon_ms"]]
    )
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure("06_auc_comparison.png")


# ============================================================
# SAVE COMBINED REPORT
# ============================================================

def save_combined_report(summary, validation_summary):

    report = summary.copy()

    if validation_summary is not None:

        report = report.merge(
            validation_summary,
            on="horizon_ms",
            how="left",
        )

    report_path = OUTPUT_DIR / "phase_5_9_combined_results.csv"

    report.to_csv(
        report_path,
        index=False,
    )

    print("\nCombined results saved:", report_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 5.9 - MODEL VISUALIZATION")
    print("=" * 65)

    summary, predictions = load_results()

    validation_summary = load_validation_results()

    plot_performance_comparison(summary)

    plot_roc_curves(predictions)

    plot_precision_recall_curves(predictions)

    plot_confusion_matrices(predictions)

    plot_validation_vs_test(
        summary,
        validation_summary,
    )

    plot_auc_comparison(summary)

    save_combined_report(
        summary,
        validation_summary,
    )

    print("\n" + "=" * 65)
    print("PHASE 5.9 COMPLETED")
    print("=" * 65)

    print("\nVisualization directory:")
    print(OUTPUT_DIR)

    print("\nAll available charts have been generated.")


if __name__ == "__main__":
    main()
