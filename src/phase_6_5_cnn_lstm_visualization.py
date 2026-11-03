
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PHASE_64_DIR = (
    ROOT / "reports" / "phase_6_4_final_test_evaluation"
)

PHASE_64C_DIR = (
    ROOT / "reports" /
    "phase_6_4C_remaining_architecture_test_evaluation"
)

OUTPUT_DIR = (
    ROOT / "reports" /
    "phase_6_5_cnn_lstm_visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [500, 1000, 1500]

# Actual filenames confirmed from your project.
SUMMARY_64 = PHASE_64_DIR / "final_test_evaluation_summary.csv"
PREDICTIONS_64 = PHASE_64_DIR / "final_test_predictions.csv"

SUMMARY_64C = (
    PHASE_64C_DIR / "remaining_architecture_test_summary.csv"
)

PREDICTIONS_64C = (
    PHASE_64C_DIR / "remaining_architecture_test_predictions.csv"
)

required_files = [
    SUMMARY_64,
    PREDICTIONS_64,
    SUMMARY_64C,
    PREDICTIONS_64C,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

# ============================================================
# 2. LOAD RESULTS
# ============================================================

summary_64 = pd.read_csv(SUMMARY_64)
summary_64c = pd.read_csv(SUMMARY_64C)

predictions_64 = pd.read_csv(PREDICTIONS_64)
predictions_64c = pd.read_csv(PREDICTIONS_64C)

# Standardize label column names across evaluation phases.
if "actual_label" in predictions_64.columns:
    predictions_64 = predictions_64.rename(
        columns={"actual_label": "true_label"}
    )

if "actual_label" in predictions_64c.columns:
    predictions_64c = predictions_64c.rename(
        columns={"actual_label": "true_label"}
    )



# Filter strictly for CNN-LSTM.
summary_64 = summary_64[
    summary_64["architecture"].str.lower() == "cnn_lstm"
].copy()

summary_64c = summary_64c[
    summary_64c["architecture"].str.lower() == "cnn_lstm"
].copy()

predictions_64 = predictions_64[
    predictions_64["architecture"].str.lower() == "cnn_lstm"
].copy()

predictions_64c = predictions_64c[
    predictions_64c["architecture"].str.lower() == "cnn_lstm"
].copy()

summary = pd.concat(
    [summary_64, summary_64c],
    ignore_index=True,
)

predictions = pd.concat(
    [predictions_64, predictions_64c],
    ignore_index=True,
)

summary["horizon_ms"] = summary["horizon_ms"].astype(int)
predictions["horizon_ms"] = predictions["horizon_ms"].astype(int)

summary = summary.sort_values("horizon_ms").reset_index(drop=True)
# Validate combined prediction labels.
if predictions["true_label"].isna().any():
    raise ValueError("Missing actual labels remain in predictions.")

if predictions["predicted_label"].isna().any():
    raise ValueError("Missing predicted labels remain in predictions.")

print("\nPrediction records by horizon:")
print(predictions.groupby("horizon_ms").size())

print("\nMissing actual labels:")
print(predictions["true_label"].isna().sum())

if summary["horizon_ms"].duplicated().any():
    raise ValueError("Duplicate CNN-LSTM horizon found.")

if sorted(summary["horizon_ms"].unique().tolist()) != HORIZONS:
    raise ValueError(
        "Expected CNN-LSTM results for all three horizons. "
        f"Found: {sorted(summary['horizon_ms'].unique().tolist())}"
    )

if predictions.empty:
    raise ValueError("No CNN-LSTM predictions were loaded.")

# Confirm the required prediction columns.
required_prediction_columns = {
    "horizon_ms",
    "architecture",
    "true_label",
    "predicted_probability",
    "predicted_label",
}

missing = required_prediction_columns.difference(predictions.columns)

if missing:
    raise ValueError(
        f"Missing prediction columns: {sorted(missing)}"
    )

# Save combined data.
summary.to_csv(
    OUTPUT_DIR / "cnn_lstm_final_test_results.csv",
    index=False,
)

predictions.to_csv(
    OUTPUT_DIR / "cnn_lstm_all_test_predictions.csv",
    index=False,
)

print("\nCNN-LSTM TEST RESULTS")
print(summary[
    [
        "horizon_ms",
        "test_samples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
].to_string(index=False))

# ============================================================
# 3. PERFORMANCE METRICS
# ============================================================

metrics = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1 Score"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (column, label) in zip(axes.flat, metrics):
    values = summary[column].to_numpy()

    ax.bar(
        summary["horizon_ms"].astype(str),
        values,
    )

    ax.set_title(label)
    ax.set_xlabel("Prediction Horizon (ms)")
    ax.set_ylabel(label)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)

    for index, value in enumerate(values):
        ax.text(
            index,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
        )

fig.suptitle(
    "CNN-LSTM Test Performance Across Prediction Horizons"
)
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "01_cnn_lstm_performance_metrics.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 4. ROC-AUC AND AVERAGE PRECISION
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(summary))
width = 0.35

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

ax.set_xticks(x)
ax.set_xticklabels(summary["horizon_ms"].astype(str))
ax.set_xlabel("Prediction Horizon (ms)")
ax.set_ylabel("Score")
ax.set_title("CNN-LSTM Ranking Performance")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "02_cnn_lstm_auc_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 5. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, horizon in zip(axes, HORIZONS):

    subset = predictions[
        predictions["horizon_ms"] == horizon
    ]

    y_true = subset["true_label"].astype(int)
    y_pred = subset["predicted_label"].astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    ax.imshow(cm, interpolation="nearest")

    ax.set_title(f"{horizon} ms")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
            )

fig.suptitle("CNN-LSTM Test Confusion Matrices")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "03_cnn_lstm_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 6. ROC CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for horizon in HORIZONS:

    subset = predictions[
        predictions["horizon_ms"] == horizon
    ]

    y_true = subset["true_label"].astype(int)
    y_score = subset["predicted_probability"].astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    ax.plot(
        fpr,
        tpr,
        marker=".",
        label=f"{horizon} ms (AUC={auc_value:.3f})",
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("CNN-LSTM ROC Curves")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "04_cnn_lstm_roc_curves.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 7. PRECISION-RECALL CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for horizon in HORIZONS:

    subset = predictions[
        predictions["horizon_ms"] == horizon
    ]

    y_true = subset["true_label"].astype(int)
    y_score = subset["predicted_probability"].astype(float)

    precision_curve, recall_curve, _ = (
        precision_recall_curve(y_true, y_score)
    )

    ap = average_precision_score(y_true, y_score)

    ax.plot(
        recall_curve,
        precision_curve,
        marker=".",
        label=f"{horizon} ms (AP={ap:.3f})",
    )

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("CNN-LSTM Precision-Recall Curves")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "05_cnn_lstm_precision_recall_curves.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 8. FALSE POSITIVE AND FALSE NEGATIVE ANALYSIS
# ============================================================

error_rows = []

for horizon in HORIZONS:

    subset = predictions[
        predictions["horizon_ms"] == horizon
    ]

    y_true = subset["true_label"].astype(int).to_numpy()
    y_pred = subset["predicted_label"].astype(int).to_numpy()

    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    error_rows.append({
        "horizon_ms": horizon,
        "false_positives": fp,
        "false_negatives": fn,
    })

error_df = pd.DataFrame(error_rows)

error_df.to_csv(
    OUTPUT_DIR / "cnn_lstm_error_counts.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(error_df))
width = 0.35

ax.bar(
    x - width / 2,
    error_df["false_positives"],
    width,
    label="False Positives",
)

ax.bar(
    x + width / 2,
    error_df["false_negatives"],
    width,
    label="False Negatives",
)

ax.set_xticks(x)
ax.set_xticklabels(error_df["horizon_ms"].astype(str))
ax.set_xlabel("Prediction Horizon (ms)")
ax.set_ylabel("Number of Errors")
ax.set_title("CNN-LSTM Test Classification Errors")
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "06_cnn_lstm_error_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 9. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("PHASE 6.5 COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nSelected architecture: CNN-LSTM")
print("\nGenerated figures:")

for filename in [
    "01_cnn_lstm_performance_metrics.png",
    "02_cnn_lstm_auc_comparison.png",
    "03_cnn_lstm_confusion_matrices.png",
    "04_cnn_lstm_roc_curves.png",
    "05_cnn_lstm_precision_recall_curves.png",
    "06_cnn_lstm_error_analysis.png",
]:
    print(OUTPUT_DIR / filename)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nNo model retraining or threshold tuning was performed.")
