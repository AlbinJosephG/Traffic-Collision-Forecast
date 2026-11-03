
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 6.6 - CNN-LSTM TEST ERROR ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_64 = (
    ROOT / "reports" / "phase_6_4_final_test_evaluation"
    / "final_test_predictions.csv"
)

PREDICTIONS_64C = (
    ROOT / "reports" / "phase_6_4C_remaining_architecture_test_evaluation"
    / "remaining_architecture_test_predictions.csv"
)

OUTPUT_DIR = ROOT / "reports" / "phase_6_6_error_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"

THRESHOLD = 0.5
HORIZONS = [500, 1000, 1500]


def load_predictions():
    """Load and standardize predictions from Phase 6.4 and 6.4C."""

    if not PREDICTIONS_64.exists():
        raise FileNotFoundError(PREDICTIONS_64)

    if not PREDICTIONS_64C.exists():
        raise FileNotFoundError(PREDICTIONS_64C)

    df_64 = pd.read_csv(PREDICTIONS_64)
    df_64c = pd.read_csv(PREDICTIONS_64C)

    # Standardize target label column.
    df_64 = df_64.rename(columns={"actual_label": "true_label"})
    df_64c = df_64c.rename(columns={"actual_label": "true_label"})

    required = {
        "architecture",
        "horizon_ms",
        "sample_id",
        "true_label",
        "predicted_probability",
        "predicted_label",
    }

    for name, frame in [
        ("Phase 6.4", df_64),
        ("Phase 6.4C", df_64c),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    predictions = pd.concat(
        [df_64, df_64c],
        ignore_index=True,
        sort=False,
    )

    # Keep only CNN-LSTM predictions.
    predictions = predictions[
        predictions["architecture"].astype(str).str.lower() == "cnn_lstm"
    ].copy()

    predictions["horizon_ms"] = pd.to_numeric(
        predictions["horizon_ms"], errors="raise"
    ).astype(int)

    predictions["true_label"] = pd.to_numeric(
        predictions["true_label"], errors="raise"
    ).astype(int)

    predictions["predicted_label"] = pd.to_numeric(
        predictions["predicted_label"], errors="raise"
    ).astype(int)

    predictions["predicted_probability"] = pd.to_numeric(
        predictions["predicted_probability"], errors="raise"
    )

    if predictions["true_label"].isna().any():
        raise ValueError("Missing true labels detected.")

    if predictions["predicted_probability"].isna().any():
        raise ValueError("Missing predicted probabilities detected.")

    if not predictions["true_label"].isin([0, 1]).all():
        raise ValueError("True labels must contain only 0 and 1.")

    if not predictions["predicted_label"].isin([0, 1]).all():
        raise ValueError("Predicted labels must contain only 0 and 1.")

    if not predictions["predicted_probability"].between(0, 1).all():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    predictions = predictions[
        predictions["horizon_ms"].isin(HORIZONS)
    ].copy()

    expected_counts = {500: 28, 1000: 28, 1500: 27}

    actual_counts = predictions.groupby("horizon_ms").size().to_dict()

    if actual_counts != expected_counts:
        raise ValueError(
            "Unexpected CNN-LSTM test sample counts. "
            f"Expected {expected_counts}, got {actual_counts}"
        )

    return predictions


def classify_errors(df):
    """Identify false positives, false negatives, and correct predictions."""

    df = df.copy()

    conditions = [
        (df["true_label"] == 0) & (df["predicted_label"] == 0),
        (df["true_label"] == 0) & (df["predicted_label"] == 1),
        (df["true_label"] == 1) & (df["predicted_label"] == 0),
        (df["true_label"] == 1) & (df["predicted_label"] == 1),
    ]

    choices = [
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive",
    ]

    df["error_type"] = np.select(
        conditions,
        choices,
        default="Unknown",
    )

    df["is_correct"] = (
        df["true_label"] == df["predicted_label"]
    )

    # Confidence in the class actually predicted.
    df["prediction_confidence"] = np.where(
        df["predicted_label"] == 1,
        df["predicted_probability"],
        1 - df["predicted_probability"],
    )

    # Distance from the decision threshold.
    df["threshold_distance"] = abs(
        df["predicted_probability"] - THRESHOLD
    )

    return df


def create_summary(df):
    """Create horizon-level error and confidence statistics."""

    rows = []

    for horizon in HORIZONS:
        subset = df[df["horizon_ms"] == horizon]

        tn = int(
            (
                (subset["true_label"] == 0)
                & (subset["predicted_label"] == 0)
            ).sum()
        )

        fp = int(
            (
                (subset["true_label"] == 0)
                & (subset["predicted_label"] == 1)
            ).sum()
        )

        fn = int(
            (
                (subset["true_label"] == 1)
                & (subset["predicted_label"] == 0)
            ).sum()
        )

        tp = int(
            (
                (subset["true_label"] == 1)
                & (subset["predicted_label"] == 1)
            ).sum()
        )

        correct = subset[subset["is_correct"]]
        incorrect = subset[~subset["is_correct"]]

        rows.append({
            "horizon_ms": horizon,
            "test_samples": len(subset),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "correct_predictions": len(correct),
            "incorrect_predictions": len(incorrect),
            "error_rate": len(incorrect) / len(subset),
            "mean_confidence_correct": (
                correct["prediction_confidence"].mean()
                if len(correct) else np.nan
            ),
            "mean_confidence_incorrect": (
                incorrect["prediction_confidence"].mean()
                if len(incorrect) else np.nan
            ),
            "mean_probability_positive_samples": (
                subset.loc[
                    subset["true_label"] == 1,
                    "predicted_probability"
                ].mean()
            ),
            "mean_probability_negative_samples": (
                subset.loc[
                    subset["true_label"] == 0,
                    "predicted_probability"
                ].mean()
            ),
            "mean_threshold_distance_errors": (
                incorrect["threshold_distance"].mean()
                if len(incorrect) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def create_confusion_plot(df, horizon):
    """Save a confusion matrix for one prediction horizon."""

    subset = df[df["horizon_ms"] == horizon]

    tn = int(
        (
            (subset["true_label"] == 0)
            & (subset["predicted_label"] == 0)
        ).sum()
    )
    fp = int(
        (
            (subset["true_label"] == 0)
            & (subset["predicted_label"] == 1)
        ).sum()
    )
    fn = int(
        (
            (subset["true_label"] == 1)
            & (subset["predicted_label"] == 0)
        ).sum()
    )
    tp = int(
        (
            (subset["true_label"] == 1)
            & (subset["predicted_label"] == 1)
        ).sum()
    )

    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(f"CNN-LSTM Confusion Matrix - {horizon} ms")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    ax.set_xticks([0, 1], labels=["Negative", "Positive"])
    ax.set_yticks([0, 1], labels=["Negative", "Positive"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                fontsize=14,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / f"confusion_matrix_{horizon}ms.png",
        dpi=300,
    )
    plt.close(fig)


def create_probability_plot(df, horizon):
    """Compare probability distributions for positive and negative samples."""

    subset = df[df["horizon_ms"] == horizon]

    negative = subset.loc[
        subset["true_label"] == 0,
        "predicted_probability"
    ]

    positive = subset.loc[
        subset["true_label"] == 1,
        "predicted_probability"
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        negative,
        bins=10,
        alpha=0.65,
        label="Actual Negative",
    )

    ax.hist(
        positive,
        bins=10,
        alpha=0.65,
        label="Actual Positive",
    )

    ax.axvline(
        THRESHOLD,
        linestyle="--",
        label=f"Decision threshold ({THRESHOLD})",
    )

    ax.set_title(
        f"Predicted Probability Distribution - {horizon} ms"
    )
    ax.set_xlabel("Predicted collision probability")
    ax.set_ylabel("Number of samples")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / f"probability_distribution_{horizon}ms.png",
        dpi=300,
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PHASE 6.6 - CNN-LSTM ERROR ANALYSIS")
    print("=" * 65)

    predictions = load_predictions()
    predictions = classify_errors(predictions)

    summary = create_summary(predictions)

    # Save every CNN-LSTM test prediction with its error classification.
    predictions.to_csv(
        OUTPUT_DIR / "cnn_lstm_all_test_predictions_with_errors.csv",
        index=False,
    )

    # Save only incorrect predictions.
    errors = predictions[
        ~predictions["is_correct"]
    ].copy()

    errors.to_csv(
        OUTPUT_DIR / "cnn_lstm_incorrect_predictions.csv",
        index=False,
    )

    # Save horizon-level statistics.
    summary.to_csv(
        OUTPUT_DIR / "cnn_lstm_error_analysis_summary.csv",
        index=False,
    )

    # Save confusion matrices and probability distributions.
    for horizon in HORIZONS:
        create_confusion_plot(predictions, horizon)
        create_probability_plot(predictions, horizon)

    print("\nCNN-LSTM ERROR ANALYSIS SUMMARY")
    print(summary.to_string(index=False))

    print("\nERROR TYPE COUNTS")
    print(
        predictions.groupby(
            ["horizon_ms", "error_type"]
        ).size().unstack(fill_value=0).to_string()
    )

    print("\nOUTPUT DIRECTORY:")
    print(OUTPUT_DIR)

    print("\nPHASE 6.6 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
