
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# ============================================================
# PHASE 6.7 - CNN-LSTM PROBABILITY CALIBRATION
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

OUTPUT_DIR = ROOT / "reports" / "phase_6_7_probability_calibration"
FIGURE_DIR = OUTPUT_DIR / "figures"

HORIZONS = [500, 1000, 1500]
N_BINS = 5


def load_predictions():

    if not PREDICTIONS_64.exists():
        raise FileNotFoundError(PREDICTIONS_64)

    if not PREDICTIONS_64C.exists():
        raise FileNotFoundError(PREDICTIONS_64C)

    df_64 = pd.read_csv(PREDICTIONS_64)
    df_64c = pd.read_csv(PREDICTIONS_64C)

    # Standardize target column names.
    df_64 = df_64.rename(columns={"actual_label": "true_label"})
    df_64c = df_64c.rename(columns={"actual_label": "true_label"})

    required = {
        "architecture",
        "horizon_ms",
        "true_label",
        "predicted_probability",
    }

    for name, frame in [
        ("Phase 6.4", df_64),
        ("Phase 6.4C", df_64c),
    ]:
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    predictions = pd.concat(
        [df_64, df_64c],
        ignore_index=True,
        sort=False,
    )

    predictions = predictions[
        predictions["architecture"].astype(str).str.lower()
        == "cnn_lstm"
    ].copy()

    predictions["horizon_ms"] = pd.to_numeric(
        predictions["horizon_ms"],
        errors="raise",
    ).astype(int)

    predictions["true_label"] = pd.to_numeric(
        predictions["true_label"],
        errors="raise",
    ).astype(int)

    predictions["predicted_probability"] = pd.to_numeric(
        predictions["predicted_probability"],
        errors="raise",
    )

    if predictions[
        ["true_label", "predicted_probability"]
    ].isna().any().any():
        raise ValueError("Missing calibration data detected.")

    if not predictions["true_label"].isin([0, 1]).all():
        raise ValueError("Labels must be binary.")

    if not predictions["predicted_probability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    predictions = predictions[
        predictions["horizon_ms"].isin(HORIZONS)
    ].copy()

    expected_counts = {500: 28, 1000: 28, 1500: 27}
    actual_counts = predictions.groupby("horizon_ms").size().to_dict()

    if actual_counts != expected_counts:
        raise ValueError(
            f"Unexpected sample counts: {actual_counts}"
        )

    return predictions


def calculate_ece(y_true, probabilities, n_bins=5):
    """
    Calculate equal-width Expected Calibration Error.

    ECE is descriptive and sensitive to bin selection,
    especially with small datasets.
    """

    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    bin_results = []

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_probability = float(probabilities[mask].mean())
        observed_frequency = float(y_true[mask].mean())

        bin_ece = (
            count / len(y_true)
        ) * abs(mean_probability - observed_frequency)

        ece += bin_ece

        bin_results.append({
            "bin_lower": lower,
            "bin_upper": upper,
            "sample_count": count,
            "mean_predicted_probability": mean_probability,
            "observed_collision_frequency": observed_frequency,
            "absolute_calibration_error": abs(
                mean_probability - observed_frequency
            ),
            "ece_contribution": bin_ece,
        })

    return ece, bin_results


def analyze_horizon(subset, horizon):

    y_true = subset["true_label"].to_numpy(dtype=int)

    probabilities = subset[
        "predicted_probability"
    ].to_numpy(dtype=float)

    brier = brier_score_loss(y_true, probabilities)

    ece, bins = calculate_ece(
        y_true,
        probabilities,
        N_BINS,
    )

    # Calibration curve uses only non-empty probability bins.
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=N_BINS,
        strategy="uniform",
    )

    return {
        "horizon_ms": horizon,
        "test_samples": len(subset),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int(len(y_true) - y_true.sum()),
        "mean_predicted_probability": float(probabilities.mean()),
        "observed_collision_frequency": float(y_true.mean()),
        "brier_score": float(brier),
        "ece_5_bins": float(ece),
        "calibration_bins": bins,
        "fraction_positive": fraction_positive,
        "mean_predicted": mean_predicted,
    }


def plot_reliability_curve(results):

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for horizon in HORIZONS:

        result = results[horizon]

        ax.plot(
            result["mean_predicted"],
            result["fraction_positive"],
            marker="o",
            linewidth=2,
            label=f"{horizon} ms",
        )

    ax.set_title("CNN-LSTM Reliability Diagram")
    ax.set_xlabel("Mean predicted collision probability")
    ax.set_ylabel("Observed collision frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "combined_reliability_diagram.png",
        dpi=300,
    )

    plt.close(fig)


def plot_probability_histogram(subset, horizon):

    fig, ax = plt.subplots(figsize=(7, 5))

    negative = subset.loc[
        subset["true_label"] == 0,
        "predicted_probability",
    ]

    positive = subset.loc[
        subset["true_label"] == 1,
        "predicted_probability",
    ]

    ax.hist(
        negative,
        bins=10,
        alpha=0.65,
        label="Actual negative",
    )

    ax.hist(
        positive,
        bins=10,
        alpha=0.65,
        label="Actual positive",
    )

    ax.set_title(
        f"Predicted Probability Distribution - {horizon} ms"
    )

    ax.set_xlabel("Predicted collision probability")
    ax.set_ylabel("Number of samples")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / f"probability_histogram_{horizon}ms.png",
        dpi=300,
    )

    plt.close(fig)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PHASE 6.7 - CNN-LSTM PROBABILITY CALIBRATION")
    print("=" * 65)

    predictions = load_predictions()

    results = {}
    summary_rows = []
    bin_rows = []

    for horizon in HORIZONS:

        subset = predictions[
            predictions["horizon_ms"] == horizon
        ].copy()

        result = analyze_horizon(subset, horizon)

        results[horizon] = result

        summary_rows.append({
            key: value
            for key, value in result.items()
            if key not in [
                "calibration_bins",
                "fraction_positive",
                "mean_predicted",
            ]
        })

        for row in result["calibration_bins"]:
            bin_rows.append({
                "horizon_ms": horizon,
                **row,
            })

        plot_probability_histogram(subset, horizon)

    plot_reliability_curve(results)

    summary = pd.DataFrame(summary_rows)
    calibration_bins = pd.DataFrame(bin_rows)

    summary.to_csv(
        OUTPUT_DIR / "calibration_summary.csv",
        index=False,
    )

    calibration_bins.to_csv(
        OUTPUT_DIR / "calibration_bin_details.csv",
        index=False,
    )

    print("\nCALIBRATION SUMMARY")

    print(
        summary.to_string(index=False)
    )

    print("\nCALIBRATION BIN DETAILS")

    print(
        calibration_bins.to_string(index=False)
    )

    print("\nOUTPUT DIRECTORY:")
    print(OUTPUT_DIR)

    print("\nPHASE 6.7 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
