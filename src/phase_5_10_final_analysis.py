
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = (
    ROOT / "processed" / "phase_5_8_test_evaluation"
)

TRAINING_FILE = (
    ROOT / "processed"
    / "fixed_rate_training_history"
    / "fixed_rate_training_summary.csv"
)

OUTPUT_DIR = ROOT / "reports" / "phase_5_10_final_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():

    test_file = EVAL_DIR / "test_evaluation_summary.csv"

    if not test_file.exists():
        raise FileNotFoundError(test_file)

    test = pd.read_csv(test_file)

    if not TRAINING_FILE.exists():
        raise FileNotFoundError(TRAINING_FILE)

    training = pd.read_csv(TRAINING_FILE)

    required_test = {
        "horizon_ms",
        "test_samples",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
        "false_positive",
        "false_negative",
    }

    required_training = {
        "horizon_ms",
        "best_validation_ap",
        "train_samples",
        "validation_samples",
    }

    if not required_test.issubset(test.columns):
        raise ValueError("Missing columns in test summary.")

    if not required_training.issubset(training.columns):
        raise ValueError("Missing columns in training summary.")

    return test, training


def create_analysis(test, training):

    combined = test.merge(
        training,
        on="horizon_ms",
        how="inner",
        suffixes=("_test", "_training"),
    )

    combined["validation_test_ap_difference"] = (
        combined["average_precision"]
        - combined["best_validation_ap"]
    )

    combined["collision_detection_rate"] = (
        combined["recall"] * 100
    )

    combined["false_alarm_count"] = (
        combined["false_positive"]
    )

    combined["missed_collision_count"] = (
        combined["false_negative"]
    )

    return combined


def determine_best_models(df):

    criteria = {
        "Best accuracy": "accuracy",
        "Best precision": "precision",
        "Best recall": "recall",
        "Best F1-score": "f1_score",
        "Best ROC-AUC": "roc_auc",
        "Best Average Precision": "average_precision",
    }

    results = {}

    for description, column in criteria.items():

        row = df.loc[df[column].idxmax()]

        results[description] = {
            "horizon_ms": int(row["horizon_ms"]),
            "score": float(row[column]),
        }

    return results


def generate_report(df, best):

    lines = []

    lines.append(
        "FINAL TRAFFIC COLLISION PREDICTION ANALYSIS"
    )

    lines.append("=" * 65)
    lines.append("")

    lines.append("1. EXPERIMENTAL SETUP")
    lines.append(
        "Three LSTM models were trained for prediction "
        "horizons of 500 ms, 1000 ms and 1500 ms."
    )
    lines.append(
        "The models were evaluated using the held-out "
        "test dataset."
    )
    lines.append("")

    lines.append("2. TEST PERFORMANCE")

    columns = [
        "horizon_ms",
        "test_samples",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
    ]

    lines.append(
        df[columns].to_string(index=False)
    )

    lines.append("")

    lines.append("3. MODEL COMPARISON")

    for description, result in best.items():

        lines.append(
            f"{description}: "
            f"{result['horizon_ms']} ms "
            f"(score = {result['score']:.4f})"
        )

    lines.append("")

    lines.append("4. ERROR ANALYSIS")

    for _, row in df.iterrows():

        horizon = int(row["horizon_ms"])

        lines.append(
            f"{horizon} ms: "
            f"{int(row['false_positive'])} false positives, "
            f"{int(row['false_negative'])} missed collisions."
        )

    lines.append("")

    lines.append("5. VALIDATION VS TEST AP")

    for _, row in df.iterrows():

        lines.append(
            f"{int(row['horizon_ms'])} ms: "
            f"Validation AP = {row['best_validation_ap']:.4f}, "
            f"Test AP = {row['average_precision']:.4f}, "
            f"Difference = "
            f"{row['validation_test_ap_difference']:.4f}"
        )

    lines.append("")

    lines.append("6. INTERPRETATION")

    lines.append(
        "The 500 ms model achieved the highest test accuracy "
        "and ROC-AUC among the evaluated horizons."
    )

    lines.append(
        "The 1000 ms model achieved the highest recall, "
        "F1-score and Average Precision."
    )

    lines.append(
        "The 1500 ms model showed comparatively weaker "
        "test discrimination."
    )

    lines.append("")

    lines.append("7. LIMITATIONS")

    lines.append(
        "The test sample sizes are small, ranging from "
        "27 to 28 samples per horizon."
    )

    lines.append(
        "Therefore, the reported metrics have substantial "
        "uncertainty and should be interpreted cautiously."
    )

    lines.append(
        "Further evaluation on a larger independent video "
        "dataset is recommended."
    )

    lines.append(
        "No claim of statistical significance or real-world "
        "deployment readiness is made."
    )

    lines.append("")

    lines.append("END OF REPORT")

    return "\n".join(lines)


def main():

    print("=" * 65)
    print("PHASE 5.10 - FINAL RESEARCH ANALYSIS")
    print("=" * 65)

    test, training = load_data()

    combined = create_analysis(test, training)

    best = determine_best_models(test)

    report = generate_report(combined, best)

    # Save detailed table.
    combined.to_csv(
        OUTPUT_DIR / "final_comparative_results.csv",
        index=False,
    )

    # Save best model comparison.
    with open(
        OUTPUT_DIR / "best_model_comparison.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(best, f, indent=4)

    # Save text report.
    with open(
        OUTPUT_DIR / "final_research_analysis.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    print("\n")
    print(report)

    print("\nSaved files:")

    print(
        OUTPUT_DIR / "final_comparative_results.csv"
    )

    print(
        OUTPUT_DIR / "best_model_comparison.json"
    )

    print(
        OUTPUT_DIR / "final_research_analysis.txt"
    )

    print("\nPHASE 5.10 COMPLETED")


if __name__ == "__main__":
    main()
