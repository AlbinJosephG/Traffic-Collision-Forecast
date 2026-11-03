
from pathlib import Path
import csv

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from lstm_model import CollisionLSTM


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "processed" / "event_aligned_sequences"
MODEL_DIR = ROOT / "models" / "lstm"
OUTPUT_DIR = ROOT / "processed" / "evaluation"

HORIZONS = [0.5, 1.0, 1.5]
THRESHOLD = 0.5

DEVICE = torch.device("cpu")


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():
    test_path = DATA_DIR / "test.npz"

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    data = np.load(test_path)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    horizons = data["horizon_sec"].astype(np.float32)
    video_names = data["video_names"].astype(str)
    sample_ids = data["sample_ids"].astype(str)

    print("\nTEST DATA")
    print("-" * 60)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Horizon shape:", horizons.shape)
    print("Number of videos:", len(video_names))
    print("Device:", DEVICE)

    return X, y, horizons, video_names, sample_ids


# ============================================================
# EVALUATE ONE HORIZON
# ============================================================

def evaluate_horizon(
    horizon,
    X,
    y,
    horizons,
    video_names,
    sample_ids,
):
    horizon_ms = int(round(horizon * 1000))

    print("\n" + "=" * 60)
    print(f"TEST EVALUATION: {horizon_ms} MS")
    print("=" * 60)

    # Select only samples belonging to this horizon.
    indices = np.where(np.isclose(horizons, horizon))[0]

    if len(indices) == 0:
        print("No test samples found for this horizon.")
        return None, []

    X_h = X[indices]
    y_h = y[indices]

    print(f"Test samples: {len(y_h)}")
    print(f"Positive: {int((y_h == 1).sum())}")
    print(f"Negative: {int((y_h == 0).sum())}")

    # Load the corresponding checkpoint.
    checkpoint_path = MODEL_DIR / f"lstm_{horizon_ms}ms.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model = CollisionLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Convert test sequences to tensors.
    X_tensor = torch.tensor(X_h, dtype=torch.float32).to(DEVICE)

    # Generate probabilities.
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    # Apply the fixed threshold.
    predictions = (probabilities >= THRESHOLD).astype(int)

    # Calculate metrics.
    accuracy = accuracy_score(y_h, predictions)
    precision = precision_score(
        y_h, predictions, zero_division=0
    )
    recall = recall_score(
        y_h, predictions, zero_division=0
    )
    f1 = f1_score(
        y_h, predictions, zero_division=0
    )

    ap = average_precision_score(y_h, probabilities)

    roc_auc = (
        roc_auc_score(y_h, probabilities)
        if len(np.unique(y_h)) == 2
        else float("nan")
    )

    tn, fp, fn, tp = confusion_matrix(
        y_h, predictions, labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "horizon_ms": horizon_ms,
        "test_samples": len(y_h),
        "positive_samples": int((y_h == 1).sum()),
        "negative_samples": int((y_h == 0).sum()),
        "threshold": THRESHOLD,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "average_precision": ap,
        "roc_auc": roc_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_ap": float(
            checkpoint["best_validation_ap"]
        ),
    }

    print("\nMETRICS")
    print("-" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")

    print("\nCONFUSION MATRIX")
    print("-" * 60)
    print("                 Predicted")
    print("                 Negative  Positive")
    print(f"Actual Negative  {tn:8d}  {fp:8d}")
    print(f"Actual Positive  {fn:8d}  {tp:8d}")

    # Keep individual predictions for later error analysis.
    prediction_rows = []

    for i in range(len(y_h)):
        prediction_rows.append({
            "video_name": video_names[indices[i]],
            "sample_id": sample_ids[indices[i]],
            "horizon_ms": horizon_ms,
            "actual_label": int(y_h[i]),
            "predicted_label": int(predictions[i]),
            "collision_probability": float(probabilities[i]),
            "correct": bool(predictions[i] == y_h[i]),
        })

    return metrics, prediction_rows


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, horizons, video_names, sample_ids = load_test_data()

    all_metrics = []
    all_predictions = []

    for horizon in HORIZONS:
        metrics, predictions = evaluate_horizon(
            horizon,
            X,
            y,
            horizons,
            video_names,
            sample_ids,
        )

        if metrics is not None:
            all_metrics.append(metrics)
            all_predictions.extend(predictions)

    # Save evaluation metrics.
    metrics_path = OUTPUT_DIR / "lstm_test_metrics.csv"

    if all_metrics:
        with open(
            metrics_path, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=all_metrics[0].keys()
            )
            writer.writeheader()
            writer.writerows(all_metrics)

    # Save individual predictions.
    predictions_path = OUTPUT_DIR / "lstm_test_predictions.csv"

    if all_predictions:
        with open(
            predictions_path, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=all_predictions[0].keys()
            )
            writer.writeheader()
            writer.writerows(all_predictions)

    print("\n" + "=" * 60)
    print("TEST EVALUATION COMPLETED")
    print("=" * 60)
    print("Metrics:", metrics_path)
    print("Predictions:", predictions_path)


if __name__ == "__main__":
    main()
