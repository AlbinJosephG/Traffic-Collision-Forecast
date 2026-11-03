
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "processed" / "fixed_rate_sequences"
MODEL_DIR = PROJECT_ROOT / "models" / "fixed_rate_lstm"
OUTPUT_DIR = PROJECT_ROOT / "processed" / "phase_5_8_test_evaluation"

TEST_FILE = DATA_DIR / "test.npz"

HORIZONS = [500, 1000, 1500]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# MODEL DEFINITION
# IMPORTANT:
# This architecture must match the Phase 5.7 training model.
# ============================================================


class CollisionLSTM(nn.Module):

    def __init__(
        self,
        input_size=96,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]

        final_hidden = self.dropout(final_hidden)

        return self.classifier(final_hidden).squeeze(1)

# ============================================================
# NPZ DATA LOADING
# ============================================================

def find_key(data, possible_names):

    for name in possible_names:
        if name in data.files:
            return name

    return None


def load_test_data():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    data = np.load(TEST_FILE, allow_pickle=False)

    print("\nAvailable NPZ keys:")
    print(data.files)

    x_key = find_key(
        data,
        ["X", "x", "X_test", "test_X", "sequences"]
    )

    y_key = find_key(
        data,
        ["y", "Y", "y_test", "test_y", "labels"]
    )

    horizon_key = find_key(
        data,
        [
            "horizon_ms",
            "prediction_horizon_ms",
            "horizons",
            "horizon",
        ]
    )

    if x_key is None or y_key is None:
        raise KeyError(
            "Could not identify the feature or label arrays. "
            f"Available keys: {data.files}"
        )

    if horizon_key is None:
        raise KeyError(
            "Could not find the prediction horizon metadata. "
            "We need it to separate the 500, 1000 and 1500 ms "
            "test samples safely."
        )

    X = np.asarray(data[x_key], dtype=np.float32)
    y = np.asarray(data[y_key]).reshape(-1)
    horizons = np.asarray(data[horizon_key]).reshape(-1)

    if len(X) != len(y) or len(X) != len(horizons):
        raise ValueError(
            "The number of sequences, labels and horizon values "
            "does not match."
        )

    if not np.isfinite(X).all():
        raise ValueError("Test features contain NaN or infinity.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    print("\nTest dataset loaded.")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)

    return X, y.astype(np.int64), horizons.astype(np.int64)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(horizon):

    checkpoint_path = (
        MODEL_DIR / f"collision_lstm_{horizon}ms.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # Support common checkpoint dictionary formats.
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise ValueError(
            f"Unexpected checkpoint format: {checkpoint_path}"
        )

    model = CollisionLSTM(
    input_size=96,
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    bidirectional=False,
)

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded {horizon} ms checkpoint.")

    return model


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(model, X):

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)

    probabilities = probabilities.cpu().numpy()

    # Use 0.5 as the fixed classification threshold.
    predictions = (probabilities >= 0.5).astype(np.int64)

    return probabilities, predictions


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, probabilities):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else None
        ),
        "average_precision": (
            average_precision_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else None
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PHASE 5.8 - FINAL LSTM TEST EVALUATION")
    print("=" * 65)

    print("\nDevice:", DEVICE)

    X_test, y_test, horizons = load_test_data()

    all_results = []

    for horizon in HORIZONS:

        print("\n" + "-" * 60)
        print(f"EVALUATING {horizon} ms MODEL")
        print("-" * 60)

        indices = np.where(horizons == horizon)[0]

        if len(indices) == 0:
            raise ValueError(
                f"No test samples found for {horizon} ms."
            )

        X_horizon = X_test[indices]
        y_horizon = y_test[indices]

        print("Test samples:", len(indices))
        print("Negative samples:", int((y_horizon == 0).sum()))
        print("Positive samples:", int((y_horizon == 1).sum()))

        model = load_checkpoint(horizon)

        probabilities, predictions = predict(
            model,
            X_horizon,
        )

        metrics = calculate_metrics(
            y_horizon,
            predictions,
            probabilities,
        )

        metrics["horizon_ms"] = horizon
        metrics["test_samples"] = len(indices)

        all_results.append(metrics)

        print("\nClassification results:")

        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        print("\nDetailed classification report:")
        print(
            classification_report(
                y_horizon,
                predictions,
                labels=[0, 1],
                target_names=["Negative", "Positive"],
                zero_division=0,
            )
        )

        # Save predictions for this horizon.
        prediction_df = pd.DataFrame({
            "original_test_index": indices,
            "horizon_ms": horizon,
            "actual_label": y_horizon,
            "predicted_label": predictions,
            "collision_probability": probabilities,
        })

        prediction_path = (
            OUTPUT_DIR /
            f"predictions_{horizon}ms.csv"
        )

        prediction_df.to_csv(
            prediction_path,
            index=False,
        )

        # Save confusion matrix.
        cm = confusion_matrix(
            y_horizon,
            predictions,
            labels=[0, 1],
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual_Negative", "Actual_Positive"],
            columns=["Predicted_Negative", "Predicted_Positive"],
        )

        cm_df.to_csv(
            OUTPUT_DIR / f"confusion_matrix_{horizon}ms.csv"
        )

        print("\nSaved predictions:", prediction_path)

        del model

    # ========================================================
    # SAVE COMPARISON
    # ========================================================

    results_df = pd.DataFrame(all_results)

    results_df = results_df[
        [
            "horizon_ms",
            "test_samples",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "average_precision",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ]
    ]

    summary_path = OUTPUT_DIR / "test_evaluation_summary.csv"

    results_df.to_csv(
        summary_path,
        index=False,
    )

    json_path = OUTPUT_DIR / "test_evaluation_summary.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=4,
        )

    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("PHASE 5.8 COMPLETED")
    print("=" * 65)

    print("\nFINAL TEST PERFORMANCE")
    print(results_df.to_string(index=False))

    print("\nSaved evaluation files:")
    print(summary_path)
    print(json_path)

    print("\nTotal evaluation time:", round(elapsed, 2), "seconds")

    print("\nTEST EVALUATION FINISHED.")
    print("No model weights were changed.")


if __name__ == "__main__":
    main()
