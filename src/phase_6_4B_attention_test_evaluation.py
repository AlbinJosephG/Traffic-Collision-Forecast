
from pathlib import Path
import json
import random

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
)


# ============================================================
# PHASE 6.4B: ATTENTION-LSTM FINAL TEST EVALUATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_DIR
    / "processed"
    / "fixed_rate_sequences"
    / "test.npz"
)

CHECKPOINT_DIR = (
    PROJECT_DIR
    / "models"
    / "phase_6_3_architecture_comparison"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "phase_6_4B_attention_test_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
THRESHOLD = 0.5
SEED = 42

HORIZONS = [500, 1000, 1500]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# ATTENTION-LSTM ARCHITECTURE
# ============================================================

class AttentionLSTM(nn.Module):

    def __init__(
        self,
        input_size=96,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.attention = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):

        outputs, _ = self.lstm(x)

        attention_scores = self.attention(outputs)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        context = torch.sum(
            outputs * attention_weights,
            dim=1,
        )

        context = self.dropout(context)

        return self.classifier(context).squeeze(-1)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(path):

    try:
        return torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def find_checkpoint(horizon):

    expected_name = f"attention_lstm_{horizon}ms.pt"

    path = CHECKPOINT_DIR / expected_name

    if not path.exists():

        raise FileNotFoundError(
            f"Attention-LSTM checkpoint not found:\n{path}\n"
            "Check the Phase 6.3 checkpoint directory."
        )

    return path


def load_model(horizon):

    checkpoint_path = find_checkpoint(horizon)

    print("\nLoading:", checkpoint_path.name)

    checkpoint = load_checkpoint(checkpoint_path)

    if not isinstance(checkpoint, dict):
        raise ValueError("Unexpected checkpoint format.")

    if checkpoint.get("horizon_ms") is not None:
        if int(checkpoint["horizon_ms"]) != horizon:
            raise ValueError("Checkpoint horizon mismatch.")

    config = checkpoint.get("model_config", {})

    input_size = int(config.get("input_size", 96))
    hidden_size = int(config.get("hidden_size", 64))
    num_layers = int(config.get("num_layers", 2))
    dropout = float(config.get("dropout", 0.3))

    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict"),
    )

    if state_dict is None:
        raise KeyError("Checkpoint has no model state dictionary.")

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model, checkpoint_path


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data(horizon):

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    with np.load(TEST_PATH, allow_pickle=True) as data:

        X_all = data["X"]
        y_all = data["y"]
        horizons_all = data["horizon_ms"]

        videos_all = (
            data["video_names"]
            if "video_names" in data.files
            else np.array(["unknown"] * len(y_all))
        )

        ids_all = (
            data["sample_ids"]
            if "sample_ids" in data.files
            else np.array(
                [f"sample_{i}" for i in range(len(y_all))]
            )
        )

    mask = horizons_all.astype(int) == horizon

    X = X_all[mask].astype(np.float32)
    y = y_all[mask].astype(np.int64)

    videos = videos_all[mask]
    sample_ids = ids_all[mask]

    if len(X) == 0:
        raise ValueError(
            f"No test samples found for {horizon} ms."
        )

    if X.ndim != 3 or X.shape[1:] != (50, 96):
        raise ValueError(
            f"Unexpected test feature shape: {X.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError("Test data contains NaN or infinity.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Unexpected test labels.")

    return X, y, videos, sample_ids


# ============================================================
# PREDICTIONS
# ============================================================

def predict(model, X):

    X_tensor = torch.from_numpy(X).float()

    probabilities = []

    with torch.no_grad():

        for start in range(0, len(X_tensor), 32):

            batch = X_tensor[start:start + 32].to(DEVICE)

            logits = model(batch)

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

    return np.asarray(probabilities, dtype=np.float64)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y, probabilities):

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "test_samples": int(len(y)),
        "negative_samples": int(np.sum(y == 0)),
        "positive_samples": int(np.sum(y == 1)),
        "decision_threshold": THRESHOLD,

        "accuracy": float(
            accuracy_score(y, predictions)
        ),

        "precision": float(
            precision_score(
                y, predictions, zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y, predictions, zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y, predictions, zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),

        "average_precision": float(
            average_precision_score(y, probabilities)
        ),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }, predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 6.4B: ATTENTION-LSTM FINAL TEST EVALUATION")
    print("=" * 70)

    summary_rows = []
    prediction_frames = []
    confusion_frames = []

    for horizon in HORIZONS:

        print("\n" + "=" * 60)
        print(f"ATTENTION-LSTM: {horizon} ms")
        print("=" * 60)

        X, y, videos, sample_ids = load_test_data(horizon)

        print("Test shape:", X.shape)
        print("Class counts:", np.unique(
            y, return_counts=True
        ))

        model, checkpoint_path = load_model(horizon)

        probabilities = predict(model, X)

        metrics, predictions = calculate_metrics(
            y,
            probabilities,
        )

        summary_rows.append({
            "architecture": "attention_lstm",
            "horizon_ms": horizon,
            "checkpoint": checkpoint_path.name,
            **metrics,
        })

        prediction_frames.append(
            pd.DataFrame({
                "sample_id": sample_ids,
                "video_name": videos,
                "horizon_ms": horizon,
                "architecture": "attention_lstm",
                "true_label": y,
                "predicted_probability": probabilities,
                "predicted_label": predictions,
                "decision_threshold": THRESHOLD,
            })
        )

        confusion_frames.append(
            pd.DataFrame(
                [
                    [
                        horizon,
                        "attention_lstm",
                        "Actual Negative",
                        metrics["true_negative"],
                        metrics["false_positive"],
                    ],
                    [
                        horizon,
                        "attention_lstm",
                        "Actual Positive",
                        metrics["false_negative"],
                        metrics["true_positive"],
                    ],
                ],
                columns=[
                    "horizon_ms",
                    "architecture",
                    "actual_class",
                    "predicted_negative",
                    "predicted_positive",
                ],
            )
        )

        print("\nTEST METRICS")

        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

        del model

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "attention_test_summary.csv",
        index=False,
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "attention_test_predictions.csv",
        index=False,
    )

    confusion_df = pd.concat(
        confusion_frames,
        ignore_index=True,
    )

    confusion_df.to_csv(
        OUTPUT_DIR / "attention_test_confusion_matrices.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "attention_test_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary_rows,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    print("\n" + "=" * 70)
    print("ATTENTION-LSTM FINAL TEST RESULTS")
    print("=" * 70)

    columns = [
        "architecture",
        "horizon_ms",
        "test_samples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    print(
        summary_df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved files:")
    print(OUTPUT_DIR)

    print("\nPHASE 6.4B COMPLETED.")


if __name__ == "__main__":
    main()
