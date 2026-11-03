from pathlib import Path
import json
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
# PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = PROJECT_ROOT / "processed" / "fixed_rate_sequences" / "test.npz"

CHECKPOINT_DIR = (
    PROJECT_ROOT / "models" / "phase_6_3_architecture_comparison"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "phase_6_4C_remaining_architecture_test_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 96
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
SEQUENCE_LENGTH = 50

DECISION_THRESHOLD = 0.5

# Only the three models that have not yet been evaluated on the test set.
MODELS_TO_EVALUATE = [
    ("baseline_lstm", 500),
    ("baseline_lstm", 1000),
    ("cnn_lstm", 1500),
]


# ============================================================
# MODEL DEFINITIONS
# These must match the Phase 6.3 checkpoint architectures.
# ============================================================

class BaselineLSTM(nn.Module):
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
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)

        return self.classifier(final_hidden).squeeze(1)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        input_size=96,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        # Keep this Sequential structure to match Phase 6.3
        # checkpoint keys: cnn.0.*, cnn.2.*
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Input: batch x time x features
        x = x.transpose(1, 2)

        # CNN input: batch x features x time
        x = self.cnn(x)

        # LSTM input: batch x time x channels
        x = x.transpose(1, 2)

        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)

        return self.classifier(final_hidden).squeeze(1)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 70)
print("PHASE 6.4C - REMAINING ARCHITECTURE TEST EVALUATION")
print("=" * 70)

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_FILE}"
    )

test_data = np.load(TEST_FILE, allow_pickle=True)

required_keys = {"X", "y", "horizon_ms"}

missing_keys = required_keys.difference(test_data.files)

if missing_keys:
    raise KeyError(
        f"Missing required keys in test NPZ: {sorted(missing_keys)}"
    )

X_all = test_data["X"].astype(np.float32)
y_all = test_data["y"].astype(np.int64)
horizons_all = np.asarray(test_data["horizon_ms"]).reshape(-1)

if X_all.ndim != 3:
    raise ValueError(
        f"Expected X to have 3 dimensions; received {X_all.shape}"
    )

if X_all.shape[1:] != (SEQUENCE_LENGTH, INPUT_SIZE):
    raise ValueError(
        "Unexpected test feature shape. "
        f"Expected (*, {SEQUENCE_LENGTH}, {INPUT_SIZE}), "
        f"received {X_all.shape}"
    )

if len(X_all) != len(y_all) or len(y_all) != len(horizons_all):
    raise ValueError(
        "The X, y, and horizon_ms arrays have inconsistent lengths."
    )

if not np.isfinite(X_all).all():
    raise ValueError("Test features contain NaN or infinite values.")

print(f"Loaded test dataset: {TEST_FILE}")
print(f"Total test sequences: {len(y_all)}")
print(f"Feature tensor shape: {X_all.shape}")
print(f"Available horizons: {np.unique(horizons_all).tolist()}")
print(f"Evaluation device: {DEVICE}")


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Supports checkpoints containing only a state dictionary.
        state_dict = checkpoint

    return checkpoint, state_dict


# ============================================================
# EVALUATE THE THREE REMAINING MODELS
# ============================================================

summary_rows = []
prediction_frames = []
confusion_rows = []

for architecture, horizon in MODELS_TO_EVALUATE:

    print("\n" + "-" * 70)
    print(f"Evaluating: {architecture} | {horizon} ms")
    print("-" * 70)

    checkpoint_name = f"{architecture}_{horizon}ms.pt"
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name

    checkpoint, state_dict = load_checkpoint(checkpoint_path)

    # Instantiate the exact architecture used in Phase 6.3.
    if architecture == "baseline_lstm":
        model = BaselineLSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
    elif architecture == "cnn_lstm":
        model = CNNLSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint architecture mismatch for {checkpoint_name}.\n"
            "The model definition does not match the saved checkpoint.\n"
            f"Original error:\n{exc}"
        ) from exc

    model.to(DEVICE)
    model.eval()

    # Select only test samples belonging to this prediction horizon.
    horizon_mask = horizons_all.astype(int) == int(horizon)

    X = X_all[horizon_mask]
    y = y_all[horizon_mask]

    if len(y) == 0:
        raise ValueError(
            f"No test samples found for horizon {horizon} ms."
        )

    if len(np.unique(y)) != 2:
        raise ValueError(
            f"Both classes must be present for horizon {horizon} ms."
        )

    X_tensor = torch.from_numpy(X).to(DEVICE)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(np.int64)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(
        y, predictions, zero_division=0
    )
    recall = recall_score(
        y, predictions, zero_division=0
    )
    f1 = f1_score(
        y, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y, probabilities)
    average_precision = average_precision_score(y, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    summary_rows.append({
        "architecture": architecture,
        "horizon_ms": horizon,
        "test_samples": len(y),
        "decision_threshold": DECISION_THRESHOLD,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "checkpoint": checkpoint_name,
    })

    confusion_rows.append({
        "architecture": architecture,
        "horizon_ms": horizon,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    })

    # Preserve sample identifiers when the NPZ contains them.
    if "sample_ids" in test_data.files:
        sample_ids = np.asarray(test_data["sample_ids"])[horizon_mask]
    else:
        sample_ids = np.arange(len(y))

    prediction_frames.append(
        pd.DataFrame({
            "architecture": architecture,
            "horizon_ms": horizon,
            "sample_id": sample_ids,
            "actual_label": y,
            "predicted_probability": probabilities,
            "predicted_label": predictions,
            "decision_threshold": DECISION_THRESHOLD,
        })
    )

    print(f"Test samples:       {len(y)}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1:                 {f1:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"Average Precision:  {average_precision:.4f}")
    print(f"Confusion matrix:   TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# ============================================================
# SAVE RESULTS
# ============================================================

summary_df = pd.DataFrame(summary_rows)
predictions_df = pd.concat(
    prediction_frames,
    ignore_index=True,
)
confusion_df = pd.DataFrame(confusion_rows)

summary_path = OUTPUT_DIR / "remaining_architecture_test_summary.csv"
predictions_path = OUTPUT_DIR / "remaining_architecture_test_predictions.csv"
confusion_path = OUTPUT_DIR / "remaining_architecture_confusion_matrices.csv"
json_path = OUTPUT_DIR / "remaining_architecture_test_summary.json"

summary_df.to_csv(summary_path, index=False)
predictions_df.to_csv(predictions_path, index=False)
confusion_df.to_csv(confusion_path, index=False)

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        summary_rows,
        file,
        indent=4,
    )

print("\n" + "=" * 70)
print("PHASE 6.4C COMPLETED")
print("=" * 70)

print("\nRemaining-model test results:")
print(
    summary_df[
        [
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
    ].to_string(index=False)
)

print("\nSaved files:")
print(summary_path)
print(predictions_path)
print(confusion_path)
print(json_path)

print("\nNo model was retrained. No test-based threshold tuning was performed.")