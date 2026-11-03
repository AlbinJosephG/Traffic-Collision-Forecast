
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
# PHASE 6.4: FINAL TEST EVALUATION
# Selected models from Phase 6.3
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_DIR / "processed" / "fixed_rate_sequences"
CHECKPOINT_DIR = PROJECT_DIR / "models" / "phase_6_3_architecture_comparison"

OUTPUT_DIR = PROJECT_DIR / "reports" / "phase_6_4_final_test_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
THRESHOLD = 0.5
SEED = 42

SELECTED_MODELS = {
    500: "cnn_lstm",
    1000: "cnn_lstm",
    1500: "baseline_lstm",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# MODEL ARCHITECTURES
# Must match Phase 6.3
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
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)

        return self.classifier(final_hidden).squeeze(-1)



class CNNLSTM(nn.Module):

    def __init__(
        self,
        input_size=96,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

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
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.cnn(x)

        x = x.transpose(1, 2)

        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)

        return self.classifier(final_hidden).squeeze(-1)


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def load_checkpoint(path):

    try:
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")

    return checkpoint


def get_model_config(checkpoint):

    config = checkpoint.get("model_config", {})

    return {
        "input_size": int(config.get("input_size", 96)),
        "hidden_size": int(config.get("hidden_size", 64)),
        "num_layers": int(config.get("num_layers", 2)),
        "dropout": float(config.get("dropout", 0.3)),
    }


def find_checkpoint(architecture, horizon):

    expected_name = f"{architecture}_{horizon}ms.pt"

    checkpoint_path = CHECKPOINT_DIR / expected_name

    if checkpoint_path.exists():
        return checkpoint_path

    raise FileNotFoundError(
        f"Expected checkpoint not found: {checkpoint_path}\n"
        f"Please check the Phase 6.3 checkpoint directory."
    )



def build_model(architecture, config):

    if architecture == "baseline_lstm":
        return BaselineLSTM(**config)

    if architecture == "cnn_lstm":
        return CNNLSTM(**config)

    raise ValueError(f"Unsupported architecture: {architecture}")


def load_selected_model(architecture, horizon):

    checkpoint_path = find_checkpoint(architecture, horizon)

    print("\nLoading checkpoint:", checkpoint_path.name)

    checkpoint = load_checkpoint(checkpoint_path)

    saved_horizon = checkpoint.get("horizon_ms")

    if saved_horizon is not None:
        if int(saved_horizon) != int(horizon):
            raise ValueError(
                f"Checkpoint horizon mismatch: expected {horizon}, "
                f"found {saved_horizon}"
            )

    saved_architecture = checkpoint.get("architecture")

    if saved_architecture is not None:
        if str(saved_architecture).lower() != architecture.lower():
            raise ValueError(
                f"Architecture mismatch: expected {architecture}, "
                f"found {saved_architecture}"
            )

    config = get_model_config(checkpoint)

    model = build_model(architecture, config)

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict"),
    )

    if state_dict is None:
        raise KeyError(
            f"No model_state_dict or state_dict in {checkpoint_path.name}"
        )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("Architecture:", architecture)
    print("Horizon:", horizon)
    print("Input size:", config["input_size"])

    return model, checkpoint_path


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data(horizon):

    test_path = TEST_DIR / "test.npz"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    with np.load(test_path, allow_pickle=True) as data:

        required = {"X", "y", "horizon_ms"}

        missing = required.difference(data.files)

        if missing:
            raise KeyError(
                f"Missing required NPZ keys: {sorted(missing)}"
            )

        X_all = data["X"]
        y_all = data["y"]
        horizons_all = data["horizon_ms"]

        video_names_all = (
            data["video_names"]
            if "video_names" in data.files
            else np.array(["unknown"] * len(y_all))
        )

        sample_ids_all = (
            data["sample_ids"]
            if "sample_ids" in data.files
            else np.array([f"sample_{i}" for i in range(len(y_all))])
        )

    mask = horizons_all.astype(int) == int(horizon)

    X = X_all[mask].astype(np.float32)
    y = y_all[mask].astype(np.int64)

    video_names = video_names_all[mask]
    sample_ids = sample_ids_all[mask]

    if len(X) == 0:
        raise ValueError(
            f"No test samples found for horizon {horizon} ms."
        )

    if X.ndim != 3:
        raise ValueError(f"Expected 3D X array, received {X.shape}")

    if X.shape[1] != 50:
        raise ValueError(
            f"Expected sequence length 50, received {X.shape[1]}"
        )

    if not np.isfinite(X).all():
        raise ValueError("Test features contain NaN or infinite values.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Test labels must contain only 0 and 1.")

    print("\nTest data:", horizon, "ms")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

    return X, y, video_names, sample_ids


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(model, X):

    X_tensor = torch.from_numpy(X).float()

    all_probabilities = []

    with torch.no_grad():

        for start in range(0, len(X_tensor), 32):

            batch = X_tensor[start:start + 32].to(DEVICE)

            logits = model(batch)

            probabilities = torch.sigmoid(logits)

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    return np.asarray(all_probabilities, dtype=np.float64)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities):

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "test_samples": int(len(y_true)),
        "negative_samples": int(np.sum(y_true == 0)),
        "positive_samples": int(np.sum(y_true == 1)),
        "decision_threshold": THRESHOLD,

        "accuracy": accuracy_score(y_true, predictions),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),

        "average_precision": (
            average_precision_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics, predictions


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 6.4: FINAL TEST EVALUATION")
    print("=" * 70)

    print("\nSelected architectures:")

    for horizon, architecture in SELECTED_MODELS.items():
        print(f"  {horizon} ms: {architecture}")

    summary_rows = []
    prediction_frames = []
    confusion_frames = []

    for horizon, architecture in SELECTED_MODELS.items():

        print("\n" + "=" * 70)
        print(f"EVALUATING {architecture.upper()} - {horizon} ms")
        print("=" * 70)

        X, y, video_names, sample_ids = load_test_data(horizon)

        model, checkpoint_path = load_selected_model(
            architecture,
            horizon,
        )

        probabilities = predict(model, X)

        metrics, predictions = calculate_metrics(
            y,
            probabilities,
        )

        row = {
            "architecture": architecture,
            "horizon_ms": horizon,
            "checkpoint": checkpoint_path.name,
            **metrics,
        }

        summary_rows.append(row)

        prediction_df = pd.DataFrame({
            "sample_id": sample_ids,
            "video_name": video_names,
            "horizon_ms": horizon,
            "architecture": architecture,
            "true_label": y,
            "predicted_probability": probabilities,
            "predicted_label": predictions,
            "decision_threshold": THRESHOLD,
        })

        prediction_frames.append(prediction_df)

        confusion_df = pd.DataFrame(
            [
                ["Actual Negative", int(metrics["true_negative"]),
                 int(metrics["false_positive"])],
                ["Actual Positive", int(metrics["false_negative"]),
                 int(metrics["true_positive"])],
            ],
            columns=[
                "actual_class",
                "predicted_negative",
                "predicted_positive",
            ],
        )

        confusion_df.insert(0, "horizon_ms", horizon)
        confusion_df.insert(1, "architecture", architecture)

        confusion_frames.append(confusion_df)

        print("\nTEST METRICS")

        for key, value in metrics.items():

            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

        del model

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_DIR / "final_test_evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions_path = OUTPUT_DIR / "final_test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    confusion_df = pd.concat(
        confusion_frames,
        ignore_index=True,
    )

    confusion_path = OUTPUT_DIR / "final_test_confusion_matrices.csv"
    confusion_df.to_csv(confusion_path, index=False)

    json_path = OUTPUT_DIR / "final_test_evaluation_summary.json"

    with open(json_path, "w", encoding="utf-8") as file:

        json.dump(
            summary_rows,
            file,
            indent=4,
            allow_nan=True,
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 6.4 COMPLETED")
    print("=" * 70)

    display_columns = [
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

    print("\nFINAL TEST RESULTS:\n")

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nOUTPUT FILES:")

    print(summary_path)
    print(predictions_path)
    print(confusion_path)
    print(json_path)

    print("\nIMPORTANT:")
    print("Test data was used only for final evaluation.")
    print("No model training or threshold tuning was performed.")
    print("Do not use these test results to revise model selection.")

    print("\nPHASE 6.4 FINISHED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
