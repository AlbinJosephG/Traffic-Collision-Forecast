
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PHASE 6.3: ARCHITECTURE COMPARISON
# Baseline LSTM vs CNN-LSTM vs Attention-LSTM
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "processed" / "fixed_rate_sequences"

MODEL_DIR = ROOT / "models" / "phase_6_3_architecture_comparison"
REPORT_DIR = ROOT / "reports" / "phase_6_3_architecture_comparison"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.npz"
VAL_FILE = DATA_DIR / "validation.npz"

HORIZONS = [500, 1000, 1500]

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 96
SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 100
PATIENCE = 12
GRADIENT_CLIP = 1.0

ARCHITECTURES = [
    "baseline_lstm",
    "cnn_lstm",
    "attention_lstm",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL 1: BASELINE LSTM
# Matches the existing Phase 5.7 architecture.
# ============================================================

class BaselineLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            HIDDEN_SIZE,
            1,
        )

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]

        final_hidden = self.dropout(final_hidden)

        return self.classifier(final_hidden).squeeze(1)


# ============================================================
# MODEL 2: CNN-LSTM
# CNN extracts local temporal patterns.
# LSTM learns longer-term dependencies.
# ============================================================

class CNNLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels=INPUT_SIZE,
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
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            HIDDEN_SIZE,
            1,
        )

    def forward(self, x):

        # Input: batch x time x features
        # Conv1d expects: batch x channels x time

        x = x.transpose(1, 2)

        x = self.cnn(x)

        # Restore: batch x time x channels
        x = x.transpose(1, 2)

        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]

        final_hidden = self.dropout(final_hidden)

        return self.classifier(final_hidden).squeeze(1)


# ============================================================
# MODEL 3: ATTENTION-BASED LSTM
# Learns the importance of individual time steps.
# ============================================================

class AttentionLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
            bidirectional=False,
        )

        self.attention = nn.Linear(
            HIDDEN_SIZE,
            1,
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            HIDDEN_SIZE,
            1,
        )

    def forward(self, x):

        # LSTM output:
        # batch x sequence_length x hidden_size

        outputs, _ = self.lstm(x)

        # Calculate attention score for each time step.
        scores = self.attention(outputs).squeeze(-1)

        # Normalize attention weights.
        weights = torch.softmax(scores, dim=1)

        # Weighted temporal representation.
        context = torch.sum(
            outputs * weights.unsqueeze(-1),
            dim=1,
        )

        context = self.dropout(context)

        return self.classifier(context).squeeze(1)


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(architecture):

    if architecture == "baseline_lstm":
        return BaselineLSTM()

    if architecture == "cnn_lstm":
        return CNNLSTM()

    if architecture == "attention_lstm":
        return AttentionLSTM()

    raise ValueError(f"Unknown architecture: {architecture}")


# ============================================================
# LOAD DATA
# ============================================================

def load_horizon_data(path, horizon):

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        horizons = data["horizon_ms"].astype(np.int64)

    mask = horizons == horizon

    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        raise ValueError(
            f"No samples found for horizon {horizon} ms."
        )

    if X.ndim != 3:
        raise ValueError(f"Unexpected input shape: {X.shape}")

    if X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"Expected 50 frames, found {X.shape[1]}")

    if X.shape[2] != INPUT_SIZE:
        raise ValueError(f"Expected 96 features, found {X.shape[2]}")

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinity.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary.")

    return X, y


# ============================================================
# CLASS WEIGHT
# Calculate separately for each horizon using training data.
# ============================================================

def calculate_positive_weight(y):

    negatives = int(np.sum(y == 0))
    positives = int(np.sum(y == 1))

    if negatives == 0 or positives == 0:
        raise ValueError("Both classes must be present in training data.")

    weight = negatives / positives

    return weight, negatives, positives


# ============================================================
# EVALUATION
# ============================================================

def calculate_metrics(y_true, probabilities):

    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
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

        "average_precision": average_precision_score(
            y_true,
            probabilities,
        ),

        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
    }

    return metrics


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

def predict_validation(model, loader):

    model.eval()

    labels = []
    probabilities = []

    with torch.no_grad():

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)

            logits = model(batch_X)

            probs = torch.sigmoid(logits)

            labels.extend(batch_y.numpy())
            probabilities.extend(probs.cpu().numpy())

    return (
        np.asarray(labels).astype(int),
        np.asarray(probabilities),
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    architecture,
    horizon,
    X_train,
    y_train,
    X_val,
    y_val,
    positive_weight,
    negative_count,
    positive_count,
):

    set_seed(SEED)

    print("\n" + "=" * 70)
    print(f"ARCHITECTURE: {architecture}")
    print(f"HORIZON: {horizon} ms")
    print("=" * 70)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = create_model(architecture).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_ap = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        # ---------------- TRAINING ----------------

        model.train()

        training_losses = []

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_X)

            loss = criterion(logits, batch_y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            optimizer.step()

            training_losses.append(loss.item())

        # ---------------- VALIDATION ----------------

        y_true, probabilities = predict_validation(
            model,
            val_loader,
        )

        val_metrics = calculate_metrics(
            y_true,
            probabilities,
        )

        val_ap = val_metrics["average_precision"]

        train_loss = float(np.mean(training_losses))

        history.append({
            "architecture": architecture,
            "horizon_ms": horizon,
            "epoch": epoch,
            "training_loss": train_loss,
            "validation_ap": val_ap,
            "validation_auc": val_metrics["roc_auc"],
            "validation_f1": val_metrics["f1"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AP: {val_ap:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # ---------------- EARLY STOPPING ----------------

        if val_ap > best_ap:

            best_ap = val_ap
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    # Evaluate the best checkpoint.
    y_true, probabilities = predict_validation(
        model,
        val_loader,
    )

    metrics = calculate_metrics(
        y_true,
        probabilities,
    )

    checkpoint_path = (
        MODEL_DIR /
        f"{architecture}_{horizon}ms.pt"
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": architecture,
            "horizon_ms": horizon,
            "input_size": INPUT_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "best_epoch": best_epoch,
            "best_validation_ap": best_ap,
            "positive_class_weight": positive_weight,
            "training_negative_samples": negative_count,
            "training_positive_samples": positive_count,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "selection_metric": "validation_average_precision",
            "decision_threshold": 0.5,
        },
        checkpoint_path,
    )

    result = {
        "architecture": architecture,
        "horizon_ms": horizon,
        "input_features": INPUT_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "train_negative_samples": negative_count,
        "train_positive_samples": positive_count,
        "positive_class_weight": positive_weight,
        "best_epoch": best_epoch,
        "validation_accuracy": metrics["accuracy"],
        "validation_precision": metrics["precision"],
        "validation_recall": metrics["recall"],
        "validation_f1": metrics["f1"],
        "validation_roc_auc": metrics["roc_auc"],
        "validation_average_precision": metrics["average_precision"],
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }

    print("\nBest model results:")
    print(json.dumps(result, indent=4))

    del model

    return result, history


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("PHASE 6.3 - ARCHITECTURE COMPARISON")
    print("=" * 70)

    print("Device:", DEVICE)

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError("Training or validation dataset is missing.")

    all_results = []
    all_histories = []

    for horizon in HORIZONS:

        print("\n\n" + "#" * 70)
        print(f"PROCESSING HORIZON: {horizon} ms")
        print("#" * 70)

        X_train, y_train = load_horizon_data(
            TRAIN_FILE,
            horizon,
        )

        X_val, y_val = load_horizon_data(
            VAL_FILE,
            horizon,
        )

        positive_weight, negatives, positives = (
            calculate_positive_weight(y_train)
        )

        print("\nDataset information:")
        print("Training shape:", X_train.shape)
        print("Validation shape:", X_val.shape)
        print("Training negatives:", negatives)
        print("Training positives:", positives)
        print("Positive class weight:", positive_weight)

        for architecture in ARCHITECTURES:

            result, history = train_model(
                architecture=architecture,
                horizon=horizon,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                positive_weight=positive_weight,
                negative_count=negatives,
                positive_count=positives,
            )

            all_results.append(result)
            all_histories.extend(history)

            # Save incrementally.
            pd.DataFrame(all_results).to_csv(
                REPORT_DIR / "architecture_comparison.csv",
                index=False,
            )

            pd.DataFrame(all_histories).to_csv(
                REPORT_DIR / "architecture_training_history.csv",
                index=False,
            )

    # ========================================================
    # SAVE CONFIGURATION
    # ========================================================

    config = {
        "phase": "6.3",
        "architectures": ARCHITECTURES,
        "horizons_ms": HORIZONS,
        "input_size": INPUT_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "gradient_clip": GRADIENT_CLIP,
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss",
        "class_weighting": "Horizon-specific training-label ratio",
        "seed": SEED,
        "selection_metric": "validation_average_precision",
        "decision_threshold": 0.5,
        "test_data_used": False,
        "baseline_retrained": True,
    }

    with open(
        REPORT_DIR / "architecture_comparison_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(config, file, indent=4)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        ["horizon_ms", "validation_average_precision"],
        ascending=[True, False],
    )

    results_df.to_csv(
        REPORT_DIR / "architecture_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("PHASE 6.3 TRAINING COMPLETED")
    print("=" * 70)

    print("\nARCHITECTURE COMPARISON:")

    print(
        results_df[
            [
                "architecture",
                "horizon_ms",
                "best_epoch",
                "validation_average_precision",
                "validation_roc_auc",
                "validation_f1",
            ]
        ].to_string(index=False)
    )

    print("\nModels saved to:")
    print(MODEL_DIR)

    print("\nReports saved to:")
    print(REPORT_DIR)

    print("\nTest data was not used.")
    print("PHASE 6.3 COMPLETED.")


if __name__ == "__main__":
    main()
