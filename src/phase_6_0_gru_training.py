
import os
import json
import random
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# PHASE 6.0 - GRU MODEL TRAINING
# Uses TRAIN and VALIDATION only. Does NOT load the test set.
# ============================================================

PROJECT_DIR = Path(r"E:\TrafficCollisionProject")

DATA_DIR = PROJECT_DIR / "processed" / "fixed_rate_sequences"

MODEL_DIR = PROJECT_DIR / "models" / "fixed_rate_gru"

HISTORY_DIR = (
    PROJECT_DIR / "processed" / "fixed_rate_gru_training_history"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [500, 1000, 1500]

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 10

INPUT_SIZE = 96
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("=" * 65)
print("PHASE 6.0 - GRU MODEL TRAINING")
print("=" * 65)

print("Device:", DEVICE)
print("Project directory:", PROJECT_DIR)


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        return self.X[index], self.y[index]


# ============================================================
# GRU MODEL
# ============================================================

class CollisionGRU(nn.Module):

    def __init__(
        self,
        input_size=96,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        # Final time-step representation
        final_features = output[:, -1, :]

        final_features = self.dropout(final_features)

        logits = self.classifier(final_features)

        return logits.squeeze(1)


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_split(split_name):

    file_path = DATA_DIR / f"{split_name}.npz"

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    data = np.load(file_path, allow_pickle=True)

    required_keys = ["X", "y", "horizon_ms"]

    missing = [
        key for key in required_keys
        if key not in data.files
    ]

    if missing:

        raise ValueError(
            f"{split_name}.npz is missing keys: {missing}. "
            f"Available keys: {data.files}"
        )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    horizons = data["horizon_ms"].astype(np.int64)

    if X.ndim != 3:

        raise ValueError(
            f"{split_name}: expected 3D X array, got {X.shape}"
        )

    if X.shape[1] != 50 or X.shape[2] != INPUT_SIZE:

        raise ValueError(
            f"{split_name}: expected X shape (N, 50, 96), "
            f"got {X.shape}"
        )

    if len(X) != len(y) or len(y) != len(horizons):

        raise ValueError(
            f"{split_name}: X, y, and horizon lengths differ."
        )

    if not np.isfinite(X).all():

        raise ValueError(
            f"{split_name}: X contains NaN or infinite values."
        )

    if not np.isin(y, [0, 1]).all():

        raise ValueError(
            f"{split_name}: labels must be 0 or 1."
        )

    print(
        f"{split_name}: X={X.shape}, "
        f"y={y.shape}, "
        f"classes={dict(zip(*np.unique(y, return_counts=True)))}"
    )

    return X, y, horizons


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "average_precision": average_precision_score(
            y_true,
            probabilities
        )
    }

    if len(np.unique(y_true)) == 2:

        metrics["roc_auc"] = roc_auc_score(
            y_true,
            probabilities
        )

    else:

        metrics["roc_auc"] = float("nan")

    return metrics


# ============================================================
# TRAINING AND VALIDATION
# ============================================================

def train_one_horizon(
    X_train,
    y_train,
    X_val,
    y_val,
    horizon
):

    print("\n" + "=" * 65)
    print(f"TRAINING GRU - {horizon} ms")
    print("=" * 65)

    train_dataset = SequenceDataset(
        X_train,
        y_train
    )

    val_dataset = SequenceDataset(
        X_val,
        y_val
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = CollisionGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0 or negative_count == 0:

        raise ValueError(
            f"{horizon} ms training data must contain "
            "both classes."
        )

    pos_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_ap = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        # -------------------------
        # TRAINING
        # -------------------------

        model.train()

        running_loss = 0.0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_X)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item() * len(batch_y)

        train_loss = (
            running_loss / len(train_dataset)
        )

        # -------------------------
        # VALIDATION
        # -------------------------

        model.eval()

        val_probabilities = []

        with torch.no_grad():

            for batch_X, batch_y in val_loader:

                batch_X = batch_X.to(DEVICE)

                logits = model(batch_X)

                probabilities = torch.sigmoid(
                    logits
                )

                val_probabilities.extend(
                    probabilities.cpu().numpy()
                )

        val_probabilities = np.asarray(
            val_probabilities
        )

        val_metrics = calculate_metrics(
            y_val,
            val_probabilities
        )

        val_ap = val_metrics["average_precision"]

        history_row = {

            "epoch": epoch,

            "train_loss": train_loss,

            "validation_accuracy": val_metrics["accuracy"],

            "validation_precision": val_metrics["precision"],

            "validation_recall": val_metrics["recall"],

            "validation_f1": val_metrics["f1"],

            "validation_roc_auc": val_metrics["roc_auc"],

            "validation_average_precision": val_ap
        }

        history.append(history_row)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val AP: {val_ap:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # -------------------------
        # EARLY STOPPING
        # -------------------------

        if val_ap > best_ap:

            best_ap = val_ap
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch}."
            )

            break

    if best_state is None:

        raise RuntimeError(
            f"No valid checkpoint was produced for {horizon} ms."
        )

    # Restore best validation checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Recalculate metrics from the best checkpoint
    best_probabilities = []

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(DEVICE)

            logits = model(batch_X)

            probabilities = torch.sigmoid(logits)

            best_probabilities.extend(
                probabilities.cpu().numpy()
            )

    best_probabilities = np.asarray(
        best_probabilities
    )

    best_metrics = calculate_metrics(
        y_val,
        best_probabilities
    )

    # Save checkpoint
    checkpoint_path = (
        MODEL_DIR / f"collision_gru_{horizon}ms.pt"
    )

    torch.save(
        {
            "model_state_dict": best_state,

            "horizon_ms": horizon,

            "input_size": INPUT_SIZE,

            "hidden_size": HIDDEN_SIZE,

            "num_layers": NUM_LAYERS,

            "dropout": DROPOUT,

            "bidirectional": False,

            "best_epoch": best_epoch,

            "best_validation_average_precision": best_ap,

            "seed": SEED
        },
        checkpoint_path
    )

    # Save training history
    history_path = (
        HISTORY_DIR / f"history_gru_{horizon}ms.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    print("\nBest checkpoint saved:", checkpoint_path)
    print("Best epoch:", best_epoch)
    print("Best validation AP:", round(best_ap, 6))

    return {

        "model": "GRU",

        "horizon_ms": horizon,

        "train_samples": len(y_train),

        "validation_samples": len(y_val),

        "best_epoch": best_epoch,

        "epochs_completed": len(history),

        "best_validation_average_precision": best_metrics[
            "average_precision"
        ],

        "validation_accuracy": best_metrics["accuracy"],

        "validation_precision": best_metrics["precision"],

        "validation_recall": best_metrics["recall"],

        "validation_f1": best_metrics["f1"],

        "validation_roc_auc": best_metrics["roc_auc"],

        "checkpoint": str(checkpoint_path)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    X_train, y_train, h_train = load_split("train")

    X_val, y_val, h_val = load_split("validation")

    results = []

    for horizon in HORIZONS:

        train_mask = h_train == horizon
        val_mask = h_val == horizon

        if not train_mask.any():

            raise ValueError(
                f"No training samples for {horizon} ms."
            )

        if not val_mask.any():

            raise ValueError(
                f"No validation samples for {horizon} ms."
            )

        result = train_one_horizon(

            X_train[train_mask],
            y_train[train_mask],

            X_val[val_mask],
            y_val[val_mask],

            horizon
        )

        results.append(result)

    summary = pd.DataFrame(results)

    summary_path = (
        HISTORY_DIR / "gru_training_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    config = {

        "model": "GRU",

        "input_size": INPUT_SIZE,

        "sequence_length": 50,

        "hidden_size": HIDDEN_SIZE,

        "num_layers": NUM_LAYERS,

        "dropout": DROPOUT,

        "bidirectional": False,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "max_epochs": MAX_EPOCHS,

        "early_stopping_patience": PATIENCE,

        "selection_metric": "validation_average_precision",

        "threshold_for_reported_classification_metrics": 0.5,

        "seed": SEED,

        "device": str(DEVICE),

        "test_dataset_used": False
    }

    config_path = (
        MODEL_DIR / "gru_training_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("PHASE 6.0 COMPLETED")
    print("=" * 65)

    print("\nGRU validation summary:")

    print(
        summary[
            [
                "horizon_ms",
                "best_epoch",
                "best_validation_average_precision",
                "validation_f1",
                "validation_roc_auc"
            ]
        ].to_string(index=False)
    )

    print("\nSummary saved:", summary_path)
    print("Configuration saved:", config_path)

    print("\nIMPORTANT:")
    print("The test dataset was not loaded or evaluated.")
    print("Your Phase 5.8 test results remain untouched.")


if __name__ == "__main__":

    main()
