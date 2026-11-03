
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PHASE 6.2: FEATURE ABLATION - 1000 ms AND 1500 ms
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "processed" / "fixed_rate_sequences"
CHECKPOINT_DIR = ROOT / "models" / "phase_6_2_feature_ablation"
REPORT_DIR = ROOT / "reports" / "phase_6_2_feature_ablation"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.npz"
VAL_FILE = DATA_DIR / "validation.npz"

ORIGINAL_CHECKPOINT = (
    ROOT / "models" / "fixed_rate_lstm" / "collision_lstm_500ms.pt"
)

SEED = 42
SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 12
GRADIENT_CLIP = 1.0

HORIZONS = [1000, 1500]

EXPERIMENTS = {
    "full_96_features": list(range(96)),
    "original_66_features": list(range(66)),
    "missingness_30_features": list(range(66, 96)),
}


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
# MODEL
# ============================================================

class CollisionLSTM(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )

        self.classifier = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        output, _ = self.lstm(x)

        # Use the final time step representation.
        final_output = output[:, -1, :]

        return self.classifier(final_output).squeeze(1)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path, horizon):

    with np.load(path, allow_pickle=True) as data:

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        horizons = data["horizon_ms"].astype(np.int64)

    mask = horizons == horizon

    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        raise ValueError(
            f"No samples found for horizon {horizon} ms in {path}"
        )

    if X.ndim != 3 or X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Unexpected sequence shape for {horizon} ms: {X.shape}"
        )

    if X.shape[2] != 96:
        raise ValueError(
            f"Expected 96 input features, found {X.shape[2]}"
        )

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features found in {path}")

    return X, y


# ============================================================
# LOAD ORIGINAL POSITIVE CLASS WEIGHT
# ============================================================

def get_positive_class_weight(y_train):

    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))

    if positives == 0 or negatives == 0:
        raise ValueError(
            "Training data must contain both positive and negative samples."
        )

    weight = negatives / positives

    print("\nHorizon-specific class-weight calculation:")
    print(f"Negative training samples: {negatives}")
    print(f"Positive training samples: {positives}")
    print(f"Positive class weight: {weight:.6f}")

    return weight


# ============================================================
# TRAIN ONE EXPERIMENT
# ============================================================

def train_experiment(
    horizon,
    experiment_name,
    feature_indices,
    X_train,
    y_train,
    X_val,
    y_val,
    positive_class_weight,
):

    set_seed(SEED)

    input_size = len(feature_indices)

    print("\n" + "=" * 65)
    print(f"HORIZON: {horizon} ms")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"INPUT FEATURES: {input_size}")
    print("=" * 65)

    X_train_selected = X_train[:, :, feature_indices]
    X_val_selected = X_val[:, :, feature_indices]

    train_dataset = TensorDataset(
        torch.tensor(X_train_selected, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val_selected, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CollisionLSTM(input_size).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_class_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    # Same optimizer choice across all six experiments.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_ap = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        # ---------------- TRAINING ----------------

        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            optimizer.step()

            train_losses.append(loss.item())

        # ---------------- VALIDATION ----------------

        model.eval()

        validation_losses = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():

            for batch_X, batch_y in val_loader:

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                probabilities = torch.sigmoid(logits)

                validation_losses.append(loss.item())

                all_labels.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_labels = np.asarray(all_labels).astype(int)
        all_probabilities = np.asarray(all_probabilities)

        val_ap = average_precision_score(
            all_labels,
            all_probabilities,
        )

        predictions = (all_probabilities >= 0.5).astype(int)

        val_f1 = f1_score(
            all_labels,
            predictions,
            zero_division=0,
        )

        val_precision = precision_score(
            all_labels,
            predictions,
            zero_division=0,
        )

        val_recall = recall_score(
            all_labels,
            predictions,
            zero_division=0,
        )

        val_auc = (
            roc_auc_score(all_labels, all_probabilities)
            if len(np.unique(all_labels)) == 2
            else float("nan")
        )

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(validation_losses))

        history.append({
            "horizon_ms": horizon,
            "experiment": experiment_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_ap": val_ap,
            "validation_f1": val_f1,
            "validation_precision": val_precision,
            "validation_recall": val_recall,
            "validation_roc_auc": val_auc,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AP: {val_ap:.4f} | "
            f"Val F1: {val_f1:.4f}"
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
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    # Recalculate metrics using the best validation checkpoint.
    final_probabilities = []

    with torch.no_grad():

        for batch_X, _ in val_loader:

            batch_X = batch_X.to(device)

            logits = model(batch_X)
            probabilities = torch.sigmoid(logits)

            final_probabilities.extend(
                probabilities.cpu().numpy()
            )

    final_probabilities = np.asarray(final_probabilities)
    final_predictions = (
        final_probabilities >= 0.5
    ).astype(int)

    final_f1 = f1_score(
        y_val,
        final_predictions,
        zero_division=0,
    )

    final_precision = precision_score(
        y_val,
        final_predictions,
        zero_division=0,
    )

    final_recall = recall_score(
        y_val,
        final_predictions,
        zero_division=0,
    )

    final_auc = (
        roc_auc_score(y_val, final_probabilities)
        if len(np.unique(y_val)) == 2
        else float("nan")
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"lstm_{experiment_name}_{horizon}ms_horizon_weighted.pt"
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "horizon_ms": horizon,
            "experiment": experiment_name,
            "input_size": input_size,
            "feature_indices": feature_indices,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "best_epoch": best_epoch,
            "best_validation_ap": best_ap,
            "positive_class_weight": positive_class_weight,
            "decision_threshold": 0.5,
            "selection_metric": "validation_average_precision",
        },
        checkpoint_path,
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation AP: {best_ap:.6f}")
    print(f"Validation F1: {final_f1:.6f}")
    print(f"Validation ROC-AUC: {final_auc:.6f}")
    print(f"Saved checkpoint: {checkpoint_path}")

    result = {
        "horizon_ms": horizon,
        "experiment": experiment_name,
        "input_features": input_size,
        "best_epoch": best_epoch,
        "validation_ap": best_ap,
        "validation_f1": final_f1,
        "validation_precision": final_precision,
        "validation_recall": final_recall,
        "validation_roc_auc": final_auc,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "train_negative_samples": int(np.sum(y_train == 0)),
        "train_positive_samples": int(np.sum(y_train == 1)),
        "positive_class_weight": positive_class_weight,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE
    }

    return result, history


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 65)
    print("PHASE 6.2: REMAINING HORIZON FEATURE ABLATION")
    print("=" * 65)

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError("Training or validation NPZ file is missing.")

    X_train_all, y_train_all = None, None

    results = []
    histories = []

    for horizon in HORIZONS:

        print(f"\n\nLOADING DATA FOR {horizon} ms")

        X_train, y_train = load_dataset(TRAIN_FILE, horizon)
        X_val, y_val = load_dataset(VAL_FILE, horizon)

        print(f"Training samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        print(f"Training classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Validation classes: {dict(zip(*np.unique(y_val, return_counts=True)))}")

        positive_weight = get_positive_class_weight(y_train)

        print(f"Positive class weight: {positive_weight:.6f}")

        for experiment_name, feature_indices in EXPERIMENTS.items():

            result, history = train_experiment(
                horizon=horizon,
                experiment_name=experiment_name,
                feature_indices=feature_indices,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                positive_class_weight=positive_weight,
            )

            results.append(result)
            histories.extend(history)

            pd.DataFrame(results).to_csv(
                REPORT_DIR / "ablation_comparison_corrected_1000_1500ms.csv",
                index=False,
            )

            pd.DataFrame(histories).to_csv(
                REPORT_DIR / "ablation_training_history_corrected_1000_1500ms.csv",
                index=False,
            )

    # Save experiment configuration.
    config = {
        "phase": "6.2",
        "description": "Feature ablation at 1000 ms and 1500 ms",
        "horizons_ms": HORIZONS,
        "experiments": {
            name: len(indices)
            for name, indices in EXPERIMENTS.items()
        },
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
        "gradient_clip": GRADIENT_CLIP,
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss",
        "selection_metric": "validation_average_precision",
        "decision_threshold": 0.5,
        "test_data_used": False,
        "note": (
            "The 500 ms results are preserved separately. "
            "This script uses the same optimizer and loss across "
            "the six experiments. Verify original Phase 5.7 "
            "optimizer settings before claiming exact reproduction."
        ),
    }

    with open(
        REPORT_DIR / "ablation_config_corrected.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=4)

    # Combine new results with the previously completed 500 ms results.
    old_results_path = REPORT_DIR / "ablation_comparison.csv"

    if old_results_path.exists():

        old_results = pd.read_csv(old_results_path)
        new_results = pd.DataFrame(results)

        combined = pd.concat(
            [old_results, new_results],
            ignore_index=True,
            sort=False,
        )

        combined.to_csv(
            REPORT_DIR / "ablation_comparison_all_horizons.csv",
            index=False,
        )

        print("\nCombined results include previous 500 ms results.")

    else:

        print(
            "\nWARNING: Previous 500 ms comparison file was not found. "
            "The new 1000 ms and 1500 ms results were saved separately."
        )

    print("\n" + "=" * 65)
    print("PHASE 6.2 REMAINING HORIZONS COMPLETED")
    print("=" * 65)

    print("\nNew results:")
    print(pd.DataFrame(results).to_string(index=False))

    print("\nOutput directory:")
    print(REPORT_DIR)


if __name__ == "__main__":
    main()
