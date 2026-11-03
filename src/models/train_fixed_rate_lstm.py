
"""
PHASE 5.7 - FIXED-RATE LSTM TRAINING

Run from project root:
python .\src\models\train_fixed_rate_lstm.py

Three independent models:
500 ms, 1000 ms, 1500 ms

Test data is NOT loaded during training.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "processed" / "fixed_rate_sequences"

MODEL_DIR = ROOT / "models" / "fixed_rate_lstm"

HISTORY_DIR = (
    ROOT / "processed" / "fixed_rate_training_history"
)

HORIZONS = [500, 1000, 1500]

CONFIG = {
    "input_size": 96,
    "sequence_length": 50,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 16,
    "learning_rate": 0.001,
    "max_epochs": 100,
    "patience": 12,
    "seed": 42,
    "gradient_clip": 1.0,
    "device": "cpu"
}


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. LSTM MODEL
# ============================================================

class CollisionLSTM(nn.Module):

    def __init__(
        self,
        input_size=96,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        logits = self.classifier(last_output)

        return logits.squeeze(-1)


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_dataset(filename):

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:

        print(f"\nAvailable keys in {filename}: {data.files}")

        X = data["X"].copy()
        y = data["y"].copy()

        if "horizon_ms" in data.files:

            horizons = data["horizon_ms"].copy()

        elif "horizon_sec" in data.files:

            horizons = np.rint(
                data["horizon_sec"].copy() * 1000
            ).astype(int)

        else:

            raise ValueError(
                f"Missing horizon information in {filename}"
            )

        if "video_names" in data.files:

            videos = data["video_names"].copy().astype(str)

        elif "video_name" in data.files:

            videos = data["video_name"].copy().astype(str)

        else:

            videos = None

        if "sample_ids" in data.files:

            sample_ids = data["sample_ids"].copy().astype(str)

        else:

            sample_ids = None

    if X.ndim != 3:

        raise ValueError(
            f"Invalid X dimensions: {X.shape}"
        )

    if X.shape[1:] != (50, 96):

        raise ValueError(
            f"Expected (N, 50, 96), received {X.shape}"
        )

    if not np.isfinite(X).all():

        raise ValueError("X contains NaN or infinite values")

    if not np.isin(y, [0, 1]).all():

        raise ValueError("Labels must be binary")

    if not (
        len(X) == len(y) == len(horizons)
    ):

        raise ValueError("Dataset array lengths do not match")

    return {
        "X": X.astype(np.float32),
        "y": y.astype(np.int64),
        "horizons": horizons.astype(np.int64),
        "videos": videos,
        "sample_ids": sample_ids
    }


# ============================================================
# 5. VALIDATION METRICS
# ============================================================

def calculate_metrics(y_true, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {

        "validation_ap": average_precision_score(
            y_true,
            probabilities
        ),

        "validation_accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "validation_precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "validation_recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "validation_f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        )
    }

    if len(np.unique(y_true)) == 2:

        metrics["validation_roc_auc"] = (
            roc_auc_score(y_true, probabilities)
        )

    else:

        metrics["validation_roc_auc"] = np.nan

    return metrics


# ============================================================
# 6. TRAIN ONE HORIZON
# ============================================================

def train_model(
    train_data,
    val_data,
    horizon
):

    print("\n" + "=" * 65)

    print(f"TRAINING LSTM - {horizon} ms")

    print("=" * 65)

    set_seed(CONFIG["seed"])

    # --------------------------------------------------------
    # Filter horizon
    # --------------------------------------------------------

    train_mask = (
        train_data["horizons"] == horizon
    )

    val_mask = (
        val_data["horizons"] == horizon
    )

    X_train = train_data["X"][train_mask]

    y_train = train_data["y"][train_mask]

    X_val = val_data["X"][val_mask]

    y_val = val_data["y"][val_mask]

    if len(X_train) == 0 or len(X_val) == 0:

        raise ValueError(
            f"No samples available for {horizon} ms"
        )

    if len(np.unique(y_train)) != 2:

        raise ValueError(
            "Training data must contain both classes"
        )

    if len(np.unique(y_val)) != 2:

        raise ValueError(
            "Validation data must contain both classes"
        )

    # --------------------------------------------------------
    # Video leakage check
    # --------------------------------------------------------

    if (
        train_data["videos"] is not None
        and val_data["videos"] is not None
    ):

        train_videos = set(
            train_data["videos"][train_mask]
        )

        val_videos = set(
            val_data["videos"][val_mask]
        )

        overlap = train_videos.intersection(
            val_videos
        )

        if overlap:

            raise ValueError(
                f"Video leakage detected: {overlap}"
            )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    negatives = int(np.sum(y_train == 0))

    positives = int(np.sum(y_train == 1))

    print(f"Training samples: {len(X_train)}")

    print(f"Validation samples: {len(X_val)}")

    print(
        f"Training classes: "
        f"Negative={negatives}, Positive={positives}"
    )

    # Positive class weight calculated from training only
    pos_weight_value = negatives / positives

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32
    )

    print(
        f"Positive class weight: {pos_weight_value:.4f}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(

        torch.from_numpy(X_train),

        torch.from_numpy(
            y_train.astype(np.float32)
        )

    )

    generator = torch.Generator()

    generator.manual_seed(CONFIG["seed"])

    train_loader = DataLoader(

        train_dataset,

        batch_size=CONFIG["batch_size"],

        shuffle=True,

        num_workers=0,

        generator=generator

    )

    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------

    device = torch.device(CONFIG["device"])

    model = CollisionLSTM(

        input_size=CONFIG["input_size"],

        hidden_size=CONFIG["hidden_size"],

        num_layers=CONFIG["num_layers"],

        dropout=CONFIG["dropout"]

    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device)
    )

    validation_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=CONFIG["learning_rate"]

    )

    X_val_tensor = torch.from_numpy(
        X_val
    ).to(device)

    y_val_tensor = torch.from_numpy(
        y_val.astype(np.float32)
    ).to(device)

    # --------------------------------------------------------
    # Training variables
    # --------------------------------------------------------

    best_ap = -1.0

    best_epoch = 0

    patience_counter = 0

    history = []

    start_time = time.time()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        MODEL_DIR /
        f"collision_lstm_{horizon}ms.pt"
    )

    # ========================================================
    # 7. EPOCH TRAINING
    # ========================================================

    for epoch in range(
        1,
        CONFIG["max_epochs"] + 1
    ):

        model.train()

        total_loss = 0.0

        total_samples = 0

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)

            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_X)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                CONFIG["gradient_clip"]
            )

            optimizer.step()

            batch_size = len(batch_y)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        train_loss = (
            total_loss / total_samples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val_tensor
            )

            val_loss = validation_criterion(
                val_logits,
                y_val_tensor
            ).item()

            probabilities = torch.sigmoid(
                val_logits
            ).cpu().numpy()

        metrics = calculate_metrics(
            y_val,
            probabilities
        )

        current_ap = metrics["validation_ap"]

        # ----------------------------------------------------
        # Record history
        # ----------------------------------------------------

        record = {

            "horizon_ms": horizon,

            "epoch": epoch,

            "train_loss": train_loss,

            "validation_loss": val_loss,

            **metrics

        }

        history.append(record)

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if current_ap > best_ap + 1e-12:

            best_ap = current_ap

            best_epoch = epoch

            patience_counter = 0

            checkpoint = {

                "model_state_dict": model.state_dict(),

                "horizon_ms": horizon,

                "model_config": {

                    "input_size": CONFIG["input_size"],

                    "hidden_size": CONFIG["hidden_size"],

                    "num_layers": CONFIG["num_layers"],

                    "dropout": CONFIG["dropout"]

                },

                "training_config": CONFIG.copy(),

                "best_epoch": best_epoch,

                "best_validation_ap": best_ap,

                "positive_class_weight": pos_weight_value,

                "decision_threshold": 0.5,

                "selection_metric": "validation_average_precision"

            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Epoch output
        # ----------------------------------------------------

        print(

            f"Epoch {epoch:03d} | "

            f"Train Loss: {train_loss:.4f} | "

            f"Val Loss: {val_loss:.4f} | "

            f"Val AP: {current_ap:.4f} | "

            f"Best AP: {best_ap:.4f}"

        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= CONFIG["patience"]:

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # 8. SAVE TRAINING HISTORY
    # ========================================================

    history_df = pd.DataFrame(history)

    history_path = (
        HISTORY_DIR /
        f"history_{horizon}ms.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    elapsed = time.time() - start_time

    result = {

        "horizon_ms": horizon,

        "train_samples": len(X_train),

        "validation_samples": len(X_val),

        "train_negative": negatives,

        "train_positive": positives,

        "best_epoch": best_epoch,

        "best_validation_ap": best_ap,

        "epochs_completed": len(history),

        "training_seconds": round(elapsed, 2),

        "checkpoint": str(checkpoint_path.relative_to(ROOT)),

        "history": str(history_path.relative_to(ROOT))

    }

    print("\nMODEL COMPLETED")

    print(f"Best epoch: {best_epoch}")

    print(f"Best validation AP: {best_ap:.4f}")

    print(f"Training time: {elapsed:.2f} seconds")

    return result


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 65)

    print("PHASE 5.7 - FIXED-RATE LSTM TRAINING")

    print("=" * 65)

    print("\nLoading training dataset...")

    train_data = load_dataset(
        "train.npz"
    )

    print("\nLoading validation dataset...")

    val_data = load_dataset(
        "validation.npz"
    )

    # Test dataset is deliberately NOT loaded.

    print("\nDATASET INFORMATION")

    print(
        "Training shape:",
        train_data["X"].shape
    )

    print(
        "Validation shape:",
        val_data["X"].shape
    )

    print(
        "\nTest dataset: NOT LOADED"
    )

    results = []

    # --------------------------------------------------------
    # Train three independent models
    # --------------------------------------------------------

    for horizon in HORIZONS:

        result = train_model(

            train_data,

            val_data,

            horizon

        )

        results.append(result)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = pd.DataFrame(results)

    summary_path = (
        HISTORY_DIR /
        "fixed_rate_training_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {

        **CONFIG,

        "horizons_ms": HORIZONS,

        "test_data_loaded": False,

        "selection_metric": "validation_average_precision",

        "early_stopping": True

    }

    config_path = (
        MODEL_DIR /
        "training_config.json"
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

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print("PHASE 5.7 COMPLETED")

    print("=" * 65)

    print(
        summary.to_string(index=False)
    )

    print("\nSAVED CHECKPOINTS")

    for horizon in HORIZONS:

        print(
            MODEL_DIR /
            f"collision_lstm_{horizon}ms.pt"
        )

    print("\nTraining summary:", summary_path)

    print("Configuration:", config_path)

    print("\nTEST DATA REMAINS UNTOUCHED")


if __name__ == "__main__":

    main()
