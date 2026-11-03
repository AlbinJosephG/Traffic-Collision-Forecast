
import copy
import json
import random
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
    average_precision_score,
    roc_auc_score
)


# ============================================================
# PHASE 6.2 - FEATURE ABLATION STUDY
# Initial experiment: 500 ms LSTM
# Training and validation only. No test data is loaded.
# ============================================================

PROJECT_DIR = Path(r"E:\TrafficCollisionProject")

DATA_DIR = PROJECT_DIR / "processed" / "fixed_rate_sequences"

BASELINE_CHECKPOINT = (
    PROJECT_DIR / "models" / "fixed_rate_lstm"
    / "collision_lstm_500ms.pt"
)

MODEL_DIR = (
    PROJECT_DIR / "models" / "phase_6_2_feature_ablation"
)

REPORT_DIR = (
    PROJECT_DIR / "reports" / "phase_6_2_feature_ablation"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 500

SEQUENCE_LENGTH = 50

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 12
SEED = 42
GRADIENT_CLIP = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EXPERIMENTS = {
    "full_96_features": {
        "indices": list(range(96)),
        "input_size": 96
    },

    "original_66_features": {
        "indices": list(range(66)),
        "input_size": 66
    },

    "missingness_30_features": {
        "indices": list(range(66, 96)),
        "input_size": 30
    }
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed():

    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


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
# LSTM MODEL
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
            bidirectional=False
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            HIDDEN_SIZE,
            1
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        final_features = output[:, -1, :]

        final_features = self.dropout(
            final_features
        )

        logits = self.classifier(
            final_features
        )

        return logits.squeeze(1)


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data(split):

    path = DATA_DIR / f"{split}.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(
        path,
        allow_pickle=True
    )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    horizons = data["horizon_ms"].astype(np.int64)

    if X.shape[1:] != (50, 96):
        raise ValueError(
            f"Unexpected {split} shape: {X.shape}"
        )

    if len(X) != len(y) or len(y) != len(horizons):
        raise ValueError(
            f"Inconsistent {split} dataset lengths."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{split} contains non-finite features."
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"{split} contains invalid labels."
        )

    mask = horizons == HORIZON

    X = X[mask]
    y = y[mask]

    if len(np.unique(y)) != 2:
        raise ValueError(
            f"{split} must contain both classes."
        )

    return X, y


# ============================================================
# LOAD ORIGINAL TRAINING LOSS WEIGHT
# ============================================================

def load_positive_class_weight(y_train):

    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(BASELINE_CHECKPOINT)

    try:

        checkpoint = torch.load(
            BASELINE_CHECKPOINT,
            map_location="cpu",
            weights_only=True
        )

    except TypeError:

        checkpoint = torch.load(
            BASELINE_CHECKPOINT,
            map_location="cpu"
        )

    if "positive_class_weight" in checkpoint:

        value = checkpoint["positive_class_weight"]

        if torch.is_tensor(value):
            value = value.detach().cpu().item()

        value = float(value)

        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                "Invalid positive_class_weight in checkpoint."
            )

        print(
            "Using original LSTM positive class weight:",
            value
        )

        return value

    # Fallback if checkpoint does not contain the weight.
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    value = negatives / positives

    print(
        "Positive class weight calculated from training data:",
        value
    )

    return value


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
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        )
    }

    return metrics


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

def predict_validation(model, X_val):

    model.eval()

    dataset = SequenceDataset(
        X_val,
        np.zeros(len(X_val), dtype=np.float32)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    probabilities = []

    with torch.no_grad():

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)

            logits = model(batch_X)

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy()
            )

    return np.asarray(probabilities)


# ============================================================
# TRAIN ONE ABLATION EXPERIMENT
# ============================================================

def train_experiment(
    experiment_name,
    experiment_config,
    X_train,
    y_train,
    X_val,
    y_val,
    positive_weight
):

    print("\n" + "=" * 65)
    print("EXPERIMENT:", experiment_name)
    print("=" * 65)

    set_seed()

    indices = experiment_config["indices"]

    input_size = experiment_config["input_size"]

    # Select the relevant features.
    X_train_exp = X_train[:, :, indices]
    X_val_exp = X_val[:, :, indices]

    print("Input feature count:", input_size)
    print("Training shape:", X_train_exp.shape)
    print("Validation shape:", X_val_exp.shape)

    train_dataset = SequenceDataset(
        X_train_exp,
        y_train
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = CollisionLSTM(
        input_size=input_size
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight,
            dtype=torch.float32,
            device=DEVICE
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_ap = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        total_loss = 0.0

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
                max_norm=GRADIENT_CLIP
            )

            optimizer.step()

            total_loss += (
                loss.item() * len(batch_y)
            )

        train_loss = (
            total_loss / len(train_dataset)
        )

        # Validation
        val_probabilities = predict_validation(
            model,
            X_val_exp
        )

        val_metrics = calculate_metrics(
            y_val,
            val_probabilities
        )

        val_ap = val_metrics["average_precision"]

        history.append({

            "experiment": experiment_name,

            "epoch": epoch,

            "train_loss": train_loss,

            "validation_ap": val_ap,

            "validation_roc_auc": val_metrics["roc_auc"],

            "validation_f1": val_metrics["f1"]

        })

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val AP: {val_ap:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

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
                "Early stopping at epoch:",
                epoch
            )

            break

    if best_state is None:
        raise RuntimeError(
            f"No checkpoint produced for {experiment_name}"
        )

    # Restore best validation checkpoint.
    model.load_state_dict(best_state)
    model.eval()

    best_probabilities = predict_validation(
        model,
        X_val_exp
    )

    best_metrics = calculate_metrics(
        y_val,
        best_probabilities
    )

    checkpoint_path = (
        MODEL_DIR / f"lstm_{experiment_name}.pt"
    )

    torch.save(
        {
            "model_state_dict": best_state,

            "model_config": {
                "input_size": input_size,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT
            },

            "experiment": experiment_name,

            "horizon_ms": HORIZON,

            "best_epoch": best_epoch,

            "best_validation_ap": best_ap,

            "feature_indices": indices,

            "positive_class_weight": positive_weight,

            "training_config": {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "seed": SEED,
                "gradient_clip": GRADIENT_CLIP
            }
        },
        checkpoint_path
    )

    history_path = (
        REPORT_DIR / f"history_{experiment_name}.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    result = {

        "experiment": experiment_name,

        "horizon_ms": HORIZON,

        "input_features": input_size,

        "train_samples": len(y_train),

        "validation_samples": len(y_val),

        "best_epoch": best_epoch,

        "best_validation_ap": best_metrics[
            "average_precision"
        ],

        "validation_roc_auc": best_metrics[
            "roc_auc"
        ],

        "validation_accuracy": best_metrics[
            "accuracy"
        ],

        "validation_precision": best_metrics[
            "precision"
        ],

        "validation_recall": best_metrics[
            "recall"
        ],

        "validation_f1": best_metrics[
            "f1"
        ],

        "checkpoint": str(checkpoint_path)
    }

    print("\nBest validation AP:", round(best_ap, 6))
    print("Best epoch:", best_epoch)
    print("Checkpoint:", checkpoint_path)

    return result, history


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 6.2 - FEATURE ABLATION STUDY")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Horizon:", HORIZON, "ms")

    # Only training and validation datasets are loaded.
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("validation")

    print("\nTraining samples:", len(y_train))
    print("Validation samples:", len(y_val))

    print(
        "Training classes:",
        dict(zip(*np.unique(
            y_train,
            return_counts=True
        )))
    )

    print(
        "Validation classes:",
        dict(zip(*np.unique(
            y_val,
            return_counts=True
        )))
    )

    positive_weight = load_positive_class_weight(
        y_train
    )

    all_results = []
    all_histories = []

    for experiment_name, experiment_config in EXPERIMENTS.items():

        result, history = train_experiment(

            experiment_name,

            experiment_config,

            X_train,
            y_train,

            X_val,
            y_val,

            positive_weight
        )

        all_results.append(result)
        all_histories.extend(history)

    # Save comparative results.
    comparison = pd.DataFrame(all_results)

    comparison_path = (
        REPORT_DIR / "ablation_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    # Save combined training histories.
    history_path = (
        REPORT_DIR / "ablation_training_history.csv"
    )

    pd.DataFrame(all_histories).to_csv(
        history_path,
        index=False
    )

    # Save configuration.
    config = {

        "phase": "6.2",

        "study": "LSTM feature ablation",

        "horizon_ms": HORIZON,

        "experiments": {
            name: {
                "input_size": values["input_size"],
                "feature_indices": values["indices"]
            }
            for name, values in EXPERIMENTS.items()
        },

        "training_config": {
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
            "positive_class_weight": positive_weight
        },

        "test_dataset_used": False
    }

    config_path = REPORT_DIR / "ablation_config.json"

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

    # Save a concise text report.
    best = comparison.sort_values(
        "best_validation_ap",
        ascending=False
    ).iloc[0]

    report_lines = [

        "PHASE 6.2 - FEATURE ABLATION STUDY",

        "=" * 60,

        f"Prediction horizon: {HORIZON} ms",

        f"Best experiment: {best['experiment']}",

        f"Best validation AP: {best['best_validation_ap']:.6f}",

        "",

        "EXPERIMENT COMPARISON",

        comparison.to_string(index=False),

        "",

        "IMPORTANT: Test dataset was not loaded or evaluated.",

        "Feature groups are defined using the saved 66+30 feature layout."

    ]

    report_path = REPORT_DIR / "ablation_analysis.txt"

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("PHASE 6.2 COMPLETED")
    print("=" * 65)

    print("\nAblation comparison:")

    print(
        comparison[
            [
                "experiment",
                "input_features",
                "best_epoch",
                "best_validation_ap",
                "validation_f1",
                "validation_roc_auc"
            ]
        ].to_string(index=False)
    )

    print("\nFiles saved:")
    print(comparison_path)
    print(history_path)
    print(config_path)
    print(report_path)

    print("\nTest dataset was not used.")


if __name__ == "__main__":
    main()
