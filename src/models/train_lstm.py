
from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score

from lstm_model import CollisionLSTM

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "processed" / "event_aligned_sequences"
MODEL_DIR = ROOT / "models" / "lstm"
HISTORY_DIR = ROOT / "processed" / "training_history"

HORIZONS = [0.5, 1.0, 1.5]

INPUT_SIZE = 96
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 12

SEED = 42

DEVICE = torch.device("cpu")


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
# DATA LOADING
# ============================================================

def load_split(split):

    path = DATA_DIR / f"{split}.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=False)

    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.float32),
        "horizon_sec": data["horizon_sec"].astype(np.float32),
        "video_names": data["video_names"].astype(str),
        "sample_ids": data["sample_ids"].astype(str),
    }


def create_loader(X, y, shuffle):

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def get_horizon_data(data, horizon):

    mask = np.isclose(
        data["horizon_sec"],
        horizon,
    )

    return (
        data["X"][mask],
        data["y"][mask],
        data["video_names"][mask],
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    probabilities = []
    actual_labels = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)

            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

            actual_labels.extend(
                y.cpu().numpy().tolist()
            )

    average_loss = total_loss / len(actual_labels)

    if len(set(actual_labels)) < 2:
        raise ValueError(
            "Validation data must contain both classes."
        )

    average_precision = average_precision_score(
        actual_labels,
        probabilities,
    )

    return average_loss, average_precision


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_one_horizon(
    horizon,
    train_data,
    validation_data,
):

    print("\n" + "=" * 60)
    print(f"TRAINING LSTM: {int(horizon * 1000)} MS")
    print("=" * 60)

    X_train, y_train, _ = get_horizon_data(
        train_data, horizon
    )

    X_val, y_val, _ = get_horizon_data(
        validation_data, horizon
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            f"No samples available for horizon {horizon}"
        )

    train_positive = int(np.sum(y_train == 1))
    train_negative = int(np.sum(y_train == 0))

    val_positive = int(np.sum(y_val == 1))
    val_negative = int(np.sum(y_val == 0))

    print(
        f"Train: {len(y_train)} "
        f"(positive={train_positive}, negative={train_negative})"
    )

    print(
        f"Validation: {len(y_val)} "
        f"(positive={val_positive}, negative={val_negative})"
    )

    if min(train_positive, train_negative) == 0:
        raise ValueError("Training data must contain both classes.")

    if min(val_positive, val_negative) == 0:
        raise ValueError("Validation data must contain both classes.")

    train_loader = create_loader(
        X_train, y_train, shuffle=True
    )

    validation_loader = create_loader(
        X_val, y_val, shuffle=False
    )

    model = CollisionLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Weighted binary cross entropy.
    positive_weight = torch.tensor(
        [train_negative / train_positive],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_ap = -1.0
    best_epoch = 0
    patience_counter = 0

    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_train_loss = 0.0

        for X, y in train_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(logits, y)

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item() * len(y)

        train_loss = total_train_loss / len(y_train)

        val_loss, val_ap = validate(
            model,
            validation_loader,
            criterion,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_average_precision": val_ap,
        })

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AP: {val_ap:.4f}"
        )

        if val_ap > best_ap + 1e-6:

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
            "No valid model checkpoint was produced."
        )

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        MODEL_DIR
        / f"lstm_{int(horizon * 1000)}ms.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "horizon_sec": horizon,
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "best_epoch": best_epoch,
            "best_validation_ap": best_ap,
            "seed": SEED,
        },
        checkpoint_path,
    )

    history_path = (
        HISTORY_DIR
        / f"lstm_{int(horizon * 1000)}ms_history.csv"
    )

    import pandas as pd

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    print("\nBest epoch:", best_epoch)
    print("Best validation AP:", round(best_ap, 4))
    print("Saved model:", checkpoint_path)

    return {
        "horizon_ms": int(horizon * 1000),
        "best_epoch": best_epoch,
        "best_validation_ap": best_ap,
        "checkpoint": str(checkpoint_path),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("Device:", DEVICE)
    print("PyTorch:", torch.__version__)

    train_data = load_split("train")
    validation_data = load_split("validation")

    results = []

    for horizon in HORIZONS:

        result = train_one_horizon(
            horizon,
            train_data,
            validation_data,
        )

        results.append(result)

    import pandas as pd

    result_path = MODEL_DIR / "training_summary.csv"

    pd.DataFrame(results).to_csv(
        result_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("ALL HORIZON MODELS TRAINED")
    print("=" * 60)

    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
