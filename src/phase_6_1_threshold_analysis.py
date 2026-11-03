
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)


# ============================================================
# PHASE 6.1 - GRU VALIDATION THRESHOLD ANALYSIS
# Validation data only. Test data is never loaded.
# ============================================================

PROJECT_DIR = Path(r"E:\TrafficCollisionProject")

DATA_DIR = PROJECT_DIR / "processed" / "fixed_rate_sequences"

MODEL_DIR = PROJECT_DIR / "models" / "fixed_rate_gru"

OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "phase_6_1_threshold_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [500, 1000, 1500]

INPUT_SIZE = 96
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Thresholds to evaluate
THRESHOLDS = np.round(
    np.arange(0.05, 1.00, 0.05),
    2
)


# ============================================================
# GRU ARCHITECTURE
# Must match Phase 6.0
# ============================================================

class CollisionGRU(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
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

        output, hidden = self.gru(x)

        final_features = output[:, -1, :]

        final_features = self.dropout(final_features)

        logits = self.classifier(final_features)

        return logits.squeeze(1)


# ============================================================
# LOAD VALIDATION DATA ONLY
# ============================================================

def load_validation_data():

    validation_path = DATA_DIR / "validation.npz"

    if not validation_path.exists():

        raise FileNotFoundError(
            f"Validation file not found: {validation_path}"
        )

    data = np.load(
        validation_path,
        allow_pickle=True
    )

    required = ["X", "y", "horizon_ms"]

    for key in required:

        if key not in data.files:

            raise ValueError(
                f"Missing key '{key}' in validation.npz"
            )

    X = data["X"].astype(np.float32)

    y = data["y"].astype(np.int64)

    horizons = data["horizon_ms"].astype(np.int64)

    if X.ndim != 3 or X.shape[1:] != (50, 96):

        raise ValueError(
            f"Unexpected validation feature shape: {X.shape}"
        )

    if len(X) != len(y) or len(y) != len(horizons):

        raise ValueError(
            "Validation arrays have inconsistent lengths."
        )

    if not np.isfinite(X).all():

        raise ValueError(
            "Validation features contain NaN or infinity."
        )

    print("Validation data loaded successfully.")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Device:", DEVICE)

    return X, y, horizons


# ============================================================
# LOAD GRU CHECKPOINT
# ============================================================

def load_gru_model(horizon):

    checkpoint_path = (
        MODEL_DIR / f"collision_gru_{horizon}ms.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

    model = CollisionGRU().to(DEVICE)

    if "model_state_dict" not in checkpoint:

        raise ValueError(
            f"Unexpected checkpoint format: {checkpoint_path}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# GET VALIDATION PROBABILITIES
# ============================================================

def predict_probabilities(model, X):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    probabilities = []

    model.eval()

    with torch.no_grad():

        for start in range(0, len(X_tensor), 32):

            batch = X_tensor[start:start + 32]

            logits = model(batch)

            batch_probabilities = torch.sigmoid(
                logits
            )

            probabilities.extend(
                batch_probabilities.cpu().numpy()
            )

    return np.asarray(probabilities)


# ============================================================
# CALCULATE METRICS FOR ONE THRESHOLD
# ============================================================

def evaluate_threshold(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {

        "threshold": float(threshold),

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

        "true_negatives": int(tn),

        "false_positives": int(fp),

        "false_negatives": int(fn),

        "true_positives": int(tp)
    }


# ============================================================
# ANALYSE ONE HORIZON
# ============================================================

def analyse_horizon(X, y, horizons, horizon):

    print("\n" + "=" * 65)
    print(f"GRU THRESHOLD ANALYSIS - {horizon} ms")
    print("=" * 65)

    mask = horizons == horizon

    X_horizon = X[mask]

    y_horizon = y[mask]

    if len(y_horizon) == 0:

        raise ValueError(
            f"No validation samples for {horizon} ms."
        )

    if len(np.unique(y_horizon)) != 2:

        raise ValueError(
            f"Both classes are required for {horizon} ms."
        )

    model = load_gru_model(horizon)

    probabilities = predict_probabilities(
        model,
        X_horizon
    )

    # Threshold-independent ranking metrics
    ap = average_precision_score(
        y_horizon,
        probabilities
    )

    auc = roc_auc_score(
        y_horizon,
        probabilities
    )

    rows = []

    for threshold in THRESHOLDS:

        metrics = evaluate_threshold(
            y_horizon,
            probabilities,
            threshold
        )

        metrics["model"] = "GRU"

        metrics["horizon_ms"] = horizon

        metrics["samples"] = len(y_horizon)

        metrics["average_precision"] = ap

        metrics["roc_auc"] = auc

        rows.append(metrics)

    results = pd.DataFrame(rows)

    # Select threshold with highest validation F1.
    # If tied, prefer the higher threshold.
    best_row = results.sort_values(
        ["f1", "threshold"],
        ascending=[False, False]
    ).iloc[0]

    print("Validation samples:", len(y_horizon))
    print("Average Precision:", round(ap, 6))
    print("ROC-AUC:", round(auc, 6))

    print("\nBest F1 threshold:")
    print(best_row.to_string())

    return results, best_row.to_dict()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 6.1 - GRU THRESHOLD ANALYSIS")
    print("=" * 65)

    X, y, horizons = load_validation_data()

    all_results = []

    best_results = []

    for horizon in HORIZONS:

        results, best = analyse_horizon(
            X,
            y,
            horizons,
            horizon
        )

        all_results.append(results)

        best_results.append(best)

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    threshold_path = (
        OUTPUT_DIR / "gru_threshold_metrics.csv"
    )

    combined.to_csv(
        threshold_path,
        index=False
    )

    best_df = pd.DataFrame(best_results)

    best_path = (
        OUTPUT_DIR / "gru_best_thresholds.csv"
    )

    best_df.to_csv(
        best_path,
        index=False
    )

    # Save best thresholds in JSON
    best_threshold_config = {}

    for row in best_results:

        key = str(row["horizon_ms"])

        best_threshold_config[key] = {

            "threshold": float(row["threshold"]),

            "validation_f1": float(row["f1"]),

            "validation_precision": float(row["precision"]),

            "validation_recall": float(row["recall"]),

            "validation_accuracy": float(row["accuracy"])
        }

    json_path = (
        OUTPUT_DIR / "gru_best_thresholds.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            best_threshold_config,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("PHASE 6.1 - GRU ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nBest validation thresholds:")

    print(
        best_df[
            [
                "horizon_ms",
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_positives",
                "false_negatives"
            ]
        ].to_string(index=False)
    )

    print("\nFiles saved:")
    print(threshold_path)
    print(best_path)
    print(json_path)

    print("\nIMPORTANT:")
    print("Only validation data was used.")
    print("The test dataset was not loaded.")
    print("Thresholds are selected using validation F1.")
    print("Do not treat these thresholds as independently")
    print("validated test performance.")


if __name__ == "__main__":

    main()
