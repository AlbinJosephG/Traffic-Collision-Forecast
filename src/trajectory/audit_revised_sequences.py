
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "processed" / "revised_event_aligned_sequences"

HORIZONS = [500, 1000, 1500]


def main():

    manifest = pd.read_csv(
        DATA_DIR / "sequence_generation_manifest.csv"
    )

    config = json.loads(
        (DATA_DIR / "preprocessing_config.json").read_text(
            encoding="utf-8"
        )
    )

    reference = pd.read_csv(
        DATA_DIR / "training_positive_anchor_distribution.csv"
    )

    print("=" * 70)
    print("PHASE 5.3 - REVISED SEQUENCE VALIDATION AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Configuration checks
    # --------------------------------------------------------

    print("\n1. CONFIGURATION")

    print("Generator version:", config["generator_version"])
    print("Random seed:", config["random_seed"])
    print("Reference split:", config["negative_anchor_reference_split"])
    print("Reference count:", len(reference))
    print("Base features:", config["base_feature_count"])
    print("Final features:", config["final_feature_count"])

    assert config["negative_anchor_reference_split"] == "train"
    assert len(reference) == 66
    assert config["base_feature_count"] == 66
    assert config["final_feature_count"] == 96

    # --------------------------------------------------------
    # 2. Verify anchor policy
    # --------------------------------------------------------

    print("\n2. ANCHOR POLICY")

    valid = manifest[
        manifest["status"] == "valid"
    ].copy()

    positives = valid[valid["label"] == 1]
    negatives = valid[valid["label"] == 0]

    # Positive anchor must equal its actual event timestamp.
    positive_anchor_error = (
        positives["anchor_sec"]
        - positives["time_of_event"]
    ).abs()

    print(
        "Maximum positive anchor error:",
        positive_anchor_error.max(),
    )

    assert np.allclose(
        positive_anchor_error.to_numpy(),
        0.0,
        atol=1e-9,
    )

    # Negative positions must come from the training reference.
    reference_positions = set(
        np.round(reference["event_position"].to_numpy(), 10)
    )

    negative_positions = np.round(
        negatives["pseudo_event_position"].to_numpy(),
        10,
    )

    unknown_positions = [
        value
        for value in negative_positions
        if value not in reference_positions
    ]

    print(
        "Negative positions not in training reference:",
        len(unknown_positions),
    )

    assert len(unknown_positions) == 0

    assert (
        negatives["anchor_type"]
        == "training_positive_distribution"
    ).all()

    assert (
        positives["anchor_type"]
        == "true_event"
    ).all()

    # --------------------------------------------------------
    # 3. Anchor-position comparison
    # --------------------------------------------------------

    print("\n3. ANCHOR POSITION DISTRIBUTIONS")

    position_report = (
        valid.groupby(["split", "label"])["anchor_position"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    print(position_report.to_string(index=False))

    position_report.to_csv(
        DATA_DIR / "anchor_position_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Samples per horizon and class
    # --------------------------------------------------------

    print("\n4. VALID SAMPLES PER HORIZON")

    horizon_report = (
        valid.groupby(["split", "horizon_ms", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    print(horizon_report.to_string(index=False))

    horizon_report.to_csv(
        DATA_DIR / "horizon_class_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Endpoint and timestamp checks
    # --------------------------------------------------------

    print("\n5. TEMPORAL VALIDATION")

    endpoint_error = (
        valid["prediction_endpoint_sec"]
        - valid["selected_endpoint_sec"]
    )

    print(
        "Maximum endpoint lag:",
        valid["endpoint_lag_sec"].max(),
    )

    print(
        "Maximum selected timestamp gap:",
        valid["max_gap_sec"].max(),
    )

    print(
        "Minimum endpoint lag:",
        valid["endpoint_lag_sec"].min(),
    )

    assert (
        valid["endpoint_lag_sec"] >= -1e-9
    ).all()

    assert (
        valid["endpoint_lag_sec"]
        <= config["max_endpoint_lag_sec"] + 1e-9
    ).all()

    assert (
        valid["max_gap_sec"]
        <= config["max_allowed_gap_sec"] + 1e-9
    ).all()

    assert (endpoint_error >= -1e-9).all()

    # --------------------------------------------------------
    # 6. Sequence counts and class distributions
    # --------------------------------------------------------

    print("\n6. SPLIT CLASS COUNTS")

    for split in ["train", "validation", "test"]:

        path = DATA_DIR / f"{split}.npz"

        with np.load(path, allow_pickle=False) as data:

            X = data["X"]
            y = data["y"]
            horizons = data["horizon_sec"]

            print(f"\n{split.upper()}")
            print("X:", X.shape)
            print("y:", y.shape)
            print("Finite inputs:", np.isfinite(X).all())

            assert X.ndim == 3
            assert X.shape[1] == 50
            assert X.shape[2] == 96
            assert np.isfinite(X).all()

            for horizon in [0.5, 1.0, 1.5]:

                mask = np.isclose(
                    horizons,
                    horizon,
                )

                horizon_labels = y[mask]

                print(
                    f"{int(horizon * 1000)}ms:",
                    {
                        "negative": int(
                            np.sum(horizon_labels == 0)
                        ),
                        "positive": int(
                            np.sum(horizon_labels == 1)
                        ),
                    },
                )

    # --------------------------------------------------------
    # 7. Alert relationship
    # --------------------------------------------------------

    print("\n7. POSITIVE ALERT-TIME RELATIONSHIP")

    positive_valid = valid[
        valid["label"] == 1
    ].copy()

    alert_report = (
        positive_valid.groupby("horizon_ms")[
            "endpoint_at_or_after_alert"
        ]
        .agg(
            samples="count",
            at_or_after_alert="sum",
        )
        .reset_index()
    )

    alert_report["before_alert"] = (
        alert_report["samples"]
        - alert_report["at_or_after_alert"]
    )

    alert_report["at_or_after_alert_percent"] = (
        100
        * alert_report["at_or_after_alert"]
        / alert_report["samples"]
    )

    print(alert_report.to_string(index=False))

    alert_report.to_csv(
        DATA_DIR / "alert_relationship_validation.csv",
        index=False,
    )

    print("\nAUDIT COMPLETED")
    print("Reports saved in:", DATA_DIR)


if __name__ == "__main__":
    main()
