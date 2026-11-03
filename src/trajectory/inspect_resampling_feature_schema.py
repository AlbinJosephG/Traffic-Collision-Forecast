
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SCENE_DIR = ROOT / "processed" / "scene_features"

OUTPUT_DIR = (
    ROOT / "processed" / "fixed_rate_resampling_audit"
)

TIMESTAMP_COLUMN = "timestamp_sec"


def main():

    train_files = sorted(
        (SCENE_DIR / "train").rglob("*.csv")
    )

    if not train_files:
        raise FileNotFoundError(
            "No training scene-feature CSV files found."
        )

    feature_values = {}

    for path in train_files:

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        for column in df.columns:

            if column == TIMESTAMP_COLUMN:
                continue

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna().to_numpy(dtype=float)

            if len(values) == 0:
                continue

            feature_values.setdefault(column, []).append(values)

    rows = []

    for feature, arrays in feature_values.items():

        values = np.concatenate(arrays)

        unique_values = np.unique(values)

        integer_like = np.allclose(
            values,
            np.round(values),
            atol=1e-6
        )

        binary = (
            len(unique_values) <= 2
            and set(unique_values.tolist()).issubset({0.0, 1.0})
        )

        if binary:
            feature_type = "binary"

        elif integer_like and len(unique_values) <= 25:
            feature_type = "discrete_or_count"

        else:
            feature_type = "continuous"

        rows.append({
            "feature": feature,
            "detected_type": feature_type,
            "observations": len(values),
            "unique_values": len(unique_values),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "sample_values": str(unique_values[:10].tolist()),
        })

    report = pd.DataFrame(rows).sort_values(
        ["detected_type", "feature"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR / "feature_interpolation_schema.csv"
    )

    report.to_csv(output_path, index=False)

    print("=" * 70)
    print("RESAMPLING FEATURE SCHEMA AUDIT")
    print("=" * 70)

    print("\nFeature count:", len(report))

    print("\nFEATURE TYPE COUNTS")

    print(
        report["detected_type"]
        .value_counts()
        .to_string()
    )

    print("\nFEATURE DETAILS")

    print(
        report.to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
