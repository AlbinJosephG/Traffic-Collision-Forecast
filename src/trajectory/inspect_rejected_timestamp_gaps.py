
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SCENE_DIR = ROOT / "processed" / "scene_features"
MANIFEST_PATH = (
    ROOT
    / "processed"
    / "revised_event_aligned_sequences"
    / "sequence_generation_manifest.csv"
)

MAX_GAP = 0.30
SEQUENCE_LENGTH = 50
HORIZONS = [0.5, 1.0, 1.5]


def main():

    manifest = pd.read_csv(MANIFEST_PATH)

    rejected = manifest[
        manifest["status"] == "timestamp_gap"
    ].copy()

    results = []

    for _, row in rejected.iterrows():

        split = row["split"]
        label = int(row["label"])
        video_name = row["video_name"]

        category = "positive" if label == 1 else "negative"

        path = (
            SCENE_DIR
            / split
            / category
            / f"{Path(video_name).stem}.csv"
        )

        if not path.exists():
            continue

        df = pd.read_csv(path)

        timestamps = pd.to_numeric(
            df["timestamp_sec"],
            errors="coerce",
        ).dropna().sort_values().drop_duplicates().to_numpy()

        endpoint = float(row["prediction_endpoint_sec"])

        eligible = timestamps[timestamps <= endpoint]

        if len(eligible) < SEQUENCE_LENGTH:
            continue

        selected = eligible[-SEQUENCE_LENGTH:]
        gaps = np.diff(selected)

        if len(gaps) == 0:
            continue

        largest_index = int(np.argmax(gaps))
        largest_gap = float(gaps[largest_index])

        results.append({
            "split": split,
            "label": label,
            "video_name": video_name,
            "horizon_ms": int(row["horizon_ms"]),
            "anchor_position": row["anchor_position"],
            "endpoint_sec": endpoint,
            "sequence_start_sec": float(selected[0]),
            "sequence_end_sec": float(selected[-1]),
            "largest_gap_sec": largest_gap,
            "gap_before_sec": float(selected[largest_index]),
            "gap_after_sec": float(selected[largest_index + 1]),
            "gaps_over_0_30": int(np.sum(gaps > MAX_GAP)),
            "gaps_over_0_40": int(np.sum(gaps > 0.40)),
            "gaps_over_0_50": int(np.sum(gaps > 0.50)),
            "gaps_over_1_00": int(np.sum(gaps > 1.00)),
            "would_pass_0_40": bool(np.max(gaps) <= 0.40),
            "would_pass_0_50": bool(np.max(gaps) <= 0.50),
        })

    report = pd.DataFrame(results)

    output_path = (
        ROOT
        / "processed"
        / "revised_event_aligned_sequences"
        / "rejected_gap_details.csv"
    )

    report.to_csv(output_path, index=False)

    print("=" * 70)
    print("REJECTED TIMESTAMP GAP INSPECTION")
    print("=" * 70)

    if report.empty:
        print("No rejected sequences available for inspection.")
        return

    print("\nDETAILS")
    print(report.to_string(index=False))

    print("\nSUMMARY BY SPLIT AND CLASS")

    summary = (
        report.groupby(["split", "label"])
        .agg(
            rejected_sequences=("video_name", "count"),
            mean_largest_gap=("largest_gap_sec", "mean"),
            median_largest_gap=("largest_gap_sec", "median"),
            max_largest_gap=("largest_gap_sec", "max"),
            pass_0_40=("would_pass_0_40", "sum"),
            pass_0_50=("would_pass_0_50", "sum"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
