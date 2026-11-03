
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

SCENE_DIR = ROOT_DIR / "processed" / "scene_features"
OUTPUT_FILE = ROOT_DIR / "processed" / "temporal_sampling_audit.csv"

SPLITS = ["train", "validation", "test"]


def main():

    results = []

    for split in SPLITS:

        files = sorted(
            SCENE_DIR.glob(f"{split}/*/*.csv")
        )

        for file_path in files:

            df = pd.read_csv(file_path)

            if df.empty or "timestamp_sec" not in df.columns:
                results.append({
                    "video_id": file_path.stem,
                    "split": split,
                    "frame_count": len(df),
                    "mean_interval": np.nan,
                    "median_interval": np.nan,
                    "min_interval": np.nan,
                    "max_interval": np.nan,
                    "intervals_near_0_2_sec": 0,
                    "intervals_over_0_3_sec": 0,
                    "status": "empty_or_missing_timestamp",
                })
                continue

            timestamps = pd.to_numeric(
                df["timestamp_sec"],
                errors="coerce"
            ).dropna().to_numpy()

            timestamps = np.sort(timestamps)

            intervals = np.diff(timestamps)

            # Remove duplicate or invalid intervals.
            intervals = intervals[intervals > 0]

            if len(intervals) == 0:
                results.append({
                    "video_id": file_path.stem,
                    "split": split,
                    "frame_count": len(timestamps),
                    "mean_interval": np.nan,
                    "median_interval": np.nan,
                    "min_interval": np.nan,
                    "max_interval": np.nan,
                    "intervals_near_0_2_sec": 0,
                    "intervals_over_0_3_sec": 0,
                    "status": "insufficient_timestamps",
                })
                continue

            results.append({
                "video_id": file_path.stem,
                "split": split,
                "frame_count": len(timestamps),
                "mean_interval": np.mean(intervals),
                "median_interval": np.median(intervals),
                "min_interval": np.min(intervals),
                "max_interval": np.max(intervals),
                "intervals_near_0_2_sec": int(
                    np.sum(
                        (intervals >= 0.15)
                        & (intervals <= 0.25)
                    )
                ),
                "intervals_over_0_3_sec": int(
                    np.sum(intervals > 0.30)
                ),
                "status": "checked",
            })

    audit = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    audit.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("PHASE 3E.2 - TEMPORAL SAMPLING AUDIT")
    print("=" * 60)

    print("\nVideos checked:", len(audit))

    valid = audit[
        audit["status"] == "checked"
    ]

    print("Videos with usable timestamps:", len(valid))

    print("\nINTERVAL STATISTICS")
    print("-" * 40)

    print(
        valid[
            [
                "mean_interval",
                "median_interval",
                "min_interval",
                "max_interval",
            ]
        ].describe().round(4)
    )

    print("\nTOTAL INTERVALS")
    print("-" * 40)

    print(
        "Intervals near 0.2 sec:",
        int(valid["intervals_near_0_2_sec"].sum())
    )

    print(
        "Intervals over 0.3 sec:",
        int(valid["intervals_over_0_3_sec"].sum())
    )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
