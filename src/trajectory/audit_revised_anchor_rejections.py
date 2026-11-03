
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "processed" / "revised_event_aligned_sequences"

MANIFEST_FILE = DATA_DIR / "sequence_generation_manifest.csv"


def main():

    df = pd.read_csv(MANIFEST_FILE)

    print("=" * 70)
    print("REVISED ANCHOR REJECTION BIAS AUDIT")
    print("=" * 70)

    # One anchor per video, rather than counting its three horizons.
    video_df = (
        df.sort_values("horizon_ms")
        .drop_duplicates(
            subset=["split", "video_name"],
            keep="first",
        )
        .copy()
    )

    # --------------------------------------------------------
    # 1. Anchor distributions before sequence rejection
    # --------------------------------------------------------

    print("\n1. ALL VIDEO ANCHORS")

    all_report = (
        video_df.groupby(["split", "label"])["anchor_position"]
        .agg(
            videos="count",
            mean="mean",
            std="std",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    print(all_report.to_string(index=False))

    # --------------------------------------------------------
    # 2. Valid vs rejected anchors
    # --------------------------------------------------------

    print("\n2. VALID VS REJECTED ANCHORS")

    video_df["anchor_status"] = np.where(
        video_df["status"] == "valid",
        "valid",
        "rejected",
    )

    status_report = (
        video_df.groupby(
            ["split", "label", "anchor_status"]
        )["anchor_position"]
        .agg(
            videos="count",
            mean="mean",
            std="std",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    print(status_report.to_string(index=False))

    status_report.to_csv(
        DATA_DIR / "anchor_rejection_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Rejection reasons
    # --------------------------------------------------------

    print("\n3. REJECTION REASONS")

    rejection_report = (
        df[df["status"] != "valid"]
        .groupby(["split", "label", "status"])
        .size()
        .reset_index(name="sample_count")
    )

    print(rejection_report.to_string(index=False))

    rejection_report.to_csv(
        DATA_DIR / "anchor_rejection_reasons_detailed.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Per-video rejection details
    # --------------------------------------------------------

    print("\n4. REJECTED VIDEO DETAILS")

    rejected = video_df[
        video_df["anchor_status"] == "rejected"
    ][[
        "split",
        "video_name",
        "label",
        "anchor_type",
        "anchor_position",
        "anchor_sec",
        "status",
    ]].sort_values(
        ["split", "label", "anchor_position"]
    )

    print(rejected.to_string(index=False))

    rejected.to_csv(
        DATA_DIR / "rejected_anchor_videos.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Compare positive and negative distributions
    #    using unique valid videos only
    # --------------------------------------------------------

    print("\n5. UNIQUE VALID VIDEO COMPARISON")

    valid = video_df[
        video_df["anchor_status"] == "valid"
    ]

    valid_report = (
        valid.groupby(["split", "label"])["anchor_position"]
        .agg(
            videos="count",
            mean="mean",
            std="std",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    print(valid_report.to_string(index=False))

    valid_report.to_csv(
        DATA_DIR / "unique_valid_anchor_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Final checks
    # --------------------------------------------------------

    print("\n6. SUMMARY")

    for split in ["train", "validation", "test"]:

        subset = video_df[
            video_df["split"] == split
        ]

        print(f"\n{split.upper()}")

        for label in [0, 1]:

            class_subset = subset[
                subset["label"] == label
            ]

            valid_count = (
                class_subset["anchor_status"] == "valid"
            ).sum()

            rejected_count = (
                class_subset["anchor_status"] == "rejected"
            ).sum()

            print(
                f"Label {label}: "
                f"{len(class_subset)} videos, "
                f"{valid_count} valid, "
                f"{rejected_count} rejected"
            )

    print("\nAUDIT COMPLETED")


if __name__ == "__main__":
    main()
