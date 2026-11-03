
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

ANNOTATION_FILE = (
    ROOT_DIR / "data" / "annotations"
    / "nexar_train_annotations.csv"
)

MANIFEST_FILE = (
    ROOT_DIR / "data" / "manifests"
    / "subset_manifest.csv"
)

DATASET_DIR = ROOT_DIR / "dataset"

OUTPUT_DIR = ROOT_DIR / "data" / "annotations"

OUTPUT_FILE = OUTPUT_DIR / "matched_200_annotations.csv"


# ============================================================
# VIDEO DURATION
# ============================================================

def get_video_duration(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return np.nan

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps <= 0 or frame_count <= 0:
        return np.nan

    return frame_count / fps


# ============================================================
# LOAD DATA
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 3E.1 - NEXAR ANNOTATION AUDIT")
    print("=" * 65)

    annotations = pd.read_csv(ANNOTATION_FILE)

    manifest = pd.read_csv(MANIFEST_FILE)

    annotations["video_name"] = (
        annotations["video_name"].astype(str)
    )

    manifest["video_name"] = (
        manifest["video_path"]
        .apply(lambda x: Path(str(x)).name)
    )

    manifest["label"] = manifest["label"].astype(int)

    if annotations["video_name"].duplicated().any():
        raise ValueError("Duplicate annotation video IDs found.")

    if manifest["video_name"].duplicated().any():
        raise ValueError("Duplicate manifest video IDs found.")

    # ========================================================
    # MATCH ANNOTATIONS TO SELECTED VIDEOS
    # ========================================================

    merged = manifest.merge(
        annotations,
        on="video_name",
        how="left",
        suffixes=("_manifest", "_annotation"),
        indicator=True
    )

    missing = merged[
        merged["_merge"] != "both"
    ]

    matched = merged[
        merged["_merge"] == "both"
    ].copy()

    matched["label"] = matched["label_manifest"]

    matched["label_match"] = (
        matched["label_manifest"]
        == matched["label_annotation"]
    )

    matched["time_of_event"] = pd.to_numeric(
        matched["time_of_event"],
        errors="coerce"
    )

    matched["time_of_alert"] = pd.to_numeric(
        matched["time_of_alert"],
        errors="coerce"
    )

    # ========================================================
    # TIMESTAMP VALIDATION
    # ========================================================

    positive = matched["label"] == 1
    negative = matched["label"] == 0

    matched["alert_lead_time"] = (
        matched["time_of_event"]
        - matched["time_of_alert"]
    )

    matched["video_duration_sec"] = np.nan

    for index, row in matched.iterrows():

        relative_path = Path(str(row["video_path"]))
        video_path = DATASET_DIR / relative_path

        if video_path.exists():
            matched.loc[
                index, "video_duration_sec"
            ] = get_video_duration(video_path)

    matched["valid_timestamps"] = False

    positive_valid = (
        positive
        & matched["time_of_event"].notna()
        & matched["time_of_alert"].notna()
        & (matched["time_of_alert"] >= 0)
        & (matched["time_of_event"] > matched["time_of_alert"])
        & matched["video_duration_sec"].notna()
        & (
            matched["time_of_event"]
            <= matched["video_duration_sec"]
        )
    )

    negative_valid = (
        negative
        & matched["time_of_event"].isna()
        & matched["time_of_alert"].isna()
    )

    matched.loc[
        positive_valid | negative_valid,
        "valid_timestamps"
    ] = True

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\nMATCHING SUMMARY")
    print("-" * 50)

    print("Selected videos:", len(manifest))
    print("Matched videos:", len(matched))
    print("Missing annotations:", len(missing))

    print(
        "Label mismatches:",
        int((~matched["label_match"]).sum())
    )

    print("\nCLASS DISTRIBUTION")
    print("-" * 50)

    print("Positive:", int(positive.sum()))
    print("Negative:", int(negative.sum()))

    print("\nTIMESTAMP SUMMARY")
    print("-" * 50)

    print(
        "Positive with event time:",
        int(
            matched.loc[
                positive, "time_of_event"
            ].notna().sum()
        )
    )

    print(
        "Positive with alert time:",
        int(
            matched.loc[
                positive, "time_of_alert"
            ].notna().sum()
        )
    )

    print(
        "Positive with valid timestamps:",
        int(
            matched.loc[
                positive, "valid_timestamps"
            ].sum()
        )
    )

    print(
        "Negative with no event timestamps:",
        int(negative_valid.sum())
    )

    print("\nALERT LEAD-TIME DISTRIBUTION")
    print("-" * 50)

    lead_times = matched.loc[
        positive, "alert_lead_time"
    ].dropna()

    if len(lead_times):

        print(lead_times.describe().round(3))

        print(
            "\nAlert lead time <= 500 ms:",
            int((lead_times <= 0.5).sum())
        )

        print(
            "Alert lead time <= 1000 ms:",
            int((lead_times <= 1.0).sum())
        )

        print(
            "Alert lead time <= 1500 ms:",
            int((lead_times <= 1.5).sum())
        )

    # ========================================================
    # SAVE
    # ========================================================

    matched.drop(
        columns=["_merge"],
        errors="ignore"
    ).to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nVALIDATION")
    print("-" * 50)

    all_valid = (
        len(missing) == 0
        and matched["label_match"].all()
        and matched["valid_timestamps"].all()
    )

    print("All annotation checks passed:", all_valid)

    print("\nSaved:", OUTPUT_FILE)

    if len(missing):
        print("\nMissing annotation IDs:")
        print(missing["video_name"].tolist())

    if not matched["label_match"].all():
        print("\nLabel mismatches:")
        print(
            matched.loc[
                ~matched["label_match"],
                "video_name"
            ].tolist()
        )

    if not matched["valid_timestamps"].all():
        print("\nInvalid timestamp records:")
        print(
            matched.loc[
                ~matched["valid_timestamps"],
                [
                    "video_name",
                    "label",
                    "time_of_event",
                    "time_of_alert",
                    "video_duration_sec",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
