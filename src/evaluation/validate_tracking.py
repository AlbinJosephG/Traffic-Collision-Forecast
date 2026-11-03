
"""
PHASE 2: TRACKING OUTPUT VALIDATION
==================================

Project:
Early Traffic Collision Prediction Using Deep Learning-Based
Temporal Interaction Analysis

Purpose:
Validate YOLOv8n + ByteTrack tracking outputs for all videos.

Input:
    data/manifests/
        train_manifest.csv
        validation_manifest.csv
        test_manifest.csv

    processed/tracking/
        train/
        validation/
        test/

Output:
    reports/tracking_validation/
        video_validation_report.csv
        validation_summary.csv
        validation_summary.txt

Run:
    python src/evaluation/validate_tracking.py
"""

from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

MANIFEST_DIR = PROJECT_DIR / "data" / "manifests"

TRACKING_DIR = PROJECT_DIR / "processed" / "tracking"

REPORT_DIR = PROJECT_DIR / "reports" / "tracking_validation"

SPLITS = ["train", "validation", "test"]

REQUIRED_COLUMNS = [
    "frame_index",
    "timestamp_sec",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "width",
    "height",
]

# Minimum detections required for a video to be considered
# structurally usable.
MIN_DETECTIONS = 1


# ============================================================
# 2. LOGGING CONFIGURATION
# ============================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            REPORT_DIR / "validation.log",
            encoding="utf-8"
        ),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_label_folder(label):
    """Convert numerical labels into folder names."""

    label = int(label)

    if label == 1:
        return "positive"

    if label == 0:
        return "negative"

    raise ValueError(f"Unexpected label: {label}")


def get_tracking_file(video_path, split, label):
    """Find the corresponding tracking CSV."""

    video_name = Path(str(video_path)).stem

    label_folder = get_label_folder(label)

    return (
        TRACKING_DIR
        / split
        / label_folder
        / f"{video_name}.csv"
    )


def safe_number(series):
    """Convert a pandas series to numeric values."""

    return pd.to_numeric(series, errors="coerce")


# ============================================================
# 4. VALIDATE ONE VIDEO
# ============================================================

def validate_video(video_row, split):

    video_path = video_row["video_path"]
    label = video_row["label"]

    video_name = Path(str(video_path)).stem

    tracking_file = get_tracking_file(
        video_path,
        split,
        label
    )

    result = {
        "video_name": video_name,
        "split": split,
        "label": int(label),
        "tracking_file": str(tracking_file.relative_to(PROJECT_DIR)),
        "file_exists": tracking_file.exists(),
        "csv_readable": False,
        "total_detections": 0,
        "unique_tracks": 0,
        "single_frame_tracks": 0,
        "multi_frame_tracks": 0,
        "tracks_5plus_frames": 0,
        "unique_frames": 0,
        "missing_values": 0,
        "invalid_track_ids": 0,
        "invalid_boxes": 0,
        "invalid_confidence": 0,
        "invalid_frame_indices": 0,
        "duplicate_track_frames": 0,
        "valid": False,
        "issues": [],
    }

    issues = []

    # --------------------------------------------------------
    # Check 1: Tracking CSV exists
    # --------------------------------------------------------

    if not tracking_file.exists():

        issues.append("Tracking CSV missing")

        result["issues"] = "; ".join(issues)

        return result

    # --------------------------------------------------------
    # Check 2: Read CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(tracking_file)

        result["csv_readable"] = True

    except Exception as error:

        issues.append(f"CSV reading failed: {error}")

        result["issues"] = "; ".join(issues)

        return result

    result["total_detections"] = len(df)

    if df.empty:

        issues.append("CSV contains no detections")

        result["issues"] = "; ".join(issues)

        return result

    # --------------------------------------------------------
    # Check 3: Required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        issues.append(
            "Missing columns: " + ", ".join(missing_columns)
        )

        result["issues"] = "; ".join(issues)

        return result

    # --------------------------------------------------------
    # Check 4: Missing values
    # --------------------------------------------------------

    important_columns = REQUIRED_COLUMNS

    result["missing_values"] = int(
        df[important_columns].isna().sum().sum()
    )

    if result["missing_values"] > 0:

        issues.append(
            f"{result['missing_values']} missing values"
        )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "frame_index",
        "timestamp_sec",
        "track_id",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "width",
        "height",
    ]

    for column in numeric_columns:

        df[column] = safe_number(df[column])

    # --------------------------------------------------------
    # Check 5: Invalid tracking IDs
    # --------------------------------------------------------

    invalid_ids = (
        df["track_id"].isna()
        | (df["track_id"] < 0)
        | (df["track_id"] % 1 != 0)
    )

    result["invalid_track_ids"] = int(invalid_ids.sum())

    if result["invalid_track_ids"] > 0:

        issues.append("Invalid tracking IDs")

    valid_id_df = df.loc[~invalid_ids].copy()

    # --------------------------------------------------------
    # Check 6: Bounding box validity
    # --------------------------------------------------------

    invalid_boxes = (
        df[
            ["x1", "y1", "x2", "y2", "width", "height"]
        ].isna().any(axis=1)
        | (df["x2"] <= df["x1"])
        | (df["y2"] <= df["y1"])
        | (df["width"] <= 0)
        | (df["height"] <= 0)
    )

    result["invalid_boxes"] = int(invalid_boxes.sum())

    if result["invalid_boxes"] > 0:

        issues.append("Invalid bounding boxes")

    # --------------------------------------------------------
    # Check 7: Confidence scores
    # --------------------------------------------------------

    invalid_confidence = (
        df["confidence"].isna()
        | (df["confidence"] < 0)
        | (df["confidence"] > 1)
    )

    result["invalid_confidence"] = int(
        invalid_confidence.sum()
    )

    if result["invalid_confidence"] > 0:

        issues.append("Invalid confidence scores")

    # --------------------------------------------------------
    # Check 8: Frame indices
    # --------------------------------------------------------

    invalid_frames = (
        df["frame_index"].isna()
        | (df["frame_index"] < 0)
        | (df["frame_index"] % 1 != 0)
    )

    result["invalid_frame_indices"] = int(
        invalid_frames.sum()
    )

    if result["invalid_frame_indices"] > 0:

        issues.append("Invalid frame indices")

    result["unique_frames"] = int(
        df["frame_index"].nunique()
    )

    # --------------------------------------------------------
    # Check 9: Tracking persistence
    # --------------------------------------------------------

    if not valid_id_df.empty:

        track_lengths = (
            valid_id_df.groupby("track_id")["frame_index"]
            .nunique()
        )

        result["unique_tracks"] = int(
            len(track_lengths)
        )

        result["single_frame_tracks"] = int(
            (track_lengths == 1).sum()
        )

        result["multi_frame_tracks"] = int(
            (track_lengths > 1).sum()
        )

        result["tracks_5plus_frames"] = int(
            (track_lengths >= 5).sum()
        )

    # --------------------------------------------------------
    # Check 10: Duplicate track detections in a frame
    # --------------------------------------------------------

    valid_records = df.loc[
        ~invalid_ids & ~invalid_frames
    ]

    duplicates = valid_records.duplicated(
        subset=["frame_index", "track_id"],
        keep=False
    )

    result["duplicate_track_frames"] = int(
        duplicates.sum()
    )

    if result["duplicate_track_frames"] > 0:

        issues.append(
            "Duplicate track detections in the same frame"
        )

    # --------------------------------------------------------
    # Final video validation status
    # --------------------------------------------------------

    result["valid"] = len(issues) == 0

    result["issues"] = (
        "; ".join(issues) if issues else "None"
    )

    return result


# ============================================================
# 5. VALIDATE ALL DATASET SPLITS
# ============================================================

def validate_all_videos():

    all_results = []

    for split in SPLITS:

        logger.info("=" * 65)
        logger.info("VALIDATING %s SPLIT", split.upper())
        logger.info("=" * 65)

        manifest_path = (
            MANIFEST_DIR / f"{split}_manifest.csv"
        )

        if not manifest_path.exists():

            logger.error(
                "Manifest not found: %s",
                manifest_path
            )

            continue

        manifest = pd.read_csv(manifest_path)

        required_manifest_columns = [
            "video_path",
            "label"
        ]

        if not all(
            column in manifest.columns
            for column in required_manifest_columns
        ):

            logger.error(
                "Invalid manifest format: %s",
                manifest_path
            )

            continue

        for index, row in manifest.iterrows():

            result = validate_video(row, split)

            all_results.append(result)

            logger.info(
                "[%s] %s | Detections: %s | Tracks: %s",
                "PASS" if result["valid"] else "FAIL",
                result["video_name"],
                result["total_detections"],
                result["unique_tracks"],
            )

    return all_results


# ============================================================
# 6. GENERATE REPORTS
# ============================================================

def generate_reports(results):

    if not results:

        logger.error("No validation results generated.")

        return

    report = pd.DataFrame(results)

    # --------------------------------------------------------
    # Video-level report
    # --------------------------------------------------------

    video_report_path = (
        REPORT_DIR / "video_validation_report.csv"
    )

    report.to_csv(
        video_report_path,
        index=False
    )

    # --------------------------------------------------------
    # Split-level summary
    # --------------------------------------------------------

    summary = (
        report.groupby("split")
        .agg(
            total_videos=("video_name", "count"),
            valid_videos=("valid", "sum"),
            total_detections=("total_detections", "sum"),
            total_tracks=("unique_tracks", "sum"),
            total_frames=("unique_frames", "sum"),
            total_missing_values=("missing_values", "sum"),
            invalid_track_ids=("invalid_track_ids", "sum"),
            invalid_boxes=("invalid_boxes", "sum"),
            invalid_confidence=("invalid_confidence", "sum"),
            duplicate_records=("duplicate_track_frames", "sum"),
        )
        .reset_index()
    )

    summary["failed_videos"] = (
        summary["total_videos"] - summary["valid_videos"]
    )

    summary["success_rate_percent"] = (
        summary["valid_videos"]
        / summary["total_videos"]
        * 100
    )

    summary_path = REPORT_DIR / "validation_summary.csv"

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    total_videos = len(report)

    valid_videos = int(report["valid"].sum())

    failed_videos = total_videos - valid_videos

    total_detections = int(
        report["total_detections"].sum()
    )

    total_tracks = int(
        report["unique_tracks"].sum()
    )

    success_rate = (
        valid_videos / total_videos * 100
        if total_videos > 0 else 0
    )

    text_path = REPORT_DIR / "validation_summary.txt"

    with open(text_path, "w", encoding="utf-8") as file:

        file.write("=" * 65 + "\n")
        file.write("PHASE 2 TRACKING VALIDATION REPORT\n")
        file.write("=" * 65 + "\n\n")

        file.write(f"Total videos: {total_videos}\n")
        file.write(f"Valid videos: {valid_videos}\n")
        file.write(f"Failed videos: {failed_videos}\n")
        file.write(f"Success rate: {success_rate:.2f}%\n\n")

        file.write(f"Total detections: {total_detections}\n")
        file.write(f"Total unique tracks: {total_tracks}\n\n")

        file.write("SPLIT-LEVEL SUMMARY\n")
        file.write("-" * 65 + "\n")

        file.write(summary.to_string(index=False))

        file.write("\n\nINVALID VIDEOS\n")
        file.write("-" * 65 + "\n")

        failed = report.loc[~report["valid"]]

        if failed.empty:

            file.write("No structural validation failures.\n")

        else:

            file.write(
                failed[
                    ["video_name", "split", "issues"]
                ].to_string(index=False)
            )

        file.write("\n")

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    logger.info("=" * 65)
    logger.info("TRACKING VALIDATION COMPLETED")
    logger.info("=" * 65)

    logger.info("Total videos: %d", total_videos)
    logger.info("Valid videos: %d", valid_videos)
    logger.info("Failed videos: %d", failed_videos)

    logger.info(
        "Success rate: %.2f%%",
        success_rate
    )

    logger.info("Total detections: %d", total_detections)
    logger.info("Total tracks: %d", total_tracks)

    logger.info("Video report: %s", video_report_path)
    logger.info("Split summary: %s", summary_path)
    logger.info("Text report: %s", text_path)

    print("\nSPLIT SUMMARY")
    print(summary.to_string(index=False))

    if failed_videos > 0:

        print("\nVIDEOS REQUIRING ATTENTION")

        print(
            report.loc[
                ~report["valid"],
                ["video_name", "split", "issues"]
            ].to_string(index=False)
        )


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main():

    logger.info("=" * 65)
    logger.info("PHASE 2: TRACKING VALIDATION STARTED")
    logger.info("=" * 65)

    logger.info("Project directory: %s", PROJECT_DIR)
    logger.info("Tracking directory: %s", TRACKING_DIR)

    results = validate_all_videos()

    generate_reports(results)

    logger.info("=" * 65)
    logger.info("VALIDATION PIPELINE FINISHED")
    logger.info("=" * 65)


if __name__ == "__main__":

    main()
