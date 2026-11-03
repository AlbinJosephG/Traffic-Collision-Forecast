
"""
PHASE 2: DETECTION AND TRACKING DIAGNOSTICS

Investigates:
- 11 videos with empty tracking CSVs.
- 2 videos containing invalid tracking IDs.

Compares YOLO detections at confidence 0.10 and 0.25
against ByteTrack output at confidence 0.25.

Does not modify existing tracking CSV files.
"""

from pathlib import Path
from collections import Counter
import logging

import pandas as pd
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\TrafficCollisionProject")

DATASET_DIR = PROJECT_DIR / "dataset"

MANIFEST_DIR = PROJECT_DIR / "data" / "manifests"

MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.pt"

REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / "tracking_validation"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRACKER_CONFIG = "bytetrack.yaml"

TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7]

IMAGE_SIZE = 640
FRAME_STRIDE = 6
DEVICE = "cpu"

CONFIDENCE_THRESHOLD = 0.25
DIAGNOSTIC_THRESHOLD = 0.10
IOU_THRESHOLD = 0.50

AFFECTED_VIDEOS = {
    "train": [
        "00597", "01416", "00776", "01510",
        "00100", "01805", "00459", "01801",
        "01998", "01372", "01449", "00271"
    ],
    "test": [
        "01456"
    ]
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD MODEL AND VIDEO PATHS
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    logger.info("Loading YOLOv8n...")

    return YOLO(str(MODEL_PATH))


def get_video_paths():

    video_paths = {}

    for split, video_names in AFFECTED_VIDEOS.items():

        manifest_path = (
            MANIFEST_DIR / f"{split}_manifest.csv"
        )

        manifest = pd.read_csv(manifest_path)

        for _, row in manifest.iterrows():

            video_path = Path(str(row["video_path"]))

            video_name = video_path.stem

            if video_name in video_names:

                full_path = DATASET_DIR / video_path

                video_paths[video_name] = {
                    "path": full_path,
                    "split": split,
                    "label": int(row["label"])
                }

    return video_paths


# ============================================================
# RAW YOLO DETECTION DIAGNOSTICS
# ============================================================

def inspect_yolo_detections(model, video_path):

    total_frames = 0
    detections_at_low_threshold = 0
    detections_at_normal_threshold = 0

    low_confidence_detections = 0

    class_counts = Counter()

    results = model.predict(
        source=str(video_path),
        stream=True,
        classes=TRAFFIC_CLASSES,
        conf=DIAGNOSTIC_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        vid_stride=FRAME_STRIDE,
        device=DEVICE,
        verbose=False
    )

    for result in results:

        total_frames += 1

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        confidences = boxes.conf.cpu().numpy()

        class_ids = (
            boxes.cls.cpu().numpy().astype(int)
        )

        detections_at_low_threshold += len(boxes)

        for confidence, class_id in zip(
            confidences,
            class_ids
        ):

            class_counts[class_id] += 1

            if confidence >= CONFIDENCE_THRESHOLD:

                detections_at_normal_threshold += 1

            else:

                low_confidence_detections += 1

    return {
        "sampled_frames": total_frames,
        "detections_at_010": detections_at_low_threshold,
        "detections_at_025": detections_at_normal_threshold,
        "below_025": low_confidence_detections,
        "class_counts": dict(class_counts)
    }


# ============================================================
# BYTETRACK DIAGNOSTICS
# ============================================================

def inspect_tracking(model, video_path):

    total_frames = 0
    total_detections = 0

    valid_track_ids = set()

    missing_id_frames = 0
    invalid_id_detections = 0

    results = model.track(
        source=str(video_path),
        tracker=TRACKER_CONFIG,
        persist=False,
        stream=True,
        classes=TRAFFIC_CLASSES,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        vid_stride=FRAME_STRIDE,
        device=DEVICE,
        verbose=False
    )

    for result in results:

        total_frames += 1

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        total_detections += len(boxes)

        if boxes.id is None:

            missing_id_frames += 1

            invalid_id_detections += len(boxes)

            continue

        track_ids = boxes.id.cpu().numpy()

        for track_id in track_ids:

            if track_id < 0 or not float(track_id).is_integer():

                invalid_id_detections += 1

            else:

                valid_track_ids.add(int(track_id))

    return {
        "tracking_frames": total_frames,
        "tracking_detections": total_detections,
        "unique_valid_tracks": len(valid_track_ids),
        "frames_without_ids": missing_id_frames,
        "invalid_id_detections": invalid_id_detections
    }


# ============================================================
# RUN DIAGNOSTICS
# ============================================================

def main():

    logger.info("=" * 65)
    logger.info("DETECTION AND TRACKING DIAGNOSTICS")
    logger.info("=" * 65)

    model = load_model()

    videos = get_video_paths()

    results = []

    for video_name, metadata in videos.items():

        video_path = metadata["path"]

        logger.info("-" * 65)
        logger.info("Inspecting video: %s", video_name)

        if not video_path.exists():

            logger.error("Video missing: %s", video_path)

            continue

        # Run YOLO at a lower threshold to inspect
        # detections excluded by the original threshold.
        yolo_result = inspect_yolo_detections(
            model,
            video_path
        )

        # Run the original tracking configuration.
        tracking_result = inspect_tracking(
            model,
            video_path
        )

        row = {
            "video_name": video_name,
            "split": metadata["split"],
            "label": metadata["label"],
            "video_exists": True,
            **yolo_result,
            **tracking_result
        }

        results.append(row)

        logger.info(
            "YOLO detections at 0.10: %d",
            row["detections_at_010"]
        )

        logger.info(
            "YOLO detections at 0.25: %d",
            row["detections_at_025"]
        )

        logger.info(
            "ByteTrack detections: %d",
            row["tracking_detections"]
        )

        logger.info(
            "Valid tracks: %d",
            row["unique_valid_tracks"]
        )

        logger.info(
            "Invalid tracking IDs: %d",
            row["invalid_id_detections"]
        )

    # Save diagnostic results.
    report = pd.DataFrame(results)

    output_path = (
        REPORT_DIR / "detection_diagnostics.csv"
    )

    report.to_csv(output_path, index=False)

    logger.info("=" * 65)
    logger.info("DIAGNOSTICS COMPLETED")
    logger.info("Report: %s", output_path)
    logger.info("=" * 65)

    if not report.empty:

        display_columns = [
            "video_name",
            "detections_at_010",
            "detections_at_025",
            "tracking_detections",
            "unique_valid_tracks",
            "invalid_id_detections"
        ]

        print("\nDIAGNOSTIC SUMMARY")

        print(
            report[display_columns].to_string(index=False)
        )


if __name__ == "__main__":
    main()
