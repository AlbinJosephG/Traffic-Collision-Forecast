
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\TrafficCollisionProject")

DATASET_DIR = PROJECT_DIR / "dataset"
MANIFEST_DIR = PROJECT_DIR / "data" / "manifests"

MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.pt"

CONFIG_DIR = PROJECT_DIR / "configs"
TRACKER_CONFIG = CONFIG_DIR / "bytetrack_low_conf.yaml"

OUTPUT_DIR = PROJECT_DIR / "processed" / "tracking_test"
SUMMARY_PATH = PROJECT_DIR / "processed" / "tracking_test_summary.csv"

LOG_DIR = PROJECT_DIR / "logs"
LOG_PATH = LOG_DIR / "yolo_tracking.log"


# ============================================================
# 2. EXPERIMENT SETTINGS
# ============================================================

DEVICE = "cpu"

FRAME_STRIDE = 6
IMAGE_SIZE = 640

DETECTION_CONFIDENCE = 0.05
IOU_THRESHOLD = 0.50

CLASSES = [0, 1, 2, 3, 5, 7]

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

TEST_ONLY_VIDEO = None
TEST_ONLY_SPLIT = None 

# Later, after successful validation, change to:
# TEST_ONLY_VIDEO = None
# TEST_ONLY_SPLIT = None

ALL_SPLITS = ["train", "validation", "test"]


# ============================================================
# 3. BYTETRACK CONFIGURATION
# ============================================================

TRACKER_SETTINGS = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.05,
    "track_low_thresh": 0.01,
    "new_track_thresh": 0.05,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": False,
}


def create_tracker_config():

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRACKER_CONFIG, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            TRACKER_SETTINGS,
            file,
            sort_keys=False
        )

    logging.info("ByteTrack configuration created: %s", TRACKER_CONFIG)


# ============================================================
# 4. LOGGING
# ============================================================

def setup_logging():

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )


# ============================================================
# 5. LOAD YOLO MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    logging.info("Loading YOLOv8n...")

    model = YOLO(str(MODEL_PATH))

    logging.info("YOLOv8n loaded successfully.")

    return model


# ============================================================
# 6. LOAD MANIFEST
# ============================================================

def load_manifest(split):

    manifest_path = MANIFEST_DIR / f"{split}_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    df = pd.read_csv(manifest_path)

    if df.empty:
        raise ValueError(
            f"Manifest is empty: {manifest_path}"
        )

    logging.info(
        "Loaded %d videos from %s manifest",
        len(df),
        split
    )

    return df


# ============================================================
# 7. RESOLVE VIDEO INFORMATION
# ============================================================

def get_video_name(row):
    return Path(str(row["video_path"])).name

def get_video_label(row):

    label = int(row["label"])

    if label == 1:
        return "positive"

    elif label == 0:
        return "negative"

    raise ValueError(f"Unknown label: {label}")

def resolve_video_path(row, split):

    relative_path = Path(str(row["video_path"]))

    video_path = DATASET_DIR / relative_path

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    return video_path


# ============================================================
# 8. VIDEO INFORMATION
# ============================================================

def get_video_information(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    if fps <= 0:
        fps = 30.0

    duration = total_frames / fps if total_frames > 0 else 0

    return fps, total_frames, duration


# ============================================================
# 9. EXTRACT TRACKER CONFIGURATION
# ============================================================

def print_loaded_tracker_config(model):

    try:

        tracker = model.predictor.trackers[0]

        args = tracker.args

        print("\n========== LOADED TRACKER CONFIG ==========")

        print("Tracker:", type(tracker).__name__)
        print("High threshold:", args.track_high_thresh)
        print("Low threshold:", args.track_low_thresh)
        print("New track threshold:", args.new_track_thresh)
        print("Track buffer:", args.track_buffer)
        print("Match threshold:", args.match_thresh)
        print("Fuse score:", args.fuse_score)

        print("===========================================\n")

        logging.info(
            "Loaded ByteTrack thresholds: high=%s low=%s new=%s",
            args.track_high_thresh,
            args.track_low_thresh,
            args.new_track_thresh
        )

    except Exception as error:

        logging.warning(
            "Could not print tracker configuration: %s",
            error
        )


# ============================================================
# 10. PROCESS ONE VIDEO
# ============================================================

def process_video(model, video_path, split, label):

    video_name = video_path.name

    logging.info("Processing: %s", video_name)

    fps, total_frames, duration = get_video_information(
        video_path
    )

    logging.info(
        "Duration: %.2fs | FPS: %.2f | Frames: %d",
        duration,
        fps,
        total_frames
    )

    records = []

    total_detections = 0
    tracked_detections = 0

    tracker_config_printed = False

    # Each video is tracked independently.
    results = model.track(
        source=str(video_path),
        tracker=str(TRACKER_CONFIG),
        conf=DETECTION_CONFIDENCE,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        classes=CLASSES,
        device=DEVICE,
        vid_stride=FRAME_STRIDE,
        persist=False,
        stream=True,
        verbose=False
    )

    for sampled_index, result in enumerate(results):

        if not tracker_config_printed:

            print_loaded_tracker_config(model)

            tracker_config_printed = True

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        coordinates = (
            boxes.xyxy.cpu().numpy()
        )

        confidences = (
            boxes.conf.cpu().numpy()
        )

        class_ids = (
            boxes.cls.cpu().numpy().astype(int)
        )

        # Only accept IDs actually assigned by ByteTrack.
        if boxes.id is not None and boxes.is_track:

            track_ids = (
                boxes.id.cpu().numpy().astype(int)
            )

        else:

            track_ids = np.full(
                len(coordinates),
                np.nan
            )

        # Convert sampled frame position to original video frame.
        frame_index = sampled_index * FRAME_STRIDE

        timestamp = frame_index / fps

        for i, coordinate in enumerate(coordinates):

            x1, y1, x2, y2 = coordinate

            class_id = int(class_ids[i])

            track_id = track_ids[i]

            if not np.isnan(track_id):
                track_id = int(track_id)
                tracked_detections += 1
            else:
                track_id = None

            total_detections += 1

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            width = x2 - x1
            height = y2 - y1

            records.append({

                "video_name": video_name,
                "split": split,
                "label": label,

                "frame_index": frame_index,
                "timestamp_sec": round(timestamp, 4),

                "track_id": track_id,

                "class_id": class_id,
                "class_name": CLASS_NAMES.get(
                    class_id,
                    "unknown"
                ),

                "confidence": float(confidences[i]),

                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),

                "center_x": float(center_x),
                "center_y": float(center_y),

                "width": float(width),
                "height": float(height)
            })

    # --------------------------------------------------------
    # SAVE VIDEO TRACKING CSV
    # --------------------------------------------------------

    output_folder = (
        OUTPUT_DIR
        / split
        / label
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_folder
        / f"{video_path.stem}.csv"
    )

    columns = [
        "video_name",
        "split",
        "label",
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
        "height"
    ]

    df = pd.DataFrame(
        records,
        columns=columns
    )

    df.to_csv(
        output_path,
        index=False
    )

    valid_tracks = df["track_id"].dropna()

    unique_tracks = valid_tracks.nunique()

    if total_detections == 0:

        status = "no_detections"

    elif tracked_detections == 0:

        status = "detections_without_ids"

    else:

        status = "tracking_successful"

    logging.info(
        "Completed: %s | Detections: %d | "
        "Tracked detections: %d | Unique tracks: %d | Status: %s",
        video_name,
        total_detections,
        tracked_detections,
        unique_tracks,
        status
    )

    return {

        "video_name": video_name,
        "split": split,
        "label": label,

        "total_detections": total_detections,
        "tracked_detections": tracked_detections,
        "unique_tracks": unique_tracks,

        "status": status,
        "output_path": str(output_path)
    }


# ============================================================
# 11. PROCESS DATASET SPLIT
# ============================================================

def process_split(model, split):

    logging.info("Starting %s split", split.upper())

    manifest = load_manifest(split)

    if TEST_ONLY_VIDEO is not None:

        if split != TEST_ONLY_SPLIT:
            return []

        manifest = manifest[
            manifest.apply(
                lambda row: (
                    get_video_name(row) == TEST_ONLY_VIDEO
                ),
                axis=1
            )
        ]

    if manifest.empty:

        logging.warning(
            "No videos selected for split: %s",
            split
        )

        return []

    results = []

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc=f"Processing {split.upper()}"
    ):

        try:

            video_name = get_video_name(row)
            label = get_video_label(row)

            video_path = resolve_video_path(
                row,
                split
            )

            result = process_video(
                model,
                video_path,
                split,
                label
            )

            results.append(result)

        except Exception as error:

            logging.exception(
                "Failed processing video in %s: %s",
                split,
                error
            )

            results.append({

                "video_name": str(
                    row.get("video_name", "unknown")
                ),

                "split": split,

                "label": str(
                    row.get("label", "unknown")
                ),

                "total_detections": 0,
                "tracked_detections": 0,
                "unique_tracks": 0,

                "status": "failed",
                "output_path": ""
            })

    return results


# ============================================================
# 12. SAVE SUMMARY
# ============================================================

def save_summary(all_results):

    if not all_results:
        logging.warning("No processing results to summarize.")
        return

    df = pd.DataFrame(all_results)

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    logging.info("Summary saved: %s", SUMMARY_PATH)


# ============================================================
# 13. MAIN
# ============================================================

def main():

    setup_logging()

    print("\n" + "=" * 60)
    print("TRAFFIC COLLISION TRACKING PIPELINE")
    print("=" * 60)

    logging.info("=" * 60)
    logging.info("TRAFFIC COLLISION TRACKING PIPELINE")
    logging.info("=" * 60)

    logging.info(
        "ByteTrack configuration: %s",
        TRACKER_CONFIG
    )

    try:

        # Create the intended configuration.
        create_tracker_config()

        # Load YOLO.
        model = load_model()

        all_results = []

        splits_to_process = (
            [TEST_ONLY_SPLIT]
            if TEST_ONLY_VIDEO is not None
            else ALL_SPLITS
        )

        for split in splits_to_process:

            results = process_split(
                model,
                split
            )

            all_results.extend(results)

        save_summary(all_results)

        successful = sum(
            1 for r in all_results
            if r["status"] != "failed"
        )

        failed = sum(
            1 for r in all_results
            if r["status"] == "failed"
        )

        logging.info("=" * 60)
        logging.info("PIPELINE EXECUTION COMPLETED")
        logging.info("=" * 60)

        logging.info(
            "Videos processed: %d",
            len(all_results)
        )

        logging.info(
            "Successful: %d",
            successful
        )

        logging.info(
            "Failed: %d",
            failed
        )

        logging.info(
            "Output directory: %s",
            OUTPUT_DIR
        )

        logging.info(
            "Summary: %s",
            SUMMARY_PATH
        )

        logging.info("=" * 60)

    except Exception:

        logging.exception(
            "PIPELINE EXECUTION FAILED"
        )

        raise


# ============================================================
# 14. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
