
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from collections import Counter
from ultralytics import YOLO


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\TrafficCollisionProject")

DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.pt"

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "tracking_validation"
    / "empty_video_diagnostics"
)

FRAME_REPORT_DIR = OUTPUT_DIR / "frame_reports"
IMAGE_REPORT_DIR = OUTPUT_DIR / "sample_frames"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_REPORT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Original Phase 2 configuration
CONFIDENCE_THRESHOLD = 0.25
DIAGNOSTIC_CONFIDENCE = 0.10

IMAGE_SIZE = 640
FRAME_STRIDE = 6
DEVICE = "cpu"

CLASSES = [0, 1, 2, 3, 5, 7]

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Save up to this many annotated frames per video
MAX_SAMPLE_IMAGES = 3


# ============================================================
# 2. VIDEOS THAT NEED INVESTIGATION
# ============================================================

PROBLEM_VIDEOS = [
    {"video_id": "00597", "split": "train", "label": "positive"},
    {"video_id": "00100", "split": "train", "label": "positive"},
    {"video_id": "00459", "split": "train", "label": "positive"},

    {"video_id": "01416", "split": "train", "label": "negative"},
    {"video_id": "01510", "split": "train", "label": "negative"},
    {"video_id": "01805", "split": "train", "label": "negative"},
    {"video_id": "01801", "split": "train", "label": "negative"},
    {"video_id": "01998", "split": "train", "label": "negative"},
    {"video_id": "01372", "split": "train", "label": "negative"},
    {"video_id": "01449", "split": "train", "label": "negative"},

    {"video_id": "01456", "split": "test", "label": "negative"},
]


# ============================================================
# 3. VIDEO PATH RESOLUTION
# ============================================================

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


def find_video(video_id, split, label):
    """
    Find a video using its ID.

    First checks the expected split/label directory.
    Then searches the dataset directory recursively.
    """

    expected_dir = DATASET_DIR / split / label

    for extension in VIDEO_EXTENSIONS:
        candidate = expected_dir / f"{video_id}{extension}"

        if candidate.exists():
            return candidate

    matches = []

    for extension in VIDEO_EXTENSIONS:
        matches.extend(
            DATASET_DIR.rglob(f"{video_id}{extension}")
        )

    # Avoid accidentally selecting a file from another split
    matching_paths = [
        path for path in matches
        if split.lower() in str(path).lower()
        and label.lower() in str(path).lower()
    ]

    if matching_paths:
        return matching_paths[0]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple videos found for ID {video_id}: {matches}"
        )

    return None


# ============================================================
# 4. RESULT EXTRACTION
# ============================================================

def extract_prediction_information(result):
    """Extract detection information from a YOLO prediction."""

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "count": 0,
            "classes": [],
            "confidences": [],
        }

    class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
    confidences = boxes.conf.cpu().numpy().tolist()

    return {
        "count": len(class_ids),
        "classes": class_ids,
        "confidences": confidences,
    }


def extract_tracking_information(result):
    """Extract valid and invalid tracking ID information."""

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "detections": 0,
            "valid_ids": [],
            "invalid_ids": 0,
            "classes": [],
        }

    class_ids = boxes.cls.cpu().numpy().astype(int).tolist()

    if boxes.id is None:
        return {
            "detections": len(boxes),
            "valid_ids": [],
            "invalid_ids": len(boxes),
            "classes": class_ids,
        }

    track_ids = boxes.id.cpu().numpy()

    valid_ids = []
    invalid_ids = 0

    for track_id in track_ids:
        if np.isfinite(track_id) and track_id >= 0:
            valid_ids.append(int(track_id))
        else:
            invalid_ids += 1

    return {
        "detections": len(boxes),
        "valid_ids": valid_ids,
        "invalid_ids": invalid_ids,
        "classes": class_ids,
    }


# ============================================================
# 5. RESET TRACKER BETWEEN VIDEOS
# ============================================================

def reset_tracker(model):
    """
    Reset ByteTrack state so IDs from one video
    cannot carry over into another video.
    """

    predictor = getattr(model, "predictor", None)

    if predictor is None:
        return

    trackers = getattr(predictor, "trackers", None)

    if trackers:
        for tracker in trackers:
            tracker.reset()


# ============================================================
# 6. PROCESS ONE VIDEO
# ============================================================

def inspect_video(model, video_info):

    video_id = video_info["video_id"]
    split = video_info["split"]
    label = video_info["label"]

    print("\n" + "=" * 65)
    print(f"INSPECTING VIDEO: {video_id}")
    print(f"SPLIT: {split} | LABEL: {label}")
    print("=" * 65)

    video_path = find_video(video_id, split, label)

    if video_path is None:
        print(f"ERROR: Video not found: {video_id}")

        return {
            "video_id": video_id,
            "split": split,
            "label": label,
            "video_path": "",
            "status": "VIDEO_NOT_FOUND",
        }, []

    print(f"Video path: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")

        return {
            "video_id": video_id,
            "split": split,
            "label": label,
            "video_path": str(video_path),
            "status": "VIDEO_OPEN_FAILED",
        }, []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        fps = 30.0

    print(f"FPS: {fps:.2f}")
    print(f"Total original frames: {total_frames}")
    print(f"Sampling every {FRAME_STRIDE} frames")

    reset_tracker(model)

    frame_index = 0
    sampled_frames = 0

    total_detections_010 = 0
    total_detections_025 = 0
    total_tracking_detections = 0
    total_invalid_ids = 0

    frames_with_detections = 0
    frames_with_tracking = 0

    all_valid_track_ids = set()

    detected_classes = Counter()
    tracked_classes = Counter()

    frame_records = []
    saved_images = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        current_frame_index = frame_index
        frame_index += 1

        # Match the original Phase 2 sampling strategy
        if current_frame_index % FRAME_STRIDE != 0:
            continue

        sampled_frames += 1

        timestamp_sec = current_frame_index / fps

        # ----------------------------------------------------
        # A. YOLO DETECTION AT CONFIDENCE 0.10
        # ----------------------------------------------------

        prediction_results = model.predict(
            source=frame,
            conf=DIAGNOSTIC_CONFIDENCE,
            classes=CLASSES,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        prediction_info = extract_prediction_information(
            prediction_results[0]
        )

        detections_010 = prediction_info["count"]

        # Count detections that meet the original 0.25 threshold
        detections_025 = sum(
            confidence >= CONFIDENCE_THRESHOLD
            for confidence in prediction_info["confidences"]
        )

        # ----------------------------------------------------
        # B. BYTE TRACK AT ORIGINAL CONFIDENCE 0.25
        # ----------------------------------------------------

        tracking_results = model.track(
            source=frame,
            tracker="bytetrack.yaml",
            persist=True,
            classes=CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            iou=0.50,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        tracking_info = extract_tracking_information(
            tracking_results[0]
        )

        tracking_detections = tracking_info["detections"]

        valid_ids = tracking_info["valid_ids"]
        invalid_ids = tracking_info["invalid_ids"]

        # ----------------------------------------------------
        # C. UPDATE VIDEO STATISTICS
        # ----------------------------------------------------

        total_detections_010 += detections_010
        total_detections_025 += detections_025
        total_tracking_detections += tracking_detections
        total_invalid_ids += invalid_ids

        if detections_010 > 0:
            frames_with_detections += 1

        if tracking_detections > 0:
            frames_with_tracking += 1

        all_valid_track_ids.update(valid_ids)

        detected_classes.update(prediction_info["classes"])
        tracked_classes.update(tracking_info["classes"])

        # ----------------------------------------------------
        # D. SAVE FRAME-LEVEL DIAGNOSTIC INFORMATION
        # ----------------------------------------------------

        frame_records.append({
            "video_id": video_id,
            "split": split,
            "label": label,
            "frame_index": current_frame_index,
            "timestamp_sec": round(timestamp_sec, 4),
            "detections_at_010": detections_010,
            "detections_at_025": detections_025,
            "tracking_detections": tracking_detections,
            "valid_track_ids": len(valid_ids),
            "invalid_track_ids": invalid_ids,
        })

        # ----------------------------------------------------
        # E. SAVE ANNOTATED SAMPLE FRAMES
        # ----------------------------------------------------

        if (
            saved_images < MAX_SAMPLE_IMAGES
            and (detections_010 > 0 or tracking_detections > 0)
        ):

            # Prefer tracking annotations when available.
            if tracking_detections > 0:
                annotated_frame = tracking_results[0].plot()
            else:
                annotated_frame = prediction_results[0].plot()

            image_name = (
                f"{video_id}_frame_{current_frame_index}.jpg"
            )

            image_path = IMAGE_REPORT_DIR / image_name

            cv2.imwrite(str(image_path), annotated_frame)

            saved_images += 1

        if sampled_frames % 100 == 0:
            print(
                f"Processed {sampled_frames} sampled frames | "
                f"YOLO detections: {total_detections_010} | "
                f"Tracked detections: {total_tracking_detections}"
            )

    cap.release()

    # --------------------------------------------------------
    # F. FINAL VIDEO SUMMARY
    # --------------------------------------------------------

    status = (
        "TRACKING_FOUND"
        if total_tracking_detections > 0
        else "NO_TRACKING_DETECTIONS"
    )

    summary = {
        "video_id": video_id,
        "split": split,
        "label": label,
        "video_path": str(video_path),
        "status": status,
        "fps": round(fps, 2),
        "original_frames": total_frames,
        "sampled_frames": sampled_frames,
        "frames_with_detections": frames_with_detections,
        "frames_with_tracking": frames_with_tracking,
        "detections_at_010": total_detections_010,
        "detections_at_025": total_detections_025,
        "tracking_detections": total_tracking_detections,
        "unique_valid_tracks": len(all_valid_track_ids),
        "invalid_tracking_ids": total_invalid_ids,
        "detected_classes": str(dict(detected_classes)),
        "tracked_classes": str(dict(tracked_classes)),
        "sample_images_saved": saved_images,
    }

    frame_report_path = FRAME_REPORT_DIR / f"{video_id}_frames.csv"

    pd.DataFrame(frame_records).to_csv(
        frame_report_path,
        index=False,
    )

    print("\nVIDEO RESULT")
    print(f"Status: {status}")
    print(f"Sampled frames: {sampled_frames}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Frames with tracking: {frames_with_tracking}")
    print(f"YOLO detections at 0.10: {total_detections_010}")
    print(f"YOLO detections at 0.25: {total_detections_025}")
    print(f"Tracking detections: {total_tracking_detections}")
    print(f"Unique valid tracks: {len(all_valid_track_ids)}")
    print(f"Invalid IDs: {total_invalid_ids}")
    print(f"Detected classes: {dict(detected_classes)}")
    print(f"Tracked classes: {dict(tracked_classes)}")
    print(f"Sample images saved: {saved_images}")

    return summary, frame_records


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("EMPTY TRACKING VIDEO - DETAILED DIAGNOSTIC")
    print("=" * 70)

    print(f"Project: {PROJECT_DIR}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Model: {MODEL_PATH}")
    print(f"Videos to inspect: {len(PROBLEM_VIDEOS)}")
    print(f"Device: {DEVICE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model weights not found: {MODEL_PATH}"
        )

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    print("\nLoading YOLOv8n model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    video_summaries = []

    for index, video_info in enumerate(PROBLEM_VIDEOS, start=1):

        print(
            f"\nProgress: {index}/{len(PROBLEM_VIDEOS)}"
        )

        try:
            summary, _ = inspect_video(model, video_info)
            video_summaries.append(summary)

        except Exception as error:

            print(
                f"ERROR processing {video_info['video_id']}: {error}"
            )

            video_summaries.append({
                "video_id": video_info["video_id"],
                "split": video_info["split"],
                "label": video_info["label"],
                "status": "PROCESSING_ERROR",
                "error": str(error),
            })

    # --------------------------------------------------------
    # SAVE FINAL REPORTS
    # --------------------------------------------------------

    summary_df = pd.DataFrame(video_summaries)

    summary_path = OUTPUT_DIR / "empty_video_summary.csv"

    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC EXECUTION COMPLETED")
    print("=" * 70)

    print("\nFINAL SUMMARY")
    print(
        summary_df[
            [
                "video_id",
                "split",
                "label",
                "status",
                "detections_at_010",
                "detections_at_025",
                "tracking_detections",
                "unique_valid_tracks",
            ]
        ].to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print(f"Summary: {summary_path}")
    print(f"Frame reports: {FRAME_REPORT_DIR}")
    print(f"Annotated images: {IMAGE_REPORT_DIR}")

    print("\nExisting tracking CSV files were not modified.")


if __name__ == "__main__":
    main()
