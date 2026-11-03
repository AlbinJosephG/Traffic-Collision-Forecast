
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\TrafficCollisionProject")

MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.pt"

VIDEO_PATH = (
    PROJECT_DIR
    / "dataset"
    / "train"
    / "positive"
    / "00100.mp4"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "tracking_validation"
    / "pipeline_experiment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.25
IMAGE_SIZE = 640
FRAME_STRIDE = 6
IOU_THRESHOLD = 0.50
DEVICE = "cpu"

CLASSES = [0, 1, 2, 3, 5, 7]


# ============================================================
# VALIDATE INPUTS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")


# ============================================================
# EXPERIMENT A: YOLO DETECTION
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT A: YOLO DETECTION")
print("=" * 65)

detection_model = YOLO(str(MODEL_PATH))

detection_results = detection_model.predict(
    source=str(VIDEO_PATH),
    conf=CONFIDENCE,
    classes=CLASSES,
    imgsz=IMAGE_SIZE,
    device=DEVICE,
    vid_stride=FRAME_STRIDE,
    stream=True,
    verbose=False
)

detection_frames = 0
detection_count = 0
frames_with_detections = 0

for result in detection_results:

    detection_frames += 1

    boxes = result.boxes

    count = 0 if boxes is None else len(boxes)

    detection_count += count

    if count > 0:
        frames_with_detections += 1

    if detection_frames % 100 == 0:
        print(
            f"Detection frames: {detection_frames} | "
            f"Detections: {detection_count}"
        )

print("\nYOLO RESULTS")
print(f"Sampled frames: {detection_frames}")
print(f"Frames with detections: {frames_with_detections}")
print(f"Total detections: {detection_count}")


# ============================================================
# EXPERIMENT B: BYTETRACK
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT B: BYTETRACK")
print("=" * 65)

tracking_model = YOLO(str(MODEL_PATH))

tracking_results = tracking_model.track(
    source=str(VIDEO_PATH),
    tracker="bytetrack.yaml",
    persist=False,
    classes=CLASSES,
    conf=CONFIDENCE,
    iou=IOU_THRESHOLD,
    imgsz=IMAGE_SIZE,
    vid_stride=FRAME_STRIDE,
    device=DEVICE,
    stream=True,
    verbose=False
)

tracking_frames = 0
tracking_detections = 0
frames_with_tracking = 0

valid_track_ids = set()
invalid_ids = 0
frames_without_ids = 0

frame_records = []

for result in tracking_results:

    tracking_frames += 1

    boxes = result.boxes

    count = 0 if boxes is None else len(boxes)

    tracking_detections += count

    if count > 0:
        frames_with_tracking += 1

    frame_valid_ids = 0
    frame_invalid_ids = 0

    if boxes is not None and count > 0:

        if boxes.id is None:

            frames_without_ids += 1
            frame_invalid_ids = count
            invalid_ids += count

        else:

            ids = boxes.id.cpu().numpy()

            for track_id in ids:

                if track_id >= 0:
                    valid_track_ids.add(int(track_id))
                    frame_valid_ids += 1
                else:
                    frame_invalid_ids += 1
                    invalid_ids += 1

    frame_records.append({
        "frame_number": tracking_frames,
        "detections": count,
        "valid_track_ids": frame_valid_ids,
        "invalid_track_ids": frame_invalid_ids
    })

    if tracking_frames % 100 == 0:
        print(
            f"Tracking frames: {tracking_frames} | "
            f"Detections: {tracking_detections} | "
            f"Unique tracks: {len(valid_track_ids)}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

report = pd.DataFrame(frame_records)

report_path = OUTPUT_DIR / "video_00100_tracking_experiment.csv"

report.to_csv(report_path, index=False)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 65)
print("FINAL EXPERIMENT RESULTS")
print("=" * 65)

print("\nYOLO DETECTION")
print(f"Sampled frames: {detection_frames}")
print(f"Frames with detections: {frames_with_detections}")
print(f"Total detections: {detection_count}")

print("\nBYTETRACK")
print(f"Sampled frames: {tracking_frames}")
print(f"Frames with detections: {frames_with_tracking}")
print(f"Total tracking detections: {tracking_detections}")
print(f"Unique valid tracks: {len(valid_track_ids)}")
print(f"Invalid IDs: {invalid_ids}")
print(f"Frames without IDs: {frames_without_ids}")

print("\nCOMPARISON")

if detection_count > 0 and tracking_detections == 0:
    print("RESULT: YOLO detects objects, but ByteTrack returns none.")

elif tracking_detections > 0:
    print("RESULT: ByteTrack successfully returns detections.")

else:
    print("RESULT: Neither pipeline detected objects.")

print(f"\nFrame report saved: {report_path}")
print("Original tracking CSV files were not modified.")
