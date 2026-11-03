
from pathlib import Path
import pandas as pd
import ultralytics
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\TrafficCollisionProject")

MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.pt"

VIDEO_PATH = (
    PROJECT_DIR / "dataset" / "train" / "positive" / "00100.mp4"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "tracking_validation"
    / "confidence_experiment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.10
IMAGE_SIZE = 640
FRAME_STRIDE = 6
DEVICE = "cpu"

CLASSES = [0, 1, 2, 3, 5, 7]


# ============================================================
# CHECK ENVIRONMENT
# ============================================================

print("=" * 65)
print("ENVIRONMENT INFORMATION")
print("=" * 65)

print("Ultralytics version:", ultralytics.__version__)
print("Model:", MODEL_PATH)
print("Video:", VIDEO_PATH)

tracker_yaml = (
    Path(ultralytics.__file__).parent
    / "cfg"
    / "trackers"
    / "bytetrack.yaml"
)

print("\nBYTE TRACK CONFIGURATION")
print("-" * 65)

if tracker_yaml.exists():
    print(tracker_yaml.read_text())
else:
    print("Could not locate installed ByteTrack YAML.")

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

if not VIDEO_PATH.exists():
    raise FileNotFoundError(VIDEO_PATH)


# ============================================================
# LOAD A FRESH MODEL INSTANCE
# ============================================================

print("\nLoading YOLOv8n...")

model = YOLO(str(MODEL_PATH))


# ============================================================
# RUN BYTETRACK AT CONFIDENCE 0.10
# ============================================================

print("\n" + "=" * 65)
print("BYTE TRACK EXPERIMENT: CONFIDENCE 0.10")
print("=" * 65)

results = model.track(
    source=str(VIDEO_PATH),
    tracker="bytetrack.yaml",
    persist=False,
    classes=CLASSES,
    conf=CONFIDENCE,
    iou=0.50,
    imgsz=IMAGE_SIZE,
    vid_stride=FRAME_STRIDE,
    device=DEVICE,
    stream=True,
    verbose=False
)


# ============================================================
# COLLECT TRACKING RESULTS
# ============================================================

sampled_frames = 0
frames_with_detections = 0
total_detections = 0

valid_track_ids = set()
invalid_ids = 0

records = []


for result in results:

    sampled_frames += 1
    boxes = result.boxes

    if sampled_frames <= 15:

        print(f"\nFRAME {sampled_frames}")

        if boxes is None:
            print("Boxes: None")

        else:
            print("Detection count:", len(boxes))
            print("Is tracking result:", boxes.is_track)
            print("IDs are None:", boxes.id is None)
            print("Tracking IDs:", boxes.id)

            if boxes.id is not None:
                print("ID values:", boxes.id.cpu().numpy())

    count = 0 if boxes is None else len(boxes)

    total_detections += count

    frame_valid_ids = 0
    frame_invalid_ids = 0
    
    if count > 0:

        frames_with_detections += 1

        if boxes.id is None:
            invalid_ids += count
            frame_invalid_ids = count

        else:
            ids = boxes.id.cpu().numpy()

            for track_id in ids:

                if track_id >= 0:
                    valid_track_ids.add(int(track_id))
                    frame_valid_ids += 1
                else:
                    invalid_ids += 1
                    frame_invalid_ids += 1

    records.append({
        "sampled_frame": sampled_frames,
        "detections": count,
        "valid_track_ids": frame_valid_ids,
        "invalid_track_ids": frame_invalid_ids
    })

    if sampled_frames % 100 == 0:
        print(
            f"Frames: {sampled_frames} | "
            f"Detections: {total_detections} | "
            f"Tracks: {len(valid_track_ids)}"
        )


# ============================================================
# SAVE REPORT
# ============================================================

report_path = OUTPUT_DIR / "video_00100_conf_010.csv"

pd.DataFrame(records).to_csv(
    report_path,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CONFIDENCE EXPERIMENT RESULTS")
print("=" * 65)

print("Confidence threshold:", CONFIDENCE)
print("Sampled frames:", sampled_frames)
print("Frames with detections:", frames_with_detections)
print("Total tracking detections:", total_detections)
print("Unique valid tracks:", len(valid_track_ids))
print("Invalid IDs:", invalid_ids)

if total_detections > 0:
    print("\nRESULT: ByteTrack returned detections.")

    if len(valid_track_ids) > 0:
        print("Valid object tracks were established.")
    else:
        print("Detections exist, but valid tracking IDs are absent.")

else:
    print("\nRESULT: ByteTrack still returns zero detections.")

print("\nReport saved:", report_path)
print("Original tracking CSV files were not modified.")
