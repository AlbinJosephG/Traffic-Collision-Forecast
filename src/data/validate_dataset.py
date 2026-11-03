
import cv2
import csv
from pathlib import Path
from collections import Counter

DATASET_DIR = Path(
    r"E:\TrafficCollisionProject\dataset"
)

manifest_path = DATASET_DIR / "subset_manifest.csv"

results = []
label_counts = Counter()
failed_videos = []

with open(manifest_path, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        relative_path = row["video_path"]
        label = int(row["label"])

        video_path = DATASET_DIR / relative_path

        label_counts[label] += 1

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            failed_videos.append(relative_path)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = (
            frame_count / fps if fps > 0 else 0
        )

        results.append({
            "video": relative_path,
            "label": label,
            "fps": fps,
            "frames": frame_count,
            "width": width,
            "height": height,
            "duration": duration
        })

        cap.release()

# Save validation report
report_path = DATASET_DIR / "validation_report.csv"

if results:
    with open(report_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

print("\n========== DATASET VALIDATION ==========")

print("Videos in manifest:", sum(label_counts.values()))
print("Readable videos:", len(results))
print("Failed videos:", len(failed_videos))

print("\nClass distribution:")
print("Positive:", label_counts[1])
print("Negative:", label_counts[0])

if results:
    print("\nFirst video information:")
    print("FPS:", results[0]["fps"])
    print("Resolution:",
          results[0]["width"],
          "x",
          results[0]["height"])
    print("Duration:", round(
        results[0]["duration"], 2
    ), "seconds")

print("\nValidation report:", report_path)

if failed_videos:
    print("\nFailed videos:")
    for video in failed_videos:
        print(video)
