import pandas as pd
from pathlib import Path

summary = pd.read_csv(
    "processed/tracking_test_summary.csv"
)

missing_files = []
row_mismatches = []
track_mismatches = []
zero_detection_videos = []

for _, row in summary.iterrows():

    path = Path(row["output_path"])

    if not path.exists():
        missing_files.append(row["video_name"])
        continue

    df = pd.read_csv(path)

    if len(df) == 0:
        zero_detection_videos.append(row["video_name"])

    if len(df) != int(row["total_detections"]):
        row_mismatches.append(row["video_name"])

    valid_ids = df["track_id"].notna().sum()

    if valid_ids != int(row["tracked_detections"]):
        track_mismatches.append(row["video_name"])

print("\n========== FINAL TRACKING VALIDATION ==========")
print("Total videos:", len(summary))
print("Missing CSV files:", len(missing_files))
print("Row-count mismatches:", len(row_mismatches))
print("Tracking ID mismatches:", len(track_mismatches))
print("Zero-detection videos:", len(zero_detection_videos))

print("\nZero-detection videos:", zero_detection_videos)
print("Missing files:", missing_files)
print("Row mismatches:", row_mismatches)
print("Tracking mismatches:", track_mismatches)