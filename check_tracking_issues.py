import pandas as pd
from pathlib import Path

summary = pd.read_csv(
    "processed/tracking_test_summary.csv"
)

videos = [
    "00459.mp4",
    "01998.mp4",
    "00271.mp4",
    "00232.mp4"
]

for video in videos:
    row = summary[summary["video_name"] == video].iloc[0]

    path = Path(row["output_path"])
    df = pd.read_csv(path)

    print("\n" + "=" * 50)
    print("VIDEO:", video)
    print("Label:", row["label"])
    print("Summary detections:", row["total_detections"])
    print("Summary tracked:", row["tracked_detections"])
    print("Actual CSV rows:", len(df))

    if len(df) > 0:
        print("Valid IDs:", df["track_id"].notna().sum())
        print("Unique tracks:", df["track_id"].nunique())
        print("Duplicate rows:", df.duplicated().sum())
        print("Classes:")
        print(df["class_name"].value_counts().to_string())
    else:
        print("CSV is empty.")