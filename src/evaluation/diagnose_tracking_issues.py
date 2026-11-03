
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_DIR
    / "reports"
    / "tracking_validation"
    / "video_validation_report.csv"
)

report = pd.read_csv(REPORT_PATH)

failed = report[report["valid"] == False]

print("=" * 70)
print("TRACKING ISSUE DIAGNOSTICS")
print("=" * 70)

for _, video in failed.iterrows():

    csv_path = PROJECT_DIR / video["tracking_file"]

    print("\n" + "-" * 70)
    print(f"Video: {video['video_name']}")
    print(f"Split: {video['split']}")
    print(f"Label: {video['label']}")
    print(f"Issue: {video['issues']}")
    print(f"CSV: {csv_path}")

    if not csv_path.exists():
        print("ERROR: CSV file does not exist.")
        continue

    print(f"File size: {csv_path.stat().st_size} bytes")

    try:
        df = pd.read_csv(csv_path)

        print(f"Rows: {len(df)}")

        if df.empty:
            print("DIAGNOSIS: CSV has headers but no detection records.")
            continue

        if "track_id" not in df.columns:
            print("DIAGNOSIS: track_id column is missing.")
            continue

        track_ids = pd.to_numeric(
            df["track_id"],
            errors="coerce"
        )

        invalid = (
            track_ids.isna()
            | (track_ids < 0)
            | (track_ids % 1 != 0)
        )

        if invalid.any():

            print("\nInvalid tracking ID records:")

            columns = [
                col for col in [
                    "frame_index",
                    "timestamp_sec",
                    "track_id",
                    "class_name",
                    "confidence"
                ]
                if col in df.columns
            ]

            print(df.loc[invalid, columns].to_string(index=False))

            print("\nInvalid ID values:")
            print(df.loc[invalid, "track_id"].unique())

    except Exception as error:
        print(f"ERROR reading CSV: {error}")

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETED")
print("=" * 70)
