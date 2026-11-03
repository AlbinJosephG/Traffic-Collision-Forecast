
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT_DIR / "processed" / "tracking_test"
OUTPUT_DIR = ROOT_DIR / "processed" / "trajectories"

SUMMARY_PATH = ROOT_DIR / "processed" / "trajectory_summary.csv"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Tracking observations are sampled at approximately 5 FPS.
# Do not calculate motion across large observation gaps.
MAX_TRACK_GAP_SEC = 0.6

# Required input columns from Phase 2
REQUIRED_COLUMNS = [
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
    "height",
]


# ============================================================
# OUTPUT SCHEMA
# ============================================================

TRAJECTORY_COLUMNS = REQUIRED_COLUMNS + [
    "center_x_norm",
    "center_y_norm",
    "width_norm",
    "height_norm",
    "bbox_area_norm",
    "velocity_x_norm",
    "velocity_y_norm",
    "speed_norm",
    "acceleration_x_norm",
    "acceleration_y_norm",
    "acceleration_norm",
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_trajectory_features(df):

    if df.empty:
        return pd.DataFrame(columns=TRAJECTORY_COLUMNS)

    df = df.copy()

    # --------------------------------------------------------
    # 1. Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "frame_index",
        "timestamp_sec",
        "track_id",
        "class_id",
        "confidence",
        "center_x",
        "center_y",
        "width",
        "height",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # 2. Remove detections without valid tracking IDs
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "track_id",
            "timestamp_sec",
            "center_x",
            "center_y",
            "width",
            "height",
        ]
    )

    if df.empty:
        return pd.DataFrame(columns=TRAJECTORY_COLUMNS)

    # --------------------------------------------------------
    # 3. Sort observations by object and timestamp
    # --------------------------------------------------------

    df = df.sort_values(
        ["track_id", "timestamp_sec", "frame_index"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Normalize object positions and bounding boxes
    # --------------------------------------------------------

    df["center_x_norm"] = (
        df["center_x"] / FRAME_WIDTH
    )

    df["center_y_norm"] = (
        df["center_y"] / FRAME_HEIGHT
    )

    df["width_norm"] = (
        df["width"] / FRAME_WIDTH
    )

    df["height_norm"] = (
        df["height"] / FRAME_HEIGHT
    )

    df["bbox_area_norm"] = (
        df["width_norm"] * df["height_norm"]
    )

    # --------------------------------------------------------
    # 5. Calculate time difference per tracked object
    # --------------------------------------------------------

    grouped = df.groupby("track_id", sort=False)

    df["dt"] = grouped["timestamp_sec"].diff()

    valid_dt = (
        (df["dt"] > 0) &
        (df["dt"] <= MAX_TRACK_GAP_SEC)
    )

    # --------------------------------------------------------
    # 6. Calculate image-space velocity
    # --------------------------------------------------------

    df["dx_norm"] = grouped["center_x_norm"].diff()
    df["dy_norm"] = grouped["center_y_norm"].diff()

    df["velocity_x_norm"] = np.where(
        valid_dt,
        df["dx_norm"] / df["dt"],
        np.nan
    )

    df["velocity_y_norm"] = np.where(
        valid_dt,
        df["dy_norm"] / df["dt"],
        np.nan
    )

    df["speed_norm"] = np.sqrt(
        df["velocity_x_norm"] ** 2 +
        df["velocity_y_norm"] ** 2
    )

    # --------------------------------------------------------
    # 7. Calculate image-space acceleration
    # --------------------------------------------------------

    grouped = df.groupby("track_id", sort=False)

    previous_vx = grouped["velocity_x_norm"].shift(1)
    previous_vy = grouped["velocity_y_norm"].shift(1)

    previous_dt = grouped["dt"].shift(1)

    valid_acceleration = (
        valid_dt &
        previous_dt.notna() &
        (previous_dt > 0) &
        np.isfinite(previous_vx) &
        np.isfinite(previous_vy)
    )

    df["acceleration_x_norm"] = np.where(
        valid_acceleration,
        (df["velocity_x_norm"] - previous_vx) / df["dt"],
        np.nan
    )

    df["acceleration_y_norm"] = np.where(
        valid_acceleration,
        (df["velocity_y_norm"] - previous_vy) / df["dt"],
        np.nan
    )

    df["acceleration_norm"] = np.sqrt(
        df["acceleration_x_norm"] ** 2 +
        df["acceleration_y_norm"] ** 2
    )

    # --------------------------------------------------------
    # 8. Select final columns
    # --------------------------------------------------------

    for column in TRAJECTORY_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    result = df[TRAJECTORY_COLUMNS].copy()

    return result


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(csv_path):

    relative_path = csv_path.relative_to(INPUT_DIR)

    output_path = OUTPUT_DIR / relative_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        df = pd.read_csv(csv_path)

        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        total_detections = len(df)

        if df.empty:
            trajectories = pd.DataFrame(
                columns=TRAJECTORY_COLUMNS
            )
        else:
            trajectories = calculate_trajectory_features(df)

        trajectories.to_csv(
            output_path,
            index=False
        )

        return {
            "video_name": csv_path.name,
            "split": relative_path.parts[0],
            "category": relative_path.parts[1],
            "total_detections": total_detections,
            "tracked_observations": len(trajectories),
            "unique_tracks": (
                trajectories["track_id"].nunique()
                if not trajectories.empty else 0
            ),
            "status": (
                "no_tracked_objects"
                if trajectories.empty
                else "success"
            ),
            "output_path": str(output_path),
        }

    except Exception as error:

        return {
            "video_name": csv_path.name,
            "split": relative_path.parts[0],
            "category": relative_path.parts[1],
            "total_detections": 0,
            "tracked_observations": 0,
            "unique_tracks": 0,
            "status": f"failed: {error}",
            "output_path": "",
        }


# ============================================================
# PROCESS ALL TRACKING FILES
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 3A - OBJECT TRAJECTORY FEATURE EXTRACTION")
    print("=" * 65)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Tracking directory not found: {INPUT_DIR}"
        )

    csv_files = sorted(
        INPUT_DIR.rglob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            "No tracking CSV files found."
        )

    print(f"Tracking files found: {len(csv_files)}")

    results = []

    for index, csv_path in enumerate(csv_files, start=1):

        result = process_video(csv_path)
        results.append(result)

        print(
            f"[{index}/{len(csv_files)}] "
            f"{result['video_name']} - {result['status']}"
        )

    summary = pd.DataFrame(results)

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print("\n" + "=" * 65)
    print("PHASE 3A SUMMARY")
    print("=" * 65)

    print(f"Videos processed: {len(summary)}")
    print(
        f"Successful: "
        f"{(summary['status'] == 'success').sum()}"
    )
    print(
        f"No tracked objects: "
        f"{(summary['status'] == 'no_tracked_objects').sum()}"
    )
    print(
        f"Failed: "
        f"{summary['status'].str.startswith('failed').sum()}"
    )

    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
