
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAJECTORY_DIR = ROOT_DIR / "processed" / "trajectories"

INTERACTION_DIR = (
    ROOT_DIR / "processed" / "interaction_features"
)

OUTPUT_DIR = ROOT_DIR / "processed" / "scene_features"

SUMMARY_PATH = (
    ROOT_DIR / "processed" / "scene_feature_summary.csv"
)

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
]

# Features to aggregate across objects in each frame.
MOTION_FEATURES = [
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

STATISTICS = [
    "mean",
    "std",
    "min",
    "max",
]

REQUIRED_TRAJECTORY_COLUMNS = [
    "video_name",
    "split",
    "label",
    "frame_index",
    "timestamp_sec",
    "track_id",
    "class_name",
] + MOTION_FEATURES

REQUIRED_INTERACTION_COLUMNS = [
    "video_name",
    "split",
    "label",
    "frame_index",
    "timestamp_sec",
    "vehicle_count",
    "bicycle_count",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "interaction_pair_count",
    "approaching_pair_count",
    "nearby_pair_count",
    "minimum_pair_distance_norm",
    "mean_pair_distance_norm",
    "minimum_ttc_proxy_sec",
    "mean_ttc_proxy_sec",
    "maximum_closing_speed_norm",
    "mean_closing_speed_norm",
]


# ============================================================
# 2. LABEL NORMALIZATION
# ============================================================

def normalize_label(value):

    if pd.isna(value):
        raise ValueError("Missing video label")

    value = str(value).strip().lower()

    if value in ["positive", "1", "1.0"]:
        return 1

    if value in ["negative", "0", "0.0"]:
        return 0

    raise ValueError(f"Unknown label: {value}")


# ============================================================
# 3. NUMERIC CONVERSION
# ============================================================

def prepare_trajectory_data(df):

    df = df.copy()

    numeric_columns = [
        "frame_index",
        "timestamp_sec",
        "track_id",
    ] + MOTION_FEATURES

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "frame_index",
            "timestamp_sec",
            "track_id",
        ]
    )

    return df


# ============================================================
# 4. AGGREGATE OBJECT FEATURES PER FRAME
# ============================================================

def aggregate_trajectory_features(df):

    if df.empty:
        return pd.DataFrame()

    records = []

    for frame_index, frame_df in df.groupby(
        "frame_index",
        sort=True
    ):

        record = {
            "frame_index": int(frame_index),
            "trajectory_timestamp_sec": float(
                frame_df["timestamp_sec"].iloc[0]
            ),
            "tracked_object_count": int(
                frame_df["track_id"].nunique()
            ),
        }

        # ----------------------------------------------------
        # Object counts by class
        # ----------------------------------------------------

        for class_name in CLASS_NAMES:

            class_df = frame_df[
                frame_df["class_name"].astype(str).str.lower()
                == class_name
            ]

            record[f"{class_name}_object_count"] = int(
                class_df["track_id"].nunique()
            )

        # ----------------------------------------------------
        # Aggregate motion and spatial features
        # ----------------------------------------------------

        for feature in MOTION_FEATURES:

            values = pd.to_numeric(
                frame_df[feature],
                errors="coerce"
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            for statistic in STATISTICS:

                feature_name = f"{feature}_{statistic}"

                if values.empty:
                    record[feature_name] = np.nan

                elif statistic == "mean":
                    record[feature_name] = values.mean()

                elif statistic == "std":
                    record[feature_name] = values.std(ddof=0)

                elif statistic == "min":
                    record[feature_name] = values.min()

                elif statistic == "max":
                    record[feature_name] = values.max()

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 5. VALIDATE INPUT FILES
# ============================================================

def validate_columns(df, required_columns, filename):

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{filename}: missing columns {missing}"
        )


# ============================================================
# 6. PROCESS ONE VIDEO
# ============================================================

def process_video(trajectory_path):

    relative_path = trajectory_path.relative_to(
        TRAJECTORY_DIR
    )

    interaction_path = INTERACTION_DIR / relative_path

    output_path = OUTPUT_DIR / relative_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        if not interaction_path.exists():
            raise FileNotFoundError(
                f"Interaction file not found: {interaction_path}"
            )

        trajectory_df = pd.read_csv(trajectory_path)

        interaction_df = pd.read_csv(interaction_path)

        validate_columns(
            trajectory_df,
            REQUIRED_TRAJECTORY_COLUMNS,
            trajectory_path.name
        )

        validate_columns(
            interaction_df,
            REQUIRED_INTERACTION_COLUMNS,
            interaction_path.name
        )

        # ----------------------------------------------------
        # Handle videos with no tracked objects
        # ----------------------------------------------------

        if trajectory_df.empty:

            if not interaction_df.empty:
                raise ValueError(
                    "Empty trajectories but interaction frames exist"
                )

            interaction_df.to_csv(
                output_path,
                index=False
            )

            return {
                "video_name": trajectory_path.name,
                "split": relative_path.parts[0],
                "frames_processed": 0,
                "trajectory_objects": 0,
                "status": "no_tracked_objects",
            }

        # ----------------------------------------------------
        # Normalize labels and validate metadata
        # ----------------------------------------------------

        trajectory_df = prepare_trajectory_data(
            trajectory_df
        )

        if trajectory_df.empty:
            raise ValueError(
                "No valid trajectory observations remain"
            )

        trajectory_labels = (
            trajectory_df["label"]
            .dropna()
            .map(normalize_label)
            .unique()
        )

        interaction_labels = (
            interaction_df["label"]
            .dropna()
            .map(normalize_label)
            .unique()
        )

        if (
            len(trajectory_labels) != 1
            or len(interaction_labels) != 1
            or trajectory_labels[0] != interaction_labels[0]
        ):
            raise ValueError(
                "Trajectory and interaction labels do not match"
            )

        # ----------------------------------------------------
        # Aggregate trajectory features by frame
        # ----------------------------------------------------

        trajectory_features = aggregate_trajectory_features(
            trajectory_df
        )

        # ----------------------------------------------------
        # Prepare interaction features
        # ----------------------------------------------------

        interaction_df = interaction_df.copy()

        interaction_df["frame_index"] = pd.to_numeric(
            interaction_df["frame_index"],
            errors="coerce"
        )

        interaction_df["timestamp_sec"] = pd.to_numeric(
            interaction_df["timestamp_sec"],
            errors="coerce"
        )

        if interaction_df.empty:
            raise ValueError(
                "No interaction frame records available"
            )

        if interaction_df["frame_index"].duplicated().any():
            raise ValueError(
                "Duplicate frame indices in interaction features"
            )

        # ----------------------------------------------------
        # Merge by video frame
        # ----------------------------------------------------

        merged = interaction_df.merge(
            trajectory_features,
            on="frame_index",
            how="left",
            validate="one_to_one"
        )

        if merged["tracked_object_count"].isna().any():
            raise ValueError(
                "Some interaction frames have no trajectory match"
            )

        # ----------------------------------------------------
        # Validate timestamps
        # ----------------------------------------------------

        timestamp_match = np.isclose(
            merged["timestamp_sec"],
            merged["trajectory_timestamp_sec"],
            atol=1e-5
        )

        if not timestamp_match.all():
            raise ValueError(
                "Timestamp mismatch between trajectory and interaction data"
            )

        merged = merged.drop(
            columns=["trajectory_timestamp_sec"]
        )

        # ----------------------------------------------------
        # Normalize video metadata
        # ----------------------------------------------------

        merged["label"] = merged["label"].map(
            normalize_label
        )

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        merged = merged.sort_values(
            ["timestamp_sec", "frame_index"]
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Save consolidated scene features
        # ----------------------------------------------------

        merged.to_csv(
            output_path,
            index=False
        )

        return {
            "video_name": trajectory_path.name,
            "split": relative_path.parts[0],
            "frames_processed": len(merged),
            "trajectory_objects": int(
                trajectory_df["track_id"].nunique()
            ),
            "status": "success",
        }

    except Exception as error:

        return {
            "video_name": trajectory_path.name,
            "split": relative_path.parts[0],
            "frames_processed": 0,
            "trajectory_objects": 0,
            "status": f"failed: {error}",
        }


# ============================================================
# 7. PROCESS ALL VIDEOS
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 3C - SCENE LEVEL FEATURE CONSOLIDATION")
    print("=" * 65)

    if not TRAJECTORY_DIR.exists():
        raise FileNotFoundError(
            f"Trajectory directory missing: {TRAJECTORY_DIR}"
        )

    if not INTERACTION_DIR.exists():
        raise FileNotFoundError(
            f"Interaction directory missing: {INTERACTION_DIR}"
        )

    files = sorted(
        TRAJECTORY_DIR.rglob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            "No trajectory CSV files found"
        )

    print(f"Videos found: {len(files)}")

    results = []

    for index, path in enumerate(files, start=1):

        result = process_video(path)

        results.append(result)

        print(
            f"[{index}/{len(files)}] "
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
    print("PHASE 3C SUMMARY")
    print("=" * 65)

    print(f"Videos processed: {len(summary)}")

    print(
        "Successful:",
        (summary["status"] == "success").sum()
    )

    print(
        "No tracked objects:",
        (summary["status"] == "no_tracked_objects").sum()
    )

    print(
        "Failed:",
        summary["status"].str.startswith("failed").sum()
    )

    print(
        "Total consolidated frames:",
        summary["frames_processed"].sum()
    )

    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
