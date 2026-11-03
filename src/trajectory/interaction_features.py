
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT_DIR / "processed" / "trajectories"

PAIR_OUTPUT_DIR = ROOT_DIR / "processed" / "interactions"

FRAME_OUTPUT_DIR = (
    ROOT_DIR / "processed" / "interaction_features"
)

SUMMARY_PATH = (
    ROOT_DIR / "processed" / "interaction_summary.csv"
)

# COCO vehicle classes
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Original video resolution
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Normalize distances by the image diagonal.
IMAGE_DIAGONAL = np.sqrt(
    FRAME_WIDTH ** 2 + FRAME_HEIGHT ** 2
)

# Heuristic: two vehicles are considered close when their
# center distance is within 10% of the image diagonal.
PROXIMITY_THRESHOLD = 0.10

# Minimum closing speed for a meaningful TTC proxy.
MIN_CLOSING_SPEED = 0.005

# Ignore extremely large TTC estimates.
MAX_TTC_SECONDS = 10.0

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
    "center_x",
    "center_y",
    "velocity_x_norm",
    "velocity_y_norm",
]


# ============================================================
# 2. OUTPUT SCHEMAS
# ============================================================

PAIR_COLUMNS = [
    "video_name",
    "split",
    "label",
    "frame_index",
    "timestamp_sec",
    "track_id_1",
    "track_id_2",
    "class_name_1",
    "class_name_2",
    "distance_norm",
    "relative_x_norm",
    "relative_y_norm",
    "closing_speed_norm",
    "is_approaching",
    "is_nearby",
    "ttc_proxy_sec",
]

FRAME_FEATURE_COLUMNS = [
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
# 3. PREPARE TRACKED VEHICLES
# ============================================================

def prepare_frame_vehicles(frame_df):

    df = frame_df.copy()

    numeric_columns = [
        "track_id",
        "class_id",
        "center_x",
        "center_y",
        "velocity_x_norm",
        "velocity_y_norm",
        "confidence",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Keep only valid tracked vehicle observations.
    df = df.dropna(
        subset=[
            "track_id",
            "class_id",
            "center_x",
            "center_y",
        ]
    )

    df = df[
        df["class_id"].isin(VEHICLE_CLASSES.keys())
    ].copy()

    if df.empty:
        return df

    # Convert image coordinates into diagonal-normalized
    # coordinates for distance calculations.
    df["position_x"] = (
        df["center_x"] / IMAGE_DIAGONAL
    )

    df["position_y"] = (
        df["center_y"] / IMAGE_DIAGONAL
    )

    # Convert normalized Phase 3A velocity components
    # into image-diagonal units per second.
    df["velocity_x"] = (
        df["velocity_x_norm"]
        * FRAME_WIDTH
        / IMAGE_DIAGONAL
    )

    df["velocity_y"] = (
        df["velocity_y_norm"]
        * FRAME_HEIGHT
        / IMAGE_DIAGONAL
    )

    # If duplicate observations exist for the same track
    # in one frame, retain the highest-confidence detection.
    df = df.sort_values(
        "confidence",
        ascending=False,
        na_position="last"
    )

    df = df.drop_duplicates(
        subset=["track_id"],
        keep="first"
    )

    return df


# ============================================================
# 4. CALCULATE PAIRWISE INTERACTIONS
# ============================================================

def calculate_pairwise_interactions(
    frame_df,
    video_name,
    split,
    label,
    frame_index,
    timestamp_sec,
):

    vehicles = prepare_frame_vehicles(frame_df)

    pair_records = []

    if len(vehicles) < 2:
        return pair_records

    records = vehicles.to_dict("records")

    for vehicle_1, vehicle_2 in combinations(records, 2):

        id_1 = int(vehicle_1["track_id"])
        id_2 = int(vehicle_2["track_id"])

        # ----------------------------------------------------
        # Relative position
        # ----------------------------------------------------

        relative_x = (
            vehicle_2["position_x"]
            - vehicle_1["position_x"]
        )

        relative_y = (
            vehicle_2["position_y"]
            - vehicle_1["position_y"]
        )

        distance = np.sqrt(
            relative_x ** 2 + relative_y ** 2
        )

        # ----------------------------------------------------
        # Relative velocity
        # ----------------------------------------------------

        vx_1 = vehicle_1["velocity_x"]
        vy_1 = vehicle_1["velocity_y"]

        vx_2 = vehicle_2["velocity_x"]
        vy_2 = vehicle_2["velocity_y"]

        velocity_valid = all(
            np.isfinite(value)
            for value in [vx_1, vy_1, vx_2, vy_2]
        )

        closing_speed = np.nan
        is_approaching = False
        ttc = np.nan

        if velocity_valid and distance > 0:

            relative_vx = vx_2 - vx_1
            relative_vy = vy_2 - vy_1

            # Positive value means the separation is decreasing.
            closing_speed = -(
                relative_x * relative_vx
                + relative_y * relative_vy
            ) / distance

            if closing_speed > MIN_CLOSING_SPEED:

                is_approaching = True

                estimated_ttc = distance / closing_speed

                if (
                    0 < estimated_ttc <= MAX_TTC_SECONDS
                ):
                    ttc = estimated_ttc

        # ----------------------------------------------------
        # Proximity
        # ----------------------------------------------------

        is_nearby = (
            distance <= PROXIMITY_THRESHOLD
        )

        # ----------------------------------------------------
        # Save pair
        # ----------------------------------------------------

        pair_records.append({

            "video_name": video_name,
            "split": split,
            "label": label,

            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,

            "track_id_1": id_1,
            "track_id_2": id_2,

            "class_name_1": vehicle_1["class_name"],
            "class_name_2": vehicle_2["class_name"],

            "distance_norm": distance,

            "relative_x_norm": relative_x,
            "relative_y_norm": relative_y,

            "closing_speed_norm": closing_speed,

            "is_approaching": is_approaching,
            "is_nearby": is_nearby,

            "ttc_proxy_sec": ttc,
        })

    return pair_records


# ============================================================
# 5. AGGREGATE PAIRWISE INTERACTIONS INTO FRAME FEATURES
# ============================================================

def aggregate_frame_features(
    frame_df,
    pair_records,
    video_name,
    split,
    label,
    frame_index,
    timestamp_sec,
):

    vehicles = prepare_frame_vehicles(frame_df)

    vehicle_count = len(vehicles)

    class_counts = vehicles["class_name"].value_counts()

    pairs = pd.DataFrame(pair_records)

    if pairs.empty:

        pair_count = 0
        approaching_count = 0
        nearby_count = 0

        min_distance = np.nan
        mean_distance = np.nan

        min_ttc = np.nan
        mean_ttc = np.nan

        max_closing_speed = np.nan
        mean_closing_speed = np.nan

    else:

        pair_count = len(pairs)

        approaching_count = int(
            pairs["is_approaching"].sum()
        )

        nearby_count = int(
            pairs["is_nearby"].sum()
        )

        min_distance = pairs[
            "distance_norm"
        ].min()

        mean_distance = pairs[
            "distance_norm"
        ].mean()

        valid_ttc = pairs[
            "ttc_proxy_sec"
        ].dropna()

        min_ttc = (
            valid_ttc.min()
            if not valid_ttc.empty
            else np.nan
        )

        mean_ttc = (
            valid_ttc.mean()
            if not valid_ttc.empty
            else np.nan
        )

        valid_closing = pairs[
            "closing_speed_norm"
        ].dropna()

        max_closing_speed = (
            valid_closing.max()
            if not valid_closing.empty
            else np.nan
        )

        mean_closing_speed = (
            valid_closing.mean()
            if not valid_closing.empty
            else np.nan
        )

    return {

        "video_name": video_name,
        "split": split,
        "label": label,

        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,

        "vehicle_count": vehicle_count,

        "bicycle_count": int(
            class_counts.get("bicycle", 0)
        ),

        "car_count": int(
            class_counts.get("car", 0)
        ),

        "motorcycle_count": int(
            class_counts.get("motorcycle", 0)
        ),

        "bus_count": int(
            class_counts.get("bus", 0)
        ),

        "truck_count": int(
            class_counts.get("truck", 0)
        ),

        "interaction_pair_count": pair_count,

        "approaching_pair_count": approaching_count,

        "nearby_pair_count": nearby_count,

        "minimum_pair_distance_norm": min_distance,

        "mean_pair_distance_norm": mean_distance,

        "minimum_ttc_proxy_sec": min_ttc,

        "mean_ttc_proxy_sec": mean_ttc,

        "maximum_closing_speed_norm": max_closing_speed,

        "mean_closing_speed_norm": mean_closing_speed,
    }


# ============================================================
# 6. PROCESS ONE VIDEO
# ============================================================

def process_video(csv_path):

    relative_path = csv_path.relative_to(INPUT_DIR)

    pair_output_path = PAIR_OUTPUT_DIR / relative_path
    frame_output_path = FRAME_OUTPUT_DIR / relative_path

    pair_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    frame_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        df = pd.read_csv(csv_path)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        pair_records_all = []
        frame_records_all = []

        if not df.empty:

            df["frame_index"] = pd.to_numeric(
                df["frame_index"],
                errors="coerce"
            )

            df["timestamp_sec"] = pd.to_numeric(
                df["timestamp_sec"],
                errors="coerce"
            )

            df = df.dropna(
                subset=["frame_index", "timestamp_sec"]
            )

            df = df.sort_values(
                ["frame_index", "timestamp_sec"]
            )

            video_name = str(df["video_name"].iloc[0])
            split = str(df["split"].iloc[0])
            label = str(df["label"].iloc[0]).strip().lower()

            if label == "positive":
                label = 1
            elif label == "negative":
                label = 0
            else:
                raise ValueError(f"Unknown video label: {label}")

            # Process each sampled frame independently.
            for frame_index, frame_df in df.groupby(
                "frame_index",
                sort=True
            ):

                timestamp_sec = float(
                    frame_df["timestamp_sec"].iloc[0]
                )

                pair_records = calculate_pairwise_interactions(
                    frame_df=frame_df,
                    video_name=video_name,
                    split=split,
                    label=label,
                    frame_index=int(frame_index),
                    timestamp_sec=timestamp_sec,
                )

                frame_features = aggregate_frame_features(
                    frame_df=frame_df,
                    pair_records=pair_records,
                    video_name=video_name,
                    split=split,
                    label=label,
                    frame_index=int(frame_index),
                    timestamp_sec=timestamp_sec,
                )

                pair_records_all.extend(pair_records)
                frame_records_all.append(frame_features)

        # ----------------------------------------------------
        # Save pairwise interaction CSV
        # ----------------------------------------------------

        pair_df = pd.DataFrame(
            pair_records_all,
            columns=PAIR_COLUMNS
        )

        pair_df.to_csv(
            pair_output_path,
            index=False
        )

        # ----------------------------------------------------
        # Save frame-level interaction features
        # ----------------------------------------------------

        frame_df = pd.DataFrame(
            frame_records_all,
            columns=FRAME_FEATURE_COLUMNS
        )

        frame_df.to_csv(
            frame_output_path,
            index=False
        )

        return {

            "video_name": csv_path.name,

            "split": relative_path.parts[0],

            "category": relative_path.parts[1],

            "frames_processed": len(frame_df),

            "pairwise_interactions": len(pair_df),

            "frames_with_vehicles": (
                int((frame_df["vehicle_count"] > 0).sum())
                if not frame_df.empty else 0
            ),

            "frames_with_interactions": (
                int((frame_df["interaction_pair_count"] > 0).sum())
                if not frame_df.empty else 0
            ),

            "status": (
                "success"
                if not frame_df.empty
                else "no_tracked_objects"
            ),

        }

    except Exception as error:

        return {

            "video_name": csv_path.name,

            "split": relative_path.parts[0],

            "category": relative_path.parts[1],

            "frames_processed": 0,

            "pairwise_interactions": 0,

            "frames_with_vehicles": 0,

            "frames_with_interactions": 0,

            "status": f"failed: {error}",

        }


# ============================================================
# 7. PROCESS ALL VIDEOS
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 3B - PAIRWISE VEHICLE INTERACTION EXTRACTION")
    print("=" * 65)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Trajectory directory not found: {INPUT_DIR}"
        )

    csv_files = sorted(
        INPUT_DIR.rglob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            "No trajectory CSV files found."
        )

    print(f"Trajectory files found: {len(csv_files)}")

    results = []

    for index, csv_path in enumerate(csv_files, start=1):

        result = process_video(csv_path)

        results.append(result)

        print(
            f"[{index}/{len(csv_files)}] "
            f"{result['video_name']} - "
            f"{result['status']}"
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

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PHASE 3B SUMMARY")
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
        "Total pairwise interactions:",
        summary["pairwise_interactions"].sum()
    )

    print(
        "Total frames processed:",
        summary["frames_processed"].sum()
    )

    print(
        "\nPairwise output:",
        PAIR_OUTPUT_DIR
    )

    print(
        "Frame features output:",
        FRAME_OUTPUT_DIR
    )

    print(
        "Summary:",
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()
