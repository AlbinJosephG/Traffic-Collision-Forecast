
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_PATH = (
    ROOT
    / "data"
    / "annotations"
    / "matched_200_annotations.csv"
)

MANIFEST_DIR = ROOT / "data" / "manifests"

SCENE_FEATURE_DIR = (
    ROOT / "processed" / "scene_features"
)

OUTPUT_DIR = (
    ROOT / "processed" / "temporal_anchor_audit"
)

SPLITS = ["train", "validation", "test"]

HORIZONS = [0.5, 1.0, 1.5]

SEQUENCE_LENGTH = 50
MAX_TIMESTAMP_GAP = 0.30
MAX_ENDPOINT_LAG = 0.25

SEED = 42

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "timestamp_sec",
    "time",
    "time_sec",
    "frame_timestamp",
    "video_timestamp",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_video_name(value):
    """Remove directory and file extension from a video name."""
    return Path(str(value)).stem


def find_column(dataframe, candidates):
    """Find a column without depending on capitalization."""

    lookup = {
        str(column).lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

def load_annotations():

    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(ANNOTATION_PATH)

    annotations = pd.read_csv(ANNOTATION_PATH)

    required_columns = [
        "video_name",
        "label",
        "time_of_event",
        "time_of_alert",
    ]

    missing = [
        column
        for column in required_columns
        if column not in annotations.columns
    ]

    if missing:
        raise ValueError(
            f"Missing annotation columns: {missing}"
        )

    annotations = annotations.copy()

    annotations["video_id"] = (
        annotations["video_name"]
        .apply(normalize_video_name)
    )

    annotations["label"] = pd.to_numeric(
        annotations["label"],
        errors="coerce",
    )

    annotations["time_of_event"] = pd.to_numeric(
        annotations["time_of_event"],
        errors="coerce",
    )

    annotations["time_of_alert"] = pd.to_numeric(
        annotations["time_of_alert"],
        errors="coerce",
    )

    if annotations["video_id"].duplicated().any():
        raise ValueError(
            "Duplicate video IDs found in annotations."
        )

    return annotations[
        [
            "video_id",
            "video_name",
            "label",
            "time_of_event",
            "time_of_alert",
        ]
    ].copy()


# ============================================================
# LOAD SPLIT MANIFESTS
# ============================================================

def load_split_manifests():

    frames = []

    for split in SPLITS:

        path = MANIFEST_DIR / f"{split}_manifest.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        manifest = pd.read_csv(path)

        required = ["video_path", "label"]

        for column in required:
            if column not in manifest.columns:
                raise ValueError(
                    f"Missing column '{column}' in {path}"
                )

        manifest = manifest.copy()

        manifest["video_id"] = (
            manifest["video_path"]
            .apply(normalize_video_name)
        )

        manifest["label"] = pd.to_numeric(
            manifest["label"],
            errors="coerce",
        )

        manifest["split"] = split

        frames.append(
            manifest[
                [
                    "video_id",
                    "video_path",
                    "label",
                    "split",
                ]
            ]
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined["video_id"].duplicated().any():

        duplicates = combined.loc[
            combined["video_id"].duplicated(keep=False),
            "video_id",
        ].tolist()

        raise ValueError(
            f"Videos occur in multiple splits: {duplicates}"
        )

    if not combined["label"].isin([0, 1]).all():
        raise ValueError(
            "Manifest labels must contain only 0 and 1."
        )

    return combined


# ============================================================
# MATCH MANIFESTS WITH ANNOTATIONS
# ============================================================

def load_matched_metadata():

    manifests = load_split_manifests()
    annotations = load_annotations()

    annotations = annotations.rename(
        columns={
            "label": "annotation_label"
        }
    )

    data = manifests.merge(
        annotations,
        on="video_id",
        how="left",
        validate="one_to_one",
    )

    if data["video_name"].isna().any():

        missing = data.loc[
            data["video_name"].isna(),
            "video_id",
        ].tolist()

        raise ValueError(
            f"Missing annotation matches: {missing}"
        )

    if data["annotation_label"].isna().any():
        raise ValueError(
            "Some videos have missing annotation labels."
        )

    if not (
        data["label"] == data["annotation_label"]
    ).all():

        raise ValueError(
            "Manifest and annotation labels do not match."
        )

    data["label"] = data["label"].astype(int)

    return data


# ============================================================
# LOCATE SCENE FEATURE FILE
# ============================================================

def find_scene_feature_file(split, label, video_id):

    class_name = (
        "positive" if label == 1 else "negative"
    )

    search_root = (
        SCENE_FEATURE_DIR / split / class_name
    )

    if not search_root.exists():
        raise FileNotFoundError(
            f"Scene feature directory not found: {search_root}"
        )

    matches = list(
        search_root.rglob(f"{video_id}.csv")
    )

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        raise ValueError(
            f"Multiple scene feature files for {video_id}: "
            f"{matches}"
        )

    return matches[0]


# ============================================================
# LOAD VIDEO TIMESTAMPS
# ============================================================

def load_video_timestamps(path):

    dataframe = pd.read_csv(path)

    timestamp_column = find_column(
        dataframe,
        TIMESTAMP_CANDIDATES,
    )

    if timestamp_column is None:
        raise ValueError(
            f"No timestamp column found in {path}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    timestamps = pd.to_numeric(
        dataframe[timestamp_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    timestamps = timestamps[
        np.isfinite(timestamps)
    ]

    timestamps = np.unique(
        np.sort(timestamps)
    )

    if len(timestamps) == 0:

        print(
            f"WARNING: Empty timestamps in {path.name}"
        )

        return (
            np.array([], dtype=float),
            timestamp_column,
        )

    return timestamps, timestamp_column


# ============================================================
# VALIDATE EVENT ANNOTATIONS
# ============================================================

def validate_positive_annotation(
    video_id,
    event_time,
    alert_time,
    first_timestamp,
    last_timestamp,
):

    if not np.isfinite(event_time):
        raise ValueError(
            f"Missing event timestamp: {video_id}"
        )

    if not np.isfinite(alert_time):
        raise ValueError(
            f"Missing alert timestamp: {video_id}"
        )

    if event_time < first_timestamp:
        raise ValueError(
            f"Event occurs before first scene timestamp: "
            f"{video_id}"
        )

    if alert_time > event_time:
        raise ValueError(
            f"Alert occurs after event: {video_id}"
        )

    event_lag = event_time - last_timestamp

    # The event may occur slightly after the last sampled
    # observation due to approximately 5 FPS sampling.
    if event_lag > MAX_TIMESTAMP_GAP:

        raise ValueError(
            f"Event is too far beyond last sampled timestamp: "
            f"{video_id}, lag={event_lag:.3f}s"
        )


# ============================================================
# BUILD VIDEO-LEVEL AUDIT
# ============================================================

def build_video_audit():

    data = load_matched_metadata()

    video_rows = []

    print("\nLoading scene timestamps...")

    for _, row in data.iterrows():

        video_id = row["video_id"]
        split = row["split"]
        label = int(row["label"])

        event_time = row["time_of_event"]
        alert_time = row["time_of_alert"]

        scene_path = find_scene_feature_file(
            split,
            label,
            video_id,
        )

        if scene_path is None:
            raise FileNotFoundError(
                f"Scene feature file missing: {video_id}"
            )

        timestamps, timestamp_column = (
            load_video_timestamps(scene_path)
        )

        # Preserve empty scene-feature videos.
        if len(timestamps) == 0:

            video_rows.append({
                "video_id": video_id,
                "video_name": row["video_name"],
                "split": split,
                "label": label,
                "scene_feature_path": str(scene_path),
                "timestamp_column": timestamp_column,
                "first_timestamp": np.nan,
                "last_timestamp": np.nan,
                "duration_sec": np.nan,
                "time_of_event": event_time,
                "time_of_alert": alert_time,
                "event_position": np.nan,
                "alert_position": np.nan,
                "current_anchor_sec": np.nan,
                "pseudo_event_position": np.nan,
                "pseudo_event_anchor_sec": np.nan,
                "timestamp_status": "empty",
                "timestamps": timestamps,
            })

            continue

        first_timestamp = float(timestamps[0])
        last_timestamp = float(timestamps[-1])

        duration = last_timestamp - first_timestamp

        if duration <= 0:

            video_rows.append({
                "video_id": video_id,
                "video_name": row["video_name"],
                "split": split,
                "label": label,
                "scene_feature_path": str(scene_path),
                "timestamp_column": timestamp_column,
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
                "duration_sec": duration,
                "time_of_event": event_time,
                "time_of_alert": alert_time,
                "event_position": np.nan,
                "alert_position": np.nan,
                "current_anchor_sec": np.nan,
                "pseudo_event_position": np.nan,
                "pseudo_event_anchor_sec": np.nan,
                "timestamp_status": "invalid_duration",
                "timestamps": timestamps,
            })

            continue

        if label == 1:

            validate_positive_annotation(
                video_id,
                event_time,
                alert_time,
                first_timestamp,
                last_timestamp,
            )

            event_position = (
                event_time - first_timestamp
            ) / duration

            alert_position = (
                alert_time - first_timestamp
            ) / duration

            current_anchor = float(event_time)

        else:

            event_position = np.nan
            alert_position = np.nan

            # Original negative anchor.
            current_anchor = last_timestamp

        video_rows.append({
            "video_id": video_id,
            "video_name": row["video_name"],
            "split": split,
            "label": label,
            "scene_feature_path": str(scene_path),
            "timestamp_column": timestamp_column,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "duration_sec": duration,
            "time_of_event": event_time,
            "time_of_alert": alert_time,
            "event_position": event_position,
            "alert_position": alert_position,
            "current_anchor_sec": current_anchor,
            "pseudo_event_position": np.nan,
            "pseudo_event_anchor_sec": np.nan,
            "timestamp_status": "valid",
            "timestamps": timestamps,
        })

    return video_rows


# ============================================================
# MATCH NEGATIVE PSEUDO-EVENT POSITIONS
# ============================================================

def assign_pseudo_event_anchors(video_rows):

    rng = np.random.default_rng(SEED)

    for split in SPLITS:

        positive_rows = [
            row for row in video_rows
            if (
                row["split"] == split
                and row["label"] == 1
                and row["timestamp_status"] == "valid"
                and np.isfinite(row["event_position"])
            )
        ]

        negative_rows = [
            row for row in video_rows
            if (
                row["split"] == split
                and row["label"] == 0
                and row["timestamp_status"] == "valid"
            )
        ]

        if len(positive_rows) == 0:
            raise ValueError(
                f"No usable positive event positions in {split}"
            )

        positive_positions = np.array([
            row["event_position"]
            for row in positive_rows
        ], dtype=float)

        # Assign the positive event-position distribution
        # to negative videos without replacement where possible.
        matched_positions = rng.choice(
            positive_positions,
            size=len(negative_rows),
            replace=(
                len(negative_rows) > len(positive_positions)
            ),
        )

        rng.shuffle(matched_positions)

        for negative_row, position in zip(
            negative_rows,
            matched_positions,
        ):

            negative_row["pseudo_event_position"] = float(
                position
            )

            negative_row["pseudo_event_anchor_sec"] = (
                negative_row["first_timestamp"]
                + position * negative_row["duration_sec"]
            )

        for positive_row in positive_rows:

            positive_row["pseudo_event_position"] = float(
                positive_row["event_position"]
            )

            positive_row["pseudo_event_anchor_sec"] = float(
                positive_row["time_of_event"]
            )


# ============================================================
# CHECK SEQUENCE AVAILABILITY
# ============================================================

def check_sequence_availability(
    timestamps,
    endpoint,
):

    if len(timestamps) == 0:

        return {
            "valid": False,
            "reason": "empty_timestamps",
            "observation_count": 0,
            "endpoint_lag": np.nan,
            "sequence_duration_sec": np.nan,
            "maximum_gap_sec": np.nan,
        }

    eligible = timestamps[
        timestamps <= endpoint
    ]

    if len(eligible) < SEQUENCE_LENGTH:

        return {
            "valid": False,
            "reason": "insufficient_history",
            "observation_count": len(eligible),
            "endpoint_lag": np.nan,
            "sequence_duration_sec": np.nan,
            "maximum_gap_sec": np.nan,
        }

    sequence = eligible[-SEQUENCE_LENGTH:]

    endpoint_lag = endpoint - sequence[-1]

    sequence_duration = (
        sequence[-1] - sequence[0]
    )

    gaps = np.diff(sequence)

    maximum_gap = (
        float(np.max(gaps))
        if len(gaps) > 0
        else np.nan
    )

    if endpoint_lag > MAX_ENDPOINT_LAG:

        return {
            "valid": False,
            "reason": "endpoint_too_far",
            "observation_count": len(eligible),
            "endpoint_lag": endpoint_lag,
            "sequence_duration_sec": sequence_duration,
            "maximum_gap_sec": maximum_gap,
        }

    if maximum_gap > MAX_TIMESTAMP_GAP:

        return {
            "valid": False,
            "reason": "timestamp_gap",
            "observation_count": len(eligible),
            "endpoint_lag": endpoint_lag,
            "sequence_duration_sec": sequence_duration,
            "maximum_gap_sec": maximum_gap,
        }

    return {
        "valid": True,
        "reason": "valid",
        "observation_count": len(eligible),
        "endpoint_lag": endpoint_lag,
        "sequence_duration_sec": sequence_duration,
        "maximum_gap_sec": maximum_gap,
    }


# ============================================================
# BUILD HORIZON-LEVEL AUDIT
# ============================================================

def build_horizon_audit(video_rows):

    rows = []

    for video in video_rows:

        for horizon in HORIZONS:

            timestamps = video["timestamps"]

            current_anchor = video["current_anchor_sec"]

            proposed_anchor = (
                video["pseudo_event_anchor_sec"]
            )

            if (
                video["timestamp_status"] != "valid"
                or not np.isfinite(current_anchor)
                or not np.isfinite(proposed_anchor)
            ):

                current_result = check_sequence_availability(
                    np.array([], dtype=float),
                    np.nan,
                )

                proposed_result = check_sequence_availability(
                    np.array([], dtype=float),
                    np.nan,
                )

                current_endpoint = np.nan
                proposed_endpoint = np.nan

            else:

                current_endpoint = (
                    current_anchor - horizon
                )

                proposed_endpoint = (
                    proposed_anchor - horizon
                )

                current_result = check_sequence_availability(
                    timestamps,
                    current_endpoint,
                )

                proposed_result = check_sequence_availability(
                    timestamps,
                    proposed_endpoint,
                )

            rows.append({
                "video_id": video["video_id"],
                "video_name": video["video_name"],
                "split": video["split"],
                "label": video["label"],
                "timestamp_status": video["timestamp_status"],
                "horizon_sec": horizon,
                "duration_sec": video["duration_sec"],
                "event_position": video["event_position"],
                "pseudo_event_position": (
                    video["pseudo_event_position"]
                ),
                "current_anchor_sec": current_anchor,
                "proposed_anchor_sec": proposed_anchor,
                "current_endpoint_sec": current_endpoint,
                "proposed_endpoint_sec": proposed_endpoint,
                "current_valid": current_result["valid"],
                "current_reason": current_result["reason"],
                "current_endpoint_lag": (
                    current_result["endpoint_lag"]
                ),
                "current_sequence_duration_sec": (
                    current_result["sequence_duration_sec"]
                ),
                "current_maximum_gap_sec": (
                    current_result["maximum_gap_sec"]
                ),
                "proposed_valid": proposed_result["valid"],
                "proposed_reason": proposed_result["reason"],
                "proposed_endpoint_lag": (
                    proposed_result["endpoint_lag"]
                ),
                "proposed_sequence_duration_sec": (
                    proposed_result["sequence_duration_sec"]
                ),
                "proposed_maximum_gap_sec": (
                    proposed_result["maximum_gap_sec"]
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(video_rows, horizon_audit):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Video-level report
    # --------------------------------------------------------

    video_dataframe = pd.DataFrame([
        {
            key: value
            for key, value in row.items()
            if key != "timestamps"
        }
        for row in video_rows
    ])

    video_dataframe.to_csv(
        OUTPUT_DIR / "video_anchor_audit.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Horizon-level report
    # --------------------------------------------------------

    horizon_audit.to_csv(
        OUTPUT_DIR / "horizon_anchor_audit.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Anchor position distributions
    # --------------------------------------------------------

    distribution_rows = []

    for split in SPLITS:

        split_data = video_dataframe[
            video_dataframe["split"] == split
        ]

        for label, group in split_data.groupby("label"):

            if label == 1:

                positions = group[
                    "event_position"
                ].dropna()

                strategy = "positive_event_position"

            else:

                # Current strategy places every negative
                # anchor at the end of its observed video.
                positions = pd.Series(
                    np.ones(len(group)),
                    dtype=float,
                )

                strategy = "negative_video_end_position"

            if len(positions) == 0:
                continue

            distribution_rows.append({
                "split": split,
                "label": int(label),
                "strategy": strategy,
                "count": len(positions),
                "mean_position": positions.mean(),
                "std_position": positions.std(),
                "minimum_position": positions.min(),
                "median_position": positions.median(),
                "maximum_position": positions.max(),
            })

    pd.DataFrame(
        distribution_rows
    ).to_csv(
        OUTPUT_DIR / "anchor_position_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Sequence availability comparison
    # --------------------------------------------------------

    summary = (
        horizon_audit
        .groupby(
            ["split", "label", "horizon_sec"],
            dropna=False,
        )
        .agg(
            videos=("video_id", "nunique"),
            usable_timestamp_videos=(
                "timestamp_status",
                lambda values: (values == "valid").sum(),
            ),
            current_valid=("current_valid", "sum"),
            proposed_valid=("proposed_valid", "sum"),
            current_mean_duration=(
                "current_sequence_duration_sec",
                "mean",
            ),
            proposed_mean_duration=(
                "proposed_sequence_duration_sec",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "sequence_availability_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Rejection reasons
    # --------------------------------------------------------

    rejection_rows = []

    for strategy in ["current", "proposed"]:

        reason_column = f"{strategy}_reason"

        grouped = (
            horizon_audit
            .groupby(
                [
                    "split",
                    "label",
                    "horizon_sec",
                    reason_column,
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )

        grouped["strategy"] = strategy

        grouped = grouped.rename(
            columns={
                reason_column: "reason"
            }
        )

        rejection_rows.append(grouped)

    pd.concat(
        rejection_rows,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "sequence_rejection_reasons.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Proposed anchor distribution
    # --------------------------------------------------------

    proposed_distribution = (
        video_dataframe
        .groupby(["split", "label"])
        .agg(
            videos=("video_id", "count"),
            valid_anchors=(
                "pseudo_event_position",
                "count",
            ),
            mean_pseudo_position=(
                "pseudo_event_position",
                "mean",
            ),
            median_pseudo_position=(
                "pseudo_event_position",
                "median",
            ),
            std_pseudo_position=(
                "pseudo_event_position",
                "std",
            ),
        )
        .reset_index()
    )

    proposed_distribution.to_csv(
        OUTPUT_DIR / "proposed_anchor_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEMPORAL ANCHOR BIAS AUDIT COMPLETED")
    print("=" * 70)

    print("\nVideo counts:")

    print(
        video_dataframe
        .groupby(["split", "label"])
        .size()
        .to_string()
    )

    print("\nTimestamp status:")

    print(
        video_dataframe
        .groupby(["split", "timestamp_status"])
        .size()
        .to_string()
    )

    print("\nSequence availability comparison:")

    print(
        summary.to_string(index=False)
    )

    print("\nReports saved to:")
    print(OUTPUT_DIR)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEMPORAL ANCHOR BIAS AUDIT")
    print("=" * 70)

    print("Project root:", ROOT)
    print("Seed:", SEED)
    print("Horizons:", HORIZONS)
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Maximum timestamp gap:", MAX_TIMESTAMP_GAP)
    print("Maximum endpoint lag:", MAX_ENDPOINT_LAG)

    video_rows = build_video_audit()

    print("\nAssigning negative pseudo-event anchors...")

    assign_pseudo_event_anchors(video_rows)

    print("\nChecking sequence availability...")

    horizon_audit = build_horizon_audit(video_rows)

    save_reports(
        video_rows,
        horizon_audit,
    )


if __name__ == "__main__":
    main()
