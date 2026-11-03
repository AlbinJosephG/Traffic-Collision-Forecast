
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENE_FEATURE_DIR = PROJECT_ROOT / "processed" / "scene_features"

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "matched_200_annotations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "revised_event_aligned_sequences"
)

SPLITS = ["train", "validation", "test"]
HORIZONS_SEC = [0.5, 1.0, 1.5]

SEQUENCE_LENGTH = 50
MAX_ALLOWED_GAP_SEC = 0.30
MAX_ENDPOINT_LAG_SEC = 0.25

RANDOM_SEED = 42

EXCLUDED_COLUMNS = {
    "video_name",
    "split",
    "label",
    "frame_index",
    "timestamp_sec",
}

GENERATOR_VERSION = "2.0"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


# ============================================================
# ANNOTATIONS
# ============================================================

def load_annotations():

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    annotations = pd.read_csv(ANNOTATION_FILE)

    required = {
        "video_name",
        "label",
        "time_of_event",
        "time_of_alert",
    }

    missing = required - set(annotations.columns)

    if missing:
        raise ValueError(
            f"Missing annotation columns: {sorted(missing)}"
        )

    annotations["video_name"] = (
        annotations["video_name"].astype(str)
    )

    if annotations["video_name"].duplicated().any():
        raise ValueError("Duplicate annotation video names.")

    annotations["label"] = pd.to_numeric(
        annotations["label"],
        errors="raise",
    ).astype(int)

    if not annotations["label"].isin([0, 1]).all():
        raise ValueError("Labels must be either 0 or 1.")

    annotations["time_of_event"] = pd.to_numeric(
        annotations["time_of_event"],
        errors="coerce",
    )

    annotations["time_of_alert"] = pd.to_numeric(
        annotations["time_of_alert"],
        errors="coerce",
    )

    positive = annotations["label"] == 1
    negative = annotations["label"] == 0

    if not np.isfinite(
        annotations.loc[positive, "time_of_event"]
    ).all():
        raise ValueError(
            "Every positive video must have a finite event timestamp."
        )

    if not np.isfinite(
        annotations.loc[positive, "time_of_alert"]
    ).all():
        raise ValueError(
            "Every positive video must have a finite alert timestamp."
        )

    invalid_alerts = (
        annotations.loc[positive, "time_of_alert"]
        >
        annotations.loc[positive, "time_of_event"]
    )

    if invalid_alerts.any():
        raise ValueError(
            "Found positive videos where alert time is after event time."
        )

    if annotations.loc[
        negative,
        ["time_of_event", "time_of_alert"],
    ].notna().any().any():
        raise ValueError(
            "Negative videos unexpectedly contain event/alert timestamps."
        )

    return annotations.set_index(
        "video_name",
        drop=False,
    )


# ============================================================
# MANIFESTS AND SPLIT VALIDATION
# ============================================================

def load_split_manifests():

    manifests = {}
    seen_videos = {}

    for split in SPLITS:

        path = (
            PROJECT_ROOT
            / "data"
            / "manifests"
            / f"{split}_manifest.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Split manifest not found: {path}"
            )

        df = pd.read_csv(path)

        required = {"video_path", "label"}

        if not required.issubset(df.columns):
            raise ValueError(
                f"Invalid manifest columns in {path}"
            )

        df["video_name"] = df["video_path"].apply(
            lambda value: Path(str(value)).name
        )

        df["label"] = pd.to_numeric(
            df["label"],
            errors="raise",
        ).astype(int)

        if df["video_name"].duplicated().any():
            raise ValueError(
                f"Duplicate video in {split} manifest."
            )

        for video_name in df["video_name"]:
            if video_name in seen_videos:
                raise ValueError(
                    f"Video leakage across splits: {video_name} "
                    f"appears in {seen_videos[video_name]} and {split}."
                )

            seen_videos[video_name] = split

        manifests[split] = df

    return manifests


# ============================================================
# SCENE FEATURES
# ============================================================

def load_scene_features(video_name, split, label):

    category = "positive" if label == 1 else "negative"

    feature_path = (
        SCENE_FEATURE_DIR
        / split
        / category
        / f"{Path(video_name).stem}.csv"
    )

    if not feature_path.exists():
        return None

    df = pd.read_csv(feature_path)

    if df.empty:
        return None

    required = {
        "video_name",
        "split",
        "label",
        "frame_index",
        "timestamp_sec",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{feature_path} is missing columns: {sorted(missing)}"
        )

    if not (df["video_name"] == video_name).all():
        raise ValueError(
            f"Video-name mismatch in {feature_path}"
        )

    if not (df["split"] == split).all():
        raise ValueError(
            f"Split mismatch in {feature_path}"
        )

    if not (df["label"] == label).all():
        raise ValueError(
            f"Label mismatch in {feature_path}"
        )

    df["timestamp_sec"] = pd.to_numeric(
        df["timestamp_sec"],
        errors="coerce",
    )

    # Remove invalid timestamps before sorting and deduplication.
    df = df[
        np.isfinite(df["timestamp_sec"])
    ].copy()

    original_count = len(df)

    df = (
        df.sort_values("timestamp_sec")
        .drop_duplicates(
            subset=["timestamp_sec"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    df.attrs["duplicate_timestamps_removed"] = (
        original_count - len(df)
    )

    if len(df) < SEQUENCE_LENGTH:
        return None

    return df


def get_feature_columns(df):

    columns = []

    for column in df.columns:

        if column in EXCLUDED_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)

    if not columns:
        raise ValueError("No numeric model features found.")

    return columns


def discover_feature_schema(manifests):

    # Use a deterministic training-manifest order.
    train_manifest = manifests["train"]

    for _, row in train_manifest.iterrows():

        video_name = row["video_name"]
        label = int(row["label"])

        df = load_scene_features(
            video_name,
            "train",
            label,
        )

        if df is not None:
            feature_columns = get_feature_columns(df)

            print(
                "Feature schema source:",
                video_name,
            )

            return feature_columns, video_name

    raise FileNotFoundError(
        "No usable training scene-feature file was found."
    )


# ============================================================
# TRAINING-POSITIVE EVENT POSITION DISTRIBUTION
# ============================================================

def build_training_event_position_distribution(
    manifests,
    annotations,
):
    """
    Use training-positive videos only.

    Keep actual positive event positions in [0, 1].
    Do not clamp events outside this interval.
    """

    train_manifest = manifests["train"]

    positions = []
    source_videos = []

    excluded_out_of_range = []
    excluded_unusable = []

    for _, row in train_manifest.iterrows():

        if int(row["label"]) != 1:
            continue

        video_name = row["video_name"]
        annotation = annotations.loc[video_name]

        event_time = float(annotation["time_of_event"])

        df = load_scene_features(
            video_name,
            "train",
            1,
        )

        if df is None:
            excluded_unusable.append(video_name)
            continue

        first_timestamp = float(
            df["timestamp_sec"].iloc[0]
        )

        last_timestamp = float(
            df["timestamp_sec"].iloc[-1]
        )

        duration = last_timestamp - first_timestamp

        if duration <= 0:
            excluded_unusable.append(video_name)
            continue

        event_position = (
            event_time - first_timestamp
        ) / duration

        if 0.0 <= event_position <= 1.0:

            positions.append(float(event_position))
            source_videos.append(video_name)

        else:

            # Preserve the actual event for positive sequence
            # generation. Exclude it only from the negative
            # reference distribution.
            excluded_out_of_range.append({
                "video_name": video_name,
                "event_position": float(event_position),
                "event_time": event_time,
                "last_timestamp": last_timestamp,
            })

    if not positions:
        raise ValueError(
            "No eligible training-positive event positions in [0, 1]."
        )

    distribution = np.asarray(
        positions,
        dtype=np.float64,
    )

    print("\nTRAINING EVENT-POSITION DISTRIBUTION")
    print(f"Reference positions: {len(distribution)}")
    print(f"Excluded out-of-range: {len(excluded_out_of_range)}")
    print(f"Excluded unusable: {len(excluded_unusable)}")
    print(f"Mean position: {distribution.mean():.6f}")
    print(f"Median position: {np.median(distribution):.6f}")
    print(f"Minimum position: {distribution.min():.6f}")
    print(f"Maximum position: {distribution.max():.6f}")

    if excluded_out_of_range:
        print("Out-of-range training events:")
        for item in excluded_out_of_range:
            print(
                item["video_name"],
                item["event_position"],
            )

    return (
        distribution,
        source_videos,
        excluded_out_of_range,
        excluded_unusable,
    )


# ============================================================
# NEGATIVE PSEUDO-EVENT ANCHORS
# ============================================================

def build_negative_anchor_assignments(
    manifests,
    distribution,
):
    """
    Assign training-derived event positions to negative videos
    in train, validation and test.

    Sampling is deterministic and uses replacement.
    """

    assignments = {}

    split_seed_offsets = {
        "train": 0,
        "validation": 1000,
        "test": 2000,
    }

    for split in SPLITS:

        negative_videos = []

        for _, row in manifests[split].iterrows():

            if int(row["label"]) != 0:
                continue

            video_name = row["video_name"]

            df = load_scene_features(
                video_name,
                split,
                0,
            )

            if df is None:
                continue

            first_timestamp = float(
                df["timestamp_sec"].iloc[0]
            )

            last_timestamp = float(
                df["timestamp_sec"].iloc[-1]
            )

            duration = last_timestamp - first_timestamp

            if duration <= 0:
                continue

            negative_videos.append({
                "video_name": video_name,
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
                "duration": duration,
            })

        # Sort before assignment to make mapping reproducible
        # even if manifest row order changes.
        negative_videos.sort(
            key=lambda item: item["video_name"]
        )

        rng = np.random.default_rng(
            RANDOM_SEED + split_seed_offsets[split]
        )

        sampled_positions = rng.choice(
            distribution,
            size=len(negative_videos),
            replace=True,
        )

        for video, position in zip(
            negative_videos,
            sampled_positions,
        ):

            pseudo_event_anchor = (
                video["first_timestamp"]
                + float(position) * video["duration"]
            )

            assignments[
                (split, video["video_name"])
            ] = {
                "pseudo_event_position": float(position),
                "pseudo_event_anchor_sec": float(
                    pseudo_event_anchor
                ),
            }

        print(
            f"{split.upper()} negative anchors assigned: "
            f"{len(negative_videos)}"
        )

    return assignments


# ============================================================
# SEQUENCE EXTRACTION
# ============================================================

def extract_sequence(df, endpoint, feature_columns):

    timestamps = df["timestamp_sec"].to_numpy(
        dtype=float
    )

    eligible_indices = np.flatnonzero(
        timestamps <= endpoint
    )

    if len(eligible_indices) < SEQUENCE_LENGTH:
        return None, "insufficient_history"

    end_index = eligible_indices[-1]

    selected = df.iloc[
        end_index - SEQUENCE_LENGTH + 1:
        end_index + 1
    ].copy()

    selected_times = selected[
        "timestamp_sec"
    ].to_numpy(dtype=float)

    endpoint_lag = endpoint - selected_times[-1]

    if endpoint_lag < -1e-9:
        return None, "future_observation"

    if endpoint_lag > MAX_ENDPOINT_LAG_SEC:
        return None, "endpoint_too_far"

    gaps = np.diff(selected_times)

    if len(gaps) != SEQUENCE_LENGTH - 1:
        return None, "invalid_sequence_length"

    if np.any(gaps <= 0):
        return None, "non_increasing_timestamps"

    max_gap = float(np.max(gaps))

    if max_gap > MAX_ALLOWED_GAP_SEC:
        return None, "timestamp_gap"

    values = selected[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    metadata = {
        "selected_endpoint_sec": float(
            selected_times[-1]
        ),
        "endpoint_lag_sec": float(endpoint_lag),
        "sequence_start_sec": float(
            selected_times[0]
        ),
        "sequence_end_sec": float(
            selected_times[-1]
        ),
        "max_gap_sec": max_gap,
    }

    return values, metadata


# ============================================================
# SAMPLE GENERATION
# ============================================================

def generate_samples(
    manifests,
    annotations,
    feature_columns,
    negative_anchor_assignments,
):
    samples_by_split = {
        split: [] for split in SPLITS
    }

    manifest_records = []
    summary_records = []

    for split in SPLITS:

        for _, video_row in manifests[split].iterrows():

            video_name = video_row["video_name"]
            label = int(video_row["label"])

            if video_name not in annotations.index:
                raise ValueError(
                    f"Missing annotation for {video_name}"
                )

            annotation = annotations.loc[video_name]

            if int(annotation["label"]) != label:
                raise ValueError(
                    f"Label mismatch for {video_name}"
                )

            df = load_scene_features(
                video_name,
                split,
                label,
            )

            if df is None:

                summary_records.append({
                    "split": split,
                    "video_name": video_name,
                    "label": label,
                    "status": "no_usable_scene_features",
                    "generated_samples": 0,
                })

                continue

            actual_features = get_feature_columns(df)

            if actual_features != feature_columns:
                raise ValueError(
                    f"Feature schema mismatch for {video_name}"
                )

            first_timestamp = float(
                df["timestamp_sec"].iloc[0]
            )

            last_timestamp = float(
                df["timestamp_sec"].iloc[-1]
            )

            event_time = annotation["time_of_event"]
            alert_time = annotation["time_of_alert"]

            if label == 1:

                # Always preserve the true event timestamp.
                anchor_sec = float(event_time)

                duration = last_timestamp - first_timestamp

                anchor_position = (
                    (anchor_sec - first_timestamp) / duration
                    if duration > 0
                    else np.nan
                )

                anchor_type = "true_event"

                pseudo_position = np.nan

            else:

                assignment_key = (split, video_name)

                if assignment_key not in negative_anchor_assignments:
                    raise ValueError(
                        f"Missing negative anchor assignment: "
                        f"{split}/{video_name}"
                    )

                assignment = negative_anchor_assignments[
                    assignment_key
                ]

                anchor_sec = assignment[
                    "pseudo_event_anchor_sec"
                ]

                anchor_position = assignment[
                    "pseudo_event_position"
                ]

                pseudo_position = anchor_position
                anchor_type = "training_positive_distribution"

            generated_count = 0

            for horizon in HORIZONS_SEC:

                endpoint = anchor_sec - horizon

                sample, sequence_info = extract_sequence(
                    df,
                    endpoint,
                    feature_columns,
                )

                sample_id = (
                    f"{Path(video_name).stem}"
                    f"_h{int(horizon * 1000)}"
                )

                alert_relationship = (
                    bool(endpoint >= float(alert_time))
                    if label == 1
                    else None
                )

                common_metadata = {
                    "sample_id": sample_id,
                    "split": split,
                    "video_name": video_name,
                    "label": label,
                    "horizon_ms": int(horizon * 1000),
                    "anchor_type": anchor_type,
                    "anchor_position": anchor_position,
                    "anchor_sec": anchor_sec,
                    "pseudo_event_position": pseudo_position,
                    "prediction_endpoint_sec": endpoint,
                    "time_of_event": event_time,
                    "time_of_alert": alert_time,
                    "endpoint_at_or_after_alert": (
                        alert_relationship
                    ),
                }

                if sample is None:

                    manifest_records.append({
                        **common_metadata,
                        "status": sequence_info,
                    })

                    continue

                # Use the exact selected 50 observations.
                sequence_times = df.loc[
                    df["timestamp_sec"].between(
                        sequence_info["sequence_start_sec"],
                        sequence_info["sequence_end_sec"],
                    ),
                    "timestamp_sec",
                ].to_numpy(dtype=float)[-SEQUENCE_LENGTH:]

                if len(sequence_times) != SEQUENCE_LENGTH:
                    raise ValueError(
                        f"Unexpected sequence timestamp count: "
                        f"{sample_id}"
                    )

                samples_by_split[split].append({
                    "X_raw": sample,
                    "label": label,
                    "horizon": horizon,
                    "video_name": video_name,
                    "sample_id": sample_id,
                    "sequence_times": sequence_times,
                })

                manifest_records.append({
                    **common_metadata,
                    "status": "valid",
                    **sequence_info,
                })

                generated_count += 1

            summary_records.append({
                "split": split,
                "video_name": video_name,
                "label": label,
                "status": "processed",
                "generated_samples": generated_count,
            })

    return (
        samples_by_split,
        pd.DataFrame(manifest_records),
        pd.DataFrame(summary_records),
    )


# ============================================================
# TRAINING-ONLY PREPROCESSING
# ============================================================

def fit_preprocessing(train_samples, feature_columns):

    if not train_samples:
        raise ValueError(
            "No valid training sequences were generated."
        )

    unique_rows = {}

    for sample in train_samples:

        times = sample["sequence_times"]
        values = sample["X_raw"]

        for timestamp, row in zip(times, values):

            key = (
                sample["video_name"],
                round(float(timestamp), 6),
            )

            unique_rows[key] = row

    training_data = pd.DataFrame(
        list(unique_rows.values()),
        columns=feature_columns,
    )

    training_data = training_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    missing_indicator_columns = [
        column
        for column in feature_columns
        if training_data[column].isna().any()
    ]

    medians = training_data.median().fillna(0.0)

    imputed_training = training_data.fillna(medians)

    means = imputed_training.mean()

    standard_deviations = (
        imputed_training.std(ddof=0)
        .replace(0, 1.0)
        .fillna(1.0)
    )

    config = {
        "generator_version": GENERATOR_VERSION,
        "random_seed": RANDOM_SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "max_allowed_gap_sec": MAX_ALLOWED_GAP_SEC,
        "max_endpoint_lag_sec": MAX_ENDPOINT_LAG_SEC,
        "horizons_sec": HORIZONS_SEC,
        "feature_columns": feature_columns,
        "missing_indicator_columns": (
            missing_indicator_columns
        ),
        "medians": {
            key: float(value)
            for key, value in medians.items()
        },
        "means": {
            key: float(value)
            for key, value in means.items()
        },
        "standard_deviations": {
            key: float(value)
            for key, value in standard_deviations.items()
        },
        "training_unique_observations": len(training_data),
        "preprocessing_fit_split": "train",
    }

    return (
        medians,
        means,
        standard_deviations,
        missing_indicator_columns,
        config,
    )


def transform_samples(
    samples,
    feature_columns,
    medians,
    means,
    standard_deviations,
    missing_indicator_columns,
):

    X_list = []
    y_list = []
    horizon_list = []
    video_names = []
    sample_ids = []

    for sample in samples:

        raw = pd.DataFrame(
            sample["X_raw"],
            columns=feature_columns,
        )

        raw = raw.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        missing = raw[
            missing_indicator_columns
        ].isna().astype(np.float32)

        imputed = raw.fillna(medians)

        normalized = (
            imputed - means
        ) / standard_deviations

        combined = pd.concat(
            [normalized, missing],
            axis=1,
        )

        values = combined.to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite model input in {sample['sample_id']}"
            )

        X_list.append(values)
        y_list.append(sample["label"])
        horizon_list.append(sample["horizon"])
        video_names.append(sample["video_name"])
        sample_ids.append(sample["sample_id"])

    final_feature_count = (
        len(feature_columns)
        + len(missing_indicator_columns)
    )

    if not X_list:

        return {
            "X": np.empty(
                (0, SEQUENCE_LENGTH, final_feature_count),
                dtype=np.float32,
            ),
            "y": np.empty((0,), dtype=np.int64),
            "horizon_sec": np.empty((0,), dtype=np.float32),
            "video_names": np.empty((0,), dtype=str),
            "sample_ids": np.empty((0,), dtype=str),
        }

    return {
        "X": np.stack(X_list).astype(np.float32),
        "y": np.asarray(y_list, dtype=np.int64),
        "horizon_sec": np.asarray(
            horizon_list,
            dtype=np.float32,
        ),
        "video_names": np.asarray(video_names, dtype=str),
        "sample_ids": np.asarray(sample_ids, dtype=str),
    }


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    samples_by_split,
    manifest,
    summary,
    feature_columns,
    feature_schema_source,
    distribution,
    distribution_source_videos,
    excluded_out_of_range,
    excluded_unusable,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        medians,
        means,
        standard_deviations,
        missing_indicator_columns,
        config,
    ) = fit_preprocessing(
        samples_by_split["train"],
        feature_columns,
    )

    for split in SPLITS:

        transformed = transform_samples(
            samples_by_split[split],
            feature_columns,
            medians,
            means,
            standard_deviations,
            missing_indicator_columns,
        )

        output_path = OUTPUT_DIR / f"{split}.npz"

        np.savez_compressed(
            output_path,
            X=transformed["X"],
            y=transformed["y"],
            horizon_sec=transformed["horizon_sec"],
            video_names=transformed["video_names"],
            sample_ids=transformed["sample_ids"],
        )

        print(f"\n{split.upper()} OUTPUT")
        print(f"X shape: {transformed['X'].shape}")
        print(f"y shape: {transformed['y'].shape}")

        if len(transformed["y"]) > 0:

            unique, counts = np.unique(
                transformed["y"],
                return_counts=True,
            )

            print(
                "Class counts:",
                dict(zip(unique.tolist(), counts.tolist())),
            )

    manifest.to_csv(
        OUTPUT_DIR / "sequence_generation_manifest.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "sequence_generation_summary.csv",
        index=False,
    )

    # Save the exact training-positive reference distribution.
    distribution_df = pd.DataFrame({
        "source_video_name": distribution_source_videos,
        "event_position": distribution,
    })

    distribution_df.to_csv(
        OUTPUT_DIR / "training_positive_anchor_distribution.csv",
        index=False,
    )

    config.update({
        "feature_schema_source": feature_schema_source,
        "base_feature_count": len(feature_columns),
        "final_feature_count": (
            len(feature_columns)
            + len(missing_indicator_columns)
        ),
        "negative_anchor_policy": (
            "Sample training-positive event positions within [0,1] "
            "with replacement. Map each sampled position onto the "
            "negative video's observed timestamp interval."
        ),
        "negative_anchor_reference_split": "train",
        "negative_anchor_reference_count": len(distribution),
        "negative_anchor_reference_videos": (
            distribution_source_videos
        ),
        "negative_anchor_reference_mean": float(
            distribution.mean()
        ),
        "negative_anchor_reference_median": float(
            np.median(distribution)
        ),
        "negative_anchor_reference_min": float(
            distribution.min()
        ),
        "negative_anchor_reference_max": float(
            distribution.max()
        ),
        "excluded_training_positive_out_of_range": (
            excluded_out_of_range
        ),
        "excluded_training_positive_unusable": (
            excluded_unusable
        ),
        "positive_anchor_policy": (
            "Use the original time_of_event without clamping."
        ),
        "preprocessing_fit_split": "train",
    })

    with open(
        OUTPUT_DIR / "preprocessing_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    print("\nPREPROCESSING")
    print(f"Base features: {len(feature_columns)}")
    print(
        f"Missingness indicators: "
        f"{len(missing_indicator_columns)}"
    )
    print(
        f"Final features: {config['final_feature_count']}"
    )
    print(
        "Training unique observations:",
        config["training_unique_observations"],
    )

    print("\nMANIFEST STATUS")

    print(
        manifest.groupby(
            ["split", "label", "status"]
        ).size().to_string()
    )

    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print("=" * 70)
    print("PHASE 5.3 - REVISED EVENT-ALIGNED SEQUENCE GENERATION")
    print("=" * 70)

    annotations = load_annotations()

    manifests = load_split_manifests()

    feature_columns, feature_schema_source = (
        discover_feature_schema(manifests)
    )

    print(f"\nBase numeric features: {len(feature_columns)}")

    (
        distribution,
        distribution_source_videos,
        excluded_out_of_range,
        excluded_unusable,
    ) = build_training_event_position_distribution(
        manifests,
        annotations,
    )

    negative_anchor_assignments = (
        build_negative_anchor_assignments(
            manifests,
            distribution,
        )
    )

    (
        samples_by_split,
        manifest,
        summary,
    ) = generate_samples(
        manifests,
        annotations,
        feature_columns,
        negative_anchor_assignments,
    )

    save_outputs(
        samples_by_split,
        manifest,
        summary,
        feature_columns,
        feature_schema_source,
        distribution,
        distribution_source_videos,
        excluded_out_of_range,
        excluded_unusable,
    )

    print("\nPHASE 5.3 COMPLETED")


if __name__ == "__main__":
    main()
