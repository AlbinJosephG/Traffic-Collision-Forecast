
from pathlib import Path
import json

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

OUTPUT_DIR = PROJECT_ROOT / "processed" / "event_aligned_sequences"

SPLITS = ["train", "validation", "test"]
HORIZONS_SEC = [0.5, 1.0, 1.5]

SEQUENCE_LENGTH = 50
MAX_ALLOWED_GAP_SEC = 0.30
MAX_ENDPOINT_LAG_SEC = 0.25

RANDOM_SEED = 42

# Metadata columns are not model input features.
EXCLUDED_COLUMNS = {
    "video_name",
    "split",
    "label",
    "frame_index",
    "timestamp_sec",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_annotations():
    """Load and validate the matched Nexar annotations."""

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
        raise ValueError(
            "Duplicate video names found in annotation file."
        )

    annotations["label"] = pd.to_numeric(
        annotations["label"], errors="raise"
    ).astype(int)

    if not annotations["label"].isin([0, 1]).all():
        raise ValueError("Labels must be either 0 or 1.")

    return annotations.set_index("video_name", drop=False)


def load_scene_features(video_name, split, label):
    """Load the consolidated frame-level features for one video."""

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
        df["timestamp_sec"], errors="coerce"
    )

    df = (
        df.dropna(subset=["timestamp_sec"])
        .sort_values("timestamp_sec")
        .drop_duplicates(subset=["timestamp_sec"])
        .reset_index(drop=True)
    )

    if len(df) < SEQUENCE_LENGTH:
        return None

    return df


def get_feature_columns(df):
    """Identify numeric model features, excluding metadata."""

    columns = []

    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)

    if not columns:
        raise ValueError("No numeric model features were found.")

    return columns


def extract_sequence(df, endpoint, feature_columns):
    """
    Extract the last 50 observations at or before endpoint.

    Reject sequences with:
    - insufficient history
    - an endpoint that is too far from the last observation
    - any timestamp gap greater than 0.30 seconds
    """

    timestamps = df["timestamp_sec"].to_numpy(dtype=float)

    # Only observations at or before the prediction endpoint.
    eligible_indices = np.flatnonzero(timestamps <= endpoint)

    if len(eligible_indices) < SEQUENCE_LENGTH:
        return None, "insufficient_history"

    end_index = eligible_indices[-1]

    selected = df.iloc[
        end_index - SEQUENCE_LENGTH + 1 : end_index + 1
    ].copy()

    selected_times = selected["timestamp_sec"].to_numpy(
        dtype=float
    )

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

    # Return the raw numeric features. Imputation and scaling
    # happen later using training data only.
    values = selected[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)

    metadata = {
        "selected_endpoint_sec": float(selected_times[-1]),
        "endpoint_lag_sec": float(endpoint_lag),
        "sequence_start_sec": float(selected_times[0]),
        "sequence_end_sec": float(selected_times[-1]),
        "max_gap_sec": max_gap,
    }

    return values, metadata


# ============================================================
# SAMPLE GENERATION
# ============================================================

def generate_samples(annotations):
    """Generate three event-aligned samples per eligible video."""

    samples_by_split = {
        split: [] for split in SPLITS
    }

    manifest_records = []
    summary_records = []

    # Establish a consistent feature schema across all videos.
    first_feature_file = next(
        SCENE_FEATURE_DIR.rglob("*.csv"),
        None
    )

    if first_feature_file is None:
        raise FileNotFoundError(
            f"No scene-feature CSV files found in {SCENE_FEATURE_DIR}"
        )

    first_df = pd.read_csv(first_feature_file)
    feature_columns = get_feature_columns(first_df)

    print(f"Numeric model features: {len(feature_columns)}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Horizons: {HORIZONS_SEC}")

    for split in SPLITS:

        split_manifest = (
            PROJECT_ROOT
            / "data"
            / "manifests"
            / f"{split}_manifest.csv"
        )

        if not split_manifest.exists():
            raise FileNotFoundError(
                f"Split manifest not found: {split_manifest}"
            )

        video_manifest = pd.read_csv(split_manifest)

        if not {"video_path", "label"}.issubset(
            video_manifest.columns
        ):
            raise ValueError(
                f"Invalid columns in {split_manifest}"
            )

        for _, video_row in video_manifest.iterrows():

            video_name = Path(
                str(video_row["video_path"])
            ).name

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
                video_name, split, label
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

            last_timestamp = float(
                df["timestamp_sec"].max()
            )

            event_time = pd.to_numeric(
                pd.Series([annotation["time_of_event"]]),
                errors="coerce"
            ).iloc[0]

            alert_time = pd.to_numeric(
                pd.Series([annotation["time_of_alert"]]),
                errors="coerce"
            ).iloc[0]

            generated_count = 0

            for horizon in HORIZONS_SEC:

                if label == 1:

                    if pd.isna(event_time):
                        raise ValueError(
                            f"Positive video {video_name} "
                            "has no event timestamp."
                        )

                    endpoint = float(event_time) - horizon

                else:

                    # Negative videos have no collision event.
                    # Use their final observed time as the reference.
                    endpoint = last_timestamp - horizon

                sample, sequence_info = extract_sequence(
                    df,
                    endpoint,
                    feature_columns,
                )

                sample_id = (
                    f"{Path(video_name).stem}"
                    f"_h{int(horizon * 1000)}"
                )

                if sample is None:

                    manifest_records.append({
                        "sample_id": sample_id,
                        "split": split,
                        "video_name": video_name,
                        "label": label,
                        "horizon_ms": int(horizon * 1000),
                        "prediction_endpoint_sec": endpoint,
                        "time_of_event": event_time,
                        "time_of_alert": alert_time,
                        "endpoint_at_or_after_alert": (
                            bool(endpoint >= alert_time)
                            if label == 1 and pd.notna(alert_time)
                            else None
                        ),
                        "status": sequence_info,
                    })

                    continue

                alert_relationship = (
                    bool(endpoint >= alert_time)
                    if label == 1 and pd.notna(alert_time)
                    else None
                )

                samples_by_split[split].append({
                    "X_raw": sample,
                    "label": label,
                    "horizon": horizon,
                    "video_name": video_name,
                    "sample_id": sample_id,
                    "sequence_times": df.loc[
                        df["timestamp_sec"].between(
                            sequence_info["sequence_start_sec"],
                            sequence_info["sequence_end_sec"],
                        ),
                        "timestamp_sec",
                    ].to_numpy(dtype=float)[-SEQUENCE_LENGTH:],
                })

                manifest_records.append({
                    "sample_id": sample_id,
                    "split": split,
                    "video_name": video_name,
                    "label": label,
                    "horizon_ms": int(horizon * 1000),
                    "prediction_endpoint_sec": endpoint,
                    "time_of_event": event_time,
                    "time_of_alert": alert_time,
                    "endpoint_at_or_after_alert": alert_relationship,
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
        feature_columns,
    )


# ============================================================
# TRAINING-ONLY PREPROCESSING
# ============================================================

def fit_preprocessing(train_samples, feature_columns):
    """
    Fit medians, missingness indicators, means and standard
    deviations using unique training observations only.
    """

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
        [np.inf, -np.inf], np.nan
    )

    missing_indicator_columns = [
        column
        for column in feature_columns
        if training_data[column].isna().any()
    ]

    medians = training_data.median()

    # Features completely missing in training receive 0 as
    # their fallback median; this is explicitly recorded.
    medians = medians.fillna(0.0)

    imputed_training = training_data.fillna(medians)

    means = imputed_training.mean()
    standard_deviations = imputed_training.std(ddof=0)

    # Constant features must not cause division by zero.
    standard_deviations = standard_deviations.replace(
        0, 1.0
    ).fillna(1.0)

    config = {
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
    """Apply training-fitted preprocessing to a split."""

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

        raw = raw.replace([np.inf, -np.inf], np.nan)

        missing = raw[
            missing_indicator_columns
        ].isna().astype(np.float32)

        imputed = raw.fillna(medians)

        normalized = (
            imputed - means
        ) / standard_deviations

        # Append missingness indicators after normalization.
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

    if not X_list:
        return {
            "X": np.empty(
                (0, SEQUENCE_LENGTH,
                 len(feature_columns) + len(missing_indicator_columns)),
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
            horizon_list, dtype=np.float32
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
        print(
            f"Horizon shape: "
            f"{transformed['horizon_sec'].shape}"
        )

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
        "Missingness indicators:",
        len(missing_indicator_columns),
    )
    print(
        "Final features per time step:",
        len(feature_columns) + len(missing_indicator_columns),
    )
    print(
        "Training observations used to fit preprocessing:",
        config["training_unique_observations"],
    )

    print("\nMANIFEST STATUS")
    print(
        manifest.groupby(
            ["split", "status"]
        ).size().to_string()
    )

    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 3E.3 - EVENT-ALIGNED SEQUENCE GENERATION")
    print("=" * 65)

    annotations = load_annotations()

    (
        samples_by_split,
        manifest,
        summary,
        feature_columns,
    ) = generate_samples(annotations)

    save_outputs(
        samples_by_split,
        manifest,
        summary,
        feature_columns,
    )

    print("\nPHASE 3E.3 COMPLETED")


if __name__ == "__main__":
    main()
