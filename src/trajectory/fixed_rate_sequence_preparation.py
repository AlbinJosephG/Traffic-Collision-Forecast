from pathlib import Path
import json
import random

import numpy as np
import pandas as pd


# ============================================================
# PHASE 5.5: FIXED-RATE SEQUENCE GENERATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SCENE_DIR = ROOT / "processed" / "scene_features"

EVENT_MANIFEST = (
    ROOT
    / "processed"
    / "revised_event_aligned_sequences"
    / "sequence_generation_manifest.csv"
)

ACCEPTANCE_FILE = (
    ROOT
    / "processed"
    / "fixed_rate_resampling_audit"
    / "resampling_acceptance.csv"
)

OUTPUT_DIR = ROOT / "processed" / "fixed_rate_sequences"

SPLITS = ["train", "validation", "test"]
HORIZONS = [500, 1000, 1500]

SEQUENCE_LENGTH = 50
TARGET_INTERVAL = 0.2

MAX_INTERPOLATION_GAP = 0.30
DIRECT_TOLERANCE = 0.05
MAX_COUNT_HOLD_AGE = 0.30

MIN_COVERAGE_PERCENT = 90.0
MAX_UNAVAILABLE_RUN = 5
MAX_ENDPOINT_LAG = 0.25

SEED = 42
TIMESTAMP_COLUMN = "timestamp_sec"

# Metadata columns are not model inputs.
EXCLUDED_COLUMNS = {
    "video_name",
    "split",
    "label",
    "frame_index",
    TIMESTAMP_COLUMN,
}

# Count-like features must not be linearly interpolated.
COUNT_FEATURES = {
    "vehicle_count",
    "bicycle_count",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "interaction_pair_count",
    "approaching_pair_count",
    "nearby_pair_count",
    "tracked_object_count",
    "person_object_count",
    "bicycle_object_count",
    "car_object_count",
    "motorcycle_object_count",
    "bus_object_count",
    "truck_object_count",
}


def set_reproducibility():
    random.seed(SEED)
    np.random.seed(SEED)


def longest_true_run(values):
    """Return the longest consecutive run of True values."""
    longest = 0
    current = 0

    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def get_video_path(row):
    """Resolve a scene-feature CSV from split, label and video name."""
    split = str(row["split"])
    label = int(row["label"])
    video_name = Path(str(row["video_name"])).stem

    category = "positive" if label == 1 else "negative"

    return (
        SCENE_DIR
        / split
        / category
        / f"{video_name}.csv"
    )


def make_target_grid(latest_observation):
    """
    Create 50 target timestamps at 5 FPS.

    The final grid point is the latest source observation available
    at or before the requested prediction endpoint.
    """
    offsets = np.arange(SEQUENCE_LENGTH, dtype=float) * TARGET_INTERVAL
    return latest_observation - offsets[::-1]


def load_scene_data(path, feature_columns):
    """Read, type-convert, sort and deduplicate a scene CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Scene-feature file not found: {path}")

    df = pd.read_csv(path)

    required = [TIMESTAMP_COLUMN] + feature_columns
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}"
        )

    df[TIMESTAMP_COLUMN] = pd.to_numeric(
        df[TIMESTAMP_COLUMN], errors="coerce"
    )

    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[TIMESTAMP_COLUMN])
    df = df[np.isfinite(df[TIMESTAMP_COLUMN].to_numpy(dtype=float))]

    df = (
        df.sort_values(TIMESTAMP_COLUMN)
        .drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(f"No usable timestamps in {path}")

    return df


def resample_sequence(
    df,
    feature_columns,
    target_grid,
    latest_observation,
):
    """
    Feature-aware causal resampling.

    Continuous features:
      - Exact source observation, or previous observation within
        DIRECT_TOLERANCE, is used directly.
      - Otherwise interpolate between bracketing observations only
        when their gap is <= MAX_INTERPOLATION_GAP.

    Count features:
      - Use the most recent observation at or before the target time
        if its age is <= MAX_COUNT_HOLD_AGE.
      - Never linearly interpolate counts.

    No source observation after latest_observation is used.
    """

    timestamps = df[TIMESTAMP_COLUMN].to_numpy(dtype=float)
    values = df[feature_columns].to_numpy(dtype=float)

    # Enforce the prediction-time boundary.
    allowed = timestamps <= latest_observation + 1e-9
    timestamps = timestamps[allowed]
    values = values[allowed]

    if len(timestamps) == 0:
        raise ValueError("No observations available before prediction endpoint")

    result = np.full(
        (SEQUENCE_LENGTH, len(feature_columns)),
        np.nan,
        dtype=np.float32,
    )

    direct_steps = 0
    interpolated_steps = 0
    unavailable_steps = 0

    count_indices = [
        index
        for index, feature in enumerate(feature_columns)
        if feature in COUNT_FEATURES
    ]

    continuous_indices = [
        index
        for index, feature in enumerate(feature_columns)
        if feature not in COUNT_FEATURES
    ]

    for step, target_time in enumerate(target_grid):
        right = int(np.searchsorted(timestamps, target_time, side="left"))
        left = right - 1

        # Exact source observation.
        if (
            right < len(timestamps)
            and np.isclose(timestamps[right], target_time, atol=1e-8)
        ):
            result[step] = values[right]
            direct_steps += 1
            continue

        # A previous source observation close enough to the target.
        previous_is_direct = False
        if left >= 0:
            previous_age = target_time - timestamps[left]
            if 0.0 <= previous_age <= DIRECT_TOLERANCE:
                result[step] = values[left]
                direct_steps += 1
                previous_is_direct = True

        if previous_is_direct:
            continue

        # Continuous features: interpolate only between valid brackets.
        interpolated_any = False

        if left >= 0 and right < len(timestamps):
            left_time = timestamps[left]
            right_time = timestamps[right]
            gap = right_time - left_time

            if 0.0 < gap <= MAX_INTERPOLATION_GAP:
                alpha = (target_time - left_time) / gap

                for feature_index in continuous_indices:
                    left_value = values[left, feature_index]
                    right_value = values[right, feature_index]

                    if np.isfinite(left_value) and np.isfinite(right_value):
                        result[step, feature_index] = (
                            left_value + alpha * (right_value - left_value)
                        )
                        interpolated_any = True

        # Count features: previous-observation hold only.
        if left >= 0:
            previous_age = target_time - timestamps[left]

            if 0.0 <= previous_age <= MAX_COUNT_HOLD_AGE:
                for feature_index in count_indices:
                    previous_value = values[left, feature_index]
                    if np.isfinite(previous_value):
                        result[step, feature_index] = previous_value

        if interpolated_any:
            interpolated_steps += 1
        else:
            unavailable_steps += 1

    # Coverage is measured in usable time steps, matching the audit:
    # a step is usable if at least one base feature is available.
    row_available = np.isfinite(result).any(axis=1)

    actual_unavailable_steps = int(np.sum(~row_available))
    longest_unavailable = longest_true_run(~row_available)

    coverage = (
        float(np.sum(row_available)) / SEQUENCE_LENGTH * 100.0
    )

    return {
        "values": result,
        "direct_steps": int(direct_steps),
        "interpolated_steps": int(interpolated_steps),
        "unavailable_steps": int(unavailable_steps),
        "actual_unavailable_steps": actual_unavailable_steps,
        "longest_unavailable_run": int(longest_unavailable),
        "coverage_percent": float(coverage),
    }


def load_and_match_manifests():
    """Load both manifests and join each video/horizon exactly once."""

    acceptance = pd.read_csv(ACCEPTANCE_FILE)
    manifest = pd.read_csv(EVENT_MANIFEST)

    keys = ["split", "label", "video_name", "horizon_ms"]

    required_acceptance = set(keys) | {
        "acceptance_status",
        "coverage_percent",
        "longest_unavailable_run",
        "endpoint_lag_sec",
        "prediction_endpoint_sec",
    }

    required_manifest = set(keys) | {
        "sample_id",
        "prediction_endpoint_sec",
        "selected_endpoint_sec",
        "endpoint_lag_sec",
        "time_of_event",
        "time_of_alert",
        "endpoint_at_or_after_alert",
        "anchor_type",
        "anchor_position",
        "status",
    }

    missing_audit = required_acceptance - set(acceptance.columns)
    missing_manifest = required_manifest - set(manifest.columns)

    if missing_audit:
        raise ValueError(
            f"Acceptance CSV missing columns: {sorted(missing_audit)}"
        )

    if missing_manifest:
        raise ValueError(
            f"Event manifest missing columns: {sorted(missing_manifest)}"
        )

    if acceptance.duplicated(keys).any():
        raise ValueError("Duplicate keys found in acceptance CSV")

    if manifest.duplicated(keys).any():
        raise ValueError("Duplicate keys found in event manifest")

    merged = manifest.merge(
        acceptance,
        on=keys,
        how="left",
        suffixes=("_manifest", "_audit"),
        validate="one_to_one",
        indicator=True,
    )

    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        examples = merged.loc[unmatched, keys].head(10).to_dict("records")
        raise ValueError(
            f"{int(unmatched.sum())} event-manifest rows did not match "
            f"the acceptance audit. Examples: {examples}"
        )

    merged = merged.drop(columns=["_merge"])

    merged["acceptance_status"] = (
        merged["acceptance_status"].astype(str).str.lower()
    )

    accepted_by_audit = merged["acceptance_status"].eq("accepted")

    audit_coverage = pd.to_numeric(
        merged["coverage_percent"], errors="coerce"
    )
    audit_gap_run = pd.to_numeric(
        merged["longest_unavailable_run"], errors="coerce"
    )
    audit_lag = pd.to_numeric(
        merged["endpoint_lag_sec_audit"], errors="coerce"
    )

    criteria = (
        audit_coverage.ge(MIN_COVERAGE_PERCENT)
        & audit_gap_run.le(MAX_UNAVAILABLE_RUN)
        & audit_lag.le(MAX_ENDPOINT_LAG)
    ).fillna(False)

    if not np.array_equal(
        accepted_by_audit.to_numpy(),
        criteria.to_numpy(),
    ):
        raise ValueError(
            "Acceptance status does not match the configured audit "
            "criteria. Recheck resampling_acceptance.csv."
        )

    merged["accepted_by_audit"] = accepted_by_audit

    return merged

def get_feature_columns():
    """
    Establish the feature schema from a usable training CSV.

    Empty scene-feature files are retained in the dataset but skipped
    during schema validation because they contain no observations.
    """

    train_files = sorted((SCENE_DIR / "train").rglob("*.csv"))

    if not train_files:
        raise FileNotFoundError(
            f"No training scene CSVs found under {SCENE_DIR / 'train'}"
        )

    expected = None
    schema_source = None
    validated_files = 0
    empty_files = []

    for path in train_files:
        df = pd.read_csv(path)

        # Preserve empty files, but do not use them to establish
        # or validate the feature schema.
        if df.empty:
            empty_files.append(path.name)
            continue

        actual = [
            column
            for column in df.columns
            if column not in EXCLUDED_COLUMNS
        ]

        if expected is None:
            expected = actual
            schema_source = path.name
        elif actual != expected:
            missing = [
                column for column in expected
                if column not in actual
            ]

            extra = [
                column for column in actual
                if column not in expected
            ]

            raise ValueError(
                f"Feature schema/order mismatch in {path.name}\n"
                f"Reference file: {schema_source}\n"
                f"Missing columns: {missing}\n"
                f"Extra columns: {extra}\n"
                f"Expected order: {expected}\n"
                f"Actual order: {actual}"
            )

        validated_files += 1

    if expected is None:
        raise ValueError(
            "No non-empty training scene-feature CSVs available "
            "to establish the feature schema."
        )

    if len(expected) != 66:
        raise ValueError(
            f"Expected 66 base features, found {len(expected)}."
        )

    print("Schema reference:", schema_source)
    print("Non-empty files validated:", validated_files)
    print("Empty files skipped:", len(empty_files))

    if empty_files:
        print("Empty files:", empty_files)

    return expected


def identify_missingness_features(samples, feature_columns):
    """
    Identify columns containing missing values in training videos only.
    This selection never uses validation or test data.
    """

    train_samples = samples[
        (samples["split"] == "train")
        & samples["accepted_by_audit"]
    ]

    missing_features = set()
    seen_videos = set()

    for _, row in train_samples.iterrows():
        video_key = str(row["video_name"])

        if video_key in seen_videos:
            continue

        seen_videos.add(video_key)

        df = load_scene_data(
            get_video_path(row),
            feature_columns,
        )

        for feature in feature_columns:
            if df[feature].isna().any():
                missing_features.add(feature)

    ordered = [
        feature for feature in feature_columns
        if feature in missing_features
    ]

    if len(ordered) != 30:
        raise ValueError(
            f"Expected 30 missingness-indicator features, found "
            f"{len(ordered)}. Do not silently change the model schema. "
            "Inspect training feature missingness first."
        )

    return ordered


def build_sequences(samples, feature_columns):
    """Build raw resampled sequences for every audit-accepted sample."""

    candidates = samples[samples["accepted_by_audit"]].copy()

    sequence_arrays = []
    sequence_records = []
    scene_cache = {}

    for _, row in candidates.iterrows():
        split = str(row["split"])
        label = int(row["label"])
        horizon = int(row["horizon_ms"])
        video_name = str(row["video_name"])

        # IMPORTANT:
        # The original selected_endpoint_sec can be NaN for samples
        # recovered by the fixed-rate audit. Use the audit endpoint
        # and audit endpoint lag instead.
        prediction_endpoint = float(
            row["prediction_endpoint_sec_manifest"]
        )
        endpoint_lag = float(
            row["endpoint_lag_sec_audit"]
        )

        if not np.isfinite(prediction_endpoint):
            raise ValueError(
                f"Invalid prediction endpoint for {video_name}, "
                f"{horizon} ms"
            )

        if not np.isfinite(endpoint_lag):
            raise ValueError(
                f"Invalid audit endpoint lag for {video_name}, "
                f"{horizon} ms"
            )

        latest_observation = prediction_endpoint - endpoint_lag

        if not np.isfinite(latest_observation):
            raise ValueError(
                f"Invalid latest observation for {video_name}"
            )

        if endpoint_lag < 0 or endpoint_lag > MAX_ENDPOINT_LAG:
            raise ValueError(
                f"Endpoint lag outside allowed range for {video_name}: "
                f"{endpoint_lag}"
            )

        target_grid = make_target_grid(latest_observation)

        if (
            len(target_grid) != SEQUENCE_LENGTH
            or np.max(target_grid) > latest_observation + 1e-8
        ):
            raise ValueError(
                f"Invalid target grid for {video_name}"
            )

        cache_key = (split, video_name)

        if cache_key not in scene_cache:
            scene_cache[cache_key] = load_scene_data(
                get_video_path(row),
                feature_columns,
            )

        resampled = resample_sequence(
            scene_cache[cache_key],
            feature_columns,
            target_grid,
            latest_observation,
        )

        values = resampled["values"]

        # Apply the generator's own acceptance check.
        generator_accepted = (
            resampled["coverage_percent"] >= MIN_COVERAGE_PERCENT
            and resampled["longest_unavailable_run"]
            <= MAX_UNAVAILABLE_RUN
        )

        sequence_arrays.append(values)

        sequence_records.append({
            "sample_id": str(row["sample_id"]),
            "split": split,
            "video_name": video_name,
            "label": label,
            "horizon_ms": horizon,
            "anchor_type": row["anchor_type"],
            "anchor_position": row["anchor_position"],
            "prediction_endpoint_sec": prediction_endpoint,
            "latest_observation_sec": latest_observation,
            "actual_endpoint_lag_sec": endpoint_lag,
            "time_of_event": row["time_of_event"],
            "time_of_alert": row["time_of_alert"],
            "endpoint_at_or_after_alert": (
                row["endpoint_at_or_after_alert"]
            ),
            "original_manifest_status": row["status"],
            "audit_coverage_percent": row["coverage_percent"],
            "audit_longest_unavailable_run": (
                row["longest_unavailable_run"]
            ),
            "coverage_percent": resampled["coverage_percent"],
            "direct_steps": resampled["direct_steps"],
            "interpolated_steps": resampled["interpolated_steps"],
            "unavailable_steps": (
                resampled["unavailable_steps"]
            ),
            "actual_unavailable_steps": (
                resampled["actual_unavailable_steps"]
            ),
            "longest_unavailable_run": (
                resampled["longest_unavailable_run"]
            ),
            "accepted": bool(generator_accepted),
        })

    raw = np.asarray(sequence_arrays, dtype=np.float32)
    records = pd.DataFrame(sequence_records)

    if raw.ndim != 3:
        raise ValueError(
            f"Expected 3D raw sequence array, got {raw.shape}"
        )

    if raw.shape[1:] != (SEQUENCE_LENGTH, len(feature_columns)):
        raise ValueError(
            f"Unexpected raw array shape: {raw.shape}"
        )

    return raw, records


def fit_preprocessing(
    raw_sequences,
    records,
    feature_columns,
    missingness_features,
):
    """
    Fit imputation and scaling from unique accepted training
    video/timestamp observations only.
    """

    train_indices = np.where(
        (records["split"].to_numpy() == "train")
        & records["accepted"].to_numpy()
    )[0]

    if len(train_indices) == 0:
        raise ValueError("No accepted training sequences for preprocessing")

    unique_rows = {}

    for index in train_indices:
        row = records.iloc[index]

        video = str(row["video_name"])
        split = str(row["split"])
        latest = float(row["latest_observation_sec"])

        grid = make_target_grid(latest)

        for step, timestamp in enumerate(grid):
            key = (
                split,
                video,
                round(float(timestamp), 6),
            )

            if key not in unique_rows:
                unique_rows[key] = raw_sequences[index, step].copy()

    unique_matrix = np.asarray(
        list(unique_rows.values()),
        dtype=np.float64,
    )

    # Column-wise median; all-empty columns receive 0.
    with np.errstate(all="ignore"):
        medians = np.nanmedian(unique_matrix, axis=0)

    medians = np.where(np.isfinite(medians), medians, 0.0)

    imputed = np.where(
        np.isfinite(unique_matrix),
        unique_matrix,
        medians,
    )

    means = np.mean(imputed, axis=0)
    stds = np.std(imputed, axis=0)

    stds = np.where(stds < 1e-8, 1.0, stds)

    return {
        "feature_columns": feature_columns,
        "missingness_features": missingness_features,
        "base_feature_count": len(feature_columns),
        "missingness_indicator_count": len(missingness_features),
        "final_feature_count": (
            len(feature_columns) + len(missingness_features)
        ),
        "medians": medians.tolist(),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "unique_training_observations": int(len(unique_matrix)),
        "scaling": (
            "Base features: train-only median imputation and "
            "standardization. Missingness indicators: binary, unscaled."
        ),
    }


def transform_sequences(
    raw_sequences,
    feature_columns,
    missingness_features,
    config,
):
    """Apply train-only imputation/scaling and append binary indicators."""

    medians = np.asarray(config["medians"], dtype=np.float32)
    means = np.asarray(config["means"], dtype=np.float32)
    stds = np.asarray(config["stds"], dtype=np.float32)

    missing_indices = [
        feature_columns.index(feature)
        for feature in missingness_features
    ]

    transformed = np.zeros(
        (
            len(raw_sequences),
            SEQUENCE_LENGTH,
            len(feature_columns) + len(missingness_features),
        ),
        dtype=np.float32,
    )

    for index, sequence in enumerate(raw_sequences):
        missing = (~np.isfinite(sequence)).astype(np.float32)

        imputed = np.where(
            np.isfinite(sequence),
            sequence,
            medians,
        )

        normalized = (imputed - means) / stds

        indicators = missing[:, missing_indices]

        transformed[index] = np.concatenate(
            [normalized, indicators],
            axis=1,
        )

    return transformed


def save_outputs(
    X,
    records,
    config,
    feature_columns,
    missingness_features,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y = records["label"].to_numpy(dtype=np.int64)
    horizons = records["horizon_ms"].to_numpy(dtype=np.int64)
    video_names = records["video_name"].to_numpy(dtype=str)
    sample_ids = records["sample_id"].to_numpy(dtype=str)

    split_values = records["split"].to_numpy()
    accepted_values = records["accepted"].to_numpy(dtype=bool)

    for split in SPLITS:
        indices = np.where(
            (split_values == split) & accepted_values
        )[0]

        np.savez_compressed(
            OUTPUT_DIR / f"{split}.npz",
            X=X[indices],
            y=y[indices],
            horizon_ms=horizons[indices],
            video_names=video_names[indices],
            sample_ids=sample_ids[indices],
        )

        print(
            f"{split.upper()}: X={X[indices].shape}, "
            f"y={y[indices].shape}"
        )

        if len(indices):
            classes, counts = np.unique(
                y[indices], return_counts=True
            )
            print(
                "  Class distribution:",
                dict(zip(classes.tolist(), counts.tolist())),
            )

    records.to_csv(
        OUTPUT_DIR / "fixed_rate_sequence_manifest.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "preprocessing_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=4)

    schema = {
        "base_features": feature_columns,
        "missingness_features": missingness_features,
        "sequence_length": SEQUENCE_LENGTH,
        "target_interval_sec": TARGET_INTERVAL,
        "sampling_frequency_hz": 1.0 / TARGET_INTERVAL,
        "temporal_span_sec": (
            (SEQUENCE_LENGTH - 1) * TARGET_INTERVAL
        ),
        "horizons_ms": HORIZONS,
        "count_features": sorted(COUNT_FEATURES),
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
    }

    with open(
        OUTPUT_DIR / "sequence_schema.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(schema, file, indent=4)


def main():
    set_reproducibility()

    print("=" * 70)
    print("PHASE 5.5 - FIXED-RATE SEQUENCE GENERATION")
    print("=" * 70)

    print("\n1. Loading and matching manifests...")

    samples = load_and_match_manifests()

    print("Manifest rows:", len(samples))
    print(
        "Accepted by audit:",
        int(samples["accepted_by_audit"].sum()),
    )
    print(
        "Rejected by audit:",
        int((~samples["accepted_by_audit"]).sum()),
    )

    print("\n2. Loading feature schema...")

    feature_columns = get_feature_columns()
    print("Base features:", len(feature_columns))

    print("\n3. Identifying training missingness features...")

    missingness_features = identify_missingness_features(
        samples,
        feature_columns,
    )
    print("Missingness indicators:", len(missingness_features))

    print("\n4. Building feature-aware fixed-rate sequences...")

    raw_sequences, records = build_sequences(
        samples,
        feature_columns,
    )

    print("Raw sequence array:", raw_sequences.shape)

    print("\n5. Applying generator-level acceptance checks...")

    accepted_count = int(records["accepted"].sum())
    rejected_count = int((~records["accepted"]).sum())

    print("Accepted after generator checks:", accepted_count)
    print("Rejected after generator checks:", rejected_count)

    if accepted_count == 0:
        raise ValueError("No sequences passed generator-level checks")

    for split in SPLITS:
        count = int(
            (
                (records["split"] == split)
                & records["accepted"]
            ).sum()
        )
        if count == 0:
            raise ValueError(
                f"No sequences passed generator checks for {split}"
            )

    print("\n6. Fitting preprocessing on training data only...")

    config = fit_preprocessing(
        raw_sequences,
        records,
        feature_columns,
        missingness_features,
    )

    print(
        "Unique training observations:",
        config["unique_training_observations"],
    )
    print(
        "Final input feature count:",
        config["final_feature_count"],
    )

    if config["final_feature_count"] != 96:
        raise ValueError(
            f"Expected 96 final features, got "
            f"{config['final_feature_count']}"
        )

    print("\n7. Transforming all sequences...")

    X = transform_sequences(
        raw_sequences,
        feature_columns,
        missingness_features,
        config,
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values found after preprocessing"
        )

    if X.shape[1:] != (SEQUENCE_LENGTH, 96):
        raise ValueError(
            f"Unexpected transformed shape: {X.shape}"
        )

    print("\n8. Saving datasets and metadata...")

    save_outputs(
        X,
        records,
        config,
        feature_columns,
        missingness_features,
    )

    accepted_records = records[records["accepted"]].copy()

    summary = (
        accepted_records
        .groupby(["split", "horizon_ms", "label"])
        .agg(
            sequences=("sample_id", "count"),
            unique_videos=("video_name", "nunique"),
            mean_coverage=("coverage_percent", "mean"),
            min_coverage=("coverage_percent", "min"),
            max_unavailable_run=(
                "longest_unavailable_run",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "generation_summary.csv",
        index=False,
    )

    rejection_summary = (
        records[~records["accepted"]]
        .groupby(["split", "horizon_ms", "label"])
        .size()
        .reset_index(name="rejected_sequences")
    )

    rejection_summary.to_csv(
        OUTPUT_DIR / "rejection_summary.csv",
        index=False,
    )

    print("\nGENERATION SUMMARY")
    if len(summary):
        print(summary.to_string(index=False))

    print("\nREJECTION SUMMARY")
    if len(rejection_summary):
        print(rejection_summary.to_string(index=False))
    else:
        print("No generator-level rejections.")

    print("\nOutput directory:", OUTPUT_DIR)
    print("\nPHASE 5.5 COMPLETED")


if __name__ == "__main__":
    main()
