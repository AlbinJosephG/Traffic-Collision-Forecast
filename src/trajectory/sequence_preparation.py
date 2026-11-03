
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT_DIR / "processed" / "scene_features"
OUTPUT_DIR = ROOT_DIR / "processed" / "sequences"

SPLITS = ["train", "validation", "test"]

SEQUENCE_LENGTH = 25
SAMPLING_RATE = 5

METADATA_COLUMNS = {
    "video_name",
    "split",
    "label",
    "frame_index",
    "timestamp_sec",
    "Unnamed: 0",
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def parse_label(value):
    """Convert positive/negative or 0/1 into numeric labels."""

    value = str(value).strip().lower()

    if value in ("positive", "1", "1.0"):
        return 1

    if value in ("negative", "0", "0.0"):
        return 0

    raise ValueError(f"Unknown label: {value}")


def get_folder_label(path):
    """Read label from the positive/negative directory."""

    folder = path.parent.name.lower()

    if folder == "positive":
        return 1

    if folder == "negative":
        return 0

    raise ValueError(f"Unknown label folder: {folder}")


def load_video_files():
    """Load all available scene-feature CSV files."""

    records = []

    for split in SPLITS:

        split_dir = INPUT_DIR / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing split directory: {split_dir}"
            )

        files = sorted(split_dir.glob("*/*.csv"))

        for file_path in files:

            try:
                df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()

            folder_label = get_folder_label(file_path)

            if not df.empty:

                if "label" not in df.columns:
                    raise ValueError(
                        f"Missing label column: {file_path}"
                    )

                labels = df["label"].dropna().unique()

                if len(labels) != 1:
                    raise ValueError(
                        f"Inconsistent labels in {file_path}"
                    )

                label = parse_label(labels[0])

                if label != folder_label:
                    raise ValueError(
                        f"Folder/CSV label mismatch: {file_path}"
                    )

            else:
                label = folder_label

            records.append({
                "path": file_path,
                "split": split,
                "video_id": file_path.stem,
                "label": label,
                "data": df,
            })

    return records


# ============================================================
# 3. IDENTIFY NUMERICAL FEATURES
# ============================================================

def identify_features(records):

    training_frames = [
        record["data"]
        for record in records
        if record["split"] == "train"
        and not record["data"].empty
    ]

    if not training_frames:
        raise ValueError("No training scene-feature data found.")

    # Check that all populated CSVs have the same schema.
    reference_columns = set(training_frames[0].columns)

    for record in records:

        df = record["data"]

        if df.empty:
            continue

        if set(df.columns) != reference_columns:
            raise ValueError(
                f"Inconsistent CSV schema: {record['path']}"
            )

    train_df = pd.concat(
        training_frames,
        ignore_index=True
    )

    candidate_columns = [
        col for col in train_df.columns
        if col not in METADATA_COLUMNS
    ]

    numeric_df = pd.DataFrame(index=train_df.index)

    for col in candidate_columns:

        numeric_df[col] = pd.to_numeric(
            train_df[col],
            errors="coerce"
        )

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove features that are entirely missing in training.
    feature_columns = [
        col for col in numeric_df.columns
        if numeric_df[col].notna().any()
    ]

    dropped_columns = [
        col for col in numeric_df.columns
        if col not in feature_columns
    ]

    if not feature_columns:
        raise ValueError("No usable numerical features found.")

    numeric_df = numeric_df[feature_columns]

    print("\nFEATURE SELECTION")
    print("------------------------------")
    print("Selected features:", len(feature_columns))
    print("Dropped all-missing features:", dropped_columns)

    return feature_columns, numeric_df, dropped_columns


# ============================================================
# 4. FIT PREPROCESSING ON TRAINING DATA ONLY
# ============================================================

def fit_preprocessing(train_df, feature_columns):

    # Calculate training-only medians.
    medians = train_df.median().fillna(0.0)

    # Identify features with missing values in training.
    missing_columns = [
        col for col in feature_columns
        if train_df[col].isna().any()
    ]

    # Impute before calculating mean and standard deviation.
    imputed = train_df.fillna(medians)

    means = imputed.mean()

    stds = imputed.std(ddof=0).replace(0, 1.0)

    return {
        "medians": medians,
        "means": means,
        "stds": stds,
        "missing_columns": missing_columns,
    }


def transform_features(df, feature_columns, preprocessing):

    raw = pd.DataFrame(index=df.index)

    for col in feature_columns:

        raw[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    raw = raw.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Record missingness before imputation.
    missing_indicators = raw[
        preprocessing["missing_columns"]
    ].isna().astype(np.float32)

    # Impute using training medians.
    imputed = raw.fillna(
        preprocessing["medians"]
    )

    # Standardize using training mean and standard deviation.
    normalized = (
        imputed - preprocessing["means"]
    ) / preprocessing["stds"]

    normalized = normalized.replace(
        [np.inf, -np.inf],
        0
    ).fillna(0)

    normalized = normalized.astype(np.float32)

    # Append missingness indicators.
    normalized = pd.concat(
        [normalized, missing_indicators],
        axis=1
    )

    return normalized.to_numpy(dtype=np.float32)


# ============================================================
# 5. GENERATE FIXED-LENGTH SEQUENCES
# ============================================================

def prepare_split(
    records,
    split,
    feature_columns,
    preprocessing
):

    sequences = []
    labels = []
    masks = []
    manifest = []
    video_summary = []

    feature_names = (
        feature_columns
        + [
            f"{col}__missing"
            for col in preprocessing["missing_columns"]
        ]
    )

    for record in records:

        if record["split"] != split:
            continue

        df = record["data"]
        video_id = record["video_id"]
        label = record["label"]

        if df.empty:

            video_summary.append({
                "video_id": video_id,
                "split": split,
                "label": label,
                "frame_count": 0,
                "sequence_count": 0,
                "status": "no_tracked_objects",
            })

            continue

        # Preserve chronological order.
        sort_columns = []

        if "timestamp_sec" in df.columns:
            sort_columns.append("timestamp_sec")

        if "frame_index" in df.columns:
            sort_columns.append("frame_index")

        if sort_columns:
            df = df.sort_values(
                sort_columns
            ).reset_index(drop=True)

        # Transform using training-only preprocessing.
        feature_matrix = transform_features(
            df,
            feature_columns,
            preprocessing
        )

        timestamps = (
            pd.to_numeric(
                df["timestamp_sec"],
                errors="coerce"
            ).to_numpy()
            if "timestamp_sec" in df.columns
            else np.full(len(df), np.nan)
        )

        frame_count = len(feature_matrix)
        sequence_count = 0

        # Non-overlapping windows.
        for start in range(
            0,
            frame_count,
            SEQUENCE_LENGTH
        ):

            end = min(
                start + SEQUENCE_LENGTH,
                frame_count
            )

            chunk = feature_matrix[start:end]

            valid_length = len(chunk)

            # Fixed-size sequence.
            sequence = np.zeros(
                (SEQUENCE_LENGTH, len(feature_names)),
                dtype=np.float32
            )

            sequence[:valid_length] = chunk

            # Mask distinguishes valid frames from padding.
            mask = np.zeros(
                SEQUENCE_LENGTH,
                dtype=np.float32
            )

            mask[:valid_length] = 1.0

            sequences.append(sequence)
            labels.append(label)
            masks.append(mask)

            valid_timestamps = timestamps[start:end]
            valid_timestamps = valid_timestamps[
                np.isfinite(valid_timestamps)
            ]

            if len(valid_timestamps) > 0:
                start_time = float(valid_timestamps[0])
                end_time = float(valid_timestamps[-1])
            else:
                start_time = None
                end_time = None

            manifest.append({
                "video_id": video_id,
                "split": split,
                "label": label,
                "window_index": sequence_count,
                "start_frame_position": start,
                "end_frame_position": end - 1,
                "start_timestamp_sec": start_time,
                "end_timestamp_sec": end_time,
                "valid_length": valid_length,
                "padded": valid_length < SEQUENCE_LENGTH,
            })

            sequence_count += 1

        video_summary.append({
            "video_id": video_id,
            "split": split,
            "label": label,
            "frame_count": frame_count,
            "sequence_count": sequence_count,
            "status": "successful",
        })

    feature_count = len(feature_names)

    if sequences:
        X = np.stack(sequences).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        mask_array = np.stack(masks).astype(np.float32)
    else:
        X = np.empty(
            (0, SEQUENCE_LENGTH, feature_count),
            dtype=np.float32
        )
        y = np.empty((0,), dtype=np.int64)
        mask_array = np.empty(
            (0, SEQUENCE_LENGTH),
            dtype=np.float32
        )

    return X, y, mask_array, manifest, video_summary, feature_names


# ============================================================
# 6. SAVE PREPROCESSING CONFIGURATION
# ============================================================

def save_preprocessing(
    feature_columns,
    preprocessing,
    dropped_columns,
    feature_names
):

    config = {
        "sequence_length": SEQUENCE_LENGTH,
        "sampling_rate_fps_approx": SAMPLING_RATE,
        "window_duration_seconds_approx": (
            SEQUENCE_LENGTH / SAMPLING_RATE
        ),
        "window_overlap": 0,
        "feature_columns": feature_columns,
        "missing_indicator_columns": (
            preprocessing["missing_columns"]
        ),
        "output_feature_names": feature_names,
        "dropped_all_missing_columns": dropped_columns,
        "imputation_medians": (
            preprocessing["medians"].to_dict()
        ),
        "training_means": (
            preprocessing["means"].to_dict()
        ),
        "training_standard_deviations": (
            preprocessing["stds"].to_dict()
        ),
        "preprocessing_fit_split": "train",
    }

    output_path = OUTPUT_DIR / "preprocessing_config.json"

    with open(output_path, "w") as file:
        json.dump(config, file, indent=4)

    print("\nSaved preprocessing configuration:", output_path)


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("PHASE 3D - TEMPORAL SEQUENCE PREPARATION")
    print("=" * 60)

    records = load_video_files()

    print("\nTotal video files:", len(records))

    feature_columns, train_df, dropped_columns = (
        identify_features(records)
    )

    preprocessing = fit_preprocessing(
        train_df,
        feature_columns
    )

    all_video_summaries = []
    saved_feature_names = None

    for split in SPLITS:

        print(f"\nPreparing {split.upper()} sequences...")

        (
            X,
            y,
            masks,
            manifest,
            video_summary,
            feature_names,
        ) = prepare_split(
            records,
            split,
            feature_columns,
            preprocessing
        )

        saved_feature_names = feature_names

        # Save arrays.
        output_file = OUTPUT_DIR / f"{split}.npz"

        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            mask=masks,
        )

        # Save sequence-level metadata.
        pd.DataFrame(manifest).to_csv(
            OUTPUT_DIR / f"{split}_manifest.csv",
            index=False
        )

        all_video_summaries.extend(video_summary)

        print("Sequences:", len(X))
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Mask shape:", masks.shape)
        print("Saved:", output_file)

    # Save video-level processing summary.
    pd.DataFrame(all_video_summaries).to_csv(
        OUTPUT_DIR / "video_sequence_summary.csv",
        index=False
    )

    save_preprocessing(
        feature_columns,
        preprocessing,
        dropped_columns,
        saved_feature_names
    )

    print("\n" + "=" * 60)
    print("PHASE 3D COMPLETED")
    print("=" * 60)

    print("Videos loaded:", len(records))
    print("Features before missingness indicators:",
          len(feature_columns))
    print("Final model input features:",
          len(saved_feature_names))
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
