
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SCENE_DIR = ROOT / "processed" / "scene_features"

INPUT_MANIFEST = (
    ROOT
    / "processed"
    / "revised_event_aligned_sequences"
    / "sequence_generation_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "processed"
    / "fixed_rate_resampling_audit"
)

SEQUENCE_LENGTH = 50
TARGET_FPS = 5.0
TARGET_INTERVAL = 1.0 / TARGET_FPS

MAX_INTERPOLATION_GAP = 0.30

# An existing observation within 0.05 sec of a target
# timestamp is considered a direct observation.
DIRECT_TOLERANCE = 0.05

TIMESTAMP_COLUMN = "timestamp_sec"


# ============================================================
# LOAD SCENE FEATURES
# ============================================================

def load_timestamps(split, label, video_name):

    category = "positive" if int(label) == 1 else "negative"

    path = (
        SCENE_DIR
        / split
        / category
        / f"{Path(video_name).stem}.csv"
    )

    if not path.exists():
        return None, "missing_file"

    try:
        df = pd.read_csv(path)
    except Exception:
        return None, "read_error"

    if TIMESTAMP_COLUMN not in df.columns:
        return None, "missing_timestamp_column"

    timestamps = pd.to_numeric(
        df[TIMESTAMP_COLUMN],
        errors="coerce"
    ).dropna().to_numpy(dtype=float)

    timestamps = np.sort(np.unique(timestamps))

    if len(timestamps) < 2:
        return None, "insufficient_timestamps"

    if not np.all(np.isfinite(timestamps)):
        return None, "invalid_timestamps"

    return timestamps, "valid"


# ============================================================
# BUILD FIXED-RATE TARGET GRID
# ============================================================

def build_target_grid(endpoint, latest_observation):

    offsets = np.arange(SEQUENCE_LENGTH) * TARGET_INTERVAL

    return latest_observation - offsets[::-1]

# ============================================================
# AUDIT TEMPORAL COVERAGE
# ============================================================

def inspect_target_grid(timestamps, target_grid):

    direct_count = 0
    interpolated_count = 0
    unavailable_count = 0

    target_status = []

    for target in target_grid:

        insertion_index = np.searchsorted(
            timestamps,
            target,
            side="left"
        )

        # Find nearest existing observation.
        candidate_indices = []

        if insertion_index > 0:
            candidate_indices.append(insertion_index - 1)

        if insertion_index < len(timestamps):
            candidate_indices.append(insertion_index)

        nearest_distance = float("inf")

        if candidate_indices:

            nearest_distance = min(
                abs(timestamps[index] - target)
                for index in candidate_indices
            )

        # Case 1: Direct observation.
        if nearest_distance <= DIRECT_TOLERANCE:

            direct_count += 1
            target_status.append("direct")
            continue

        # Case 2: Interpolation between two observations.
        if insertion_index > 0 and insertion_index < len(timestamps):

            left_time = timestamps[insertion_index - 1]
            right_time = timestamps[insertion_index]

            bracket_gap = right_time - left_time

            if (
                left_time < target < right_time
                and bracket_gap <= MAX_INTERPOLATION_GAP
            ):

                interpolated_count += 1
                target_status.append("interpolated")
                continue

        # Case 3: No reliable observation or interpolation.
        unavailable_count += 1
        target_status.append("unavailable")

    # Longest consecutive unavailable interval in target steps.
    longest_unavailable_run = 0
    current_run = 0

    for status in target_status:

        if status == "unavailable":

            current_run += 1

            longest_unavailable_run = max(
                longest_unavailable_run,
                current_run
            )

        else:
            current_run = 0

    total = len(target_grid)

    return {
        "direct_steps": direct_count,
        "interpolated_steps": interpolated_count,
        "unavailable_steps": unavailable_count,
        "usable_steps": direct_count + interpolated_count,
        "coverage_percent": (
            (direct_count + interpolated_count) / total
        ) * 100,
        "direct_percent": direct_count / total * 100,
        "interpolated_percent": interpolated_count / total * 100,
        "longest_unavailable_run": longest_unavailable_run,
        "target_status": target_status,
    }


# ============================================================
# PROCESS MANIFEST
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(INPUT_MANIFEST)

    required_columns = {
        "split",
        "label",
        "video_name",
        "horizon_ms",
        "prediction_endpoint_sec",
        "status",
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    results = []

    print("=" * 70)
    print("FIXED-RATE RESAMPLING AUDIT")
    print("=" * 70)

    print("Manifest rows:", len(manifest))
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Target FPS:", TARGET_FPS)
    print("Target interval:", TARGET_INTERVAL)
    print(
        "Window duration:",
        (SEQUENCE_LENGTH - 1) * TARGET_INTERVAL,
        "seconds"
    )

    # Cache timestamps so each video is read only once.
    timestamp_cache = {}

    for _, row in manifest.iterrows():

        split = str(row["split"])
        label = int(row["label"])
        video_name = str(row["video_name"])
        horizon_ms = int(row["horizon_ms"])

        endpoint = pd.to_numeric(
            pd.Series([row["prediction_endpoint_sec"]]),
            errors="coerce"
        ).iloc[0]

        original_status = str(row["status"])

        base_result = {
            "split": split,
            "label": label,
            "video_name": video_name,
            "horizon_ms": horizon_ms,
            "original_status": original_status,
            "prediction_endpoint_sec": endpoint,
        }

        if not np.isfinite(endpoint):

            results.append({
                **base_result,
                "audit_status": "invalid_endpoint",
            })
            continue

        cache_key = (split, label, video_name)

        if cache_key not in timestamp_cache:

            timestamp_cache[cache_key] = load_timestamps(
                split,
                label,
                video_name
            )

        timestamps, timestamp_status = timestamp_cache[cache_key]

        if timestamps is None:
            results.append({
                **base_result,
                "audit_status": timestamp_status,
            })
            continue

        # Strictly exclude observations after the prediction endpoint.
        timestamps = timestamps[timestamps <= float(endpoint)]

        if len(timestamps) == 0:
            results.append({
                **base_result,
                "audit_status": "no_past_observations",
            })
            continue

        latest_observation = float(timestamps[-1])

        endpoint_lag = float(endpoint) - latest_observation

        if endpoint_lag > 0.25:
            results.append({
                **base_result,
                "audit_status": "endpoint_too_far",
                "endpoint_lag_sec": endpoint_lag,
            })
            continue

        target_grid = build_target_grid(
            float(endpoint),
            latest_observation
        )

        coverage = inspect_target_grid(
            timestamps,
            target_grid
        )

        target_grid = build_target_grid(
            float(endpoint),
            latest_observation
        )

        coverage = inspect_target_grid(
            timestamps,
            target_grid
        )

        results.append({
            **base_result,
            "audit_status": "completed",
            "first_target_sec": float(target_grid[0]),
            "last_target_sec": float(target_grid[-1]),
            "direct_steps": coverage["direct_steps"],
            "interpolated_steps": coverage["interpolated_steps"],
            "unavailable_steps": coverage["unavailable_steps"],
            "usable_steps": coverage["usable_steps"],
            "coverage_percent": coverage["coverage_percent"],
            "direct_percent": coverage["direct_percent"],
            "interpolated_percent": coverage["interpolated_percent"],
            "endpoint_lag_sec": endpoint_lag,
            "longest_unavailable_run": coverage[
                "longest_unavailable_run"
            ],
        })

    report = pd.DataFrame(results)

    # ========================================================
    # SAVE VIDEO-LEVEL RESULTS
    # ========================================================

    video_output = OUTPUT_DIR / "fixed_rate_video_audit.csv"

    report.to_csv(video_output, index=False)

    completed = report[
        report["audit_status"] == "completed"
    ].copy()

    if completed.empty:

        print("\nNo sequences could be audited.")
        print("Saved:", video_output)
        return

    # ========================================================
    # SUMMARY BY SPLIT AND CLASS
    # ========================================================

    summary = (
        completed
        .groupby(["split", "label"])
        .agg(
            sequences=("video_name", "count"),
            mean_coverage=("coverage_percent", "mean"),
            median_coverage=("coverage_percent", "median"),
            min_coverage=("coverage_percent", "min"),
            mean_direct_steps=("direct_steps", "mean"),
            mean_interpolated_steps=("interpolated_steps", "mean"),
            mean_unavailable_steps=("unavailable_steps", "mean"),
            max_unavailable_run=("longest_unavailable_run", "max"),
        )
        .reset_index()
    )

    summary_output = OUTPUT_DIR / "fixed_rate_summary.csv"

    summary.to_csv(summary_output, index=False)

    # ========================================================
    # COVERAGE BY ORIGINAL GENERATION STATUS
    # ========================================================

    status_summary = (
        completed
        .groupby(["split", "label", "original_status"])
        .agg(
            sequences=("video_name", "count"),
            mean_coverage=("coverage_percent", "mean"),
            median_coverage=("coverage_percent", "median"),
            mean_unavailable_steps=("unavailable_steps", "mean"),
        )
        .reset_index()
    )

    status_output = (
        OUTPUT_DIR / "fixed_rate_status_comparison.csv"
    )

    status_summary.to_csv(status_output, index=False)

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n1. SUMMARY BY SPLIT AND CLASS")

    print(
        summary.round(3).to_string(index=False)
    )

    print("\n2. COVERAGE BY ORIGINAL GENERATION STATUS")

    print(
        status_summary.round(3).to_string(index=False)
    )

    print("\n3. LOWEST COVERAGE SEQUENCES")

    lowest = completed.sort_values(
        "coverage_percent",
        ascending=True
    )

    print(
        lowest[
            [
                "split",
                "label",
                "video_name",
                "horizon_ms",
                "original_status",
                "direct_steps",
                "interpolated_steps",
                "unavailable_steps",
                "coverage_percent",
                "longest_unavailable_run",
            ]
        ].head(20).to_string(index=False)
    )

    print("\n4. AUDIT STATUS")

    print(
        report["audit_status"]
        .value_counts()
        .to_string()
    )

    print("\nOUTPUT FILES")
    print(video_output)
    print(summary_output)
    print(status_output)

    print("\nAUDIT COMPLETED")


if __name__ == "__main__":
    main()
