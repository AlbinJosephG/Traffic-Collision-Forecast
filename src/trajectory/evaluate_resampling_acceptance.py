
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "processed"
    / "fixed_rate_resampling_audit"
    / "fixed_rate_video_audit.csv"
)

OUTPUT_DIR = (
    ROOT
    / "processed"
    / "fixed_rate_resampling_audit"
)


MIN_COVERAGE = 90.0
MAX_UNAVAILABLE_RUN = 5
MAX_ENDPOINT_LAG = 0.25


def main():

    df = pd.read_csv(INPUT_PATH)

    # Apply all acceptance criteria.
    df["accepted"] = (
        (df["audit_status"] == "completed")
        & (df["coverage_percent"] >= MIN_COVERAGE)
        & (
            df["longest_unavailable_run"]
            <= MAX_UNAVAILABLE_RUN
        )
        & (
            df["endpoint_lag_sec"]
            <= MAX_ENDPOINT_LAG
        )
    )

    df["acceptance_status"] = df["accepted"].map({
        True: "accepted",
        False: "rejected",
    })

    # Save individual sequence decisions.
    output_path = OUTPUT_DIR / "resampling_acceptance.csv"

    df.to_csv(output_path, index=False)

    # Summary by split and class.
    summary = (
        df.groupby(["split", "label", "acceptance_status"])
        .agg(
            sequences=("video_name", "count"),
            mean_coverage=("coverage_percent", "mean"),
        )
        .reset_index()
    )

    print("=" * 70)
    print("FIXED-RATE RESAMPLING ACCEPTANCE AUDIT")
    print("=" * 70)

    print("\n1. ACCEPTANCE SUMMARY")

    print(summary.round(2).to_string(index=False))

    print("\n2. ORIGINAL STATUS COMPARISON")

    comparison = (
        df.groupby([
            "split",
            "label",
            "original_status",
            "acceptance_status",
        ])
        .size()
        .reset_index(name="sequences")
    )

    print(comparison.to_string(index=False))

    print("\n3. UNIQUE VIDEO COUNTS")

    # A video qualifies only if all three horizons qualify.
    per_video = (
        df.groupby(["split", "label", "video_name"])
        .agg(
            valid_horizons=("accepted", "sum"),
            total_horizons=("accepted", "count"),
        )
        .reset_index()
    )

    per_video["all_horizons_accepted"] = (
        per_video["valid_horizons"]
        == per_video["total_horizons"]
    ) & (per_video["total_horizons"] == 3)

    video_summary = (
        per_video.groupby(["split", "label"])
        .agg(
            videos=("video_name", "count"),
            all_three_horizons=(
                "all_horizons_accepted",
                "sum",
            ),
        )
        .reset_index()
    )

    print(video_summary.to_string(index=False))

    print("\n4. REJECTED SEQUENCES")

    rejected = df[~df["accepted"]]

    print(
        rejected[
            [
                "split",
                "label",
                "video_name",
                "horizon_ms",
                "original_status",
                "coverage_percent",
                "longest_unavailable_run",
                "endpoint_lag_sec",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", output_path)
    print("\nAUDIT COMPLETED")


if __name__ == "__main__":
    main()
