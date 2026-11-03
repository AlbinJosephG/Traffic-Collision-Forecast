from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "processed" / "fixed_rate_sequences"
SPLITS = ["train", "validation", "test"]
EXPECTED_HORIZONS = {500, 1000, 1500}
SEQ_LEN = 50
FEATURES = 96
BASE_FEATURES = 66
MISSING_FEATURES = 30


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(path.exists(), f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def audit_config(config, schema):
    print("\n1. PREPROCESSING CONFIGURATION")

    require(config.get("base_feature_count") == BASE_FEATURES,
            "Expected 66 base features.")
    require(config.get("missingness_indicator_count") == MISSING_FEATURES,
            "Expected 30 missingness indicators.")
    require(config.get("final_feature_count") == FEATURES,
            "Expected 96 final features.")

    medians = np.asarray(config.get("medians", []), dtype=float)
    means = np.asarray(config.get("means", []), dtype=float)
    stds = np.asarray(config.get("stds", []), dtype=float)

    for name, values in [
        ("medians", medians), ("means", means), ("stds", stds)
    ]:
        require(values.shape == (BASE_FEATURES,),
                f"{name} must contain 66 values; got {values.shape}.")
        require(np.isfinite(values).all(), f"{name} contains non-finite values.")

    require((stds > 0).all(), "All standard deviations must be positive.")
    require(len(schema.get("base_features", [])) == BASE_FEATURES,
            "Schema must list 66 base features.")
    require(len(schema.get("missingness_features", [])) == MISSING_FEATURES,
            "Schema must list 30 missingness features.")
    require(schema.get("sequence_length") == SEQ_LEN,
            "Schema sequence length must be 50.")
    require(np.isclose(schema.get("target_interval_sec", np.nan), 0.2),
            "Schema target interval must be 0.2 seconds.")
    require(set(schema.get("horizons_ms", [])) == EXPECTED_HORIZONS,
            "Schema horizons must be 500, 1000 and 1500 ms.")

    unique_count = config.get("unique_training_observations")
    require(isinstance(unique_count, int) and unique_count > 0,
            "Invalid unique_training_observations value.")

    print("Base features: 66")
    print("Missingness indicators: 30")
    print("Final features: 96")
    print("Sequence length: 50")
    print("Sampling frequency: 5 FPS")
    print("Unique training observations:", unique_count)
    print("Configuration checks: PASSED")


def audit_manifest(manifest):
    print("\n2. ACCEPTED MANIFEST AUDIT")

    required_columns = {
        "sample_id", "split", "video_name", "label", "horizon_ms",
        "accepted", "coverage_percent", "longest_unavailable_run",
        "actual_endpoint_lag_sec",
    }
    missing = required_columns - set(manifest.columns)
    require(not missing, f"Manifest missing columns: {sorted(missing)}")

    accepted = manifest[manifest["accepted"].astype(bool)].copy()
    require(not accepted.empty, "No accepted samples in manifest.")
    require(set(accepted["split"].unique()).issubset(set(SPLITS)),
            "Unexpected split in manifest.")
    require(set(accepted["horizon_ms"].astype(int).unique()) == EXPECTED_HORIZONS,
            "Manifest does not contain all three horizons.")
    require(set(accepted["label"].astype(int).unique()) == {0, 1},
            "Manifest must contain both classes.")

    key = ["split", "video_name", "horizon_ms"]
    require(not accepted.duplicated(key).any(),
            "Duplicate split/video/horizon rows found.")
    require(accepted["sample_id"].astype(str).is_unique,
            "Duplicate sample IDs found.")
    require((pd.to_numeric(accepted["coverage_percent"]) >= 90).all(),
            "Accepted row has coverage below 90%.")
    require((pd.to_numeric(accepted["longest_unavailable_run"]) <= 5).all(),
            "Accepted row has unavailable run longer than 5.")
    require((pd.to_numeric(accepted["actual_endpoint_lag_sec"]) <= 0.25).all(),
            "Accepted row has endpoint lag above 0.25 sec.")

    print("Accepted samples:", len(accepted))
    print("Duplicate video-horizon rows: 0")
    print("Duplicate sample IDs: 0")
    print("Acceptance thresholds: PASSED")
    return accepted


def load_npz(split):
    path = DATA_DIR / f"{split}.npz"
    require(path.exists(), f"Missing NPZ file: {path}")

    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "y", "horizon_ms", "video_names", "sample_ids"}
        missing = required - set(archive.files)
        require(not missing, f"{split}.npz missing fields: {sorted(missing)}")
        return {key: archive[key].copy() for key in required}


def audit_split(split, accepted_manifest):
    print(f"\n3. {split.upper()} NPZ AUDIT")
    data = load_npz(split)

    X = data["X"]
    y = data["y"]
    horizons = data["horizon_ms"].astype(int)
    videos = data["video_names"].astype(str)
    sample_ids = data["sample_ids"].astype(str)

    require(X.ndim == 3 and X.shape[1:] == (SEQ_LEN, FEATURES),
            f"{split}: unexpected X shape {X.shape}.")
    require(X.dtype == np.float32, f"{split}: X must be float32, got {X.dtype}.")
    require(np.isfinite(X).all(), f"{split}: X contains NaN or infinity.")

    n = len(X)
    require(all(len(data[k]) == n for k in
                ["y", "horizon_ms", "video_names", "sample_ids"]),
            f"{split}: NPZ arrays have inconsistent lengths.")
    require(set(np.unique(y).tolist()).issubset({0, 1}),
            f"{split}: labels are not binary.")
    require(set(np.unique(horizons).tolist()) == EXPECTED_HORIZONS,
            f"{split}: missing one or more horizons.")
    require(len(np.unique(sample_ids)) == n,
            f"{split}: duplicate sample IDs in NPZ.")

    expected = accepted_manifest[accepted_manifest["split"] == split].copy()
    expected["sample_id"] = expected["sample_id"].astype(str)
    expected = expected.set_index("sample_id", drop=False)

    require(set(sample_ids) == set(expected.index),
            f"{split}: NPZ sample IDs do not match accepted manifest.")

    expected_rows = expected.loc[sample_ids]
    require(np.array_equal(y.astype(int), expected_rows["label"].to_numpy(dtype=int)),
            f"{split}: labels differ from manifest.")
    require(np.array_equal(horizons, expected_rows["horizon_ms"].to_numpy(dtype=int)),
            f"{split}: horizons differ from manifest.")
    require(np.array_equal(videos, expected_rows["video_name"].astype(str).to_numpy()),
            f"{split}: video names differ from manifest.")

    classes, counts = np.unique(y, return_counts=True)
    print("X shape:", X.shape)
    print("X dtype:", X.dtype)
    print("Finite values: PASSED")
    print("Class distribution:", dict(zip(classes.astype(int), counts.astype(int))))

    print("Per-horizon class counts:")
    for horizon in sorted(EXPECTED_HORIZONS):
        mask = horizons == horizon
        hc, hn = np.unique(y[mask], return_counts=True)
        distribution = dict(zip(hc.astype(int), hn.astype(int)))
        print(f"  {horizon} ms: total={int(mask.sum())}, classes={distribution}")

    print("NPZ/manifest correspondence: PASSED")
    return {
        "X": X, "y": y, "horizons": horizons,
        "videos": videos, "sample_ids": sample_ids,
    }


def audit_split_isolation(results):
    print("\n4. VIDEO-LEVEL SPLIT ISOLATION")
    video_sets = {
        split: set(result["videos"].tolist())
        for split, result in results.items()
    }

    for split, values in video_sets.items():
        print(f"{split}: {len(values)} unique videos")

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = video_sets[first] & video_sets[second]
        require(not overlap,
                f"Video leakage between {first} and {second}: {sorted(overlap)}")
        print(f"{first} vs {second} overlap: 0")

    print("Video-level split isolation: PASSED")


def main():
    print("=" * 70)
    print("PHASE 5.6 - FINAL FIXED-RATE DATASET AUDIT")
    print("=" * 70)

    require(DATA_DIR.exists(), f"Dataset directory not found: {DATA_DIR}")
    config = read_json(DATA_DIR / "preprocessing_config.json")
    schema = read_json(DATA_DIR / "sequence_schema.json")
    audit_config(config, schema)

    manifest_path = DATA_DIR / "fixed_rate_sequence_manifest.csv"
    require(manifest_path.exists(), f"Missing manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    accepted = audit_manifest(manifest)

    results = {}
    for split in SPLITS:
        results[split] = audit_split(split, accepted)

    audit_split_isolation(results)

    print("\n5. FINAL TOTALS")
    total = 0
    for split in SPLITS:
        count = len(results[split]["y"])
        total += count
        print(f"{split.upper():12s}: {count:4d} sequences")
    print("-" * 40)
    print(f"{'TOTAL':12s}: {total:4d} sequences")

    print("\nALL PHASE 5.6 AUDIT CHECKS PASSED")
    print("Artifacts are structurally ready for fixed-rate LSTM training.")
    print("This audit does not establish model generalization or independently")
    print("prove preprocessing provenance beyond the saved configuration.")


if __name__ == "__main__":
    main()
