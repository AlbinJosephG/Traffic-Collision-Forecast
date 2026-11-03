
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATASET_DIR = Path(
    r"E:\TrafficCollisionProject\dataset"
)

manifest = pd.read_csv(
    DATASET_DIR / "subset_manifest.csv"
)

# First split: 70% training, 30% remaining
train, remaining = train_test_split(
    manifest,
    test_size=0.30,
    stratify=manifest["label"],
    random_state=42
)

# Second split: 15% validation, 15% testing
validation, test = train_test_split(
    remaining,
    test_size=0.50,
    stratify=remaining["label"],
    random_state=42
)

# Save split manifests
train.to_csv(
    DATASET_DIR / "train_manifest.csv",
    index=False
)

validation.to_csv(
    DATASET_DIR / "validation_manifest.csv",
    index=False
)

test.to_csv(
    DATASET_DIR / "test_manifest.csv",
    index=False
)

# Display distribution
for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    print(f"\n{name} SET")
    print("Total:", len(data))
    print("Positive:", (data["label"] == 1).sum())
    print("Negative:", (data["label"] == 0).sum())

print("\nVideo-level splitting completed!")
