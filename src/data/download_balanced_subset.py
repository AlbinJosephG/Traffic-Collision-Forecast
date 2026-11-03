
import random
import csv
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Dataset details
REPO_ID = "nexar-ai/nexar_collision_prediction"
REPO_TYPE = "dataset"

BASE_DIR = Path(r"E:\TrafficCollisionProject\dataset")
SAMPLE_SIZE = 100
RANDOM_SEED = 42

# Connect to Hugging Face
api = HfApi()

print("Inspecting Nexar dataset files...")

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    token=True
)

# Find videos in the two training classes
positive_videos = sorted([
    f for f in files
    if f.startswith("train/positive/")
    and f.lower().endswith(".mp4")
])

negative_videos = sorted([
    f for f in files
    if f.startswith("train/negative/")
    and f.lower().endswith(".mp4")
])

print("Positive videos available:", len(positive_videos))
print("Negative videos available:", len(negative_videos))

if len(positive_videos) < SAMPLE_SIZE:
    raise ValueError("Not enough positive videos.")

if len(negative_videos) < SAMPLE_SIZE:
    raise ValueError("Not enough negative videos.")

# Reproducible random selection
random.seed(RANDOM_SEED)

selected_positive = random.sample(
    positive_videos, SAMPLE_SIZE
)

selected_negative = random.sample(
    negative_videos, SAMPLE_SIZE
)

selected = (
    [(f, 1) for f in selected_positive]
    + [(f, 0) for f in selected_negative]
)

random.shuffle(selected)

# Save the selection manifest
BASE_DIR.mkdir(parents=True, exist_ok=True)

manifest_path = BASE_DIR / "subset_manifest.csv"

with open(manifest_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["video_path", "label"])
    writer.writerows(selected)

print("\nSelected videos:")
print("Positive:", len(selected_positive))
print("Negative:", len(selected_negative))
print("Total:", len(selected))
print("\nManifest saved:", manifest_path)

# Download selected videos only
print("\nStarting balanced subset download...")

for index, (filename, label) in enumerate(selected, start=1):
    print(f"\n[{index}/200] Downloading {filename}")

    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=str(BASE_DIR),
        token=True
    )

print("\nDownload completed!")
print("Dataset location:", BASE_DIR)
print("Manifest:", manifest_path)
