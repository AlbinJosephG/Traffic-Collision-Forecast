
from pathlib import Path
from urllib.parse import urlparse
import requests
import pandas as pd

from huggingface_hub import get_token


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "nexar-ai/nexar_collision_prediction"
BASE_URL = "https://datasets-server.huggingface.co"

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT_DIR / "data" / "annotations"
OUTPUT_FILE = OUTPUT_DIR / "nexar_train_annotations.csv"

PAGE_SIZE = 100


# ============================================================
# AUTHENTICATION
# ============================================================

token = get_token()

if not token:
    raise RuntimeError(
        "Hugging Face token not found. Run: hf auth login"
    )

headers = {
    "Authorization": f"Bearer {token}"
}


# ============================================================
# DOWNLOAD ALL TRAINING ANNOTATIONS
# ============================================================

def download_annotations():

    all_rows = []
    offset = 0

    print("=" * 60)
    print("NEXAR ANNOTATION RETRIEVAL")
    print("=" * 60)

    while True:

        params = {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": PAGE_SIZE,
        }

        response = requests.get(
            f"{BASE_URL}/rows",
            params=params,
            headers=headers,
            timeout=120,
        )

        response.raise_for_status()

        result = response.json()
        rows = result.get("rows", [])

        if not rows:
            break

        for item in rows:

            row = item["row"]

            video_info = row.get("video")

            if isinstance(video_info, dict):
                video_url = video_info.get("src")
            else:
                video_url = None

            if not video_url:
                continue

            video_path = urlparse(video_url).path
            parts = video_path.strip("/").split("/")

            video_name = parts[-1]

            # Expected path structure:
            # train/positive/video.mp4
            # train/negative/video.mp4

            category = parts[-2].lower()

            if category == "positive":
                label = 1
            elif category == "negative":
                label = 0
            else:
                raise ValueError(
                    f"Unexpected video category: {category}"
                )

            all_rows.append({
                "video_name": video_name,
                "label": label,
                "category": category,
                "video_url": video_url,
                "time_of_event": row.get("time_of_event"),
                "time_of_alert": row.get("time_of_alert"),
                "time_to_accident": row.get("time_to_accident"),
                "light_conditions": row.get("light_conditions"),
                "weather": row.get("weather"),
                "scene": row.get("scene"),
            })

        print(
            f"Retrieved {len(all_rows)} annotation rows..."
        )

        offset += len(rows)

        # Stop when the final page is reached.
        if len(rows) < PAGE_SIZE:
            break

    return pd.DataFrame(all_rows)


# ============================================================
# VALIDATE AND SAVE
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = download_annotations()

    if df.empty:
        raise RuntimeError("No annotations were retrieved.")

    if df["video_name"].duplicated().any():
        duplicates = df.loc[
            df["video_name"].duplicated(),
            "video_name"
        ].tolist()

        raise ValueError(
            f"Duplicate annotation video IDs: {duplicates[:10]}"
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    positive = df[df["label"] == 1]
    negative = df[df["label"] == 0]

    print("\nANNOTATION SUMMARY")
    print("=" * 60)

    print("Total annotations:", len(df))
    print("Positive videos:", len(positive))
    print("Negative videos:", len(negative))

    print(
        "Positive videos with event timestamps:",
        positive["time_of_event"].notna().sum()
    )

    print(
        "Positive videos with alert timestamps:",
        positive["time_of_alert"].notna().sum()
    )

    print(
        "Positive videos with time-to-accident:",
        positive["time_to_accident"].notna().sum()
    )

    print("\nPOSITIVE ANNOTATION EXAMPLES")
    print("=" * 60)

    columns = [
        "video_name",
        "time_of_event",
        "time_of_alert",
        "time_to_accident",
    ]

    print(
        positive[columns].head(10).to_string(index=False)
    )

    print("\nSAVED:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
