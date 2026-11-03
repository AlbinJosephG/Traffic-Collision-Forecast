
import requests
import json
from huggingface_hub import get_token

DATASET = "nexar-ai/nexar_collision_prediction"
BASE_URL = "https://datasets-server.huggingface.co"

token = get_token()

if not token:
    raise RuntimeError("Hugging Face token not found.")

headers = {
    "Authorization": f"Bearer {token}"
}

params = {
    "dataset": DATASET,
    "config": "default",
    "split": "train"
}

print("Inspecting Nexar training metadata...")

response = requests.get(
    f"{BASE_URL}/first-rows",
    params=params,
    headers=headers,
    timeout=120
)

print("HTTP status:", response.status_code)
response.raise_for_status()

data = response.json()

# Display available feature definitions.
print("\nDATASET FEATURES")
print("=" * 60)

for feature in data.get("features", []):
    print(feature)

# Display sample rows without dumping large video data.
print("\nSAMPLE ANNOTATION ROWS")
print("=" * 60)

for item in data.get("rows", [])[:3]:

    row = item.get("row", {})

    print("\nRow index:", item.get("row_idx"))

    for key, value in row.items():

        # Avoid printing potentially large binary/video content.
        if isinstance(value, dict):
            value = json.dumps(value, default=str)[:300]

        elif isinstance(value, list):
            value = str(value)[:300]

        else:
            value = str(value)[:300]

        print(f"{key}: {value}")
