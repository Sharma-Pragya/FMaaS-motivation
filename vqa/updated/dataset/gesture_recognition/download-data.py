# Download HaGRID (Hand Gesture Recognition) dataset
import json
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

ROOT = Path(__file__).parent
N = 100  # number of images to export (matching other tasks)

# HaGRID 19 gesture class names
GESTURE_NAMES = [
    "call",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "ok",
    "one",
    "palm",
    "peace",
    "peace_inverted",
    "rock",
    "stop",
    "stop_inverted",
    "three",
    "three2",
    "two_up",
    "two_up_inverted",
    "no_gesture",
]

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Login to HuggingFace if token exists
    hf_token_path = "../../../../hf-token.txt"
    if os.path.exists(hf_token_path):
        with open(hf_token_path, "r") as f:
            token = f.read().strip()
        if token:
            login(token=token)
            print("Logged in to HuggingFace")

    print(f"Loading HaGRID gesture dataset (first {N} samples)...")
    print("Using downscaled 384p version for faster download")

    try:
        # Load HaGRID sample dataset (384p version, smaller download)
        # Using streaming to avoid downloading entire dataset
        ds = load_dataset("cj-mills/hagrid-sample-500k-384p", split="train", streaming=True, trust_remote_code=True)

        # Take first N samples
        print(f"Streaming first {N} samples...")

    except Exception as e:
        print(f"\n❌ Error loading HaGRID: {e}")
        print("\nTrying alternative method...")
        try:
            # Fallback: non-streaming mode
            ds_full = load_dataset("cj-mills/hagrid-sample-500k-384p", split="train", trust_remote_code=True)
            ds = ds_full.select(range(min(N, len(ds_full))))
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return

    records = []
    count = 0

    # Process samples
    print(f"\nDownloading {N} gesture images...")
    for i, sample in enumerate(tqdm(ds, total=N, desc="Downloading images")):
        if count >= N:
            break

        try:
            # Get image and label
            image = sample["image"]
            label_id = sample["label"]

            # Convert label ID to gesture name
            if 0 <= label_id < len(GESTURE_NAMES):
                gesture_name = GESTURE_NAMES[label_id]
            else:
                print(f"⚠️ Warning: Unknown label {label_id}, skipping")
                continue

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Save image
            img_filename = f"{count:05d}.jpg"
            img_path = img_dir / img_filename
            image.save(img_path)

            # Create record
            records.append({
                "id": f"train_{count:05d}",
                "image_path": str(Path("images") / img_filename),
                "label": gesture_name,
            })

            count += 1

        except Exception as e:
            print(f"⚠️ Error processing sample {i}: {e}")
            continue

    # Save labels.json
    with open(ROOT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\n✅ Done! Exported {len(records)} images to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Labels: {ROOT / 'labels.json'}")

    # Show sample labels
    print(f"\nSample labels (first 5):")
    for rec in records[:5]:
        print(f"  {rec['id']}: {rec['label']}")

    # Show class distribution
    from collections import Counter
    label_counts = Counter([r['label'] for r in records])
    print(f"\nClass distribution (top 10):")
    for label, count_val in label_counts.most_common(10):
        print(f"  {label}: {count_val}")

if __name__ == "__main__":
    main()
