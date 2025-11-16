# Download Human Action Recognition dataset
import json
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

ROOT = Path(__file__).parent
N = 100  # number of images to export (matching other tasks)

# Human Action Recognition 15 activity class names (0-14)
ACTIVITY_NAMES = [
    "calling",
    "clapping",
    "cycling",
    "dancing",
    "drinking",
    "eating",
    "fighting",
    "hugging",
    "laughing",
    "listening_to_music",
    "running",
    "sitting",
    "sleeping",
    "texting",
    "using_laptop",
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

    print(f"Loading Human Action Recognition dataset (first {N} samples)...")

    try:
        # Load test split
        ds = load_dataset("Bingsu/Human_Action_Recognition", split="test", trust_remote_code=True)

        # Select first N samples
        subset = ds.select(range(min(N, len(ds))))
        print(f"Selected {len(subset)} samples from test split")

    except Exception as e:
        print(f"\n❌ Error loading Human Action Recognition dataset: {e}")
        return

    records = []

    # Process each sample
    for i, sample in tqdm(enumerate(subset), total=len(subset), desc="Downloading activity images"):
        try:
            # Get image and label
            image = sample["image"]
            label_id = sample["labels"]

            # Convert label ID to activity name
            if 0 <= label_id < len(ACTIVITY_NAMES):
                activity_name = ACTIVITY_NAMES[label_id]
            else:
                print(f"⚠️ Warning: Unknown label {label_id}, skipping")
                continue

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Save image
            img_filename = f"{i:05d}.jpg"
            img_path = img_dir / img_filename
            image.save(img_path)

            # Create record
            records.append({
                "id": f"test_{i:05d}",
                "image_path": str(Path("images") / img_filename),
                "label": activity_name,
            })

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
    print(f"\nClass distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()
