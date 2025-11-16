# Download Human Action Recognition dataset with balanced sampling
import json
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
import random

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
            login(token=token, add_to_git_credential=False)
            print("Logged in to HuggingFace")

    print(f"Loading Human Action Recognition dataset...")

    try:
        # Load test split
        ds = load_dataset("Bingsu/Human_Action_Recognition", split="test")
        print(f"Loaded {len(ds)} samples from test split")
        
        # Group samples by label to ensure diversity
        samples_by_label = {i: [] for i in range(15)}
        for idx, sample in enumerate(ds):
            label_id = sample["labels"]
            if 0 <= label_id < 15:
                samples_by_label[label_id].append(idx)
        
        print(f"\nSamples per class in dataset:")
        for label_id in range(15):
            print(f"  {ACTIVITY_NAMES[label_id]}: {len(samples_by_label[label_id])}")
        
        # Sample N images with balanced class distribution
        samples_per_class = N // 15
        remainder = N % 15
        selected_indices = []
        
        for label_id in range(15):
            # Each class gets samples_per_class samples
            n_for_this_class = samples_per_class + (1 if label_id < remainder else 0)
            available = samples_by_label[label_id]
            if len(available) >= n_for_this_class:
                selected_indices.extend(random.sample(available, n_for_this_class))
            else:
                # If not enough samples, take all available
                selected_indices.extend(available)
                print(f"⚠️  {ACTIVITY_NAMES[label_id]}: only {len(available)} available")
        
        # Shuffle to mix classes
        random.shuffle(selected_indices)
        selected_indices = selected_indices[:N]
        
        print(f"\nSelected {len(selected_indices)} balanced samples")

    except Exception as e:
        print(f"\n❌ Error loading Human Action Recognition dataset: {e}")
        return

    records = []

    # Process each selected sample
    for new_idx, orig_idx in tqdm(enumerate(selected_indices), total=len(selected_indices), desc="Downloading activity images"):
        try:
            sample = ds[orig_idx]
            
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
            img_filename = f"{new_idx:05d}.jpg"
            img_path = img_dir / img_filename
            image.save(img_path)

            # Create record
            records.append({
                "id": f"test_{new_idx:05d}",
                "image_path": str(Path("images") / img_filename),
                "label": activity_name,
            })

        except Exception as e:
            print(f"⚠️ Error processing sample {orig_idx}: {e}")
            continue

    # Save labels.json
    with open(ROOT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\n✅ Done! Exported {len(records)} images to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Labels: {ROOT / 'labels.json'}")

    # Show sample labels
    print(f"\nSample labels (first 10):")
    for rec in records[:10]:
        print(f"  {rec['id']}: {rec['label']}")

    # Show class distribution
    from collections import Counter
    label_counts = Counter([r['label'] for r in records])
    print(f"\nClass distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()
