# Download and process SUN397 dataset for scene classification
import json
import os
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from PIL import Image
import subprocess
import sys

ROOT = Path(__file__).parent
N = 100  # number of images to export (matching other tasks)

def install_datasets():
    """Install datasets package if not available."""
    try:
        import datasets
        return True
    except ImportError:
        print("📦 Installing datasets package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
            print("✅ datasets installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install datasets: {e}")
            return False

def main():
    print("=" * 60)
    print("SUN397 Scene Classification Dataset Download Script")
    print("=" * 60)

    # Install dependencies
    if not install_datasets():
        print("\n⚠️  Could not install datasets. Please install manually:")
        print("   pip install datasets")
        return

    from datasets import load_dataset

    # Create output directories
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading SUN397 dataset from HuggingFace...")
    print(f"   This may take a few minutes...")

    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("tanganke/sun397", split="train", trust_remote_code=True)

        print(f"✅ Dataset loaded: {len(dataset)} total images")

        # Get unique labels and their counts
        label_counts = Counter(dataset['label'])
        unique_labels = list(label_counts.keys())

        print(f"📊 Found {len(unique_labels)} unique scene categories")

        # Calculate images per category to get diverse sampling
        images_per_category = max(1, N // len(unique_labels))

        print(f"Sampling ~{images_per_category} image(s) from each category for diversity...")

        # Group images by label
        images_by_label = {}
        for idx, item in enumerate(dataset):
            label = item['label']
            if label not in images_by_label:
                images_by_label[label] = []
            images_by_label[label].append(idx)

        # Sample images from each category
        records = []
        category_counts = Counter()
        processed_count = 0

        # Sort labels for consistent ordering
        sorted_labels = sorted(images_by_label.keys())

        for label in tqdm(sorted_labels, desc="Processing categories"):
            if processed_count >= N:
                break

            # Get indices for this label
            indices = images_by_label[label]

            # Sample evenly from this category
            if len(indices) <= images_per_category:
                sample_indices = indices
            else:
                # Evenly space samples across available images
                step = len(indices) // images_per_category
                sample_indices = [indices[i * step] for i in range(images_per_category)]

            # Process sampled images
            for idx in sample_indices[:images_per_category]:
                if processed_count >= N:
                    break

                item = dataset[idx]
                image = item['image']
                label_id = item['label']

                # Convert numeric label to scene name
                label_name = dataset.features['label'].int2str(label_id)

                category_counts[label_name] += 1

                # Save image
                dest_img = img_dir / f"{processed_count:05d}.jpg"

                # Convert to RGB if needed (some images might be grayscale)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                image.save(dest_img)

                # Create record
                records.append({
                    "id": f"sun397_{processed_count:05d}",
                    "image_path": str(Path("images") / dest_img.name),
                    "label": label_name,
                })

                processed_count += 1

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

        # Show category distribution (top 10)
        print(f"\nCategory distribution (top 10):")
        for category, count in category_counts.most_common(10):
            print(f"  {category}: {count} images")

        print(f"\nTotal unique categories sampled: {len(category_counts)}")

    except Exception as e:
        print(f"❌ Error downloading/processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
