# Download ImageNet-1k validation dataset for image classification
import json
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

ROOT = Path(__file__).parent
N = 1000  # number of images to export

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Login to HuggingFace if token exists
    hf_token_path = "../../hf-token.txt"
    if os.path.exists(hf_token_path):
        with open(hf_token_path, "r") as f:
            token = f.read().strip()
        if token:
            login(token=token)
            print("Logged in to HuggingFace")

    print(f"Loading ImageNet-1k validation dataset (streaming first {N} samples)...")
    print("Note: You may need to accept ImageNet terms on HuggingFace first:")
    print("https://huggingface.co/datasets/imagenet-1k")

    # Load ImageNet-1k validation split with STREAMING to avoid downloading everything
    try:
        ds_stream = load_dataset("imagenet-1k", split="validation", streaming=True)
        # Get label names from a non-streaming version (just metadata)
        ds_meta = load_dataset("imagenet-1k", split="validation[:1]")
        label_names = ds_meta.features['label'].names if hasattr(ds_meta.features['label'], 'names') else None
    except Exception as e:
        print(f"\n❌ Error loading ImageNet: {e}")
        print("\nMake sure you have:")
        print("1. Accepted ImageNet terms at: https://huggingface.co/datasets/imagenet-1k")
        print("2. Valid HuggingFace token in ../../hf-token.txt")
        return

    if not label_names:
        print("⚠️ Warning: Could not get label names from dataset")
        print("Using label indices instead")

    records = []

    # Take only first N samples from the stream
    for i, sample in tqdm(enumerate(ds_stream.take(N)), total=N, desc="Downloading ImageNet images"):
        # Get image and label
        image = sample["image"].convert("RGB")
        label_id = sample["label"]

        # Get label name
        if label_names:
            label = label_names[label_id]
        else:
            label = str(label_id)

        # Save image
        img_filename = f"{i:05d}.jpg"
        img_path = img_dir / img_filename
        image.save(img_path)

        # Create record
        records.append({
            "id": f"val_{i:05d}",
            "image_path": str(Path("images") / img_filename),
            "label": label,
        })

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

if __name__ == "__main__":
    main()
