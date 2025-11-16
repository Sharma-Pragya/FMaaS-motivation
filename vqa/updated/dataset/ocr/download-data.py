# save as export_mnist_flat.py
from datasets import load_dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

ROOT = Path(__file__).parent
N = 1000  # number of samples to export

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ylecun/mnist")
    split = "test"
    subset = ds[split].select(range(min(N, len(ds[split]))))

    records = []
    for i, row in tqdm(enumerate(subset), total=len(subset), desc="Export MNIST"):
        img = row["image"].convert("RGB")
        fname = f"{i:05d}.png"
        path = img_dir / fname
        img.save(path)

        records.append({
            "id": f"{split}_{i:05d}",
            "image_path": str(path.relative_to(ROOT)),
            "label": int(row["label"]),
        })

    with open(ROOT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Done. Wrote {len(records)} samples to {ROOT}")

if __name__ == "__main__":
    main()