# Download MS COCO val2017 dataset for object detection
import json
import urllib.request
import zipfile
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path(__file__).parent
N = 1000  # number of images to export

def download_file(url, dest_path):
    """Download file with progress bar"""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}")

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Download annotations
    ann_zip_path = ROOT / "annotations_trainval2017.zip"
    if not ann_zip_path.exists():
        download_file(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            ann_zip_path
        )

    # Extract annotations
    print("Extracting annotations...")
    with zipfile.ZipFile(ann_zip_path, 'r') as zip_ref:
        zip_ref.extractall(ROOT)

    # Load instances_val2017.json
    instances_path = ROOT / "annotations" / "instances_val2017.json"
    with open(instances_path, 'r') as f:
        coco_data = json.load(f)

    # Build category id -> name mapping
    cat_map = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Build image_id -> image info mapping
    img_map = {img['id']: img for img in coco_data['images']}

    # Group annotations by image_id
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # Download images (first N)
    img_zip_path = ROOT / "val2017.zip"
    if not img_zip_path.exists():
        download_file(
            "http://images.cocodataset.org/zips/val2017.zip",
            img_zip_path
        )

    # Extract images
    print(f"Extracting first {N} images from val2017.zip...")
    records = []
    extracted_count = 0

    with zipfile.ZipFile(img_zip_path, 'r') as zip_ref:
        all_files = [f for f in zip_ref.namelist() if f.endswith('.jpg')]

        for img_file in tqdm(all_files[:N], desc="Processing images"):
            if extracted_count >= N:
                break

            # Extract image
            zip_ref.extract(img_file, ROOT)

            # Get image info
            img_filename = Path(img_file).name
            img_id = int(img_filename.replace('.jpg', ''))

            if img_id not in img_map:
                continue

            # Move image to images/ directory
            src_path = ROOT / img_file
            dest_path = img_dir / img_filename
            src_path.rename(dest_path)

            # Get all unique categories for this image
            anns = img_to_anns.get(img_id, [])
            unique_categories = sorted(set([cat_map[ann['category_id']] for ann in anns]))

            records.append({
                "id": f"val_{extracted_count:05d}",
                "image_path": str(Path("images") / img_filename),
                "categories": unique_categories,
            })

            extracted_count += 1

    # Clean up extracted val2017 directory if it exists
    val2017_dir = ROOT / "val2017"
    if val2017_dir.exists():
        import shutil
        shutil.rmtree(val2017_dir)

    # Save annotations.json
    with open(ROOT / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone! Exported {len(records)} images to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Annotations: {ROOT / 'annotations.json'}")

    # Clean up zip files and temporary annotations directory
    print("\nCleaning up temporary files...")
    ann_zip_path.unlink()
    img_zip_path.unlink()
    import shutil
    shutil.rmtree(ROOT / "annotations")
    print("Cleanup complete!")

if __name__ == "__main__":
    main()
