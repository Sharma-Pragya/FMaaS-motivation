# Download and process DroneCrowd dataset for categorical crowd counting
import json
import os
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import Counter
import subprocess
import sys

ROOT = Path(__file__).parent
N = 100  # number of images to export (matching other tasks)

# Google Drive file ID for DroneCrowd dataset
GDRIVE_FILE_ID = "1HY3V4QObrVjzXUxL_J86oxn2bi7FMUgd"
DATASET_ZIP = ROOT / "DroneCrowd.zip"

# Crowd density categories based on person count
# These ranges are designed for drone-based aerial crowd scenes
CROWD_CATEGORIES = [
    (0, 20, "very_sparse"),      # 0-20 people
    (21, 50, "sparse"),          # 21-50 people
    (51, 100, "moderate"),       # 51-100 people
    (101, 300, "dense"),         # 101-300 people
    (301, float('inf'), "very_dense")  # 301+ people
]

def install_gdown():
    """Install gdown package if not available."""
    try:
        import gdown
        return True
    except ImportError:
        print("📦 Installing gdown package for Google Drive downloads...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
            print("✅ gdown installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install gdown: {e}")
            return False

def download_dataset():
    """Download DroneCrowd dataset from Google Drive."""
    if DATASET_ZIP.exists():
        print(f"✅ Dataset ZIP already downloaded: {DATASET_ZIP}")
        return True

    if not install_gdown():
        print("\n⚠️  Could not install gdown. Please install manually:")
        print("   pip install gdown")
        return False

    try:
        import gdown
        print(f"\n📥 Downloading DroneCrowd dataset from Google Drive...")
        print(f"   Size: ~1.03 GB (this may take a few minutes)")

        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(DATASET_ZIP), quiet=False)

        print(f"✅ Download complete: {DATASET_ZIP}")
        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n🔗 Manual download option:")
        print(f"   https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
        return False

def extract_dataset():
    """Extract the downloaded ZIP file."""
    dataset_path = ROOT / "DroneCrowd"

    if dataset_path.exists():
        print(f"✅ Dataset already extracted: {dataset_path}")
        return True

    if not DATASET_ZIP.exists():
        print(f"❌ ZIP file not found: {DATASET_ZIP}")
        return False

    try:
        print(f"\n📦 Extracting dataset...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(ROOT)

        print(f"✅ Extraction complete: {dataset_path}")

        # Optionally remove ZIP file to save space
        # DATASET_ZIP.unlink()
        # print(f"🗑️  Removed ZIP file to save space")

        return True

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def get_category(count):
    """Convert exact count to crowd density category."""
    for min_count, max_count, category in CROWD_CATEGORIES:
        if min_count <= count <= max_count:
            return category
    return "very_dense"  # fallback

def count_people_in_frame(annotation_file, frame_number):
    """
    Count the number of people in a specific frame.
    VisDrone annotation format: frame_number,x_coordinate,y_coordinate
    Each line represents one person's head position.
    """
    if not os.path.exists(annotation_file):
        return None

    try:
        count = 0
        with open(annotation_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    frame_num = int(parts[0])
                    if frame_num == frame_number:
                        count += 1
        return count
    except Exception as e:
        print(f"Error reading {annotation_file}: {e}")
        return None

def main():
    print("=" * 60)
    print("DroneCrowd Dataset Download and Processing Script")
    print("=" * 60)

    # Step 1: Download dataset if needed
    dataset_path = ROOT / "VisDrone2020-CC"

    if not dataset_path.exists():
        print("\n📥 Dataset not found. Starting automatic download...")

        # Download from Google Drive
        if not download_dataset():
            print("\n❌ Automatic download failed.")
            print("\n📝 Manual download instructions:")
            print("   1. Download from: https://drive.google.com/file/d/1HY3V4QObrVjzXUxL_J86oxn2bi7FMUgd/view")
            print(f"   2. Save as: {DATASET_ZIP}")
            print("   3. Run this script again")
            return

        # Extract the downloaded ZIP
        if not extract_dataset():
            print("\n❌ Extraction failed. Please extract manually.")
            return

    print(f"\n✅ Found VisDrone2020-CC dataset at: {dataset_path}")

    # Create output directories
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Read training sequences list
    trainlist_file = dataset_path / "trainlist.txt"
    if not trainlist_file.exists():
        print(f"\n❌ Error: trainlist.txt not found")
        return

    with open(trainlist_file, 'r') as f:
        train_sequences = [line.strip() for line in f if line.strip()]

    print(f"\n📊 Found {len(train_sequences)} training sequences")
    print(f"Sampling frames to get {N} images...")

    records = []
    category_counts = Counter()
    processed_count = 0

    # Calculate frames per sequence to sample
    frames_per_seq = max(1, N // len(train_sequences) + 1)

    print(f"Sampling ~{frames_per_seq} frame(s) from each sequence for diversity...")

    # Process sequences
    for seq_id in tqdm(train_sequences, desc="Processing sequences"):
        if processed_count >= N:
            break

        seq_dir = dataset_path / "sequences" / seq_id
        ann_file = dataset_path / "annotations" / f"{seq_id}.txt"

        if not seq_dir.exists() or not ann_file.exists():
            continue

        # Get all frames in this sequence
        frame_files = sorted(list(seq_dir.glob("*.jpg")))

        if len(frame_files) == 0:
            continue

        # Sample frames evenly from this sequence
        # Take frames from beginning, middle, end to get diversity within sequence
        total_frames = len(frame_files)
        sample_indices = []

        if frames_per_seq == 1:
            # Take middle frame
            sample_indices = [total_frames // 2]
        elif frames_per_seq == 2:
            # Take frames from 1/3 and 2/3 positions
            sample_indices = [total_frames // 3, 2 * total_frames // 3]
        else:
            # Evenly space frames
            step = total_frames // frames_per_seq
            sample_indices = [i * step for i in range(frames_per_seq) if i * step < total_frames]

        for frame_idx in sample_indices[:frames_per_seq]:
            if processed_count >= N:
                break

            frame_path = frame_files[frame_idx]
            frame_number = frame_idx + 1  # Frame numbers start at 1

            # Count people in this frame
            person_count = count_people_in_frame(ann_file, frame_number)

            if person_count is None or person_count == 0:
                continue

            # Convert count to category
            category = get_category(person_count)
            category_counts[category] += 1

            # Copy image to our directory
            dest_img = img_dir / f"{processed_count:05d}.jpg"
            shutil.copy(frame_path, dest_img)

            # Create record
            records.append({
                "id": f"seq{seq_id}_frame{frame_number:05d}",
                "image_path": str(Path("images") / dest_img.name),
                "exact_count": person_count,
                "label": category,
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
        print(f"  {rec['id']}: {rec['label']} ({rec['exact_count']} people)")

    # Show category distribution
    print(f"\nCategory distribution:")
    for category in ["very_sparse", "sparse", "moderate", "dense", "very_dense"]:
        count = category_counts[category]
        if count > 0:
            print(f"  {category}: {count} images")

    # Show category definitions
    print(f"\nCategory definitions:")
    for min_c, max_c, cat_name in CROWD_CATEGORIES:
        if max_c == float('inf'):
            print(f"  {cat_name}: {min_c}+ people")
        else:
            print(f"  {cat_name}: {min_c}-{max_c} people")

    # Cleanup: Remove extracted dataset and ZIP file to save space
    print(f"\n🧹 Cleaning up temporary files...")
    try:
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            print(f"  ✅ Removed: {dataset_path.name}/")

        if DATASET_ZIP.exists():
            DATASET_ZIP.unlink()
            print(f"  ✅ Removed: {DATASET_ZIP.name}")

        print(f"\n✨ Cleanup complete! Directory now contains only:")
        print(f"  - images/ ({len(records)} images)")
        print(f"  - labels.json")
        print(f"  - download-data.py")

    except Exception as e:
        print(f"  ⚠️ Cleanup warning: {e}")
        print(f"  You can manually delete {dataset_path.name}/ and {DATASET_ZIP.name} to save space")

if __name__ == "__main__":
    main()
