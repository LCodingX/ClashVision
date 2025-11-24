import shutil
import random
from pathlib import Path
from global_stuff.constants import BASE_DIR
# Paths
DS_DIR = BASE_DIR / "new-clock-dataset"
images_dir = DS_DIR / "images"
labels_dir = DS_DIR / "labels"

# New split directories
for subset in ["train", "val"]:
    (images_dir / subset).mkdir(parents=True, exist_ok=True)
    (labels_dir / subset).mkdir(parents=True, exist_ok=True)

# Get all image files
image_files = list(images_dir.glob("*.png"))
random.shuffle(image_files)

# Split ratio (e.g., 80% train, 20% val)
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_files(files, subset):
    for img_path in files:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        if label_path.exists():
            shutil.move(str(img_path), str(images_dir / subset / img_path.name))
            shutil.move(str(label_path), str(labels_dir / subset / label_path.name))

move_files(train_files, "train")
move_files(val_files, "val")
