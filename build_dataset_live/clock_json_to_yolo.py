import random
from global_stuff.constants import BASE_DIR
import json
from PIL import Image
from pathlib import Path
import shutil

# === Paths ===
DATASET_PATH = {
    "train_img": BASE_DIR / "clock_dataset/images/train",
    "train_txt": BASE_DIR / "clock_dataset/labels/train",
    "val_img": BASE_DIR / "clock_dataset/images/val",
    "val_txt": BASE_DIR / "clock_dataset/labels/val",
}

# Create all directories
for path in DATASET_PATH.values():
    path.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = BASE_DIR / "frames"
JSON_FILE = BASE_DIR / "live_annotations/clock.json"

with open(JSON_FILE, "r") as json_file:
    annotations = json.load(json_file)

file_id_train = 1000
file_id_val = 200

for img_id, entry in annotations.items():
    filename = entry['filename']
    regions = entry['regions']
    img_path = IMAGE_DIR / filename

    if not img_path.exists():
        print(f"Warning: Image {filename} not found.")
        continue

    with Image.open(img_path) as img:
        img_w, img_h = img.size

    yolo_lines = []

    for region in regions:
        shape = region['shape_attributes']
        try:
            class_name = region['region_attributes']['class']
        except:
            print(f"failure with {region}")
            continue
        if shape['name'] != 'rect':
            continue

        if class_name not in ["clock-ally", "clock-enemy"]:
            continue

        x = shape['x']
        y = shape['y']
        w = shape['width']
        h = shape['height']

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        class_id = 0 if class_name=="clock-ally" else 1  # only 1 class: "clock"
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Determine split
    if random.random() < 0.2:
        split = "val"
        basename = f"{file_id_val:06d}.jpg"
        file_id_val += 1
    else:
        split = "train"
        basename = f"{file_id_train:06d}.jpg"
        file_id_train += 1

    # Save label (may be empty)
    label_path = DATASET_PATH[f"{split}_txt"] / basename.replace(".jpg", ".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # Save image
    image_path = DATASET_PATH[f"{split}_img"] / basename
    shutil.copy(img_path, image_path)

print("Train image count:", file_id_train)
print("Val image count:", file_id_val)
