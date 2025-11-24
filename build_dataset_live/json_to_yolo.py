import random
from global_stuff.constants import (
    small_troops, medium_troops, large_troops, buildings,
    id_small_troops, id_medium_troops, id_large_troops, id_buildings,
    BASE_DIR
)
import json
from PIL import Image
from pathlib import Path
import shutil

# === Paths ===
DATASET_PATHS = {
    "large": {
        "train_img": BASE_DIR / "large-troop-live/images/train",
        "train_txt": BASE_DIR / "large-troop-live/labels/train",
        "val_img": BASE_DIR / "large-troop-live/images/val",
        "val_txt": BASE_DIR / "large-troop-live/labels/val",
    },
    "medium": {
        "train_img": BASE_DIR / "medium-troop-live/images/train",
        "train_txt": BASE_DIR / "medium-troop-live/labels/train",
        "val_img": BASE_DIR / "medium-troop-live/images/val",
        "val_txt": BASE_DIR / "medium-troop-live/labels/val",
    },
    "small": {
        "train_img": BASE_DIR / "small-troop-live/images/train",
        "train_txt": BASE_DIR / "small-troop-live/labels/train",
        "val_img": BASE_DIR / "small-troop-live/images/val",
        "val_txt": BASE_DIR / "small-troop-live/labels/val",
    },
    "building": {
        "train_img": BASE_DIR / "building-live/images/train",
        "train_txt": BASE_DIR / "building-live/labels/train",
        "val_img": BASE_DIR / "building-live/images/val",
        "val_txt": BASE_DIR / "building-live/labels/val",
    },
}

# Create all directories
for dataset in DATASET_PATHS.values():
    for path in dataset.values():
        path.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = BASE_DIR / "frames"
JSON_FILE = BASE_DIR / "live_annotations/annotation7.json"

with open(JSON_FILE, "r") as json_file:
    annotations = json.load(json_file)

file_id_train = 815
file_id_val = 182

for img_id, entry in annotations.items():
    filename = entry['filename']
    regions = entry['regions']
    img_path = IMAGE_DIR / filename

    if not img_path.exists():
        print(f"Warning: Image {filename} not found.")
        continue

    with Image.open(img_path) as img:
        img_w, img_h = img.size

    yolo_by_type = {"small": [], "medium": [], "large": [], "building": []}

    for region in regions:
        shape = region['shape_attributes']
        try:
            class_name = region['region_attributes']['class']
        except:
            print(f"failure with {region}")
            continue
        team = 1 if class_name.endswith("-enemy") else 0
        if shape['name'] != 'rect':
            continue

        x = shape['x']
        y = shape['y']
        w = shape['width']
        h = shape['height']

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        if class_name is None or (
            class_name not in buildings
            and class_name not in small_troops
            and class_name not in medium_troops
            and class_name not in large_troops
        ):
            print(f"Skipping unknown class '{class_name}' in {filename}")
            continue

        if class_name in buildings:
            class_id = id_buildings[class_name]
            yolo_by_type["building"].append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        elif class_name in small_troops:
            class_id = id_small_troops[class_name]
            yolo_by_type["small"].append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        elif class_name in medium_troops:
            class_id = id_medium_troops[class_name]
            yolo_by_type["medium"].append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        elif class_name in large_troops:
            class_id = id_large_troops[class_name]
            yolo_by_type["large"].append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Determine split
    if random.random() < 0.2:
        split = "val"
        basename = f"{file_id_val:06d}.jpg"
        file_id_val += 1
    else:
        split = "train"
        basename = f"{file_id_train:06d}.jpg"
        file_id_train += 1

    # Save each troop type to correct folder if it has any labels
    # Save the image and a label file for every troop type, even if it's empty
    for troop_type, dataset_paths in DATASET_PATHS.items():
        label_path = dataset_paths[f"{split}_txt"] / f"{basename.replace('.jpg', '.txt')}"
        image_path = dataset_paths[f"{split}_img"] / basename

        yolo_lines = yolo_by_type[troop_type]

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))  # may be empty

        shutil.copy(img_path, image_path)

print(file_id_train)
print(file_id_val)