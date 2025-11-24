import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import uuid
from global_stuff.constants import BASE_DIR, non_troops, id_non_troops

# === Paths ===
OUTPUT_DIR = BASE_DIR / "human_annotated_yolo" / "spells-live"
TRAIN_IMG_DIR = OUTPUT_DIR / "images" / "train"
TRAIN_LABEL_DIR = OUTPUT_DIR / "labels" / "train"

# Clock validation paths
CLOCK_VALIDATION_DIR = BASE_DIR / "clock-validation"
CLOCK_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
CLOCK_LABELS_CSV = CLOCK_VALIDATION_DIR / "labels.csv"

# Create directories
TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Initialize clock validation CSV
if not CLOCK_LABELS_CSV.exists():
    with open(CLOCK_LABELS_CSV, 'w') as f:
        f.write("filename,label\n")

# Input files
IMAGE_DIR = BASE_DIR / "frames"
MODEL_ANNOTATIONS_FILE = BASE_DIR / "model-annotations.json"  # Original model predictions
HUMAN_ANNOTATIONS_FILE = BASE_DIR / "live-annotations.json"          # Human corrections

# Load both annotation files
with open(MODEL_ANNOTATIONS_FILE, "r") as json_file:
    model_annotations = json.load(json_file)

with open(HUMAN_ANNOTATIONS_FILE, "r") as json_file:
    human_annotations = json.load(json_file)

def convert_to_yolo_format(x, y, w, h, img_w, img_h):
    """Convert bounding box to YOLO format (normalized center coordinates)"""
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def find_model_mistakes(human_boxes, model_boxes, img_w, img_h):
    """Find frames where model made predictions but humans corrected them"""
    # If model made predictions but humans have no annotations, 
    # this suggests humans corrected the model's mistakes
    return len(model_boxes) > 0 and len(human_boxes) == 0

def save_clock_patch(image, x, y, w, h, clock_class):
    """Save clock patch with 10-digit UID and return filename"""
    # Generate 10-digit UID
    uid = str(uuid.uuid4().int)[:10]
    filename = f"{uid}.jpg"
    
    # Crop the patch
    cropped = image[y:y+h, x:x+w]
    
    # Save patch
    patch_path = CLOCK_VALIDATION_DIR / filename
    cv2.imwrite(str(patch_path), cropped)
    
    # Return filename for CSV
    return filename

# Process each image by going through actual frame files
file_counter = 30266
frame_counter = 0

# Get all frame files and sort them
frame_files = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

for img_path in frame_files:
    filename = img_path.name
    
    # Look up annotations for this frame in both files
    model_entry = model_annotations.get(filename, None)
    human_entry = human_annotations.get(filename, None)
    
    # Get model predictions and human corrections
    model_regions = model_entry['regions'] if model_entry else []
    human_regions = human_entry['regions'] if human_entry else []

    if not img_path.exists():
        print(f"Warning: Image {filename} not found.")
        continue

    # Get image dimensions
    try:
        import cv2
        import numpy as np
        with open(img_path, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Could not read image {filename}")
                continue
            img_h, img_w = img.shape[:2]
    except ImportError:
        # Fallback if cv2 not available
        from PIL import Image
        img = Image.open(img_path)
        img_w, img_h = img.size

    # Convert human corrections to YOLO format
    human_boxes = []
    yolo_lines = []
    
    # Process clock patches from model annotations
    for region in model_regions:
        if region['region_attributes']['class'].startswith('clock'):
            shape = region['shape_attributes']
            x = shape['x']
            y = shape['y']
            w = shape['width']
            h = shape['height']
            
            # Extract clock class (0, 1, or 2)
            clock_class = region['region_attributes']['class'].split(' ')[1]
            
            # Save clock patch
            filename = save_clock_patch(img, x, y, w, h, clock_class)
            
            # Add to CSV
            with open(CLOCK_LABELS_CSV, 'a') as f:
                f.write(f"{filename},{clock_class}\n")
            
            print(f"🕐 Saved clock patch {filename} with class 'clock {clock_class}'")
    
    for region in human_regions:
        shape = region['shape_attributes']
        try:
            class_name = region['region_attributes']['class']
            # If class name contains a space (e.g., "poison 0.77"), extract just the class part
            if ' ' in class_name:
                class_name = class_name.split(' ')[0]
        except:
            print(f"failure with {region}")
            continue
            
        if shape['name'] != 'rect':
            continue

        x = shape['x']
        y = shape['y']
        w = shape['width']
        h = shape['height']

        # Store human box for comparison
        human_boxes.append({'x': x, 'y': y, 'w': w, 'h': h})

        # Convert to YOLO format
        if class_name in id_non_troops:
            class_id = id_non_troops[class_name]
            x_center, y_center, w_norm, h_norm = convert_to_yolo_format(x, y, w, h, img_w, img_h)
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    # Check model predictions vs human corrections
    human_corrections = len(yolo_lines) > 0
    
    # Check if model made predictions but humans removed them
    model_removed_by_humans = len(model_regions) > 0 and not human_corrections
    
    # Save file if either condition is met
    if human_corrections or model_removed_by_humans:
        # Generate filename with 5 digits starting from 982
        new_filename = f"{file_counter:05d}.jpg"
        new_label_filename = f"{file_counter:05d}.txt"
        
        # Save image
        output_img_path = TRAIN_IMG_DIR / new_filename
        shutil.copy(img_path, output_img_path)
        
        # Save labels
        output_label_path = TRAIN_LABEL_DIR / new_label_filename
        with open(output_label_path, "w") as f:
            if human_corrections:
                # Human corrections to model predictions
                f.write("\n".join(yolo_lines))
                print(f"Saved {new_filename} with {len(yolo_lines)} human annotations (model corrected)")
            else:
                # Model made predictions but humans removed them
                f.write("")
                print(f"Saved {new_filename} with empty labels (model predictions removed by humans)")
        
        file_counter += 1
    else:
        # Neither model predictions nor human corrections - skip this frame
        print(f"Skipping {filename} - no model predictions or human corrections")

print(f"\n✅ Dataset creation complete!")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"📊 Total images processed: {file_counter}")
print(f"🖼️ Images: {TRAIN_IMG_DIR}")
print(f"🏷️ Labels: {TRAIN_LABEL_DIR}")
print(f"📝 Note: spell-annotations.json contains original model predictions")
print(f"🔍 Human annotations represent corrections to model mistakes")

# Count clock patches
if CLOCK_LABELS_CSV.exists():
    with open(CLOCK_LABELS_CSV, 'r') as f:
        lines = f.readlines()
        clock_count = len(lines) - 1  # Subtract header
        print(f"🕐 Clock patches saved: {clock_count}")
        print(f"📁 Clock validation directory: {CLOCK_VALIDATION_DIR}")
        print(f"📊 Clock labels CSV: {CLOCK_LABELS_CSV}")