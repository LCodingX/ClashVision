import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os
from PIL import Image
import uuid
from global_stuff.constants import BASE_DIR, non_troops
# Configuration
FRAMES_DIR = BASE_DIR / "frames"
JSON_FILE = BASE_DIR / "model-annotations-10.json"
MODEL_PATH = BASE_DIR / "models/non-troop-finetuning-9/last.pt"
OUTPUT_DIR = BASE_DIR / "spell-validation-dataset"

# Spell classes to collect
SPELL_CLASSES = ["zap", "earthquake", "goblin-curse", "poison", "ice-wizard", "graveyard"]

def ensure_output_dir():
    """Ensure the output directory exists"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each spell class
    for spell in SPELL_CLASSES:
        spell_dir = OUTPUT_DIR / spell
        spell_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Created subdirectories for: {', '.join(SPELL_CLASSES)}")

def crop_detection(image, x, y, width, height):
    """Crop detection without padding for clean validation samples"""
    h, w = image.shape[:2]
    
    # Crop exactly to the detection bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + width)
    y2 = min(h, y + height)
    
    return image[y1:y2, x1:x2]

def process_json_mode():
    """Process frames from JSON annotations"""
    print("Processing JSON annotations mode...")
    
    if not JSON_FILE.exists():
        print(f"JSON file not found: {JSON_FILE}")
        return
    
    with open(JSON_FILE, 'r') as f:
        annotations = json.load(f)
    
    processed_count = 0
    spell_count = {spell: 0 for spell in SPELL_CLASSES}
    
    for frame_name, frame_data in annotations.items():
        if not frame_data.get('regions'):
            continue
            
        frame_path = FRAMES_DIR / frame_name
        if not frame_path.exists():
            print(f"Frame not found: {frame_path}")
            continue
        
        # Load image
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"Failed to load image: {frame_path}")
            continue
        
        # Process each detection
        for region in frame_data['regions']:
            class_name = region['region_attributes'].get('class', '').lower()
            if " " in class_name:
                class_name = class_name.split(" ")[0]
            
            if class_name in SPELL_CLASSES:
                # Get bounding box coordinates
                shape = region['shape_attributes']
                x = shape['x']
                y = shape['y']
                width = shape['width']
                height = shape['height']
                
                # Crop the detection
                cropped = crop_detection(image, x, y, width, height)
                
                if cropped.size > 0:
                    # Save the patch to the appropriate spell subdirectory
                    uid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
                    output_filename = f"{uid}.jpg"
                    output_path = OUTPUT_DIR / class_name / output_filename
                    
                    cv2.imwrite(str(output_path), cropped)
                    spell_count[class_name] += 1
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} detections...")
    
    print(f"\nJSON mode completed!")
    print(f"Total detections processed: {processed_count}")
    for spell, count in spell_count.items():
        print(f"{spell}: {count} patches")

def process_live_mode():
    """Process frames using YOLO model"""
    print("Processing live detection mode...")
    
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return
    
    # Load YOLO model
    model = YOLO(str(MODEL_PATH))
    print(f"Loaded model: {MODEL_PATH}")
    
    # Get list of frame files
    frame_files = list(FRAMES_DIR.glob("*.jpg")) + list(FRAMES_DIR.glob("*.png"))
    if not frame_files:
        print(f"No frame files found in {FRAMES_DIR}")
        return
    
    print(f"Found {len(frame_files)} frame files")
    
    processed_count = 0
    spell_count = {spell: 0 for spell in SPELL_CLASSES}
    
    for frame_path in frame_files:
        print(f"Processing {frame_path.name}...")
        
        # Run YOLO detection
        results = model(str(frame_path))
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy()
                coords = boxes.xyxy.cpu().numpy()
                
                for i, (cls, coord) in enumerate(zip(classes, coords)):
                    class_name = model.names[int(cls)].lower()
                    
                    if class_name in SPELL_CLASSES:
                        # Get coordinates
                        x1, y1, x2, y2 = coord.astype(int)
                        x, y, width, height = x1, y1, x2 - x1, y2 - y1
                        
                        # Load image for cropping
                        image = cv2.imread(str(frame_path))
                        if image is not None:
                            # Crop the detection
                            cropped = crop_detection(image, x, y, width, height)
                            
                            if cropped.size > 0:
                                # Save the patch to the appropriate spell subdirectory
                                uid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
                                output_filename = f"{uid}_{class_name}.jpg"
                                output_path = OUTPUT_DIR / class_name / output_filename
                                
                                cv2.imwrite(str(output_path), cropped)
                                spell_count[class_name] += 1
                                processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"Processed {processed_count} detections...")
    
    print(f"\nLive mode completed!")
    print(f"Total detections processed: {processed_count}")
    for spell, count in spell_count.items():
        print(f"{spell}: {count} patches")

def main(from_json=False, json_file=None):
    """Main function with two modes"""
    print("Spell Patch Collection Program")
    print("=" * 40)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    if from_json:
        # Use provided json_file if specified, otherwise use default
        global JSON_FILE
        if json_file:
            JSON_FILE = Path(json_file)
        process_json_mode()
    else:
        process_live_mode()
    
    print("\nProgram completed successfully!")

if __name__ == "__main__":
    import argparse
    
    main(from_json=True)
