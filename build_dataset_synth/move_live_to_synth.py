#!/usr/bin/env python3
"""
Move live clock validation data to synthetic dataset folder
Copies images and merges labels.csv entries
"""

import shutil
import csv
from pathlib import Path
from global_stuff.constants import BASE_DIR

# Paths
SOURCE_DIR = BASE_DIR / "clock-validation"
TARGET_DIR = BASE_DIR / "clock-validation-synth"
SOURCE_CSV = SOURCE_DIR / "labels.csv"
TARGET_CSV = TARGET_DIR / "labels.csv"

def main():
    print("Moving live clock validation data to synthetic dataset...")
    
    # Ensure target directory exists
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if source directory and CSV exist
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        return
    
    if not SOURCE_CSV.exists():
        print(f"❌ Source labels.csv not found: {SOURCE_CSV}")
        return
    
    # Get all image files from source
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    source_images = [f for f in SOURCE_DIR.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"📁 Found {len(source_images)} images in source directory")
    
    # Read source labels
    source_labels = {}
    with open(SOURCE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename'].strip()
            label = row['label'].strip()
            source_labels[filename] = label
    
    print(f"📊 Found {len(source_labels)} labels in source CSV")
    
    # Read existing target labels (if any)
    target_labels = {}
    if TARGET_CSV.exists():
        with open(TARGET_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename'].strip()
                label = row['label'].strip()
                target_labels[filename] = label
        print(f"📋 Found {len(target_labels)} existing labels in target CSV")
    
    # Copy images and collect labels
    copied_count = 0
    skipped_count = 0
    new_labels = {}
    
    for img_file in source_images:
        target_path = TARGET_DIR / img_file.name
        
        # Skip if file already exists in target
        if target_path.exists():
            print(f"⏭️  Skipping {img_file.name} (already exists)")
            skipped_count += 1
            continue
        
        # Copy the image file
        try:
            shutil.copy2(img_file, target_path)
            copied_count += 1
            
            # Add label if available
            if img_file.name in source_labels:
                new_labels[img_file.name] = source_labels[img_file.name]
            else:
                print(f"⚠️  No label found for {img_file.name}")
                
        except Exception as e:
            print(f"❌ Failed to copy {img_file.name}: {e}")
    
    # Merge labels (existing + new)
    all_labels = {**target_labels, **new_labels}
    
    # Write updated labels.csv
    with open(TARGET_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        
        # Sort by filename for consistency
        for filename in sorted(all_labels.keys()):
            writer.writerow([filename, all_labels[filename]])
    
    print(f"\n✅ Copy completed!")
    print(f"📁 Copied: {copied_count} images")
    print(f"⏭️  Skipped: {skipped_count} images (already existed)")
    print(f"📊 Total labels in target CSV: {len(all_labels)}")
    print(f"📂 Target directory: {TARGET_DIR}")
    print(f"📋 Target labels: {TARGET_CSV}")

if __name__ == "__main__":
    main()
