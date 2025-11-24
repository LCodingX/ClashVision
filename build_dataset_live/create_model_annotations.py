from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from global_stuff.constants import BASE_DIR

# === CONFIG ===
MODEL_PATHS = [
    BASE_DIR / "models/finetuning-3/small.pt",
    BASE_DIR / "models/finetuning-3/medium.pt",
    BASE_DIR / "models/finetuning-3/large.pt",
    BASE_DIR / "models/finetuning-3/buildings.pt"
]

INPUT_DIR = BASE_DIR / "frames"
OUTPUT_DIR = BASE_DIR / "json-model-annotations"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "json8.json"

# Load models once and pair with their paths
models = [(YOLO(model_path), model_path) for model_path in MODEL_PATHS]

def run_prediction(model, frame_path):
    return model.predict(
        source=str(frame_path),
        imgsz=896,
        conf=0.25,
        save=False,
        device="cpu",
        verbose=False
    )

data = {}
for frame_path in tqdm(list(INPUT_DIR.glob("*.jpg"))):
    image = cv2.imread(str(frame_path))
    if image is None:
        print(f"Warning: Could not read image {frame_path}")
        continue

    height, width = image.shape[:2]
    size = Path(frame_path).stat().st_size
    file_key = f"{frame_path.name}{size}"

    data[file_key] = {
        "filename": frame_path.name,
        "size": size,
        "regions": [],
        "file_attributes": {}
    }

    # Run model predictions in parallel
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_info = {
            executor.submit(run_prediction, model, frame_path): (model, model_path)
            for model, model_path in models
        }
        for future in as_completed(future_to_info):
            model, model_path = future_to_info[future]
            try:
                results = future.result()
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        xyxy = [int(c) for c in boxes.xyxy[i].tolist()]
                        confidence = float(boxes.conf[i].item())
                        class_name = model.model.names[cls_id]

                        model_path_str = str(model_path)
                        if "medium" in model_path_str and "bush-goblin" in class_name:
                            continue  # skip for medium model
                        if "large" in model_path_str and "elixir-golem-small" in class_name:
                            continue  # skip for large model

                        x1, y1, x2, y2 = xyxy
                        x1 = max(0, min(x1, width))
                        y1 = max(0, min(y1, height))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))
                        rect_width = x2 - x1
                        rect_height = y2 - y1
                        if rect_width <= 0 or rect_height <= 0:
                            continue

                        region = {
                            "shape_attributes": {
                                "name": "rect",
                                "x": x1,
                                "y": y1,
                                "width": rect_width,
                                "height": rect_height
                            },
                            "region_attributes": {
                                "class": class_name,
                            }
                        }

                        data[file_key]["regions"].append(region)

            except Exception as e:
                print(f"Error running model from {model_path} on {frame_path}: {e}")

# Write the JSON output
with open(OUTPUT_FILE, "w") as output_file:
    json.dump(data, output_file, indent=2)

print(f"Generated annotations for {len(data)} images")
print(f"Output saved to: {OUTPUT_FILE}")

# Optional: print a sample entry for verification
if data:
    sample_key = next(iter(data))
    print(f"\nSample entry structure:")
    print(f"Key: {sample_key}")
    print(f"Regions count: {len(data[sample_key]['regions'])}")
    if data[sample_key]['regions']:
        print(f"First region: {data[sample_key]['regions'][0]}")
