from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import torch
import torchvision.models as models
from time import perf_counter
from global_stuff.constants import BASE_DIR
import os
import numpy as np

# === CONFIG ===
MODEL_PATH = BASE_DIR / "models/non-troops-5/last.pt"
INPUT_DIR = BASE_DIR / "frames"
OUTPUT_DIR = BASE_DIR
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "model-annotations.json"

# Clock classifier input size (requested)
CLOCK_INPUT_SIZE = 96

# Prefer MPS; fallback to CPU
USE_MPS = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
DEVICE_STR = "mps" if USE_MPS else "cpu"
TORCH_DEVICE = torch.device(DEVICE_STR)

def load_clock_model():
    """Load the trained ResNet-50 model for 3-class clock detection (0/1/2)."""
    try:
        model_path = BASE_DIR / "models/clock-classifier-7/resnet50_best.pt"
        if not model_path.exists():
            print("⚠️ Clock detection model not found, using default confidence")
            return None, None

        # CPU threading hints (still useful for data prep even if using MPS)
        try:
            torch.set_num_threads(os.cpu_count())
            torch.set_num_interop_threads(max(1, (os.cpu_count() or 2)//2))
        except Exception:
            pass

        # Build ResNet-50 head for 3 classes
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)

        # Load checkpoint; strip 'module.' if present
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        def strip_module(sd):
            out = {}
            for k, v in sd.items():
                out[k[7:] if k.startswith("module.") else k] = v
            return out

        state_dict = strip_module(state_dict)
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"⚠️ strict load failed ({e}); retrying with strict=False")
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        model.to(TORCH_DEVICE)

        print(f"✅ Clock detection model (ResNet-50) loaded on {DEVICE_STR}")
        return model, None
    except Exception as e:
        print(f"⚠️ Failed to load clock detection model: {e}, using default confidence")
        return None, None

def get_clock_detection_count(image, clock_model, _unused_transform):
    """
    Get detection count for clock using the classifier model (ResNet-50).

    - If `image` is a single BGR crop (np.ndarray HxWx3), returns (int_prediction, transform_ms, detect_ms).
    - If `image` is a list of BGR crops, returns (list_of_int_predictions, transform_ms_total, detect_ms_total).
    """
    if clock_model is None:
        if isinstance(image, list):
            return [1] * len(image), 0.0, 0.0
        return 1, 0.0, 0.0

    _IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _prep_one(bgr, size=(CLOCK_INPUT_SIZE, CLOCK_INPUT_SIZE)):
        # Resize to 96x96 as requested
        r = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)       # (H,W,3) BGR
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r = (r - _IM_MEAN) / _IM_STD
        r = np.transpose(r, (2, 0, 1))  # CHW
        return r  # float32

    try:
        if isinstance(image, list):
            if len(image) == 0:
                return [], 0.0, 0.0

            t0 = perf_counter()
            batch_arr = np.stack([_prep_one(c) for c in image], axis=0)  # NCHW, float32
            inputs = torch.from_numpy(batch_arr).to(TORCH_DEVICE, non_blocking=False)
            t1 = perf_counter()

            t2 = perf_counter()
            with torch.inference_mode():
                outputs = clock_model(inputs)
                preds = torch.argmax(outputs, dim=1).tolist()
            t3 = perf_counter()

            return preds, (t1 - t0) * 1000.0, (t3 - t2) * 1000.0

        # Single-crop path
        t0 = perf_counter()
        arr = _prep_one(image)  # CHW float32
        inp = torch.from_numpy(arr)[None, ...].to(TORCH_DEVICE, non_blocking=False)  # NCHW
        t1 = perf_counter()

        t2 = perf_counter()
        with torch.inference_mode():
            output = clock_model(inp)
            prediction = int(torch.argmax(output, dim=1).item())
        t3 = perf_counter()

        return prediction, (t1 - t0) * 1000.0, (t3 - t2) * 1000.0

    except Exception as e:
        print(f"⚠️ Clock detection failed: {e}, using default")
        if isinstance(image, list):
            return [1] * len(image), 0.0, 0.0
        return 1, 0.0, 0.0

def main():
    # Load models
    yolo_model = YOLO(MODEL_PATH)
    clock_model, clock_transform = load_clock_model()

    data = {}
    frame_paths = sorted(INPUT_DIR.glob("*.jpg"), key=lambda x: int(x.stem.split("_")[1]))

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"Warning: Could not read image {frame_path}")
            continue

        height, width = image.shape[:2]
        size = Path(frame_path).stat().st_size
        file_key = frame_path.name

        data[file_key] = {
            "filename": frame_path.name,
            "size": size,
            "regions": [],
            "file_attributes": {}
        }

        try:
            yolo_start = perf_counter()
            results = yolo_model.predict(
                source=str(frame_path),
                imgsz=896,
                conf=0.7,
                save=False,
                device=DEVICE_STR,  # use MPS if available
                verbose=False
            )
            yolo_time = (perf_counter() - yolo_start) * 1000.0  # ms
            print(f"🔍 YOLO on {frame_path.name} ({DEVICE_STR}): {yolo_time:.2f}ms")

            result = results[0]
            boxes = result.boxes

            # Collect clock crops for batched classifier inference
            clock_indices = []
            clock_crops = []

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    class_name = yolo_model.model.names[cls_id].lower()
                    xyxy = [int(c) for c in boxes.xyxy[i].tolist()]

                    x1, y1, x2, y2 = xyxy
                    x1 = max(0, min(x1, width));  y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width));  y2 = max(0, min(y2, height))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    if class_name == "clock" and clock_model is not None:
                        clock_indices.append(i)
                        clock_crops.append(image[y1:y2, x1:x2])  # BGR crop

                # Batched call for all clocks in this frame
                batched_preds = []
                transform_time = 0.0
                detect_time = 0.0
                if clock_indices:
                    batched_preds, transform_time, detect_time = get_clock_detection_count(
                        clock_crops, clock_model, clock_transform
                    )
                    print(f"🕐 Batched ResNet50@{CLOCK_INPUT_SIZE} on {len(clock_indices)} crop(s) [{DEVICE_STR}]: "
                          f"Transform {transform_time:.2f}ms | Detect {detect_time:.2f}ms")

                # Emit regions
                clock_ptr = 0
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    xyxy = [int(c) for c in boxes.xyxy[i].tolist()]
                    confidence = float(boxes.conf[i].item())
                    class_name = yolo_model.model.names[cls_id]

                    x1, y1, x2, y2 = xyxy
                    x1 = max(0, min(x1, width));  y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width));  y2 = max(0, min(y2, height))
                    rect_width = x2 - x1
                    rect_height = y2 - y1
                    if rect_width <= 0 or rect_height <= 0:
                        continue

                    if class_name.lower() == "clock" and clock_model is not None and clock_ptr < len(batched_preds):
                        detection_count = int(batched_preds[clock_ptr])
                        clock_ptr += 1
                        class_with_value = f"{class_name} {detection_count}"
                    else:
                        class_with_value = f"{class_name} {confidence:.2f}"

                    region = {
                        "shape_attributes": {
                            "name": "rect",
                            "x": x1,
                            "y": y1,
                            "width": rect_width,
                            "height": rect_height
                        },
                        "region_attributes": {
                            "class": class_with_value,
                        }
                    }

                    data[file_key]["regions"].append(region)

        except Exception as e:
            print(f"Error processing {frame_path}: {e}")

    # Sort data by filename numerically before writing
    sorted_data = dict(sorted(data.items(), key=lambda x: int(Path(x[0]).stem.split("_")[1])))

    # Write the JSON output
    with open(OUTPUT_FILE, "w") as output_file:
        json.dump(sorted_data, output_file, indent=2)

    print(f"Generated annotations for {len(data)} images")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
