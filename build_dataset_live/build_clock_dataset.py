import cv2
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from global_stuff.constants import BASE_DIR
# === CONFIGURATION ===
CONFIDENCE_THRESHOLD = 0.5
MISS_LIMIT = 30
IOU_THRESHOLD = 0.05
VAL_SPLIT = 0.2

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def is_same_clock(box1, box2):
    return iou(box1, box2) >= IOU_THRESHOLD

def build_clock_dataset(input_dir, output_dir, clock_model_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    clock_model = YOLO(clock_model_path)
    frame_paths = sorted(input_dir.glob("*.jpg"), key=lambda p: int(p.stem.split("_")[-1]))

    detections_by_frame = {}
    active_candidates = []  # candidates waiting to hit 15-frame streak
    confirmed_clocks = []   # clocks that have passed 15-frame streak

    for frame_path in tqdm(frame_paths, desc="Detecting clocks"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        results = clock_model.predict(frame, conf=CONFIDENCE_THRESHOLD, imgsz=896, device='cpu', verbose=False)[0]
        current_detections = []
        for j in range(len(results.boxes)):
            cls = int(results.boxes.cls[j].item())
            class_name = clock_model.model.names[cls]
            if "enemy" in class_name.lower():
                x1, y1, x2, y2 = map(int, results.boxes.xyxy[j].tolist())
                current_detections.append([x1, y1, x2, y2])

        # Step 1: Match against active candidates
        for cand in active_candidates:
            cand["matched"] = False  # Reset match status

        for dbox in current_detections:
            found = False
            for cand in active_candidates:
                if is_same_clock(cand["box"], dbox):
                    cand["box"] = dbox
                    cand["streak"] += 1
                    cand["frames"].append((frame_path, dbox))
                    cand["matched"] = True
                    cand["miss_count"] = 0
                    found = True
                    if cand["streak"] == 15:
                        print(f"✅ Confirmed clock after 15 detections: {dbox}")
                        confirmed_clocks.append({
                            "box": dbox,
                            "frames": cand["frames"].copy(),
                        })
                    break
            if not found:
                active_candidates.append({
                    "box": dbox,
                    "streak": 1,
                    "frames": [(frame_path, dbox)],
                    "matched": True,
                    "miss_count": 0,
                })

        # Step 2: Update miss counters and filter
        new_candidates = []
        for cand in active_candidates:
            if not cand["matched"]:
                cand["miss_count"] += 1
            if cand["streak"] >= 15:
                continue  # keep confirmed ones
            if cand["miss_count"] >= MISS_LIMIT:
                print(f"❌ Clock candidate expired (missed {MISS_LIMIT} frames): {cand['box']}")
            else:
                new_candidates.append(cand)
        active_candidates = new_candidates

        # Step 3: Store detections from confirmed clocks
        for clock in confirmed_clocks:
            for frame_path_entry, box in clock["frames"]:
                x1, y1, x2, y2 = box
                box_w = x2 - x1
                box_h = y2 - y1
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                rel_w = box_w / w
                rel_h = box_h / h
                label_line = f"0 {x_center:.6f} {y_center:.6f} {rel_w:.6f} {rel_h:.6f}"
                detections_by_frame.setdefault(frame_path_entry, []).append(label_line)
            clock["frames"] = []
        confirmed_clocks.clear()

    # === Split and export dataset ===
    valid_items = list(detections_by_frame.items())
    random.shuffle(valid_items)
    split_idx = int(len(valid_items) * (1 - VAL_SPLIT))
    train_items = valid_items[:split_idx]
    val_items = valid_items[split_idx:]

    for subset_name, subset_data in [("train", train_items), ("val", val_items)]:
        img_dir = output_dir / "images" / subset_name
        lbl_dir = output_dir / "labels" / subset_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for frame_path, label_lines in tqdm(subset_data, desc=f"Exporting {subset_name}"):
            frame_name = frame_path.stem
            out_img = img_dir / f"{frame_name}.png"
            out_txt = lbl_dir / f"{frame_name}.txt"

            shutil.copy(str(frame_path), str(out_img))
            with open(out_txt, "w") as f:
                f.write("\n".join(label_lines))

    print(f"\n✅ YOLO train/val dataset written to: {output_dir}")


def get_frames_without_clock(input_dir, clock_model_path, conf_threshold=0.15):
    input_dir = Path(input_dir)
    clock_model = YOLO(clock_model_path)
    frame_paths = sorted(list(Path(input_dir/"images/train/").glob("*.png"))+list(Path(input_dir/"images/labels/").glob("*.png")), key=lambda p: int(p.stem.split("_")[-1]))

    no_clock_frames = []

    for frame_path in tqdm(frame_paths, desc="Checking for frames without clocks"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        result = clock_model.predict(frame, conf=conf_threshold, imgsz=568, device='mps', verbose=False)[0]
        has_clock = False
        for j in range(len(result.boxes)):
            cls = int(result.boxes.cls[j].item())
            class_name = clock_model.model.names[cls]
            if "enemy" in class_name.lower():
                has_clock = True
                break

        if not has_clock:
            print(frame_path)
            no_clock_frames.append(frame_path.name)

    print("\n🟨 Frames with no clock detection at ≥15% confidence:")
    for f in no_clock_frames:
        print(f" {f},",end="")
    print()
    return no_clock_frames

def get_frames_without_clock_and_delete(input_dir, clock_model_path, conf_threshold=0.5):
    input_dir = Path(input_dir)
    clock_model = YOLO(clock_model_path)
    image_dir = input_dir / "images/train"
    label_dir = input_dir / "labels/train"
    frame_paths = sorted(list(image_dir.glob("*.png")), key=lambda p: int(p.stem.split("_")[-1]))

    no_clock_frames = []

    for frame_path in tqdm(frame_paths, desc="Checking for and removing frames without clocks"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        result = clock_model.predict(frame, conf=conf_threshold, imgsz=896, device='mps', verbose=False)[0]
        has_clock = False
        for j in range(len(result.boxes)):
            cls = int(result.boxes.cls[j].item())
            class_name = clock_model.model.names[cls]
            if "enemy" in class_name.lower():
                has_clock = True
                break

        if not has_clock:
            print(frame_path)
            no_clock_frames.append(frame_path.name)
            continue
            # Delete image
            frame_path.unlink(missing_ok=True)

            # Delete corresponding label
            label_path = label_dir / f"{frame_path.stem}.txt"
            label_path.unlink(missing_ok=True)

    print(f"\n🗑️ Removed {len(no_clock_frames)} frames with no clock detected (≥{int(conf_threshold*100)}% conf):")
    for f in no_clock_frames:
        print(f" - {f}")

import shutil
import random

def copy_half_yolo_dataset_subset(yolo_base_dir: Path, split="train", suffix="_sampled"):
    img_dir = yolo_base_dir / "images" / split
    lbl_dir = yolo_base_dir / "labels" / split

    sampled_img_dir = yolo_base_dir / "images" / f"{split}{suffix}"
    sampled_lbl_dir = yolo_base_dir / "labels" / f"{split}{suffix}"

    sampled_img_dir.mkdir(parents=True, exist_ok=True)
    sampled_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    random.shuffle(image_files)
    selected_images = image_files[:len(image_files) // 2]

    for img_path in tqdm(selected_images, desc=f"Copying 50% of {split} set"):
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        if not lbl_path.exists():
            continue  # Skip if label doesn't exist

        shutil.copy(img_path, sampled_img_dir / img_path.name)
        shutil.copy(lbl_path, sampled_lbl_dir / lbl_path.name)

    print(f"\n✅ Sampled YOLO subset written to:")
    print(f"   Images → {sampled_img_dir}")
    print(f"   Labels → {sampled_lbl_dir}")

def print_filenames_with_detections(input_dir, model_path, conf_threshold=0.5, device='cpu'):
    """
    Prints the filename for each image in input_dir if an 'enemy' detection is found.

    Args:
        input_dir (str or Path): Folder containing image files (.jpg or .png).
        model_path (str or Path): Path to a trained YOLO model (.pt).
        conf_threshold (float): Minimum confidence for a detection to count.
        device (str): 'cpu', 'mps', or 'cuda' depending on hardware.
    """
    input_dir = Path(input_dir)
    model_path = Path(model_path)

    model = YOLO(model_path)
    frame_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")),
                         key=lambda p: int(p.stem.split("_")[-1]))
    count=0
    for frame_path in tqdm(frame_paths, desc="Scanning images"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        result = model.predict(frame, conf=conf_threshold, imgsz=896, device=device, verbose=False)[0]
        detection=False
        for j in range(len(result.boxes)):
            cls = int(result.boxes.cls[j].item())
            class_name = model.model.names[cls]
            if "enemy" in class_name.lower():
                print(frame_path.name)
                detection=True
                count+=1
                print(count)
                break  # Only print once per image
        if not detection:
            count=0

            
        

if __name__ == "__main__":
    input_dir = BASE_DIR / "frames"
    output_dir = BASE_DIR / "some-clocks"
    clock_model_path = BASE_DIR / "models/clock/last.pt"
    #get_frames_without_clock_and_delete(BASE_DIR / "clock-dataset-jul30", clock_model_path)
    print_filenames_with_detections(input_dir, clock_model_path)
    #copy_half_yolo_dataset_subset(output_dir, "val")