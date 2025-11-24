import cv2
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from global_stuff.constants import BASE_DIR
import time
import torch
torch.mps.empty_cache()
# === CONFIGURATION ===
CONFIDENCE_THRESHOLD = 0.7

def is_same_clock(boxA, boxB, max_center_distance=20):
    # Get center points
    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2

    # Euclidean distance
    dist = ((cxA - cxB) ** 2 + (cyA - cyB) ** 2) ** 0.5
    return dist <= max_center_distance



def build_clock_dataset(input_dir, output_dir, clock_model_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_idx = 0

    clock_model = YOLO(clock_model_path)
    frame_paths = sorted(input_dir.glob("*.jpg"), key=lambda p: int(p.stem.split("_")[-1]))

    clocks_active = {} #(int, int, int, int): (int, int)

    clock_standard_area = None

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️ Failed to load frame: {frame_path}")
            continue
        h, w = frame.shape[:2]

        results = clock_model.predict(frame, conf=CONFIDENCE_THRESHOLD, imgsz=896, device='cpu', verbose=False)[0]

        max_confidence=0
        active_before = [False for _ in range(len(results.boxes))]
        for xyxy_active in list(clocks_active.keys()):
            print(f"xyxy active {xyxy_active}")
            still_active=False
            for i,detection in enumerate(results.boxes):
                xyxy = tuple(*detection.xyxy.int().tolist())

                if is_same_clock(xyxy_active, xyxy):
                    active_before[i]=True
                    still_active=True
                    clocks_active[xyxy_active][0]+=1
                    clocks_active[xyxy_active][1]=0
                    if clocks_active[xyxy_active][0] in [3, 10, 20]:
                        #print(output_idx, frame_path)
                        x1 = max(0, xyxy[0]-30)
                        x2 = min(w, xyxy[2]+30)
                        y1 = max(0, xyxy[1]-150)
                        y2 = min(h, xyxy[3])

                        patch = frame[y1:y2, x1:x2]

                        # Draw clock bbox in patch-relative coordinates
                        #rel_x1 = xyxy[0] - x1
                        #rel_y1 = xyxy[1] - y1
                        #rel_x2 = xyxy[2] - x1
                        #rel_y2 = xyxy[3] - y1

                        #cv2.rectangle(
                        #    patch,
                        #    (rel_x1, rel_y1),
                        #    (rel_x2, rel_y2),
                        #    color=(0, 255, 0),  # green box
                        #    thickness=2
                        #)

                        cv2.imwrite(str(output_dir / f"{output_idx}.png"), patch)
                        output_idx += 1
                    break # break inner loop

            if not still_active:
                print(f"not active at {frame_path}")
                clocks_active[xyxy_active][1]+=1
                if clocks_active[xyxy_active][1]==3:
                    del clocks_active[xyxy_active]
        for i in range(len(results.boxes)):
            if not active_before[i]:
                x1, y1, x2, y2 = tuple(*results.boxes[i].xyxy.int().tolist())
                area = (x2 - x1) * (y2 - y1)
                if clock_standard_area is None:
                    clock_standard_area = area
                rel_diff = abs(area-clock_standard_area) / clock_standard_area
                if rel_diff< 0.3:
                    clocks_active[tuple(*results[i].boxes.xyxy.int().tolist())] = [1,0]

    print(f"\n✅ YOLO train/val dataset written to: {output_dir}")

build_clock_dataset(BASE_DIR / "frames", BASE_DIR / "patches", BASE_DIR / "models/clock/last.pt")