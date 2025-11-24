import argparse
import cv2
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import time


CONFIDENCE_THRESHOLD = 0.5

def is_same_clock(boxA, boxB, max_center_distance=20):
    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2
    dist = ((cxA - cxB) ** 2 + (cyA - cyB) ** 2) ** 0.5
    return dist <= max_center_distance

def build_clock_dataset(input_dir, output_dir, clock_model_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_idx = 0

    clock_model = YOLO(clock_model_path)
    frame_paths = sorted(input_dir.glob("*.jpg"), key=lambda p: int(p.stem.split("_")[-1]))

    clocks_active = {}
    clock_standard_area = None

    for frame_path in tqdm(frame_paths[1000:]):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️ Failed to load frame: {frame_path}")
            continue
        h, w = frame.shape[:2]

        results = clock_model.predict(frame, conf=CONFIDENCE_THRESHOLD, imgsz=896, device='cuda', verbose=False)[0]

        active_before = [False for _ in range(len(results.boxes))]
        for xyxy_active in list(clocks_active.keys()):
            still_active = False
            for i, detection in enumerate(results.boxes):
                xyxy = tuple(*detection.xyxy.int().tolist())
                if is_same_clock(xyxy_active, xyxy):
                    active_before[i] = True
                    still_active = True
                    clocks_active[xyxy_active][0] += 1
                    clocks_active[xyxy_active][1] = 0
                    if clocks_active[xyxy_active][0] in [5, 10, 20]:
                        if clocks_active[xyxy_active][0]==3:
                            print(f"patch found at {frame_path}")
                        x1 = max(0, xyxy[0] - 75)
                        x2 = min(w, xyxy[2] + 75)
                        y1 = max(0, xyxy[1] - 150)
                        y2 = min(h, xyxy[3])
                        patch = frame[y1:y2, x1:x2]
                        cv2.imwrite(str(output_dir / f"{output_idx}.png"), patch)
                        output_idx += 1
                    break

            if not still_active:
                clocks_active[xyxy_active][1] += 1
                if clocks_active[xyxy_active][1] == 7:
                    del clocks_active[xyxy_active]

        for i in range(len(results.boxes)):
            if not active_before[i]:
                x1, y1, x2, y2 = tuple(*results.boxes[i].xyxy.int().tolist())
                area = (x2 - x1) * (y2 - y1)
                if clock_standard_area is None:
                    clock_standard_area = area
                rel_diff = abs(area - clock_standard_area) / clock_standard_area
                if rel_diff < 0.3:
                    clocks_active[tuple(*results.boxes[i].xyxy.int().tolist())] = [1, 0]

    print(f"\nYOLO train/val dataset written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input frames directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output patch directory")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model file")

    args = parser.parse_args()

    build_clock_dataset(args.input, args.output, args.model)
