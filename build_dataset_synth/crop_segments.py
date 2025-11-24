import json, uuid, argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from global_stuff.constants import BASE_DIR

# === Args ===
parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, help='Name of the json file in live_annotations')
parser.add_argument('--frames', type=str, help='Name of the folder containing annotated frames')
args = parser.parse_args()

json_file = args.json
frames_folder = args.frames
if not json_file or not frames_folder:
    raise ValueError("Please provide both --json and --frames arguments.")

# === Your Paths ===
VIA_JSON_PATH = BASE_DIR / json_file
IMAGE_DIR = BASE_DIR / frames_folder
OUTPUT_ROOT = BASE_DIR / "segment-log-ally"
SAM_CHECKPOINT = BASE_DIR / "models/sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Init SAM ===
sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# === Load Annotations ===
with open(VIA_JSON_PATH, 'r') as f:
    annotations = json.load(f)

# === Process Frames with tqdm
for entry in tqdm(annotations.values(), desc="🔍 Segmenting frames"):
    fname = entry["filename"]
    regions = entry["regions"]
    if not regions:
        continue

    img_path = IMAGE_DIR / fname
    if not img_path.exists():
        print(f"⚠️ Skipping missing image: {img_path}")
        continue

    image = np.array(Image.open(img_path).convert("RGB"))
    predictor.set_image(image)

    for region in regions:
        shape = region["shape_attributes"]
        troop = region["region_attributes"].get("troop-id", "unknown")
        x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
        box = np.array([x, y, x + w, y + h])

        masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)
        mask = masks[0]

        # === Crop and apply mask ===
        x0, y0, x1, y1 = box.astype(int)
        crop = image[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]
        alpha = (mask_crop * 255).astype(np.uint8)
        rgba = np.dstack([crop, alpha])

        # === Save as RGBA PNG ===
        if troop.endswith("-ally"):
            troop = troop[:-5]
            troop_dir = OUTPUT_ROOT / troop
            troop_dir.mkdir(parents=True, exist_ok=True)
            uid = uuid.uuid4().hex[:8]
            out_path = troop_dir / f"{troop}_0_{uid}.png"
        elif troop.endswith("-enemy"):
            troop = troop[:-6]
            troop_dir = OUTPUT_ROOT / troop
            troop_dir.mkdir(parents=True, exist_ok=True)
            uid = uuid.uuid4().hex[:8]
            out_path = troop_dir / f"{troop}_1_{uid}.png"
        else:
            troop_dir = OUTPUT_ROOT / troop
            troop_dir.mkdir(parents=True, exist_ok=True)
            uid = uuid.uuid4().hex[:8]
            out_path = troop_dir / f"{troop}_0_{uid}.png"

        img_out = Image.fromarray(rgba)
        img_out.save(out_path)

print("✅ All RGBA troop cutouts saved to:", OUTPUT_ROOT)
