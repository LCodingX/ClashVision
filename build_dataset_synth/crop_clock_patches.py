import argparse
from pathlib import Path
import random, uuid, csv
from PIL import Image, ImageFile
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from global_stuff.constants import BASE_DIR

# ---------- Args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--ratio", type=float, default=0.10, help="Fraction of files to process")
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

# ---------- Paths ----------
INPUT_ROOT = BASE_DIR / "clock-validation"
OUTPUT_DIR = BASE_DIR / "clock-segments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_CSV = INPUT_ROOT / "labels.csv"
filename_to_label = {}
if LABELS_CSV.exists():
    with open(LABELS_CSV, "r") as f:
        for row in csv.DictReader(f):
            fn = (row.get("filename") or "").strip()
            lab = (row.get("label") or "").strip()
            if fn:
                filename_to_label[fn] = lab if lab in {"0","1","2"} else "unknown"

# ---------- Device / dtype ----------
USE_MPS = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
DEVICE = torch.device("mps" if USE_MPS else "cpu")
torch.set_default_dtype(torch.float32)
print(f"[INFO] device: {DEVICE}")

# ---------- Files ----------
EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
files = [p for p in INPUT_ROOT.rglob("*") if p.suffix.lower() in EXTS]
if not files:
    print("No images found."); raise SystemExit(0)
random.seed(args.seed)
k = max(1, int(len(files) * args.ratio))
sampled = random.sample(files, k)
print(f"[INFO] total {len(files)} | processing {k} ({args.ratio:.0%})")

# ---------- SAM (Predictor, not Automatic) ----------
SAM_CHECKPOINT = BASE_DIR / "models" / "sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
sam = sam.to(device=DEVICE, dtype=torch.float32).eval()
predictor = SamPredictor(sam)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pick_mask(image, predictor):
    """
    Prompt with 1 positive (center) and 8 negatives along the borders.
    Return best mask biased toward center blob (not background).
    """
    H, W = image.shape[:2]
    predictor.set_image(image)

    # Positive at center
    cx, cy = W / 2.0, H / 2.0
    pos = np.array([[cx, cy]], dtype=np.float32)

    # Negatives: 4 corners + midpoints of 4 edges (pull mask away from borders)
    e = 3.0  # a few px in from edges
    neg = np.array([
        [e, e], [W-e, e], [e, H-e], [W-e, H-e],         # corners
        [W/2.0, e], [W/2.0, H-e], [e, H/2.0], [W-e, H/2.0]  # edge midpoints
    ], dtype=np.float32)

    pts = np.vstack([pos, neg])
    lbs = np.concatenate([np.ones(1, np.int32), np.zeros(len(neg), np.int32)])

    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbs,
        multimask_output=True
    )

    # Filter candidates: must include center, avoid giant "whole image" masks
    area = masks.reshape(masks.shape[0], -1).sum(axis=1)
    center_ok = masks[:, int(cy), int(cx)] > 0
    not_huge = area < 0.8 * (H * W)
    candidates = np.where(center_ok & not_huge)[0]

    idx = int(candidates[np.argmax(scores[candidates])]) if len(candidates) else int(np.argmax(scores))
    m = masks[idx].astype(np.uint8)

    # Keep only the connected component that contains the center
    # (simple flood-fill via mask of pixels connected to center)
    from collections import deque
    if m[int(cy), int(cx)]:
        visited = np.zeros_like(m, dtype=bool)
        q = deque([(int(cy), int(cx))])
        visited[int(cy), int(cx)] = True
        m_keep = np.zeros_like(m, dtype=np.uint8)
        while q:
            y, x = q.popleft()
            m_keep[y, x] = 1
            for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=ny<H and 0<=nx<W and not visited[ny,nx] and m[ny,nx]:
                    visited[ny,nx] = True
                    q.append((ny,nx))
        m = m_keep

    return m

def tight_crop_rgba(image, mask, pad=2):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = max(0, xs.min()-pad), min(image.shape[1], xs.max()+1+pad)
    y0, y1 = max(0, ys.min()-pad), min(image.shape[0], ys.max()+1+pad)
    crop = image[y0:y1, x0:x1]
    alpha = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    return np.dstack([crop, alpha])

# ---------- Run ----------
for img_path in tqdm(sampled, desc="Segmenting clocks"):
    try:
        img = np.array(Image.open(img_path).convert("RGB"))
    except Exception as e:
        print(f"[WARN] {img_path.name}: {e}"); continue

    try:
        mask = pick_mask(img, predictor)
    except Exception as e:
        print(f"[WARN] SAM failed on {img_path.name}: {e}"); continue

    rgba = tight_crop_rgba(img, mask, pad=2)
    if rgba is None:
        continue

    label = filename_to_label.get(img_path.name, "unknown")
    uid = uuid.uuid4().hex[:8]
    out_path = OUTPUT_DIR / f"{label}_{uid}.png"
    try:
        Image.fromarray(rgba).save(out_path)
    except Exception as e:
        print(f"[WARN] save failed {out_path.name}: {e}")

print(f"\n✅ Done. Cropped PNGs in: {OUTPUT_DIR.resolve()}")
