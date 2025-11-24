# synth_place_clocks.py
from pathlib import Path
import random, uuid, csv
from typing import Tuple
from PIL import Image, ImageFile
import numpy as np

from global_stuff.constants import BASE_DIR

# =========================
# CONFIG - set these paths
# =========================
ARENA_DIR      = BASE_DIR / "non-troop-dataset"/"images/val"     # folder of arena / live gameplay frames (RGB)
CLOCK_DIR      = BASE_DIR / "clock-segments"     # first dir (filenames like "0_*.png", "1_*.png", etc.)
BLUE_CLOCK_DIR = BASE_DIR /"segments"/"clock"       # second dir (only use files matching "clock_0_*.png", label them as 2)
OUTPUT_DIR     = BASE_DIR / "clock-validation-synth"   # where to save cropped composites + labels.csv

# per arena image: 10 from CLOCK_DIR, 1 from BLUE_CLOCK_DIR (label 2)
NUM_GENERAL_PER_ARENA = 2
NUM_BLUE_PER_ARENA    = 1

# crop padding around the visible clock (in pixels, computed as a fraction of the placed clock size)
PADDING_FRAC = 0.05  # 25% of max(w,h). Increase for more arena context.

# reproducibility
SEED = 1337
random.seed(SEED)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# helpers
# =========================
def list_images(folder: Path, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

def ensure_output():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_csv = OUTPUT_DIR / "labels.csv"
    if not labels_csv.exists():
        with labels_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "label"])
    return labels_csv

def parse_label_from_filename(fname: str) -> str:
    """
    For CLOCK_DIR files like '0_abcd1234.png' → '0'
    If not parsable, returns None.
    """
    stem = Path(fname).stem
    token = stem.split("_")[0]
    return token if token.isdigit() and token in ["0", "1", "2"] else None

def random_center(w: int, h: int) -> Tuple[int, int]:
    """Uniformly sample a center anywhere in the canvas."""
    cx = random.randint(0, max(0, w - 1))
    cy = random.randint(0, max(0, h - 1))
    return cx, cy

def place_patch_and_crop(arena_rgb: Image.Image, patch_rgba: Image.Image, center_xy=None, pad_frac=PADDING_FRAC):
    """
    Paste patch (with alpha) at random center (or provided), allow partial off-canvas,
    then crop a bounding box around the visible (non-transparent) pixels with padding.
    Returns (cropped_rgba or None).
    """
    W, H = arena_rgb.size
    patch_rgba = patch_rgba.convert("RGBA")
    pw, ph = patch_rgba.size

    # choose a random center if not provided
    if center_xy is None:
        cx, cy = random_center(W, H)
    else:
        cx, cy = center_xy

    # initial top-left where we'd place the full patch
    x0 = int(round(cx - pw / 2.0))
    y0 = int(round(cy - ph / 2.0))
    x1 = x0 + pw
    y1 = y0 + ph

    # intersection with arena bounds
    ix0 = max(0, x0); iy0 = max(0, y0)
    ix1 = min(W, x1); iy1 = min(H, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return None  # fully off-canvas, skip

    # crop the patch & mask to the intersection
    px0 = ix0 - x0; py0 = iy0 - y0
    px1 = px0 + (ix1 - ix0); py1 = py0 + (iy1 - iy0)
    visible_patch = patch_rgba.crop((px0, py0, px1, py1))  # RGBA
    mask = visible_patch.split()[3]  # alpha

    # compute tight bbox of non-zero alpha inside visible_patch
    mask_np = np.array(mask, dtype=np.uint8)
    ys, xs = np.where(mask_np > 0)
    if xs.size == 0:
        return None  # no visible pixels after clipping

    vx0, vx1 = int(xs.min()), int(xs.max()) + 1
    vy0, vy1 = int(ys.min()), int(ys.max()) + 1

    # paste onto a copy of arena to produce composite
    composite = arena_rgb.convert("RGBA").copy()
    composite.paste(visible_patch, (ix0, iy0), mask=mask)

    # bbox in arena coordinates for the visible alpha region
    bx0 = ix0 + vx0; by0 = iy0 + vy0
    bx1 = ix0 + vx1; by1 = iy0 + vy1

    # padding
    bw = bx1 - bx0; bh = by1 - by0
    pad = int(round(pad_frac * max(bw, bh)))
    cx0 = max(0, bx0 - pad); cy0 = max(0, by0 - pad)
    cx1 = min(W, bx1 + pad); cy1 = min(H, by1 + pad)

    if cx0 >= cx1 or cy0 >= cy1:
        return None

    crop_rgba = composite.crop((cx0, cy0, cx1, cy1))
    return crop_rgba

def save_patch_and_label(img_rgba: Image.Image, label: str, labels_csv: Path):
    uid = uuid.uuid4().hex[:8]
    out_name = f"{label}_{uid}.png"
    out_path = OUTPUT_DIR / out_name
    img_rgba.save(out_path)
    with labels_csv.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([out_name, label])

# =========================
# main
# =========================
def main():
    labels_csv = ensure_output()

    # gather files
    arenas = list_images(ARENA_DIR)
    general = list_images(CLOCK_DIR)
    blue    = [p for p in list_images(BLUE_CLOCK_DIR) if p.stem.startswith("clock_0_")]

    if not arenas:
        print(f"[ERR] No arena images in {ARENA_DIR}"); return
    if not general:
        print(f"[ERR] No clock patches in {CLOCK_DIR}"); return
    if not blue:
        print(f"[WARN] No blue 'clock_0_*' patches found in {BLUE_CLOCK_DIR} (label 2).")

    print(f"[INFO] arenas={len(arenas)} | general_clocks={len(general)} | blue_clocks={len(blue)}")
    print(f"[INFO] output -> {OUTPUT_DIR}")

    for arena_path in arenas:
        # load arena once
        try:
            arena = Image.open(arena_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] skip arena {arena_path.name}: {e}")
            continue

        # choose clocks for this arena
        # general: sample with replacement if fewer than needed
        chosen_general = [random.choice(general) for _ in range(NUM_GENERAL_PER_ARENA)] \
                         if len(general) < NUM_GENERAL_PER_ARENA \
                         else random.sample(general, NUM_GENERAL_PER_ARENA)

        # blue: at most NUM_BLUE_PER_ARENA (label=2). If none found, skip that part silently.
        chosen_blue = []
        if blue and NUM_BLUE_PER_ARENA > 0:
            chosen_blue = [random.choice(blue) for _ in range(NUM_BLUE_PER_ARENA)] \
                          if len(blue) < NUM_BLUE_PER_ARENA \
                          else random.sample(blue, NUM_BLUE_PER_ARENA)

        # place and crop general clocks
        for patch_path in chosen_general:
            try:
                patch = Image.open(patch_path).convert("RGBA")
            except Exception as e:
                print(f"[WARN] bad patch {patch_path.name}: {e}")
                continue

            crop = place_patch_and_crop(arena, patch, pad_frac=PADDING_FRAC)
            if crop is None:
                # try a couple more centers before giving up
                tries = 3
                ok = False
                for _ in range(tries):
                    crop = place_patch_and_crop(arena, patch, pad_frac=PADDING_FRAC)
                    if crop is not None:
                        ok = True; break
                if not ok:
                    continue

            label = parse_label_from_filename(patch_path.name)  # from '0_*.png', '1_*.png', etc.
            if label is None:
                print(f"[WARN] skipping {patch_path.name} - no valid label (0, 1, 2)")
                continue
            save_patch_and_label(crop, label, labels_csv)

        # place and crop blue clocks (label=2)
        for patch_path in chosen_blue:
            try:
                patch = Image.open(patch_path).convert("RGBA")
            except Exception as e:
                print(f"[WARN] bad blue patch {patch_path.name}: {e}")
                continue

            crop = place_patch_and_crop(arena, patch, pad_frac=PADDING_FRAC)
            if crop is None:
                tries = 3
                ok = False
                for _ in range(tries):
                    crop = place_patch_and_crop(arena, patch, pad_frac=PADDING_FRAC)
                    if crop is not None:
                        ok = True; break
                if not ok:
                    continue

            save_patch_and_label(crop, "2", labels_csv)  # force label=2 for blue clocks

    print("\n✅ Done.")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
