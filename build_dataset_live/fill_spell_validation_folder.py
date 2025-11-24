from pathlib import Path
from PIL import Image

from global_stuff.constants import BASE_DIR

IM_DIR = BASE_DIR / "human_annotated_yolo/spells-live/images/train"
LBL_DIR = BASE_DIR / "human_annotated_yolo/spells-live/labels/train"
OUT_DIR = BASE_DIR / "human_annotated_yolo/goblin-curse-validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOBLIN_CURSE_CLASS = "12"

def yolo_to_xyxy(xc, yc, w, h, W, H):
    """Convert normalized YOLO (xc,yc,w,h) to integer pixel (x1,y1,x2,y2), clamped to image bounds."""
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H

    # clamp & convert to int
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W,     int(round(x2))))
    y2 = max(0, min(H,     int(round(y2))))
    # ensure proper ordering
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2

def process_one(image_path: Path):
    label_path = LBL_DIR / (image_path.stem + ".txt")
    if not label_path.exists():
        return 0

    img = Image.open(image_path).convert("RGBA") if image_path.suffix.lower() == ".png" else Image.open(image_path).convert("RGB")
    W, H = img.size

    saved = 0
    with label_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            if cls != GOBLIN_CURSE_CLASS:
                continue

            try:
                xc, yc, w, h = map(float, (xc, yc, w, h))
            except ValueError:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
            patch = img.crop((x1, y1, x2, y2))

            out_name = f"{image_path.stem}_gc_{saved:03d}.png"
            patch.save(OUT_DIR / out_name)
            saved += 1
    return saved

def main():
    total = 0
    for p in IM_DIR.iterdir():
        if not p.is_file() or p.suffix.lower() != ".jpg":
            continue
        total += process_one(p)
    print(f"Saved {total} goblin-curse patches to: {OUT_DIR}")

if __name__ == "__main__":
    main()
