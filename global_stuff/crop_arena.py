from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os

# === Arena crop rules from constant.py ===
CROP_RULES = {
    '2.16': (0.021, 0.073, 0.960, 0.700),
    '2.22': (0.020, 0.070, 0.960, 0.690),
    '2.13': (0.026, 0.064, 0.960, 0.73),
    '1.78': (0.078, 0.023, 0.939-0.078, 0.777-0.023),  # 16:9
    '1.43': (0.156, 0.027, 0.848-0.156, 0.77-0.027),  # 4:3
    #CHANGE X, Y DEPENDING ON VIDEO EDITING (USUALLY KEEP WIDTH, HEIGHT SAME):
    '0.56': (0.08, 0.02, 0.335, 0.965)
    #'0.56': (0.568, 0.025, 0.352, 0.965) #NORMAL IAN ONE
    #'0.56': (0.5, 0.032, 0.792-0.5, 0.786-0.032), #Tag video
    #'0.56': (0.073, 0.024, 0.28, 0.746) # KEN VIDEO
}

def crop_arena(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ratio = round(h / w, 2)
    if 2.16 <= ratio <= 2.17:
        x, y, w_ratio, h_ratio = CROP_RULES['2.16']
    elif 2.22 <= ratio <= 2.23:
        x, y, w_ratio, h_ratio = CROP_RULES['2.22']
    elif 2.13 <= ratio <= 2.14:
        x, y, w_ratio, h_ratio = CROP_RULES['2.13']
    elif 1.77 <= ratio <= 1.81:
        x, y, w_ratio, h_ratio = CROP_RULES['1.78']
    elif 1.42 <= ratio <= 1.44:
        x, y, w_ratio, h_ratio = CROP_RULES['1.43']
    elif 0.55 <= ratio <= 0.57:
        x, y, w_ratio, h_ratio = CROP_RULES['0.56']
    else:
        raise ValueError(f"Unsupported frame ratio: {ratio:.3f} for image shape {w}x{h}")

    x0 = int(x * w)
    y0 = int(y * h)
    x1 = int((x + w_ratio) * w)
    y1 = int((y + h_ratio) * h)
    return img[y0:y1, x0:x1]

def crop_folder(input_dir, output_dir, resize_to=(568, 896)):
    input_dir = Path(input_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.png"))

    for img_path in tqdm(files, desc="Cropping arena frames"):
        img = cv2.imread(str(img_path))
        
        try:
            h0, w0 = img.shape[:2]
            cropped = crop_arena(img)
            h1, w1 = cropped.shape[:2]

            resized = cv2.resize(cropped, resize_to, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(output_dir / img_path.name), resized)
        except Exception as e:
            print(f"⚠️ Skipped {img_path.name}: {e}")

    print(f"✅ Done. Cropped frames saved to: {output_dir}")

def crop_single_image(img, save_path=None, resize_to=(568, 896)):
    """
    Crop a single Clash Royale frame image using arena crop rules.
    
    Args:
        img_path (str or Path): Path to the input image.
        save_path (str or Path, optional): If provided, saves the result to this path.
        resize_to (tuple): Target size for resizing (width, height).
    
    Returns:
        cropped_resized (np.ndarray): Cropped and resized image as NumPy array.
    """


    cropped = crop_arena(img)
    cropped_resized = cv2.resize(cropped, resize_to, interpolation=cv2.INTER_AREA)

    if save_path:
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cropped_resized)

    return cropped_resized
