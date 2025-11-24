from PIL import Image
from pathlib import Path
import numpy as np
import random
from global_stuff.constants import BASE_DIR, towers_bottom_center_grid_position
from build_dataset_synth.helpers import get_seed, cell_to_pixel

IMG_WIDTH, IMG_HEIGHT = 568, 896
NUM_TRAIN = 1000
NUM_VAL = 200
CLOCK_CLASSES = ["clock-ally", "clock-enemy"]

OUT_PATH = BASE_DIR / "new-clock-dataset"
IMAGE_DIR = OUT_PATH / "images"
LABEL_DIR = OUT_PATH / "labels"

for split in ["train", "val"]:
    (IMAGE_DIR / split).mkdir(parents=True, exist_ok=True)
    (LABEL_DIR / split).mkdir(parents=True, exist_ok=True)

def get_background():
    idx = random.randint(1, 27)
    path = BASE_DIR / "segments/backgrounds" / f"background{idx:02}.jpg"
    assert path.exists()
    img = Image.open(path).convert("RGBA")
    assert img.size == (IMG_WIDTH, IMG_HEIGHT)
    return img

def paste_tower(base, name, key):
    team = 0 if '0' in key else 1
    tower_dir = BASE_DIR / f"segments/{name}"
    sprite_path = random.choice(list(tower_dir.glob(f"{name}_{team}_*.png")))
    sprite = Image.open(sprite_path).convert("RGBA")

    x, y = towers_bottom_center_grid_position[key]
    cx, by = cell_to_pixel((x, y))
    top_left = (int(cx - sprite.width // 2), int(by - sprite.height))
    base.paste(sprite, top_left, sprite)

def paste_clock(base, team):
    clock_path = random.choice(list((BASE_DIR / "segments/clock").glob(f"clock_{team}_*.png")))
    sprite = Image.open(clock_path).convert("RGBA")

    # Random location in arena
    grid_size = (18, 32)  # tiles wide, tiles tall

    if random.random() < 0.15:  # 15% chance to be clipped at border
        edge = random.choice(["top", "bottom", "left", "right"])
        margin = 0.2  # how far to go outside the border
        if edge == "top":
            x_cell = random.uniform(0, grid_size[0])
            y_cell = random.uniform(-grid_size[1] * margin, 1.5)
        elif edge == "bottom":
            x_cell = random.uniform(0, grid_size[0])
            y_cell = random.uniform(grid_size[1] - 1.5, grid_size[1] * (1 + margin))
        elif edge == "left":
            x_cell = random.uniform(-grid_size[0] * margin, 1.5)
            y_cell = random.uniform(0, grid_size[1])
        elif edge == "right":
            x_cell = random.uniform(grid_size[0] - 1.5, grid_size[0] * (1 + margin))
            y_cell = random.uniform(0, grid_size[1])
    else:
        x_cell = random.uniform(0, grid_size[0])
        y_cell = random.uniform(0, grid_size[1])

    cx, by = cell_to_pixel((x_cell, y_cell))
    top_left = (int(cx - sprite.width // 2), int(by - sprite.height))
    base.paste(sprite, top_left, sprite)

    x_center = max(0.0, min(cx / IMG_WIDTH, 1.0))
    y_center = max(0.0, min((by - sprite.height / 2) / IMG_HEIGHT, 1.0))
    width = sprite.width / IMG_WIDTH
    height = sprite.height / IMG_HEIGHT
    return f"{team - 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def generate(split, idx):
    seed = get_seed()
    random.seed(seed)
    np.random.seed(seed)

    base = get_background()
    labels = []

    # Add towers
    for key in ["king-tower0", "king-tower1", "queen0_0", "queen0_1", "queen1_0", "queen1_1"]:
        name = "king-tower" if "king" in key else random.choice([
            "dagger-duchess-tower", "cannoneer-tower", "princess-tower"
        ])
        paste_tower(base, name, key)

    # Add clocks
    for team in [0, 1]:
        for _ in range(random.randint(1, 3)):
            if team==1:
                labels.append(paste_clock(base, team))
            else:
                paste_clock(base, team)

    img_path = IMAGE_DIR / split / f"{idx:06}.jpg"
    txt_path = LABEL_DIR / split / f"{idx:06}.txt"
    base.convert("RGB").save(img_path)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(labels))

for i in range(NUM_TRAIN):
    generate("train", i)
for i in range(NUM_VAL):
    generate("val", i)
