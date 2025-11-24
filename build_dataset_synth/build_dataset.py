from PIL import Image
from pathlib import Path
import sys
import numpy as np
import time
import random
from tqdm import tqdm
from build_dataset_synth.helpers import cell_to_pixel, pixel_to_cell, get_seed, get_random_valid_xy
from global_stuff.constants import (
  small_troops, medium_troops, large_troops, buildings, id_small_troops, id_medium_troops, id_large_troops, id_buildings,
  ground_spots, flying_unit_list, swarm_troops, towers_bottom_center_grid_position, BASE_DIR
  )


class Unit:
    def __init__(self, name, path, x, y, level, team):
        self.name = name
        self.path = path
        self.x = x #cell
        self.y = y #cell
        self.team = team
        self.level = level #1=ground, 2=flying
    def get_path(self):
        return self.path
    def __str__(self):
        return f"Unit {self.name} at x {self.x} y {self.y} at level {self.level} from path {self.path}"

class Tower(Unit):
    # name is king-tower, princess-tower, dagger-duchess-tower", "cannoneer-tower", or "royal-chef-tower"
    def __init__(self, name, path, team):#team 0 or 1
        if name != "king-tower":
            x = towers_bottom_center_grid_position[f"queen{team}_{0 if name.endswith('left') else 1}"][0]
            y = towers_bottom_center_grid_position[f"queen{team}_{0 if name.endswith('left') else 1}"][1]
        else:
            x = towers_bottom_center_grid_position[f"{name}{team}"][0]
            y = towers_bottom_center_grid_position[f"{name}{team}"][1]
        super().__init__(name, path, x, y, 1, team)
    def __str__(self):
        return f"Tower {self.name} at x {self.x} y {self.y} from path {self.path}"

class Generator:
    def __init__(self, avail_names, img_path, label_path, troops_labelled, id_troops, n=0, weights=None):
        self.avail_names = avail_names
        if weights is None:
            self.weights = [1]*len(avail_names)
        else:
            self.weights = weights
        self.seed = get_seed()
        self.units = [] #list of units to be added to the image once draw() is called
        self.towers = []
        self.n = random.randint(1,n) #number of units to add to the image
        #initialize self.image as np.array of size (896, 568, 3)
        self.image = np.zeros((896, 568, 3), dtype=np.uint8)
        self.set_background()
        self.img_path = img_path
        self.label_path = label_path
        self.troops_labelled = troops_labelled
        self.id_troops = id_troops
        random.seed(self.seed)
        np.random.seed(self.seed)


    def set_background(self):
        """
        Set the background for the generator by selecting a random background file
        from background01 to background27 in the Clash-Royale-Detection-Dataset.
        """
        background_index = random.randint(1, 27)
        background_path = BASE_DIR / "segments/backgrounds" / f"background{background_index:02}.jpg"
        #assert background path is the proper size and exists
        assert background_path.exists(), f"Background file {background_path} does not exist."
        background_image = Image.open(background_path).convert("RGB")
        assert background_image.size == (568, 896), f"Background image {background_path} is not the correct size (568, 896)."
        if background_path.exists():
            self.image = np.array(Image.open(background_path))
        else:
            raise FileNotFoundError(f"Background file {background_path} does not exist.")

    def add_towers(self):
        king_path = Path(BASE_DIR, "segments/king-tower/")
        king_ally_path = random.choice(list(king_path.glob("king-tower_0_*.png")))
        king_enemy_path = random.choice(list(king_path.glob("king-tower_1_*.png")))
        self.towers.append(Tower("king-tower", king_ally_path, 0))
        self.towers.append(Tower("king-tower", king_enemy_path, 1))

        ally_tower_name = random.choice(["dagger-duchess-tower", "cannoneer-tower", "princess-tower"])
        enemy_tower_name = random.choice(["dagger-duchess-tower", "cannoneer-tower", "princess-tower"])
        queen_path_ally_left = random.choice(list(Path(BASE_DIR, \
            f"segments/{ally_tower_name}") \
            .glob(f"{ally_tower_name}_0_*.png")))
        queen_path_ally_right = random.choice(list(Path(BASE_DIR, \
            f"segments/{ally_tower_name}") \
            .glob(f"{ally_tower_name}_0_*.png")))    
        queen_path_enemy_left = random.choice(list(Path(BASE_DIR, \
            f"segments/{enemy_tower_name}")\
            .glob(f"{enemy_tower_name}_1_*.png")))
        queen_path_enemy_right = random.choice(list(Path(BASE_DIR, \
            f"segments/{enemy_tower_name}")\
            .glob(f"{enemy_tower_name}_1_*.png")))
        self.towers.append(Tower(ally_tower_name+"-left", queen_path_ally_left, 0))
        self.towers.append(Tower(ally_tower_name+"-right", queen_path_ally_right, 0))
        self.towers.append(Tower(ally_tower_name+"-left", queen_path_enemy_left, 1))
        self.towers.append(Tower(ally_tower_name+"-right", queen_path_enemy_right, 1))

    def add_units(self):
        unit_names = random.choices(self.avail_names, weights=self.weights, k=self.n)
        for unit_name in unit_names:
            team = 0 if random.random() < 0.5 else 1
            xy = get_random_valid_xy(
                self.units,
                2 if unit_name.split("-ally")[0].split("-enemy")[0] in flying_unit_list else 1,
                None if unit_name.startswith("goblin-drill") or unit_name.startswith("cannon-cart") else (team if unit_name in buildings else None)
            )
            troop_path = random.choice(list(Path(BASE_DIR, 
                f"segments/{unit_name.split('-enemy')[0].split('-ally')[0]}")
                .glob(f"{unit_name.split('-enemy')[0].split('-ally')[0]}_{team}_*.png")))
            
            # Add original unit
            unit = Unit(unit_name.split("-ally")[0].split("-enemy")[0], troop_path, xy[0], xy[1], 2 if unit_name.split("-ally")[0].split("-enemy")[0] in flying_unit_list else 1, team)
            self.units.append(unit)

            if unit.name in swarm_troops and (random.random() < 0.5):
                SWARM_UPPER_BOUND = 5
                if (unit.name == 'skeleton'): SWARM_UPPER_BOUND = 10
                num_swarm_units = random.randint(1,SWARM_UPPER_BOUND)
                for _ in range(num_swarm_units):
                    troop_path = random.choice(list(Path(BASE_DIR, 
                        f"segments/{unit.name}")
                        .glob(f"{unit.name}_{team}_*.png")))
                    dx = random.uniform(-1.5, 1.5)
                    dy = random.uniform(-2, 2)
                    swarm_unit = Unit(unit.name, troop_path, xy[0] + dx, xy[1] + dy, 1, team)
                    unit_works=True
                    for other_unit in self.units:
                        if abs(other_unit.x-swarm_unit.x)<=0.4 and abs(other_unit.y-swarm_unit.y)<=0.4 and other_unit.level == swarm_unit.level:
                            unit_works=False
                    if unit_works: self.units.append(swarm_unit)
        


    def draw(self):
        # Sort units for correct layering (ground first, upper units later)
        self.units += self.towers
        self.units = sorted(self.units, key=lambda unit: (unit.level, unit.y))

        # Convert base image (numpy array) to RGBA PIL Image
        base = Image.fromarray(self.image).convert("RGBA")
        labels = []

        for unit in self.units:
            sprite = Image.open(unit.path).convert("RGBA")
            alpha = sprite.getchannel("A")
            if random.random()<0.05 and not isinstance(unit, Tower):#25% chance of overlaying a red tint to replicate damage taking state
                # Create red tint (fully red but transparent outside unit shape)
                red_overlay = Image.new("RGBA", sprite.size, (255, 0, 0, 0))
                red_pixels = Image.new("RGBA", sprite.size, (255, 0, 0, 100))  # semi-transparent red
                red_overlay = Image.composite(red_pixels, red_overlay, alpha)
                sprite = Image.alpha_composite(sprite, red_overlay)
            elif random.random()<0.05 and not isinstance(unit,Tower) and "evolution" in unit.name:
                purple_overlay = Image.new("RGBA", sprite.size, (128, 0, 128, 0))
                purple_pixels = Image.new("RGBA", sprite.size, (128, 0, 128, 100))  # semi-transparent red
                purple_overlay = Image.composite(purple_pixels, purple_overlay, alpha)
                sprite = Image.alpha_composite(sprite, purple_overlay)
            elif random.random()<0.05 and not isinstance(unit,Tower) and "evolution" in unit.name:
                yellow_overlay = Image.new("RGBA", sprite.size, (255, 215, 0, 0))
                yellow_pixels = Image.new("RGBA", sprite.size, (255, 215, 0, 100))  # semi-transparent red
                yellow_overlay = Image.composite(yellow_pixels, yellow_overlay, alpha)
                sprite = Image.alpha_composite(sprite, yellow_overlay)
            elif random.random()<0.05 and not isinstance(unit,Tower):
                blue_overlay = Image.new("RGBA", sprite.size, (0, 0, 255, 0))
                blue_pixels = Image.new("RGBA", sprite.size, (0, 0, 255, 100))  # semi-transparent red
                blue_overlay = Image.composite(blue_pixels, blue_overlay, alpha)
                sprite = Image.alpha_composite(sprite, blue_overlay)

            # Convert unit (x, y) in cell space to pixel position
            center_x, bottom_y = cell_to_pixel((unit.x, unit.y))
            top_left_x = int(center_x - sprite.width // 2)
            top_left_y = int(bottom_y - sprite.height)
            #print(unit)



            # Check if the top-left corner is within bounds
            if not (-10 <= top_left_x < base.width and 0 <= top_left_y < base.height):
                #print(f"Skipping {unit.full_name}: out of bounds (x={top_left_x}, y={top_left_y})")
                continue

            # Paste sprite using alpha channel as mask
            base.paste(sprite, (top_left_x, top_left_y), sprite)

            bar_level_path = random.choice(list(Path(BASE_DIR / 
                "segments/bar-level")
                .glob(f"bar-level_{unit.team}_*.png")))
            bar_path = random.choice(list(Path(BASE_DIR / 
                "segments/bar")
                .glob(f"bar_{unit.team}_*.png")))
            if not isinstance(unit, Tower) and (random.random()<0.7 or unit.name in swarm_troops) and unit.team==1:
                sprite_np = np.array(sprite)
                alpha_np = sprite_np[:, :, 3]
                nonzero_y, nonzero_x = np.nonzero(alpha_np)

                if len(nonzero_x) == 0:
                    avg_x = center_x
                else:
                    avg_x = top_left_x + int(np.mean(nonzero_x))

                bar_level_sprite = Image.open(bar_level_path).convert("RGBA")
                bar_level_left_x = avg_x - bar_level_sprite.width // 2
                bar_level_top_y = top_left_y-bar_level_sprite.height+5
                base.paste(bar_level_sprite, (bar_level_left_x, bar_level_top_y), bar_level_sprite)
            elif not isinstance(unit, Tower) and unit.team==1:
                sprite_np = np.array(sprite)
                alpha_np = sprite_np[:, :, 3]
                nonzero_y, nonzero_x = np.nonzero(alpha_np)

                if len(nonzero_x) == 0:
                    left_x = center_x - sprite.width // 2
                    right_x = center_x + sprite.width // 2
                else:
                    left_x = top_left_x + np.min(nonzero_x)
                    right_x = top_left_x + np.max(nonzero_x)

                start_x = int(left_x + 0.4 * (right_x - left_x))
                end_x = int(left_x + 0.8 * (right_x - left_x))

                bar_width = end_x - start_x

                bar_level_sprite = Image.open(bar_level_path).convert("RGBA")
                bar_sprite = Image.open(bar_path).convert("RGBA")

                bar_sprite = bar_sprite.resize((bar_width, bar_sprite.height))

                bar_level_top_y = top_left_y - bar_level_sprite.height + 5
                bar_top_y = bar_level_top_y + 7

                # Paste the resized health bar and level box
                base.paste(bar_sprite, (start_x, bar_top_y), bar_sprite)
                base.paste(bar_level_sprite, (start_x - bar_level_sprite.width, bar_level_top_y), bar_level_sprite)
            elif not isinstance(unit, Tower) and unit.team==0 and random.random()<0.5:
                sprite_np = np.array(sprite)
                alpha_np = sprite_np[:, :, 3]
                nonzero_y, nonzero_x = np.nonzero(alpha_np)

                if len(nonzero_x) == 0:
                    left_x = center_x - sprite.width // 2
                    right_x = center_x + sprite.width // 2
                else:
                    left_x = top_left_x + np.min(nonzero_x)
                    right_x = top_left_x + np.max(nonzero_x)

                start_x = int(left_x + 0.4 * (right_x - left_x))
                end_x = int(left_x + 0.8 * (right_x - left_x))

                bar_width = end_x - start_x

                bar_level_sprite = Image.open(bar_level_path).convert("RGBA")
                bar_sprite = Image.open(bar_path).convert("RGBA")

                # Resize bar to span between 10% and 90% of the mask's width
                bar_sprite = bar_sprite.resize((bar_width, bar_sprite.height))

                bar_level_top_y = top_left_y - bar_level_sprite.height + 5
                bar_top_y = bar_level_top_y + 7

                # Paste the resized health bar and level box
                base.paste(bar_sprite, (start_x, bar_top_y), bar_sprite)
                base.paste(bar_level_sprite, (start_x - bar_level_sprite.width, bar_level_top_y), bar_level_sprite)

            if not isinstance(unit, Tower) and random.random() < (0.15 if unit.name not in swarm_troops else 0.05):
                clock_path = Path(BASE_DIR, "segments/clock/")
                clock_path = random.choice(list(clock_path.glob(f"clock_{unit.team}_*.png")))
                if clock_path.exists():
                    clock = Image.open(clock_path).convert("RGBA")

                    # Center it horizontally under the unit
                    clock_top_left_x = int(center_x - clock.width // 2)

                    # Position bottom of clock 10px below image bottom
                    clock_top_left_y = bottom_y + 10 - clock.height

                    # Paste onto the image
                    base.paste(clock, (clock_top_left_x, clock_top_left_y), clock)

            box_x_center = max(0.0,min(center_x / base.width,1.0))
            box_y_center = max(0.0,min((bottom_y - sprite.height / 2) / base.height,1.0))
            box_width = sprite.width / base.width
            box_height = sprite.height / base.height
            #currently making training data for small troops dataset
            if unit.name+("-ally" if unit.team==0 else "-enemy") in self.troops_labelled:
                labels.append(f"{self.id_troops[unit.name+('-ally' if unit.team==0 else '-enemy')]} {box_x_center:.6f} {box_y_center:.6f} {box_width:.6f} {box_height:.6f}")
                print(unit.name+('-ally' if unit.team==0 else '-enemy'))
                print(self.id_troops[unit.name+('-ally' if unit.team==0 else '-enemy')])
                print(labels[-1])
        img = base.convert("RGB")
        img.save(self.img_path)

        with open(self.label_path, 'w') as f:
            f.write('\n'.join(labels))
        # Convert back to RGB numpy array

    
if __name__ == "__main__":
    DATASET_PATH = Path(BASE_DIR / "finetuning-batch-3")
    DATASET_PATH.mkdir(exist_ok=True)
    NUM_TRAIN = 1633
    NUM_VAL = 402
    dataset_configs = [
    {
        "name": "buildings-dataset",
        "classes": buildings,
        "id_map": id_buildings
    },
    {
        "name": "large-dataset",
        "classes": large_troops,
        "id_map": id_large_troops
    },
    {
        "name": "medium-dataset",
        "classes": medium_troops,
        "id_map": id_medium_troops
    },
    {
        "name": "small-dataset",
        "classes": small_troops,
        "id_map": id_small_troops
    }
    ]
    avail_troops = small_troops + medium_troops + large_troops + buildings
    from global_stuff.create_yaml_file import create_yaml_file
    for config in dataset_configs:
        dataset_name = config["name"]
        troop_list = config["classes"]
        id_map = config["id_map"]
        dataset_path = DATASET_PATH / dataset_name
        create_yaml_file(dataset_path/"data.yaml",troop_list,f"finetuning-batch-3/{dataset_name}")
        # Create directories
        for subfolder in ["images/train", "images/val", "labels/train", "labels/val"]:
            (dataset_path / subfolder).mkdir(parents=True, exist_ok=True)

        # Generate train images
        for i in range(NUM_TRAIN,NUM_TRAIN*2):
            img_path = dataset_path / f"images/train/{i:06}.jpg"
            label_path = dataset_path / f"labels/train/{i:06}.txt"
            generator = Generator(avail_troops, img_path, label_path, troop_list, id_map, n=10)
            generator.add_towers()
            generator.add_units()
            generator.draw()

        # Generate val images
        for i in range(NUM_VAL,NUM_VAL*2):
            img_path = dataset_path / f"images/val/{i:06}.jpg"
            label_path = dataset_path / f"labels/val/{i:06}.txt"
            generator = Generator(avail_troops, img_path, label_path, troop_list, id_map, n=10)
            generator.add_towers()
            generator.add_units()
            generator.draw()

        # Create data.yaml