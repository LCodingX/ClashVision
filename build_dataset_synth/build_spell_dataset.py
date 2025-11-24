from PIL import Image, ImageDraw
from pathlib import Path
import sys
import numpy as np
import time
import random
import argparse
from tqdm import tqdm
from build_dataset_synth.helpers import cell_to_pixel, pixel_to_cell, get_seed, get_random_valid_xy
from global_stuff.constants import (
  towers_bottom_center_grid_position, BASE_DIR, non_troops, id_non_troops, 
  medium_troops, small_troops, large_troops, buildings, flying_unit_list,
  fixed_spells, fixed_positions, transparent_spells
  )

# Define air and ground spell types for proper layering
AIR_SPELLS = [
    'fireball', 'rocket', 'goblin-barrel', 'clock', 
    'evolution-symbol', 'royal-delivery',
    "snowball", 
    'vines',
]

GROUND_SPELLS = [
    'zap', 'earthquake', 'freeze', 'poison', 'clone',
    'tornado', 'void', 'miner', 'goblin-drill', 'ice-wizard',
    'goblin-curse', 'graveyard', 'mirror', 'arrows', 'the-log', 'barbarian-barrel', "lightning"
]

def remove_random_pixels(sprite, removal_percentage):
    """
    Remove a random percentage of pixels from a sprite to simulate partial occlusion.
    
    Args:
        sprite: PIL Image in RGBA format
        removal_percentage: Float between 0 and 1 indicating percentage of pixels to remove
    
    Returns:
        PIL Image with randomly removed pixels
    """
    # Convert to numpy array for easier manipulation
    sprite_array = np.array(sprite)
    height, width = sprite_array.shape[:2]
    
    # Create a mask of non-transparent pixels (alpha > 0)
    non_transparent_mask = sprite_array[:, :, 3] > 0
    non_transparent_indices = np.where(non_transparent_mask)
    
    if len(non_transparent_indices[0]) == 0:
        return sprite  # No non-transparent pixels to remove
    
    # Calculate number of pixels to remove
    total_non_transparent = len(non_transparent_indices[0])
    pixels_to_remove = int(total_non_transparent * removal_percentage)
    
    if pixels_to_remove > 0:
        # Randomly select indices to remove
        indices_to_remove = np.random.choice(total_non_transparent, pixels_to_remove, replace=False)
        
        # Set selected pixels to transparent
        for idx in indices_to_remove:
            y, x = non_transparent_indices[0][idx], non_transparent_indices[1][idx]
            sprite_array[y, x, 3] = 0  # Set alpha to 0 (transparent)
    
    # Convert back to PIL Image
    return Image.fromarray(sprite_array, 'RGBA')



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
    def __init__(self, avail_names, img_path, label_path, troops_labelled, id_troops, weights=None):
        self.avail_names = avail_names
        if weights is None:
            self.weights = [1]*len(avail_names)
        else:
            self.weights = weights
        self.seed = get_seed()
        self.units = [] #list of units to be added to the image once draw() is called
        self.towers = []
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
        
        
        if random.random() < 0.25: #25% chance of adding a fixed spell
            available_fixed = fixed_spells
            if available_fixed:
                spell_name = random.choice(available_fixed)
                folder = BASE_DIR / "segment-log" / spell_name
                png_files = list(folder.glob("*.png"))
                team = 1
                troop_path = random.choice(png_files)
                xy = fixed_positions[spell_name]
                # print(f"🔍 DEBUG: Creating fixed spell {spell_name} with xy={xy}")
                unit = Unit(spell_name, troop_path, xy[0], xy[1], 1, team)
                # print(f"🔍 DEBUG: Unit created with x={unit.x}, y={unit.y}")
                self.units.append(unit)
                # print(f"📍 Added fixed spell: {spell_name} at ({xy[0]:.1f}, {xy[1]:.1f})")
        
        ground_spells_no_fixed = [s for s in GROUND_SPELLS if s not in fixed_spells]
        available_ground = ground_spells_no_fixed
        # print(f"🔍 Available ground spells: {available_ground}")
        if available_ground:
            selected_ground = random.sample(available_ground, 4)
            for spell_name in selected_ground:
                folder = BASE_DIR/f"segment-log/{spell_name}"
                png_files = list(folder.glob("*.png"))
                team = 1
                
                # Allow ground spells to be positioned closer to edges for more realistic placement
                # Sometimes use extended positioning that can leak off sides
                if random.random() < 0.3:  # 30% chance for edge placement
                    # Extended positioning - can be placed closer to/beyond edges
                    x = random.uniform(-2, 20)  # Extended range (normal is ~0-18)
                    y = random.uniform(2, 30)   # Full height range
                    xy = (x, y)
                else:
                    # Normal positioning
                    xy = get_random_valid_xy(self.units, 1, None)
                
                troop_path = random.choice(png_files)
                unit = Unit(spell_name, troop_path, xy[0], xy[1], 0, team)
                self.units.append(unit)
                # print(f"🌍 Added ground spell: {spell_name} at ({xy[0]:.1f}, {xy[1]:.1f})")
        
        available_air = AIR_SPELLS  # Use AIR_SPELLS directly
        # print(f"🔍 Available air spells: {available_air}")
        if available_air:
            selected_air = random.sample(available_air, min(2, len(available_air)))
            for spell_name in selected_air:
                folder = BASE_DIR/f"segment-log/{spell_name}"
                png_files = list(folder.glob("*.png"))
                team = 1
                xy = get_random_valid_xy(self.units, 2, None)
                troop_path = random.choice(png_files)
                unit = Unit(spell_name, troop_path, xy[0], xy[1], 3, team)
                self.units.append(unit)
                # print(f"✈️ Added air spell: {spell_name} at ({xy[0]:.1f}, {xy[1]:.1f})")
        

    def add_negatives(self):
        """Add random troops and buildings as negative examples (not included in YOLO labels)"""
        # Define troop categories with their segment folders
        troops = medium_troops+small_troops+large_troops+buildings
        n_negatives = random.randint(0, 15)
        for _ in range(n_negatives):
            #get one troop from troops list
            troop_name = "-".join(random.choice(troops).split("-")[:-1])
            #print(troop_name)
            troop_folder = (BASE_DIR/"segments"/troop_name)
            if troop_folder.exists():
                troop_files = list(troop_folder.glob(f"*.png"))
                if troop_files:
                    troop_path = random.choice(troop_files)
                    xy = get_random_valid_xy(self.units, 1, None)
                    unit = Unit(troop_name, troop_path, xy[0], xy[1], 2 if troop_name in flying_unit_list else 1, 0)
                    self.units.append(unit)
                    # print(f"🌍 Added negative troop: {troop_name} at ({xy[0]:.1f}, {xy[1]:.1f})")
        
        # Sometimes add royal delivery allies (level 3)
        if random.random() < 0.05:  # 5% chance
            delivery_folder = BASE_DIR / "segments" / "royal-delivery"
            if delivery_folder.exists():
                delivery_files = list(delivery_folder.glob("royal-delivery_0_*.png"))
                if delivery_files:
                    #print("selected a clock")
                    selected_delivery = random.choice(delivery_files)
                    xy = get_random_valid_xy(self.units, 3, None)
                    #print(f"path {selected_clock}")
                    self.units.append(Unit(
                        name="royal-delivery",
                        path=selected_delivery,
                        x=xy[0],
                        y=xy[1],
                        level=3,  # Highest level for clocks
                        team=0
                    ))
        #sometimes add clone, earthquake, graveyard, poison, freeze to the ground with 5% chance each
        
        if random.random() < 0.05:
            clone_folder = BASE_DIR / "segment" / "clone"
            if clone_folder.exists():
                clone_files = list(clone_folder.glob("clone_0_*.png"))
                if clone_files:
                    selected_clone = random.choice(clone_files)
                    xy = get_random_valid_xy(self.units, 1, None)
                    self.units.append(Unit(
                        name="clone",
                        path=selected_clone,
                        x=xy[0],
                        y=xy[1],
                        level=1,
                        team=0
                    ))
        if random.random() < 0.05:
            freeze_folder = BASE_DIR / "segment" / "freeze"
            if freeze_folder.exists():
                freeze_files = list(freeze_folder.glob("freeze_0_*.png"))
                if freeze_files:
                    selected_freeze = random.choice(freeze_files)
                    xy = get_random_valid_xy(self.units, 1, None)
                    self.units.append(Unit(
                        name="freeze",
                        path=selected_freeze,
                        x=xy[0],
                        y=xy[1],
                        level=1,
                        team=0
                    ))
        if random.random() < 0.05:
                earthquake_folder = BASE_DIR / "segment" / "earthquake"
                if earthquake_folder.exists():
                    earthquake_files = list(earthquake_folder.glob("earthquake_0_*.png"))
                    if earthquake_files:
                        selected_earthquake = random.choice(earthquake_files)
                        xy = get_random_valid_xy(self.units, 1, None)
                        self.units.append(Unit(
                            name="earthquake",
                            path=selected_earthquake,
                            x=xy[0],
                            y=xy[1],
                            level=1,
                            team=0
                        ))
        if random.random() < 0.05:
            graveyard_folder = BASE_DIR / "segment" / "graveyard"
            if graveyard_folder.exists():
                graveyard_files = list(graveyard_folder.glob("graveyard_0_*.png"))
                if graveyard_files:
                    selected_graveyard = random.choice(graveyard_files)
                    xy = get_random_valid_xy(self.units, 1, None)
                    self.units.append(Unit(
                        name="graveyard",
                        path=selected_graveyard,
                        x=xy[0],
                        y=xy[1],
                        level=1,
                        team=0
                    ))
        if random.random() < 0.05:
            poison_folder = BASE_DIR / "segment" / "poison"
            if poison_folder.exists():
                poison_files = list(poison_folder.glob("poison_0_*.png"))
                if poison_files:
                    selected_poison = random.choice(poison_files)
                    xy = get_random_valid_xy(self.units, 1, None)
                    self.units.append(Unit(
                        name="poison",
                        path=selected_poison,
                        x=xy[0],
                        y=xy[1],
                        level=1,
                        team=0
                    ))  
                
        if random.random() < 0.3:  # 30% chance
            clock_folder = BASE_DIR / "segments" / "clock"
            if clock_folder.exists():
                clock_files = list(clock_folder.glob("clock_0_*.png"))
                if clock_files:
                    #print("selected a clock")
                    selected_clock = random.choice(clock_files)
                    xy = get_random_valid_xy(self.units, 3, None)
                    #print(f"path {selected_clock}")
                    self.units.append(Unit(
                        name="clock",
                        path=selected_clock,
                        x=xy[0],
                        y=xy[1],
                        level=3,  # Highest level for clocks
                        team=0
                    ))
                    
                    # print(f"🕐 Added negative clock at ({x}, {y}) level 3")

    def draw(self):
        # Sort units for correct layering (ground first, upper units later)
        self.units += self.towers
        self.units = sorted(self.units, key=lambda unit: (unit.level, unit.y))

        # Convert base image (numpy array) to RGBA PIL Image
        base = Image.fromarray(self.image).convert("RGBA")
        labels = []

        for unit in self.units:
            sprite = Image.open(unit.path).convert("RGBA")
            
            # Apply pixel removal to spells only (not towers or other units)
            # 1/3 chance to remove 0-50% of pixels from spell sprites
            
            if unit.name in non_troops and random.random() < 1/3:  # 1/3 chance
                removal_percentage = random.uniform(0, 0.1)  # 0-50% removal
                sprite = remove_random_pixels(sprite, removal_percentage)
                # print(f"Applied {removal_percentage*100:.1f}% pixel removal to {unit.name}")

            # Convert unit (x, y) in cell space to pixel position
            center_x, bottom_y = cell_to_pixel((unit.x, unit.y))
            top_left_x = int(center_x - sprite.width // 2)
            top_left_y = int(bottom_y - sprite.height)
            
            # Debug output for fixed spells
            if unit.name in fixed_spells:
                # print(f"🔍 DEBUG: Drawing fixed spell {unit.name} at cell ({unit.x}, {unit.y}) -> pixel ({center_x}, {bottom_y})")
                pass

            # Simple bounds check: center must be within image bounds
            center_y = bottom_y - sprite.height // 2
            if not (0 <= center_x < base.width and 0 <= center_y < base.height):
                #print(f"Skipping {unit.name}: center out of bounds")
                continue

            # Paste sprite using alpha channel as mask (allows partial overlap at edges)
            base.paste(sprite, (top_left_x, top_left_y), sprite)

            box_x_center = max(0.0,min(center_x / base.width,1.0))
            box_y_center = max(0.0,min((bottom_y - sprite.height / 2) / base.height,1.0))
            box_width = sprite.width / base.width
            box_height = sprite.height / base.height
            # Generate labels for all spells that are drawn (both teams)
            if not isinstance(unit, Tower) and unit.team==1:
                labels.append(f"{self.id_troops[unit.name]} {box_x_center:.6f} {box_y_center:.6f} {box_width:.6f} {box_height:.6f}")
                # print(f"Added spell: {unit.name} (team {unit.team})")
                # print(f"Class ID: {self.id_troops[unit.name]}")
                # print(f"Label: {labels[-1]}")
        img = base.convert("RGB")
        img.save(self.img_path)

        with open(self.label_path, 'w') as f:
            f.write('\n'.join(labels))
        # Convert back to RGB numpy array

    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build spell dataset')
    args = parser.parse_args()
    
    DATASET_PATH = Path(BASE_DIR / "non-troop-dataset-3")
    DATASET_PATH.mkdir(exist_ok=True)
    NUM_TRAIN = 30266 # match number of human annotated images
    NUM_VAL = 2000 # Use synthetic val set for now
    # Use non_troops and id_non_troops for the dataset
    dataset_configs = [
    {
        "name": "non-troop-dataset-3",
        "classes": non_troops,
        "id_map": id_non_troops
    }
    ]
    from global_stuff.create_yaml_file import create_yaml_file
    for config in dataset_configs:
        dataset_name = config["name"]
        spell_list = config["classes"]
        id_map = config["id_map"]
        dataset_path = BASE_DIR/dataset_name
        create_yaml_file(dataset_path/"data.yaml",spell_list,"non-troop-dataset-3")
        # Create directories
        for subfolder in ["images/train", "images/val", "labels/train", "labels/val"]:
            (dataset_path / subfolder).mkdir(parents=True, exist_ok=True)

        # Generate train images
        for i in range(NUM_TRAIN,int(NUM_TRAIN*1.5)):
            img_path = dataset_path / f"images/train/{i:06}.jpg"
            label_path = dataset_path / f"labels/train/{i:06}.txt"
            generator = Generator(non_troops, img_path, label_path, spell_list, id_map)
            generator.add_towers()             # Add towers
            generator.add_units()              # Add all spells with proper layering
            generator.add_negatives()
            generator.draw()

        # Generate val images
        for i in range(NUM_VAL):
            img_path = dataset_path / f"images/val/{i:06}.jpg"
            label_path = dataset_path / f"labels/val/{i:06}.txt"
            generator = Generator(non_troops, img_path, label_path, spell_list, id_map)
            generator.add_towers()             # Add towers
            generator.add_units()              # Add all spells with proper layering
            generator.draw()
