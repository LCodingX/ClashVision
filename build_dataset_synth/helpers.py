import numpy as np
import time
import random
from global_stuff.constants import ground_spots
xyxy_grids = np.array([6, 64, 562, 864]) #x1, y1, x2, y2 of arena tile grid
grid_size = (18, 32) #tiles wide, tiles tall
cell_size = np.array([30+8/9,25])

def cell_to_pixel(xy):
    if type(xy) != np.ndarray: 
        xy = np.array(xy)
    return (xy * cell_size + xyxy_grids[:2]).astype(np.int32)

def pixel_to_cell(xy):
    if type(xy) != np.ndarray: 
        xy = np.array(xy)
    return ((xy - xyxy_grids[:2]) / cell_size).astype(np.float32)

def get_seed():
    return int(time.time_ns() % (2**32))

def get_random_valid_xy(units_placed, level, team):
    while True:
        spot = random.choice(ground_spots)[::-1]
        spot_works = True
        for unit in units_placed:
            if abs(unit.x-spot[0])<=0.5 and abs(unit.y-spot[1])<=0.5 and unit.level==level:
                spot = random.choice(ground_spots)
                spot_works=False
        if spot_works and (team == None or (team==0 and spot[1]>=16) or (team==1 and spot[1]<=16)): return spot


