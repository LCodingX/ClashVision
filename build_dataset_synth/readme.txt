Python files in this folder assist in the development of a dataset of images used for training of Clash Royale vision models.
The dataset consists of frames from live gameplay as well as synthetically generated images with labels.
build_dataset.py: core logic for building synthetic image dataset with labels
find_arena_rect.py: useful for finding the frames of the arena to be cropped for gathering segments
helpers.py: contains helper functions used by build_dataset
crop_segments.py: helper function to crop segments from annotated images

