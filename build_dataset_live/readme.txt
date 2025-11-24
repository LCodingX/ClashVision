Python files in this folder assist in the development of a dataset of images used for training of Clash Royale vision models.
Specifically, frames are taken from live videos, cropped, and annotated by models and then checked by humans before being added to the dataset.

create_model_annotations.py: uses models given to create initial image annotations in json format for input to VIA.
crop_arena.py: contains a function to automate cropping of arena pixels from a gameplay image and for different sized images.
download_video.py: Downloads an image and stores it in BASE_DIR / videos
extract_frames.py: extracts frames from a set of videos in BASE_DIR / videos and crops them for arena pixels.
json_to_yolo.py: Converts JSON format compatible in the VIA image annotator to YOLO format for different unit groups(small/medium/large/buildings)

