# build_dataset_live

Utilities for turning raw Clash Royale gameplay videos into labeled training data. Workflows here generally:
1) download/collect videos,
2) extract frames and crop to the arena,
3) run existing models to propose annotations,
4) convert/review annotations and prepare YOLO datasets.

Key scripts (conceptual):
- `download_video.py` — fetch source videos into `BASE_DIR/videos`.
- `extract_frames.py` / `crop_arena.py` — break videos into frames and crop to the playable arena region.
- `create_model_annotations.py`, `collect_cards.py`, `collect_spell_patches.py`, `create_spell_annotations.py`, `annotate_non_troops.py` — run current models to draft JSON annotations for VIA or to collect patches for classes.
- `json_to_yolo.py`, `spell_json_to_yolo.py`, `clock_json_to_yolo.py` — convert VIA-style JSON labels into YOLO text files for the different class groups.
- `fill_spell_validation_folder.py`, `split_clock_dataset.py`, `label_patches_with_class.py` — helpers to organize/label validation splits and patch datasets.
- `start_labeling.ipynb`, `via.html` — convenience assets for manual review in VIA.

Outputs land under the YOLO-style `images/` and `labels/` trees in your `BASE_DIR`, alongside datasets like `frames/`, `clock-validation/`, and `spell-validation-dataset/`.
