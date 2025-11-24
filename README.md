# ClashVision

Tools and datasets for training Clash Royale computer vision models. The repo mixes live-frame processing, synthetic data generation, and supporting assets/scripts. Most paths are rooted at `BASE_DIR` (see `global_stuff/constants.py`) and expect the on-disk datasets that live alongside this repo.

## Layout (tracked items)
- `build_dataset_live/` — scripts to pull frames from live gameplay videos, auto-label them with existing models, convert annotations, and prep human review.
- `build_dataset_synth/` — synthetic data generators that composite segmented sprites onto arenas for troops, spells, and clocks.
- `build_patch_dataset/` — pipelines for patch extraction; see its `bash/pipeline.sh` and Python helpers for details.
- `global_stuff/` — shared constants (e.g., `BASE_DIR`, class lists) and utilities used across generators.
- `non-troop-segments/`, `troop-segments-plus-arena/` — sprite assets used by the synthetic generators (ally/enemy variants by filename).
- `clock-segments/`, `clock-validation*`, `frames/`, `spell-validation-dataset/`, `train_validation_models/`, `troop-class-dataset/` — datasets and validation assets used by the training pipelines.
- `copy_segment_to_non_troop_segments.py` — helper to mirror legacy `segment/` sprites into `non-troop-segments/`.

## Credits
- Portions of the segment library (including arena backgrounds and some enemy sprites) and the synthetic data workflow are adapted from the [Clash Royale Detection Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset).
- Additional segments were produced in-house using Meta’s Segment Anything Model (SAM) on live gameplay frames.

## Disclaimer
- This project is unaffiliated with and not endorsed by Supercell.
- Intended for research/education on computer vision; not for creating bots, automation, or gameplay abuse.

## Expectations and conventions
- Filenames encode team where relevant: `_0_` for ally, `_1_` for enemy.
- Synthetic generators look for sprites in `troop-segments-plus-arena/` (troops, towers, UI elements) and spells in `non-troop-segments/`.
- Live pipelines assume videos/frames under `BASE_DIR` with outputs written to YOLO-style `images/` + `labels/` trees.

For folder-level details, see `build_dataset_live/README.md` and `build_dataset_synth/README.md`.
