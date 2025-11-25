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

Screenshots showcasing finetuned YOLO model performance on non troops (spells, clocks, evolution symbols). All of these are detections on new images not in the training or validation set:

<img width="462" height="718" alt="image" src="https://github.com/user-attachments/assets/341efe2b-4cd7-4368-8bc3-7fe247722ec1" />
<img width="455" height="721" alt="image" src="https://github.com/user-attachments/assets/2bbe4f38-e0e2-485c-be45-aab317ceb0d7" />
<img width="454" height="717" alt="image" src="https://github.com/user-attachments/assets/6d9afa31-e835-469d-8f23-d83dc93885eb" />
<img width="454" height="717" alt="image" src="https://github.com/user-attachments/assets/af87afa9-77b1-4728-a7e9-9749f6bd7ba8" />
<img width="451" height="716" alt="image" src="https://github.com/user-attachments/assets/c95cb969-37a3-4580-b251-5642ff45fcc3" />

Clock validater(resnet-50) decides if a clock is 0 or 1 depending on whether the clock is within the first 1/4 of its progression cycle
Screenshots showcasing finetuned YOLO model performance on troops (units, buildings): 

<img width="455" height="719" alt="image" src="https://github.com/user-attachments/assets/e82a26dd-f3c8-490a-973a-57e5569777e6" />
<img width="451" height="716" alt="image" src="https://github.com/user-attachments/assets/67967926-49b1-4f15-ba36-fb0e8cf2052f" />
<img width="455" height="713" alt="image" src="https://github.com/user-attachments/assets/3e5a8960-fdcc-4a86-ad96-6ac85c4cefce" />

As you can see, the troop detectors right now generalize worse than the spell one, even though the models are larger size (YOLO11l and YOLO11m vs. YOLO11s). The troop detector is split across 4 different models (small, medium, large, builing) that run in parallel.

TODO:
- Create a "patch" dataset of spawn in screen regions so that I can train a ResNet 50 to classify troop spawn in animations and identify troop placements. Use troop classifier to help me build this dataset.
- Clean up spell dataset so that only enemy spells are included as labels. Train validater ResNet 50 models to eliminate errors (eg. ice wizard spawn, etc.)
- Create an elixir counting AI for educational purposes, with special rules for spells that are not obviously ally or enemy (zap, log, barbarian barrel, etc.) and evolution detection
- Post my dataset for open source CV experiments.
