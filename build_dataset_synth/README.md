# build_dataset_synth

Synthetic data generators that composite segmented sprites onto arena backgrounds to produce labeled YOLO datasets. Assets come from:
- `BASE_DIR/troop-segments-plus-arena/` for troops, towers, health bars, backgrounds, and clocks (team encoded as `_0_` ally / `_1_` enemy).
- `BASE_DIR/non-troop-segments/` for spells and other non-troop effects.

Key pieces:
- `build_dataset.py` — core troop/building synthesizer; samples units, places them on backgrounds, draws bars/clocks, and writes images + YOLO labels.
- `build_spell_dataset.py` — similar synthesizer focused on spells/non-troops using `non-troop-segments/`.
- `build_clock_dataset.py` — clock-specific composites to augment clock detection.
- `helpers.py` — shared geometry/randomization helpers (cell/pixel conversion, valid placement, seeding).
- `crop_segments.py`, `get_arena_patches_from_segment.py`, `find_arena_rect.py`, `crop_clock_patches.py` — utilities for cropping arenas and segments from annotated frames to create sprite banks.
- `move_live_to_synth.py` — copy/align live-captured validation assets into synthetic datasets.

Datasets are emitted under `BASE_DIR` (e.g., `finetuning-batch-3/`, `non-troop-dataset-3/`) using YOLO folder conventions (`images/train|val`, `labels/train|val`).
