"""
Copy all PNGs from BASE_DIR/segment/* into matching folders under
BASE_DIR/non-troop-segments/.

Usage:
    python copy_segment_to_non_troop_segments.py
"""

from pathlib import Path
import shutil

from global_stuff.constants import BASE_DIR


def main() -> None:
    src_root = BASE_DIR / "segment"
    dst_root = BASE_DIR / "non-troop-segments"

    if not src_root.exists():
        raise SystemExit(f"Source directory missing: {src_root}")
    if not dst_root.exists():
        raise SystemExit(f"Destination directory missing: {dst_root}")

    copied = 0
    for src_dir in src_root.iterdir():
        if not src_dir.is_dir():
            continue

        dst_dir = dst_root / src_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)

        for png in src_dir.glob("*.png"):
            shutil.copy2(png, dst_dir / png.name)
            copied += 1

    print(f"Copied {copied} file(s) from {src_root} to {dst_root}.")


if __name__ == "__main__":
    main()
