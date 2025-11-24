# annotate_non_troops.py
import os, sys, shutil, random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import re  # regex for numeric sorting

# --- Matplotlib setup (BEFORE importing pyplot) ---
import matplotlib
# If keys/windows don't work on your setup, uncomment ONE of these:
# matplotlib.use("QtAgg")   # pip install PyQt5
# matplotlib.use("TkAgg")   # platform dependent
import matplotlib as mpl
mpl.rcParams['keymap.save'] = []     # unbind 's' for save
mpl.rcParams['toolbar'] = 'None'     # hide toolbar

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.patches import Rectangle
from tqdm import tqdm

from global_stuff.constants import BASE_DIR, non_troops  # list of class names

print(non_troops)
# ============ CONFIG ============
FRAMES_DIR = Path(BASE_DIR) / "frames"
OUTPUT_DIR = Path(BASE_DIR) / "non-troops-dataset-2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_SPLIT = 0.8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RANDOM_SEED = 1337
FILENAME_PAD = 6

# Batch processing options
START_FRAME_INDEX = 1021  # Start from this frame index (0-based)
FORCE_SPLIT = "train"  # Set to "train" or "val" to force all frames to one split, or None for auto-split

# ================================

@dataclass
class Selection:
    sel_id: int
    start_idx: int
    end_idx: int
    cls_id: int
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

class Annotator:
    def __init__(self, show_nav_buttons: bool = False, start_frame_index: int = START_FRAME_INDEX, force_split: Optional[str] = FORCE_SPLIT):
        self.frames = self._load_frames()
        if not self.frames:
            print(f"❌ No frames found in {FRAMES_DIR}")
            sys.exit(1)
        
        # Apply start frame index
        if start_frame_index >= len(self.frames):
            print(f"⚠️ Start frame index {start_frame_index} is beyond available frames ({len(self.frames)}). Starting from frame 0.")
            start_frame_index = 0
        
        self.start_frame_index = start_frame_index
        self.force_split = force_split
        
        print(f"Found {len(self.frames)} frames. Starting from frame {start_frame_index + 1} (index {start_frame_index})")
        print(f"First frame in batch: {self.frames[start_frame_index].name}")

        self.idx = start_frame_index
        self.img = self._read(self.frames[self.idx])
        self.h, self.w = self.img.shape[:2]

        # selection state
        self.range_start: Optional[int] = None
        self.range_end: Optional[int] = None
        self.cls_id: Optional[int] = None
        self.bbox_xyxy: Optional[Tuple[int, int, int, int]] = None

        # history + per-frame annotations
        self.queue: List[Selection] = []
        self.next_sel_id: int = 1
        # frame_index -> list of (sel_id, cls_id, x1, y1, x2, y2)
        self.ann_by_frame: Dict[int, List[Tuple[int,int,int,int,int,int]]] = {}

        # class picker state (right panel)
        self.picking_class = False
        self.class_cursor = 0
        self.cls_view_start = 0
        self.cls_items_per_page = 24

        # --- UI: one figure with two subplots (image left, class list right) ---
        self.fig, (self.ax_img, self.ax_list) = plt.subplots(
            1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [6, 1.5], 'wspace': 0.1}
        )
        try:
            self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        except Exception:
            pass

        # Custom positioning for compact layout - call immediately after subplot creation
        self._adjust_layout()

        # image axis
        self.im = self.ax_img.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        self.ax_img.set_xticks([]); self.ax_img.set_yticks([])
        self.ax_img.set_title(self._title())

        # list axis
        self.ax_list.set_facecolor((0.1, 0.1, 0.1))
        self.ax_list.set_xticks([]); self.ax_list.set_yticks([])
        self._render_class_list()
        
        # Add no detection button at the bottom of the class list
        self._add_no_detection_button()

        # rectangle selector on the image axis only
        self.rs: Optional[RectangleSelector] = None
        self._init_rectangle_selector()

        # events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_list)

        # --- Nav buttons: ◀ -50 and ▶ +50 (optional) ---
        if show_nav_buttons:
            self._add_nav_buttons()

        plt.show(block=True)

    def _load_frames(self) -> List[Path]:
        files = [p for p in FRAMES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]

        def num_key(p: Path):
            # Grab all digit runs in the filename (no extension)
            nums = re.findall(r'\d+', p.stem)
            if nums:
                # Use the LAST number group as the frame index (common in names like frame_000123)
                return (int(nums[-1]), p.stem.lower())
            # If no digits, push to the end but keep a stable order
            return (float('inf'), p.stem.lower())

        return sorted(files, key=num_key)

    def _read(self, p: Path) -> np.ndarray:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        return img

    # ---------- UI helpers ----------
    def _title(self) -> str:
        name = self.frames[self.idx].name
        rs = f"{self.range_start+1}" if self.range_start is not None else "-"
        re = f"{self.range_end+1}" if self.range_end is not None else "-"
        cls_str = f"{self.cls_id}" if self.cls_id is not None else "-"
        bbox_str = f"{self.bbox_xyxy}" if self.bbox_xyxy is not None else "-"
        pick = " | [CLASS MODE] ↑/↓ PgUp/PgDn Home/End, Enter=confirm, Esc=exit" if self.picking_class else ""
        
        return (f"Frame {self.idx+1}/{len(self.frames)} :: {name} | Start(S): {rs} End(E): {re} Class(C): {cls_str} BBox: {bbox_str}\n"
                f"Selections: {len(self.queue)} | Labeled frames: {len(self.ann_by_frame)} | Enter=Add, U=Undo, Q=Export & Quit{pick}")

    def _init_rectangle_selector(self):
        if self.rs is not None:
            self.rs.set_active(False)
            self.rs = None

        def onselect(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            x1, x2 = sorted([max(0, x1), min(self.w - 1, x2)])
            y1, y2 = sorted([max(0, y1), min(self.h - 1, y2)])
            if x2 > x1 and y2 > y1:
                self.bbox_xyxy = (x1, y1, x2, y2)
                self._refresh_image()
            else:
                print("⚠️ Invalid bbox; try again.")

        try:
            self.rs = RectangleSelector(
                self.ax_img, onselect,
                useblit=True, button=[1],
                minspanx=3, minspany=3, spancoords='pixels',
                interactive=True,
                props=dict(edgecolor='white', linewidth=2, fill=False)
            )
        except TypeError:
            self.rs = RectangleSelector(
                self.ax_img, onselect,
                drawtype='box', useblit=True, button=[1],
                minspanx=3, minspany=3, spancoords='pixels',
                interactive=True
            )

    def _refresh_image(self):
        # update image (do NOT clear axes)
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        if getattr(self, "im", None) is None or self.im.axes is None:
            self.im = self.ax_img.imshow(rgb)
            self._init_rectangle_selector()
        else:
            self.im.set_data(rgb)

        # remove old rectangles (keep selector internals)
        for p in list(self.ax_img.patches):
            try:
                is_selector = hasattr(p, "artists") or p.get_label() == "Selector"
            except Exception:
                is_selector = False
            if not is_selector:
                p.remove()

        # draw bbox
        if self.bbox_xyxy is not None:
            x1, y1, x2, y2 = self.bbox_xyxy
            self.ax_img.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2))



        self.ax_img.set_title(self._title())
        self.fig.canvas.draw_idle()

    def _add_no_detection_button(self):
        """Add no detection button at the bottom of the class list"""
        # Position the button at the bottom of the class list area
        try:
            bbox = self.ax_list.get_position()  # figure coords
            width = 0.2  # button width
            height = 0.04  # button height
            bottom = max(0.005, bbox.y0 - 0.05)  # below the class list
            left = bbox.x0 + (bbox.width - width) / 2  # center in class list area
        except Exception:
            left, bottom, width, height = 0.72, 0.005, 0.2, 0.04

        self.btn_no_detection_ax = self.fig.add_axes([left, bottom, width, height])
        self.btn_no_detection = Button(self.btn_no_detection_ax, "No Detection")

        # Styling - black background with white text
        try:
            self.btn_no_detection_ax.set_facecolor('black')
            self.btn_no_detection.label.set_color('white')
            self.btn_no_detection.hovercolor = '0.3'
            
            # Fix text positioning to center vertically - more aggressive approach
            self.btn_no_detection.label.set_va('center')
            self.btn_no_detection.label.set_position((0.5, 0.5))  # Center both horizontally and vertically
            self.btn_no_detection.label.set_transform(self.btn_no_detection.ax.transAxes)  # Use axes coordinates
        except Exception:
            pass

        self.btn_no_detection.on_clicked(self._mark_no_detection)

    def _mark_no_detection(self, _event=None):
        """Mark current frame as having no detections"""
        if self.idx in self.ann_by_frame:
            print(f"⚠️ Frame {self.idx+1} already has annotations. Remove them first or use a different frame.")
            return
        
        # Add to no-detection frames set
        if not hasattr(self, 'no_detection_frames'):
            self.no_detection_frames = set()
        
        if self.idx in self.no_detection_frames:
            self.no_detection_frames.remove(self.idx)
            print(f"✅ Removed frame {self.idx+1} from no-detection frames")
        else:
            self.no_detection_frames.add(self.idx)
            print(f"✅ Marked frame {self.idx+1} as no-detection frame")
        
        self._refresh_image()

    def _adjust_layout(self):
        """Adjust layout for compact, left-aligned positioning"""
        # Position the image axis (left side, full height) - more aggressive left positioning
        self.ax_img.set_position([0.02, 0.08, 0.68, 0.84])
        
        # Position the class list (right side, compact)
        self.ax_list.set_position([0.72, 0.08, 0.25, 0.84])
        
        # Force the layout to update
        self.fig.canvas.draw_idle()

    # ---------- Nav buttons ----------
    def _add_nav_buttons(self):
        # Place two buttons below the class list, left-aligned and compact
        try:
            bbox = self.ax_list.get_position()  # figure coords
            width = 0.08  # narrower buttons
            height = 0.04  # shorter height
            gap = 0.005  # smaller gap
            bottom = max(0.005, bbox.y0 - 0.05)  # closer to class list
            left = bbox.x0  # left-aligned with class list
            back_left = left
            fwd_left = left + width + gap
        except Exception:
            back_left, fwd_left, bottom, width, height = 0.72, 0.81, 0.005, 0.08, 0.04

        self.btn_back_ax = self.fig.add_axes([back_left, bottom, width, height])
        self.btn_fwd_ax  = self.fig.add_axes([fwd_left,  bottom, width, height])

        self.btn_back = Button(self.btn_back_ax, "◀-50")
        self.btn_fwd  = Button(self.btn_fwd_ax,  "▶+50")

        # Optional styling
        try:
            for ax in (self.btn_back_ax, self.btn_fwd_ax):
                ax.set_facecolor((0.2, 0.2, 0.2))
            self.btn_back.label.set_color('white')
            self.btn_fwd.label.set_color('white')
            self.btn_back.hovercolor = '0.4'
            self.btn_fwd.hovercolor = '0.4'
            
            # Fix text positioning to center vertically - more aggressive approach
            for btn in (self.btn_back, self.btn_fwd):
                btn.label.set_va('center')
                btn.label.set_position((0.5, 0.5))  # Center both horizontally and vertically
                btn.label.set_transform(btn.ax.transAxes)  # Use axes coordinates
        except Exception:
            pass

        self.btn_back.on_clicked(self._go_back_50)
        self.btn_fwd.on_clicked(self._go_forward_50)







    def _go_forward_50(self, _event=None):
        old = self.idx
        self.idx = min(self.idx + 50, len(self.frames) - 1)
        if self.idx != old:
            self._load_current()

    def _go_back_50(self, _event=None):
        old = self.idx
        self.idx = max(self.idx - 50, 0)
        if self.idx != old:
            self._load_current()

    # ---------- Class list (right panel) ----------
    def _render_class_list(self):
        n = len(non_troops)
        self.ax_list.clear()
        self.ax_list.set_facecolor((0.1, 0.1, 0.1))
        self.ax_list.set_xticks([]); self.ax_list.set_yticks([])
        self.ax_list.set_title("Class Picker (C)", color='white', fontsize=10)

        if n == 0:
            self.ax_list.text(0.5, 0.5, "No classes", color='w', ha='center', va='center', fontsize=14)
            self.fig.canvas.draw_idle()
            return

        # keep cursor in [0..n-1]
        self.class_cursor = max(0, min(self.class_cursor, n-1))

        # paging
        per = self.cls_items_per_page
        if self.class_cursor < self.cls_view_start:
            self.cls_view_start = (self.class_cursor // per) * per
        if self.class_cursor >= self.cls_view_start + per:
            self.cls_view_start = (self.class_cursor // per) * per

        start = self.cls_view_start
        end = min(n, start + per)

        # draw header hints
        hint = "Click any class to select it"
        self.ax_list.text(0.02, 0.04, hint, color='0.7', transform=self.ax_list.transAxes, fontsize=9, va='bottom', ha='left')



        # layout rows - compact and left-aligned
        self.ax_list.set_xlim(0, 1)
        self.ax_list.set_ylim(per + 0.5, -0.5)  # tighter margins
        y = 0.2  # start higher up, less padding
        for idx in range(start, end):
            label = f"{idx:2d} {non_troops[idx]}"  # reduced padding, left-aligned
            color = 'yellow' if idx == self.class_cursor else 'white'
            weight = 'bold' if idx == self.class_cursor else 'normal'
            
            self.ax_list.text(0.02, y, label, color=color, fontsize=11, va='top', fontweight=weight, ha='left')
            y += 0.8  # tighter spacing between rows

        self.fig.canvas.draw_idle()



    def _confirm_class(self):
        self.cls_id = self.class_cursor
        print(f"✅ Class set to {self.cls_id} ({non_troops[self.cls_id]})")
        self._render_class_list()
        self._refresh_image()

    def on_click_list(self, event):
        if event.inaxes != self.ax_list:
            return
        if event.ydata is None:
            return
        
        # Handle class list clicks - adjusted for new compact layout
        # Convert y coordinate to row index based on new spacing
        y_offset = 0.2  # starting y position from _render_class_list
        row_height = 0.8  # spacing between rows from _render_class_list
        
        # Calculate which row was clicked
        row = int(round((event.ydata - y_offset) / row_height))
        idx = self.cls_view_start + row
        
        if 0 <= idx < len(non_troops):
            self.class_cursor = idx
            # Single click now sets the class immediately
            self._confirm_class()

    # ---------- Key handling ----------
    def on_key(self, event):
        if event.key is None:
            return
        key = event.key.lower()

        # Class picker keyboard control
        if self.picking_class:
            n = len(non_troops)
            if key in ['up']:
                self.class_cursor = (self.class_cursor - 1) % n
                self._render_class_list()
            elif key in ['down']:
                self.class_cursor = (self.class_cursor + 1) % n
                self._render_class_list()
            elif key in ['pageup']:
                self.class_cursor = max(0, self.class_cursor - self.cls_items_per_page)
                self._render_class_list()
            elif key in ['pagedown']:
                self.class_cursor = min(n-1, self.class_cursor + self.cls_items_per_page)
                self._render_class_list()
            elif key in ['home']:
                self.class_cursor = 0; self._render_class_list()
            elif key in ['end']:
                self.class_cursor = n-1; self._render_class_list()
            elif key in ['enter', 'return']:
                self._confirm_class()
            elif key in ['escape', 'esc', 'c']:
                self.picking_class = False
                self.ax_img.set_title(self._title())
                self.fig.canvas.draw_idle()
            return

        # Normal mode (image/nav)
        if key in ['right']:
            self.idx = min(self.idx + 1, len(self.frames) - 1)
            self._load_current()
        elif key in ['left']:
            self.idx = max(self.idx - 1, 0)
            self._load_current()
        elif key == 's':
            self.range_start = self.idx
            print(f"✅ Start set to frame index {self.range_start}")
            self._refresh_image()
        elif key == 'e':
            if self.range_start is None:
                print("⚠️ Set start first (S).")
            else:
                self.range_end = self.idx
                print(f"✅ End set to frame index {self.range_end}")
                self._refresh_image()
        elif key == 'c':
            self.picking_class = not self.picking_class
            if self.cls_id is not None:
                self.class_cursor = self.cls_id
            self.ax_img.set_title(self._title())
            self._render_class_list()
        elif key == 'n':
            self._mark_no_detection()
        elif key in ['enter', 'return']:
            self._add_selection()
        elif key == 'u':
            self._undo_last_selection()
            self._refresh_image()
            self._render_class_list()
        elif key == 'q':
            plt.close(self.fig)
            self._export_dataset()

    def _load_current(self):
        self.img = self._read(self.frames[self.idx])
        self.h, self.w = self.img.shape[:2]
        self.im.set_data(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        self.ax_img.set_title(self._title())
        self.fig.canvas.draw_idle()

    # ---------- Add / Undo / Export ----------
    def _add_selection(self):
        if self.range_start is None or self.range_end is None:
            print("⚠️ Set start (S) and end (E) first.")
            return
        if self.cls_id is None:
            print("⚠️ Choose class (C) first.")
            return
        if self.bbox_xyxy is None:
            print("⚠️ Draw bbox (mouse drag) first.")
            return

        sel = Selection(self.next_sel_id, self.range_start, self.range_end, self.cls_id, self.bbox_xyxy)
        self.queue.append(sel)
        self.next_sel_id += 1

        x1, y1, x2, y2 = sel.bbox_xyxy
        for i in range(sel.start_idx, sel.end_idx + 1):
            self.ann_by_frame.setdefault(i, []).append((sel.sel_id, sel.cls_id, x1, y1, x2, y2))

        print(f"➕ Added selection #{sel.sel_id}: frames {sel.start_idx}-{sel.end_idx}, "
              f"class {sel.cls_id} ({non_troops[sel.cls_id]}), box {sel.bbox_xyxy}")

        self.range_start, self.range_end = None, None
        self.cls_id = None
        self.bbox_xyxy = None
        self._refresh_image()
        self._render_class_list()

    def _undo_last_selection(self):
        if not self.queue:
            print("Nothing to undo.")
            return
        sel = self.queue.pop()
        removed_frames = 0
        for i in range(sel.start_idx, sel.end_idx + 1):
            if i in self.ann_by_frame:
                before = len(self.ann_by_frame[i])
                self.ann_by_frame[i] = [rec for rec in self.ann_by_frame[i] if rec[0] != sel.sel_id]
                if not self.ann_by_frame[i]:
                    del self.ann_by_frame[i]
                if len(self.ann_by_frame.get(i, [])) < before:
                    removed_frames += 1
        print(f"↩️  Undid selection #{sel.sel_id}; affected frames with labels removed: {removed_frames}")

    def _numeric_from_stem(self, stem: str) -> Optional[int]:
        nums = re.findall(r'\d+', stem)
        return int(nums[-1]) if nums else None

    def _next_export_index_for_split(self, split: str) -> int:
        img_dir = OUTPUT_DIR / "images" / split
        max_int = -1
        if img_dir.exists():
            for p in img_dir.iterdir():
                if p.is_file():
                    n = self._numeric_from_stem(p.stem)
                    if n is not None and n > max_int:
                        max_int = n
        return max_int + 1

    def _export_dataset(self):
        # Get all frames to export (annotated + no-detection)
        annotated_frames = set(self.ann_by_frame.keys())
        no_detection_frames = getattr(self, 'no_detection_frames', set())
        
        if not annotated_frames and not no_detection_frames:
            print("No frames to export. Add annotations or mark frames as no-detection.")
            return
        
        # Combine all frames to export
        all_frames = list(annotated_frames | no_detection_frames)
        all_frames.sort()
        
        print(f"📝 Exporting {len(all_frames)} frames:")
        print(f"  - {len(annotated_frames)} annotated frames")
        print(f"  - {len(no_detection_frames)} no-detection frames")
        
        # Handle forced split vs auto-split
        if self.force_split is not None:
            # Force all frames to one split
            if self.force_split == "train":
                train_indices = set(all_frames)
                val_indices = set()
                print(f"📝 All frames will go to TRAIN split (forced)")
            elif self.force_split == "val":
                train_indices = set()
                val_indices = set(all_frames)
                print(f"📝 All frames will go to VAL split (forced)")
            else:
                print(f"⚠️ Invalid force_split value: {self.force_split}. Using auto-split.")
                random.Random(RANDOM_SEED).shuffle(all_frames)
                split_point = int(len(all_frames) * TRAIN_SPLIT)
                train_indices = set(all_frames[:split_point])
                val_indices = set(all_frames[split_point:])
        else:
            # Auto-split based on TRAIN_SPLIT ratio
            random.Random(RANDOM_SEED).shuffle(all_frames)
            split_point = int(len(all_frames) * TRAIN_SPLIT)
            train_indices = set(all_frames[:split_point])
            val_indices = set(all_frames[split_point:])
            print(f"📝 Using auto-split with {TRAIN_SPLIT*100:.0f}% train")

        # Ensure folders exist
        (OUTPUT_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # Start from highest existing index in EACH split
        next_idx = {
            "train": self._next_export_index_for_split("train"),
            "val":   self._next_export_index_for_split("val"),
        }
        print(f"Starting at train={next_idx['train']}, val={next_idx['val']}.")

        for i in tqdm(all_frames):
            img_path = self.frames[i]
            im = cv2.imread(str(img_path))
            h, w = im.shape[:2]

            split = "train" if i in train_indices else "val"

            # Find the next FREE stem in this split (avoid any overwrite)
            while True:
                stem = str(next_idx[split]).zfill(FILENAME_PAD)
                dst_img = OUTPUT_DIR / "images" / split / f"{stem}{img_path.suffix.lower()}"
                dst_lbl = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
                if not dst_img.exists() and not dst_lbl.exists():
                    break
                next_idx[split] += 1

            # Copy image
            shutil.copy2(img_path, dst_img)

            # Write label file
            if i in annotated_frames:
                # Frame has annotations - write bounding boxes
                lines = []
                for (_sel_id, cls_id, x1, y1, x2, y2) in self.ann_by_frame[i]:
                    bw, bh = x2 - x1, y2 - y1
                    cx, cy = x1 + bw / 2, y1 + bh / 2
                    lines.append(f"{cls_id} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")
                with open(dst_lbl, "w") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                # No-detection frame - create empty label file
                with open(dst_lbl, "w") as f:
                    f.write("")  # Empty file indicates no detections

            next_idx[split] += 1  # advance per-split counter

        # dataset.yaml
        names_yaml = "\n".join([f"  {i}: {name}" for i, name in enumerate(non_troops)])
        (OUTPUT_DIR / "dataset.yaml").write_text(
            f"# Auto-generated\npath: {OUTPUT_DIR.as_posix()}\n"
            f"train: images/train\nval: images/val\n\nnames:\n{names_yaml}\n"
        )
        print(f"✅ Exported to {OUTPUT_DIR}")
        print("Done.")


if __name__ == "__main__":
    print(f"📂 Frames: {FRAMES_DIR}")
    print(f"🧾 Output: {OUTPUT_DIR}")
    print("Controls: ←/→ prev/next | S start | E end | C toggle class keyboard | N no detection | draw box | Enter add | U undo | Q export")
    print("Options:")
    print("  - Navigation buttons: Annotator(show_nav_buttons=True)")
    print("  - Start from specific frame: Annotator(start_frame_index=1000)")
    print("  - Force all frames to train: Annotator(force_split='train')")
    print("  - Force all frames to val: Annotator(force_split='val')")
    print("  - Combine options: Annotator(show_nav_buttons=True, start_frame_index=1000, force_split='train')")
    
    # Example: Start from frame 1000 and force all to train split
    # Annotator(start_frame_index=1000, force_split='train')
    
    # Default: Start from beginning with auto-split
    Annotator()
