# label_patches_with_class.py

import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from global_stuff.constants import BASE_DIR
from IPython.display import clear_output

class PatchLabeler:
    def __init__(self, patch_dir, cards_json_path, output_csv="classified_patches.csv", current_index=0, added_class=["bomb,unclear"]):
        self.patch_dir = BASE_DIR / patch_dir
        self.cards_path = BASE_DIR / cards_json_path
        self.output_csv = BASE_DIR / "classes_csv" / output_csv
        self.current_index = current_index

        # Check paths
        if not self.patch_dir.exists():
            raise FileNotFoundError(f"❌ Patch directory does not exist: {self.patch_dir}")
        if not self.cards_path.exists():
            raise FileNotFoundError(f"❌ Cards JSON file does not exist: {self.cards_path}")
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Load card names + special tags
        with open(self.cards_path) as f:
            cards = json.load(f)
        self.card_names = sorted(
            [card["name"] for card in cards if card.get("elixirCost") is not None] + added_class
        )
        self.card_names_lower = {name.lower(): name for name in self.card_names}

        # Load or create DataFrame
        if self.output_csv.exists():
            try:
                self.df = pd.read_csv(self.output_csv)
                assert "path" in self.df.columns and "label" in self.df.columns
            except Exception as e:
                print(f"⚠️ Error reading existing CSV, creating new one: {e}")
                self.df = pd.DataFrame(columns=["path", "label"])
        else:
            self.df = pd.DataFrame(columns=["path", "label"])

        # Collect images not yet labeled
        labeled_files = set(self.df["path"].tolist())
        all_images = sorted(
            self.patch_dir.glob("*.png"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else float('inf')
        )
        self.unlabeled_images = [img for img in all_images if str(img) not in labeled_files]

    def show_image(self, img_path):
        try:
            clear_output(wait=True)  # ⬅️ This clears the previous image
            print(f"\n🖼️  Now labeling: {img_path.name}")
            img = mpimg.imread(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(str(img_path.name))
            plt.show()
        except Exception as e:
            print(f"⚠️ Failed to load image {img_path}: {e}")

    def start(self, start_idx=0):
        print("🧠 Type the card name for each patch.")
        print("↩️ Press Enter to skip, or type 'done' to save and exit.")
        print(f"✅ Example card names: {', '.join(self.card_names[:5])}... [+{len(self.card_names)-5} more]\n")

        for idx in range(self.current_index, len(self.unlabeled_images)):
            img_path = self.unlabeled_images[idx]
            self.show_image(img_path)

            while True:
                user_input = input(f"Label for '{img_path.name}' (Enter=skip, done=exit): ").strip()
                if user_input.lower() == "done":
                    print("💾 Saving progress...")
                    self.df.to_csv(self.output_csv, index=False)
                    print(f"✅ Progress saved to {self.output_csv}. Exiting.")
                    return
                if user_input == "":
                    break  # skip
                normalized = self.card_names_lower.get(user_input.lower())
                if normalized:
                    self.df.loc[len(self.df)] = [str(img_path), normalized]
                    self.df.to_csv(self.output_csv, index=False)
                    print(f"✅ Saved: {normalized}")
                    break
                else:
                    print("❌ Invalid card name. Try again or type 'done' to quit.")

        print("🎉 All done! All patches labeled.")
        print(f"💾 Final dataset saved to {self.output_csv}")
