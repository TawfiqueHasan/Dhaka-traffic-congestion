"""
============================================================
02_remap_trafic_data.py  —  Dhaka Traffic Project (Phase 3)
============================================================
Remaps the trafic_data dataset from its original 21 classes
to our unified 7-class scheme.

SAFE MODE: reads from the original dataset, writes to a NEW
folder. Your original data is never touched.

What it does, per split (train/valid):
  - copies each image to the new folder
  - rewrites each label file: old class id -> new class id
  - DROPS boxes whose class is in the drop list (e.g. wheelbarrow)
  - skips a label file entirely if it becomes empty after dropping
  - reports a full summary of what was remapped/dropped

Run from project root:
    python src/02_remap_trafic_data.py
============================================================
"""

import os
import shutil
from collections import Counter

# ---------------- CONFIG ----------------
SRC = os.path.join("data", "datasets", "trafic_data")
OUT = os.path.join("data", "processed", "unified", "from_trafic_data")

# The original 21 classes, in their original ID order (from data_1.yaml):
ORIGINAL = [
    "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus", "car",
    "garbagevan", "human hauler", "minibus", "minivan", "motorbike", "pickup",
    "policecar", "rickshaw", "scooter", "suv", "taxi", "three wheelers -CNG-",
    "truck", "van", "wheelbarrow",
]

# Our unified 7-class scheme (new IDs = index in this list):
UNIFIED = ["car", "bus_truck", "rickshaw", "cng", "motorbike", "bicycle", "human_hauler"]
NEW_ID = {name: i for i, name in enumerate(UNIFIED)}

# Map each ORIGINAL class name -> unified class name (or None to DROP):
MAP = {
    "car": "car",
    "taxi": "car",
    "suv": "car",
    "minivan": "car",
    "policecar": "car",
    "ambulance": "car",
    "army vehicle": "car",

    "bus": "bus_truck",
    "minibus": "bus_truck",
    "truck": "bus_truck",
    "pickup": "bus_truck",
    "van": "bus_truck",
    "garbagevan": "bus_truck",

    "rickshaw": "rickshaw",
    "auto rickshaw": "cng",
    "three wheelers -CNG-": "cng",

    "motorbike": "motorbike",
    "scooter": "motorbike",

    "bicycle": "bicycle",

    "human hauler": "human_hauler",

    "wheelbarrow": None,   # DROP
}
# ----------------------------------------


def remap_split(split):
    src_img = os.path.join(SRC, split, "images")
    src_lbl = os.path.join(SRC, split, "labels")
    out_img = os.path.join(OUT, split, "images")
    out_lbl = os.path.join(OUT, split, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    if not os.path.isdir(src_lbl):
        print(f"  ⚠️  No labels folder for split '{split}', skipping.")
        return Counter(), Counter()

    kept = Counter()
    dropped = Counter()
    images_copied = 0
    images_skipped_empty = 0

    label_files = [f for f in os.listdir(src_lbl) if f.endswith(".txt")]
    for lf in label_files:
        with open(os.path.join(src_lbl, lf), "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        new_lines = []
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue  # malformed, skip
            old_id = int(float(parts[0]))
            if old_id < 0 or old_id >= len(ORIGINAL):
                continue
            old_name = ORIGINAL[old_id]
            target = MAP.get(old_name, "__UNKNOWN__")

            if target == "__UNKNOWN__":
                print(f"  ⚠️  Unmapped class '{old_name}' in {lf} — left out.")
                dropped[old_name] += 1
                continue
            if target is None:
                dropped[old_name] += 1
                continue

            new_id = NEW_ID[target]
            new_lines.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
            kept[target] += 1

        # find matching image (any extension)
        stem = os.path.splitext(lf)[0]
        match_img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            cand = os.path.join(src_img, stem + ext)
            if os.path.exists(cand):
                match_img = cand
                break

        if not new_lines:
            images_skipped_empty += 1
            continue  # nothing left after dropping -> skip image+label entirely

        # write new label
        with open(os.path.join(out_lbl, lf), "w", encoding="utf-8") as fh:
            fh.write("\n".join(new_lines) + "\n")
        # copy image
        if match_img:
            shutil.copy2(match_img, os.path.join(out_img, os.path.basename(match_img)))
            images_copied += 1

    print(f"\n  [{split}] images copied: {images_copied}  | skipped (became empty): {images_skipped_empty}")
    return kept, dropped


def main():
    print("=" * 58)
    print("  REMAP trafic_data  (21 classes -> 7)   [SAFE: new folder]")
    print("=" * 58)
    print(f"  Source : {SRC}")
    print(f"  Output : {OUT}\n")

    total_kept = Counter()
    total_dropped = Counter()
    for split in ["train", "valid"]:
        print(f"--- {split} ---")
        k, d = remap_split(split)
        total_kept.update(k)
        total_dropped.update(d)

    print("\n" + "=" * 58)
    print("  RESULT — objects kept per unified class")
    print("=" * 58)
    for name in UNIFIED:
        print(f"  {NEW_ID[name]} {name:16s} {total_kept.get(name, 0):7d}")
    print(f"\n  Total kept   : {sum(total_kept.values())}")
    print(f"  Total dropped: {sum(total_dropped.values())}  {dict(total_dropped)}")
    print("\n  ✅ Done. Originals untouched. Next: convert Poribohon.")


if __name__ == "__main__":
    main()