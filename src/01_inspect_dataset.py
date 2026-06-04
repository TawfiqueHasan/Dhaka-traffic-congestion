"""
============================================================
01_inspect_dataset.py  —  Dhaka Traffic Project (Phase 3)
============================================================
READ-ONLY. This script changes NOTHING. It only looks at your
dataset and reports what's there, so you understand the mess
BEFORE you start cleaning.

It reports, for both train and valid:
  - how many images and how many label files
  - whether every image has a matching label (and vice versa)
  - empty label files (images with no objects)
  - total objects, and the count of EACH class (imbalance check)
  - any label rows that look malformed

Run from project root:
    python src/01_inspect_dataset.py
============================================================
"""

import os
from collections import Counter

# ---- EDIT THIS if your folder name is different ----
DATASET = os.path.join("data", "datasets", "trafic_data")
YAML = os.path.join(DATASET, "data_1.yaml")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_class_names(yaml_path):
    """Read class names from the data.yaml 'names:' line (no yaml lib needed)."""
    if not os.path.exists(yaml_path):
        print(f"⚠️  Could not find {yaml_path} — class names will show as numbers.")
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        text = f.read()
    # find the names: [...] part
    start = text.find("names:")
    if start == -1:
        return None
    bracket = text.find("[", start)
    end = text.find("]", bracket)
    if bracket == -1 or end == -1:
        return None
    inside = text[bracket + 1:end]
    names = [n.strip().strip("'").strip('"') for n in inside.split(",")]
    return names


def inspect_split(split_dir, split_name, class_names):
    img_dir = os.path.join(split_dir, "images")
    lbl_dir = os.path.join(split_dir, "labels")

    print("\n" + "=" * 58)
    print(f"  SPLIT: {split_name}")
    print("=" * 58)

    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        print(f"⚠️  Missing images/ or labels/ folder in {split_dir}")
        return Counter()

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
    labels = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")]

    img_stems = {os.path.splitext(f)[0] for f in images}
    lbl_stems = {os.path.splitext(f)[0] for f in labels}

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    # Pairing checks
    imgs_without_label = img_stems - lbl_stems
    lbls_without_image = lbl_stems - img_stems
    print(f"\nImages with NO label file : {len(imgs_without_label)}")
    print(f"Labels with NO image      : {len(lbls_without_image)}")
    if imgs_without_label:
        sample = list(imgs_without_label)[:3]
        print(f"   e.g. {sample}")

    # Read all labels: count objects per class, empties, malformed
    class_counter = Counter()
    empty_files = 0
    malformed = 0
    for lf in labels:
        path = os.path.join(lbl_dir, lf)
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if not lines:
            empty_files += 1
            continue
        for ln in lines:
            parts = ln.split()
            # YOLO format: class_id x y w h  (5 values)
            if len(parts) != 5:
                malformed += 1
                continue
            try:
                cid = int(float(parts[0]))
                class_counter[cid] += 1
            except ValueError:
                malformed += 1

    print(f"\nEmpty label files (image, no objects): {empty_files}")
    print(f"Malformed label rows                 : {malformed}")
    print(f"Total objects (boxes)                : {sum(class_counter.values())}")

    # Per-class breakdown, sorted most-common first
    print("\nObjects per class:")
    for cid, count in class_counter.most_common():
        name = class_names[cid] if class_names and cid < len(class_names) else f"class_{cid}"
        bar = "█" * min(40, count // max(1, sum(class_counter.values()) // 200 + 1))
        print(f"  {name:22s} {count:7d}  {bar}")

    return class_counter


def main():
    print("=" * 58)
    print("  DATASET INSPECTION (read-only) — Phase 3 start")
    print("=" * 58)
    print(f"Dataset folder: {DATASET}")

    class_names = load_class_names(YAML)
    if class_names:
        print(f"\nClasses defined in yaml ({len(class_names)}):")
        for i, n in enumerate(class_names):
            print(f"  {i:2d} = {n}")

    total = Counter()
    for split in ["train", "valid"]:
        c = inspect_split(os.path.join(DATASET, split), split, class_names)
        total.update(c)

    # Combined imbalance summary
    print("\n" + "=" * 58)
    print("  COMBINED (train + valid) — class balance")
    print("=" * 58)
    grand = sum(total.values())
    if grand:
        for cid, count in total.most_common():
            name = class_names[cid] if class_names and cid < len(class_names) else f"class_{cid}"
            pct = 100 * count / grand
            print(f"  {name:22s} {count:7d}  ({pct:5.1f}%)")
        print(f"\n  TOTAL objects: {grand}")
        rarest = total.most_common()[-1]
        commonest = total.most_common()[0]
        rname = class_names[rarest[0]] if class_names else rarest[0]
        cname = class_names[commonest[0]] if class_names else commonest[0]
        print(f"  Most common : {cname} ({commonest[1]})")
        print(f"  Rarest      : {rname} ({rarest[1]})")
        print(f"  Imbalance ratio (most/rarest): {commonest[1] / max(1, rarest[1]):.0f}x")

    print("\n" + "=" * 58)
    print("  Inspection done. Nothing was changed.")
    print("  Next: we'll decide class merges + handle imbalance.")
    print("=" * 58)


if __name__ == "__main__":
    main()
