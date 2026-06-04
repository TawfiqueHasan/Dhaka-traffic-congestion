"""
============================================================
04_merge_and_verify.py  —  Dhaka Traffic Project (Phase 3)
============================================================
Merges the two converted datasets:
    data/processed/unified/from_trafic_data  (train/valid)
    data/processed/unified/from_poribohon    (flat)
into ONE training-ready YOLO dataset:
    data/processed/unified/final/
        train/images, train/labels
        valid/images, valid/labels
        data.yaml

Poribohon (no split of its own) is split ~85% train / 15% valid.
A filename prefix is added to every copied file to guarantee no
name collisions between sources.

Run from project root:
    python src/04_merge_and_verify.py
============================================================
"""

import os
import shutil
import random
from collections import Counter

random.seed(42)  # reproducible split

UNIFIED = ["car", "bus_truck", "rickshaw", "cng", "motorbike", "bicycle", "human_hauler"]

BASE = os.path.join("data", "processed", "unified")
SRC_TRAFIC = os.path.join(BASE, "from_trafic_data")
SRC_PORI = os.path.join(BASE, "from_poribohon")
FINAL = os.path.join(BASE, "final")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_dirs():
    for split in ("train", "valid"):
        os.makedirs(os.path.join(FINAL, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(FINAL, split, "labels"), exist_ok=True)


def copy_pair(img_path, lbl_path, split, prefix):
    """Copy one image+label into FINAL/<split>, with a source prefix."""
    img_name = prefix + "_" + os.path.basename(img_path)
    lbl_name = prefix + "_" + os.path.basename(lbl_path)
    shutil.copy2(img_path, os.path.join(FINAL, split, "images", img_name))
    shutil.copy2(lbl_path, os.path.join(FINAL, split, "labels", lbl_name))


def merge_trafic():
    """Copy trafic_data's existing train/valid as-is."""
    n = 0
    for split in ("train", "valid"):
        img_dir = os.path.join(SRC_TRAFIC, split, "images")
        lbl_dir = os.path.join(SRC_TRAFIC, split, "labels")
        if not os.path.isdir(lbl_dir):
            continue
        for lf in os.listdir(lbl_dir):
            if not lf.endswith(".txt"):
                continue
            stem = os.path.splitext(lf)[0]
            img = None
            for ext in IMG_EXTS:
                cand = os.path.join(img_dir, stem + ext)
                if os.path.exists(cand):
                    img = cand
                    break
            if img:
                copy_pair(img, os.path.join(lbl_dir, lf), split, "td")
                n += 1
    return n


def merge_poribohon(valid_frac=0.15):
    """Split Poribohon's flat set into train/valid and copy."""
    img_dir = os.path.join(SRC_PORI, "images")
    lbl_dir = os.path.join(SRC_PORI, "labels")
    if not os.path.isdir(lbl_dir):
        print("  ⚠️  No Poribohon labels found — skipping.")
        return 0, 0

    pairs = []
    for lf in os.listdir(lbl_dir):
        if not lf.endswith(".txt"):
            continue
        stem = os.path.splitext(lf)[0]
        for ext in IMG_EXTS:
            cand = os.path.join(img_dir, stem + ext)
            if os.path.exists(cand):
                pairs.append((cand, os.path.join(lbl_dir, lf)))
                break

    random.shuffle(pairs)
    n_valid = max(1, int(len(pairs) * valid_frac))
    valid_set = pairs[:n_valid]
    train_set = pairs[n_valid:]

    for img, lbl in train_set:
        copy_pair(img, lbl, "train", "pb")
    for img, lbl in valid_set:
        copy_pair(img, lbl, "valid", "pb")
    return len(train_set), len(valid_set)


def write_yaml():
    yaml_path = os.path.join(FINAL, "data.yaml")
    lines = [
        "# Unified Dhaka traffic dataset (trafic_data + Poribohon-BD)",
        "# 7 congestion-relevant super-classes",
        "train: ../train/images",
        "val: ../valid/images",
        "",
        f"nc: {len(UNIFIED)}",
        f"names: {UNIFIED}",
        "",
    ]
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return yaml_path


def inspect_final():
    print("\n" + "=" * 58)
    print("  FINAL UNIFIED DATASET — distribution")
    print("=" * 58)
    grand = Counter()
    for split in ("train", "valid"):
        lbl_dir = os.path.join(FINAL, split, "labels")
        img_dir = os.path.join(FINAL, split, "images")
        n_img = len([f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]) if os.path.isdir(img_dir) else 0
        c = Counter()
        if os.path.isdir(lbl_dir):
            for lf in os.listdir(lbl_dir):
                if not lf.endswith(".txt"):
                    continue
                with open(os.path.join(lbl_dir, lf), encoding="utf-8") as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if ln:
                            c[int(float(ln.split()[0]))] += 1
        print(f"\n  [{split}] images: {n_img}")
        for cid in range(len(UNIFIED)):
            print(f"     {cid} {UNIFIED[cid]:14s} {c.get(cid, 0):7d}")
        grand.update(c)

    print("\n  --- COMBINED ---")
    total = sum(grand.values())
    for cid in range(len(UNIFIED)):
        cnt = grand.get(cid, 0)
        pct = 100 * cnt / total if total else 0
        print(f"     {cid} {UNIFIED[cid]:14s} {cnt:7d}  ({pct:4.1f}%)")
    if total:
        counts = [grand.get(i, 0) for i in range(len(UNIFIED))]
        print(f"\n  Total objects: {total}")
        print(f"  Imbalance ratio (max/min): {max(counts) / max(1, min(counts)):.0f}x")


def main():
    print("=" * 58)
    print("  MERGE both datasets -> final training-ready dataset")
    print("=" * 58)
    ensure_dirs()
    n_td = merge_trafic()
    print(f"  trafic_data images merged : {n_td}")
    pb_tr, pb_va = merge_poribohon()
    print(f"  poribohon images merged   : {pb_tr} train + {pb_va} valid")
    yaml_path = write_yaml()
    print(f"  data.yaml written         : {yaml_path}")
    inspect_final()
    print("\n" + "=" * 58)
    print("  ✅ PHASE 3 (dataset wrangling) COMPLETE.")
    print(f"  Train on: {os.path.join(FINAL, 'data.yaml')}")
    print("=" * 58)


if __name__ == "__main__":
    main()