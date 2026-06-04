"""
============================================================
06_build_unified_dataset.py  —  Dhaka Traffic Project
============================================================
MASTER build script. Rebuilds the entire unified detection
dataset from the THREE untouched raw sources, applying ONE
consistent 10-class scheme. Replaces scripts 02/03/04.

Sources:
  1. trafic_data   (YOLO, 21 classes, has train/valid)
  2. poribohon     (Pascal VOC xml, flat)
  3. dataset3/ODS_New (YOLO, 17 classes, flat)

Output:
  data/processed/unified/final/{train,valid}/{images,labels} + data.yaml

SAFE: reads raw sources only; writes to a fresh final/ folder
(wipes a previous final/ build first so reruns are clean).

Run from project root:
    python src/06_build_unified_dataset.py
============================================================
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from collections import Counter

random.seed(42)

# ---------------- UNIFIED 10-CLASS SCHEME ----------------
UNIFIED = [
    "car",                        # 0
    "bus_truck",                  # 1
    "rickshaw",                   # 2
    "cng",                        # 3
    "motorbike",                  # 4
    "bicycle",                    # 5
    "human_hauler",               # 6
    "pedestrian",                 # 7
    "paddlefoot_transportation",  # 8  (van + shoppingvan + wheelbarrow)
    "auto_transportation",        # 9  (powertiller + bhotbhoti)
]
NEW_ID = {n: i for i, n in enumerate(UNIFIED)}

# Universal name->unified map (lowercased keys). None = drop. Missing = reported.
def U(name):  # helper: return unified target for a raw name (lowercased)
    return UNIVERSAL_MAP.get(name.strip().lower(), "__UNKNOWN__")

UNIVERSAL_MAP = {
    # cars
    "car": "car", "taxi": "car", "suv": "car", "minivan": "car",
    "policecar": "car", "ambulance": "car", "army vehicle": "car", "mpv": "car",
    # bus/truck
    "bus": "bus_truck", "minibus": "bus_truck", "truck": "bus_truck", "pickup": "bus_truck",
    # rickshaw (human pedal)
    "rickshaw": "rickshaw",
    # cng (motor 3-wheelers)
    "three wheelers -cng-": "cng", "auto rickshaw": "cng", "autorickshaw": "cng",
    "cng": "cng", "easybike": "cng", "easy-bike": "cng",
    # motorbike
    "motorbike": "motorbike", "motorcycle": "motorbike", "scooter": "motorbike", "bike": "motorbike",
    # bicycle
    "bicycle": "bicycle", "cycle": "bicycle",
    # human hauler
    "human hauler": "human_hauler", "leguna": "human_hauler",
    # pedestrian
    "pedestrian": "pedestrian",
    # paddlefoot_transportation
    "van": "paddlefoot_transportation", "shoppingvan": "paddlefoot_transportation",
    "wheelbarrow": "paddlefoot_transportation",
    # auto_transportation
    "powertiller": "auto_transportation", "bhotbhoti": "auto_transportation",
    # explicit drops (non-road)
    "boat": None, "launch": None, "horse-cart": None, "horsecart": None, "tractor": None,
    "garbagevan": "bus_truck",  # garbage van -> treat as truck-like
}

# ---------------- SOURCE DEFINITIONS ----------------
DS1 = os.path.join("data", "datasets", "trafic_data")
DS1_CLASSES = [
    "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus", "car",
    "garbagevan", "human hauler", "minibus", "minivan", "motorbike", "pickup",
    "policecar", "rickshaw", "scooter", "suv", "taxi", "three wheelers -CNG-",
    "truck", "van", "wheelbarrow",
]

DS2 = os.path.join("data", "datasets", "poribohon", "BangladeshiVehiclesDataset-master")

DS3 = os.path.join("data", "datasets", "dataset3", "ODS_New")
DS3_CLASSES = [
    "Bicycle", "Bus", "Bhotbhoti", "Car", "CNG", "Easybike", "Leguna",
    "Motorbike", "MPV", "Pedestrian", "Pickup", "PowerTiller", "Rickshaw",
    "ShoppingVan", "Truck", "Van", "Wheelbarrow",
]

FINAL = os.path.join("data", "processed", "unified", "final")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

unknowns = Counter()


def fresh_final():
    if os.path.exists(FINAL):
        shutil.rmtree(FINAL)
    for split in ("train", "valid"):
        os.makedirs(os.path.join(FINAL, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(FINAL, split, "labels"), exist_ok=True)


def find_image(img_dir, stem):
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
        p2 = os.path.join(img_dir, stem + ext.upper())
        if os.path.exists(p2):
            return p2
    return None


def write_pair(img_src, new_lines, split, prefix, stem):
    safe = (prefix + "_" + stem).replace(" ", "_").replace("(", "").replace(")", "")
    ext = os.path.splitext(img_src)[1].lower()
    shutil.copy2(img_src, os.path.join(FINAL, split, "images", safe + ext))
    with open(os.path.join(FINAL, split, "labels", safe + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")


# ---------- YOLO source remap (datasets 1 and 3) ----------
def remap_yolo(img_dir, lbl_dir, classes, split, prefix, kept):
    n = 0
    if not os.path.isdir(lbl_dir):
        return 0
    for lf in os.listdir(lbl_dir):
        if not lf.endswith(".txt"):
            continue
        stem = os.path.splitext(lf)[0]
        with open(os.path.join(lbl_dir, lf), encoding="utf-8") as fh:
            rows = [r.strip() for r in fh if r.strip()]
        new_lines = []
        for r in rows:
            p = r.split()
            if len(p) != 5:
                continue
            cid = int(float(p[0]))
            if not (0 <= cid < len(classes)):
                continue
            target = U(classes[cid])
            if target == "__UNKNOWN__":
                unknowns[classes[cid]] += 1
                continue
            if target is None:
                continue
            new_lines.append(f"{NEW_ID[target]} {p[1]} {p[2]} {p[3]} {p[4]}")
            kept[target] += 1
        if not new_lines:
            continue
        img = find_image(img_dir, stem)
        if img:
            write_pair(img, new_lines, split, prefix, stem)
            n += 1
    return n


# ---------- VOC source convert (dataset 2) ----------
def convert_voc(src, prefix, kept, valid_frac=0.15):
    xml_dir = os.path.join(src, "annotations")
    img_dir = os.path.join(src, "images")
    if not os.path.isdir(xml_dir):
        return 0, 0
    pairs = []  # (img_path, new_lines, stem)
    for xf in os.listdir(xml_dir):
        if not xf.lower().endswith(".xml"):
            continue
        try:
            root = ET.parse(os.path.join(xml_dir, xf)).getroot()
        except ET.ParseError:
            continue
        size = root.find("size")
        if size is None:
            continue
        W = float(size.findtext("width", 0)); H = float(size.findtext("height", 0))
        if W <= 0 or H <= 0:
            continue
        stem = os.path.splitext(xf)[0]
        img = find_image(img_dir, stem)
        if img is None:
            fn = root.findtext("filename", "").strip()
            if fn and os.path.exists(os.path.join(img_dir, fn)):
                img = os.path.join(img_dir, fn)
        if img is None:
            continue
        new_lines = []
        for obj in root.findall("object"):
            target = U(obj.findtext("name", ""))
            if target == "__UNKNOWN__":
                unknowns[obj.findtext("name", "")] += 1
                continue
            if target is None:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = max(0, min(float(bb.findtext("xmin", 0)), W))
            xmax = max(0, min(float(bb.findtext("xmax", 0)), W))
            ymin = max(0, min(float(bb.findtext("ymin", 0)), H))
            ymax = max(0, min(float(bb.findtext("ymax", 0)), H))
            if xmax <= xmin or ymax <= ymin:
                continue
            xc = (xmin + xmax) / 2 / W; yc = (ymin + ymax) / 2 / H
            bw = (xmax - xmin) / W; bh = (ymax - ymin) / H
            new_lines.append(f"{NEW_ID[target]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            kept[target] += 1
        if new_lines:
            pairs.append((img, new_lines, stem))
    random.shuffle(pairs)
    nv = max(1, int(len(pairs) * valid_frac))
    for img, nl, stem in pairs[:nv]:
        write_pair(img, nl, "valid", prefix, stem)
    for img, nl, stem in pairs[nv:]:
        write_pair(img, nl, "train", prefix, stem)
    return len(pairs) - nv, nv


def write_yaml():
    with open(os.path.join(FINAL, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# Unified Dhaka traffic dataset (3 sources, 10 classes)\n")
        f.write("train: ../train/images\nval: ../valid/images\n\n")
        f.write(f"nc: {len(UNIFIED)}\n")
        f.write(f"names: {UNIFIED}\n")


def inspect():
    print("\n" + "=" * 60)
    print("  FINAL UNIFIED DATASET — distribution")
    print("=" * 60)
    grand = Counter()
    for split in ("train", "valid"):
        lbl = os.path.join(FINAL, split, "labels")
        img = os.path.join(FINAL, split, "images")
        ni = len([f for f in os.listdir(img) if f.lower().endswith(IMG_EXTS)]) if os.path.isdir(img) else 0
        c = Counter()
        for lf in os.listdir(lbl):
            if lf.endswith(".txt"):
                with open(os.path.join(lbl, lf), encoding="utf-8") as fh:
                    for r in fh:
                        if r.strip():
                            c[int(float(r.split()[0]))] += 1
        print(f"\n  [{split}] images: {ni}")
        for i in range(len(UNIFIED)):
            print(f"     {i} {UNIFIED[i]:26s} {c.get(i,0):7d}")
        grand.update(c)
    total = sum(grand.values())
    print("\n  --- COMBINED ---")
    for i in range(len(UNIFIED)):
        cnt = grand.get(i, 0)
        print(f"     {i} {UNIFIED[i]:26s} {cnt:7d}  ({100*cnt/total if total else 0:4.1f}%)")
    counts = [grand.get(i, 0) for i in range(len(UNIFIED))]
    print(f"\n  Total objects: {total}")
    nz = [c for c in counts if c > 0]
    if nz:
        print(f"  Imbalance ratio (max/min nonzero): {max(nz)/min(nz):.0f}x")
    zero = [UNIFIED[i] for i in range(len(UNIFIED)) if counts[i] == 0]
    if zero:
        print(f"  ⚠️  Classes with ZERO objects: {zero}")


def main():
    print("=" * 60)
    print("  BUILD UNIFIED DATASET — 3 sources -> 10 classes")
    print("=" * 60)
    fresh_final()
    kept = Counter()

    print("\n[1/3] trafic_data (YOLO)...")
    n1 = 0
    for split in ("train", "valid"):
        n1 += remap_yolo(os.path.join(DS1, split, "images"),
                         os.path.join(DS1, split, "labels"),
                         DS1_CLASSES, split, "td", kept)
    print(f"      images: {n1}")

    print("[2/3] poribohon (VOC)...")
    p_tr, p_va = convert_voc(DS2, "pb", kept)
    print(f"      images: {p_tr} train + {p_va} valid")

    print("[3/3] dataset3 / ODS_New (YOLO, flat)...")
    # flat: split its images into train/valid as we remap
    lbl_dir = os.path.join(DS3, "labels")
    img_dir = os.path.join(DS3, "images")
    all_lbls = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")] if os.path.isdir(lbl_dir) else []
    random.shuffle(all_lbls)
    nv = int(len(all_lbls) * 0.15)
    valid_lbls = set(all_lbls[:nv])
    n3 = 0
    for lf in all_lbls:
        split = "valid" if lf in valid_lbls else "train"
        stem = os.path.splitext(lf)[0]
        with open(os.path.join(lbl_dir, lf), encoding="utf-8") as fh:
            rows = [r.strip() for r in fh if r.strip()]
        new_lines = []
        for r in rows:
            p = r.split()
            if len(p) != 5:
                continue
            cid = int(float(p[0]))
            if not (0 <= cid < len(DS3_CLASSES)):
                continue
            target = U(DS3_CLASSES[cid])
            if target == "__UNKNOWN__":
                unknowns[DS3_CLASSES[cid]] += 1
                continue
            if target is None:
                continue
            new_lines.append(f"{NEW_ID[target]} {p[1]} {p[2]} {p[3]} {p[4]}")
            kept[target] += 1
        if not new_lines:
            continue
        img = find_image(img_dir, stem)
        if img:
            write_pair(img, new_lines, split, "od", stem)
            n3 += 1
    print(f"      images: {n3}")

    write_yaml()
    inspect()

    if unknowns:
        print("\n  ⚠️  UNKNOWN class names encountered (tell Claude):")
        for n, c in unknowns.most_common():
            print(f"       '{n}' ({c})")
    print("\n" + "=" * 60)
    print(f"  ✅ DONE. Train on: {os.path.join(FINAL, 'data.yaml')}")
    print("=" * 60)


if __name__ == "__main__":
    main()