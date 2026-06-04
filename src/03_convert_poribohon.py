"""
============================================================
03_convert_poribohon.py  —  Dhaka Traffic Project (Phase 3)
============================================================
Converts the Poribohon-BD dataset from Pascal VOC (.xml) to
YOLO format, AND maps its class names onto our unified
7-class scheme. Writes to a NEW folder; originals untouched.

The conversion math (VOC pixel box -> YOLO normalized box):
   VOC gives:  xmin, ymin, xmax, ymax  (pixel corners)
   YOLO wants: x_center, y_center, width, height  (all 0-1)
     x_center = (xmin + xmax) / 2 / image_width
     y_center = (ymin + ymax) / 2 / image_height
     width    = (xmax - xmin)     / image_width
     height   = (ymax - ymin)     / image_height

SAFETY: any class name NOT in the map is REPORTED (not guessed),
so you can tell me and we add it. Nothing is silently mishandled.

Run from project root:
    python src/03_convert_poribohon.py
============================================================
"""

import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

# ---------------- CONFIG ----------------
SRC = os.path.join("data", "datasets", "poribohon", "BangladeshiVehiclesDataset-master")
SRC_IMG = os.path.join(SRC, "images")
SRC_XML = os.path.join(SRC, "annotations")
OUT = os.path.join("data", "processed", "unified", "from_poribohon")
OUT_IMG = os.path.join(OUT, "images")
OUT_LBL = os.path.join(OUT, "labels")

UNIFIED = ["car", "bus_truck", "rickshaw", "cng", "motorbike", "bicycle", "human_hauler"]
NEW_ID = {name: i for i, name in enumerate(UNIFIED)}

# Map Poribohon class names (lowercased) -> unified name, or None to DROP.
# Names are matched case-insensitively. Unknown names are REPORTED.
MAP = {
    "car": "car",
    "bus": "bus_truck",
    "truck": "bus_truck",
    "van": "bus_truck",
    "minibus": "bus_truck",
    "pickup": "bus_truck",
    "rickshaw": "rickshaw",
    "cng": "cng",
    "auto rickshaw": "cng",
    "autorickshaw": "cng",
    "easy-bike": "cng",
    "easybike": "cng",
    "bike": "motorbike",
    "motorbike": "motorbike",
    "motorcycle": "motorbike",
    "scooter": "motorbike",
    "bicycle": "bicycle",
    "cycle": "bicycle",
    "leguna": "human_hauler",
    "human hauler": "human_hauler",
    # explicit drops (not road traffic):
    "boat": None,
    "launch": None,
    "horse-cart": None,
    "horsecart": None,
    "tractor": None,
    "wheelbarrow": None,
}
# ----------------------------------------


def convert():
    os.makedirs(OUT_IMG, exist_ok=True)
    os.makedirs(OUT_LBL, exist_ok=True)

    if not os.path.isdir(SRC_XML):
        print(f"❌ Annotations folder not found: {SRC_XML}")
        return

    xml_files = [f for f in os.listdir(SRC_XML) if f.lower().endswith(".xml")]
    print(f"Found {len(xml_files)} XML files.\n")

    kept = Counter()
    dropped = Counter()
    unknown = Counter()
    converted_imgs = 0
    missing_imgs = 0
    bad_xml = 0

    for xf in xml_files:
        path = os.path.join(SRC_XML, xf)
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except ET.ParseError:
            bad_xml += 1
            print(f"  ⚠️  Could not parse {xf} — skipped.")
            continue

        size = root.find("size")
        if size is None:
            bad_xml += 1
            continue
        W = float(size.findtext("width", 0))
        H = float(size.findtext("height", 0))
        if W <= 0 or H <= 0:
            bad_xml += 1
            continue

        # figure out the image filename from <filename>, fall back to xml stem
        fname = root.findtext("filename", "").strip()
        stem = os.path.splitext(xf)[0]
        candidates = []
        if fname:
            candidates.append(fname)
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidates.append(stem + ext)

        src_img_path = None
        for c in candidates:
            cand = os.path.join(SRC_IMG, c)
            if os.path.exists(cand):
                src_img_path = cand
                break

        new_lines = []
        for obj in root.findall("object"):
            raw = (obj.findtext("name", "") or "").strip()
            key = raw.lower()
            if key not in MAP:
                unknown[raw] += 1
                continue
            target = MAP[key]
            if target is None:
                dropped[raw] += 1
                continue

            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = float(bb.findtext("xmin", 0))
            ymin = float(bb.findtext("ymin", 0))
            xmax = float(bb.findtext("xmax", 0))
            ymax = float(bb.findtext("ymax", 0))

            # clamp to image bounds (some boxes spill over)
            xmin = max(0, min(xmin, W)); xmax = max(0, min(xmax, W))
            ymin = max(0, min(ymin, H)); ymax = max(0, min(ymax, H))
            if xmax <= xmin or ymax <= ymin:
                continue

            xc = (xmin + xmax) / 2 / W
            yc = (ymin + ymax) / 2 / H
            bw = (xmax - xmin) / W
            bh = (ymax - ymin) / H
            new_lines.append(f"{NEW_ID[target]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            kept[target] += 1

        if not new_lines:
            continue  # nothing usable in this file
        if src_img_path is None:
            missing_imgs += 1
            continue  # annotation with no image -> skip

        # sanitize output filename (Poribohon has spaces/parens like "mixed (1)")
        safe_stem = stem.replace(" ", "_").replace("(", "").replace(")", "")
        out_img_name = safe_stem + os.path.splitext(src_img_path)[1].lower()
        shutil.copy2(src_img_path, os.path.join(OUT_IMG, out_img_name))
        with open(os.path.join(OUT_LBL, safe_stem + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(new_lines) + "\n")
        converted_imgs += 1

    # ---- report ----
    print("=" * 58)
    print("  PORIBOHON CONVERSION — result")
    print("=" * 58)
    print(f"  Images converted        : {converted_imgs}")
    print(f"  Annotations w/o image   : {missing_imgs}")
    print(f"  Bad/unparseable XML     : {bad_xml}")
    print("\n  Objects kept per unified class:")
    for name in UNIFIED:
        print(f"    {NEW_ID[name]} {name:16s} {kept.get(name, 0):6d}")
    print(f"\n  Objects dropped (intended): {sum(dropped.values())}  {dict(dropped)}")
    if unknown:
        print("\n  ⚠️  UNKNOWN class names found (NOT mapped — tell Claude to add these):")
        for n, c in unknown.most_common():
            print(f"       '{n}'  ({c} boxes)")
    else:
        print("\n  ✅ No unknown class names — every class was mapped.")
    print("\n  Done. Originals untouched. Next: merge both into one dataset.")


if __name__ == "__main__":
    convert()