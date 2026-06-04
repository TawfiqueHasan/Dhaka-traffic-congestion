"""
============================================================
08_validate_labels.py  —  Dhaka Traffic Project
============================================================
READ-ONLY. Scans the FINAL merged dataset for label errors
that cause CUDA "device-side assert" during training:
  - class id  < 0  or  >= NUM_CLASSES (10)
  - coordinates outside [0, 1]
  - wrong number of columns

Reports the offending files so we can fix the build.

Run from project root:
    python src/08_validate_labels.py
============================================================
"""

import os
from collections import Counter

FINAL = os.path.join("data", "processed", "unified", "final")
NUM_CLASSES = 10

bad_class = []      # (file, line_no, raw)
bad_coord = []      # (file, line_no, raw)
bad_format = []     # (file, line_no, raw)
class_hist = Counter()
max_id = -1


def scan(split):
    global max_id
    lbl_dir = os.path.join(FINAL, split, "labels")
    if not os.path.isdir(lbl_dir):
        print(f"  (no {split} labels)")
        return
    files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    print(f"  [{split}] scanning {len(files)} label files...")
    for lf in files:
        path = os.path.join(lbl_dir, lf)
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad_format.append((lf, i, line))
                    continue
                try:
                    cid = int(float(parts[0]))
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    bad_format.append((lf, i, line))
                    continue
                class_hist[cid] += 1
                max_id = max(max_id, cid)
                if cid < 0 or cid >= NUM_CLASSES:
                    bad_class.append((lf, i, line))
                if any(c < 0.0 or c > 1.0 for c in coords):
                    bad_coord.append((lf, i, line))


def main():
    print("=" * 58)
    print("  LABEL VALIDATION (read-only)")
    print("=" * 58)
    print(f"  Expecting class ids 0..{NUM_CLASSES - 1}\n")

    for split in ("train", "valid"):
        scan(split)

    print("\n  Class id histogram (what's actually in the labels):")
    for cid in sorted(class_hist):
        flag = "  <-- OUT OF RANGE!" if (cid < 0 or cid >= NUM_CLASSES) else ""
        print(f"     id {cid:3d} : {class_hist[cid]:7d}{flag}")
    print(f"\n  Highest class id found: {max_id}")

    print("\n" + "=" * 58)
    print("  PROBLEMS FOUND")
    print("=" * 58)
    print(f"  Out-of-range class ids : {len(bad_class)} rows")
    print(f"  Out-of-bounds coords   : {len(bad_coord)} rows")
    print(f"  Malformed rows         : {len(bad_format)} rows")

    def show(label, items):
        if items:
            print(f"\n  --- {label} (first 5) ---")
            for f, ln, raw in items[:5]:
                print(f"     {f} line {ln}: {raw}")

    show("OUT-OF-RANGE CLASS", bad_class)
    show("OUT-OF-BOUNDS COORDS", bad_coord)
    show("MALFORMED", bad_format)

    if not (bad_class or bad_coord or bad_format):
        print("\n  ✅ No problems found. (CUDA error may be something else.)")
    else:
        print("\n  ⚠️  These cause the CUDA device-side assert. We'll fix the build.")


if __name__ == "__main__":
    main()