"""
============================================================
09_fix_label_coords.py  —  Dhaka Traffic Project
============================================================
Fixes the out-of-bounds coordinate rows found by the validator.
Clamps every x,y,w,h value into [0, 1]. Also drops any box that
becomes degenerate (w<=0 or h<=0) after clamping.

Operates IN-PLACE on the FINAL dataset labels only (not raw sources).
Only rewrites files that actually contain a bad row, and prints
exactly what it changed.

Run from project root:
    python src/09_fix_label_coords.py
============================================================
"""

import os

FINAL = os.path.join("data", "processed", "unified", "final")
NUM_CLASSES = 10
EPS = 1e-6


def clamp01(v):
    return min(1.0, max(0.0, v))


def fix_split(split):
    lbl_dir = os.path.join(FINAL, split, "labels")
    if not os.path.isdir(lbl_dir):
        return 0, 0
    files_changed = 0
    rows_fixed = 0
    for lf in os.listdir(lbl_dir):
        if not lf.endswith(".txt"):
            continue
        path = os.path.join(lbl_dir, lf)
        with open(path, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        new_lines = []
        changed = False
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                changed = True  # drop malformed
                continue
            cid = int(float(parts[0]))
            x, y, w, h = (float(p) for p in parts[1:])

            nx, ny, nw, nh = clamp01(x), clamp01(y), clamp01(w), clamp01(h)

            # if a box edge spilled over, clamping w/h alone can misplace it;
            # safest: recompute extents, clamp corners, rebuild center/size
            left = clamp01(x - w / 2)
            right = clamp01(x + w / 2)
            top = clamp01(y - h / 2)
            bottom = clamp01(y + h / 2)
            nw = right - left
            nh = bottom - top
            nx = (left + right) / 2
            ny = (top + bottom) / 2

            if nw <= EPS or nh <= EPS:
                changed = True  # degenerate after clamp -> drop
                rows_fixed += 1
                continue

            if (abs(nx - x) > EPS or abs(ny - y) > EPS or
                    abs(nw - w) > EPS or abs(nh - h) > EPS):
                changed = True
                rows_fixed += 1

            new_lines.append(f"{cid} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

        if changed:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(new_lines) + ("\n" if new_lines else ""))
            files_changed += 1
    return files_changed, rows_fixed


def main():
    print("=" * 58)
    print("  FIX out-of-bounds coordinates (clamp to [0,1])")
    print("=" * 58)
    total_files = 0
    total_rows = 0
    for split in ("train", "valid"):
        fc, rf = fix_split(split)
        print(f"  [{split}] files changed: {fc}  | rows fixed/dropped: {rf}")
        total_files += fc
        total_rows += rf
    print(f"\n  Total files changed: {total_files}")
    print(f"  Total rows fixed   : {total_rows}")
    print("\n  ✅ Done. Re-run 08_validate_labels.py to confirm clean,")
    print("     then start training again.")


if __name__ == "__main__":
    main()