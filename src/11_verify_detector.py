"""
============================================================
11_verify_detector.py  —  Dhaka Traffic Project (Phase 4 check)
============================================================
Loads your TRAINED detector (best.pt) and runs it on test
image(s), so you can confirm it correctly detects Dhaka
vehicles before building the tracking pipeline on top of it.

It will:
  - load best.pt and print the 10 class names it knows
  - run detection on any images you point it at
  - print what it found (class + confidence)
  - save annotated images to reports/ so you can SEE the boxes

Usage (from project root):
  # test on the validation images (uses a few automatically):
  python src/11_verify_detector.py

  # OR test on your own image(s) - put them in a folder and pass it:
  python src/11_verify_detector.py path\\to\\some\\images
============================================================
"""

import os
import sys
import glob
from ultralytics import YOLO

# Path to your trained best weights (adjust if your run folder differs)
WEIGHTS = os.path.join("runs", "detect", "runs", "detect",
                       "dhaka_yolov8m-8", "weights", "best.pt")

OUT_DIR = os.path.join("reports", "detector_check")
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def gather_images(arg):
    """Decide which images to test on."""
    if arg and os.path.isdir(arg):
        files = []
        for ext in IMG_EXTS:
            files += glob.glob(os.path.join(arg, ext))
        return sorted(files)[:8]
    if arg and os.path.isfile(arg):
        return [arg]
    # default: pull a few from the validation set
    valid_img = os.path.join("data", "processed", "unified", "final", "valid", "images")
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(valid_img, ext))
    return sorted(files)[:6]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if not os.path.exists(WEIGHTS):
        print(f"❌ Could not find trained weights at:\n   {WEIGHTS}")
        print("   Check the run folder name under runs\\detect\\runs\\detect\\")
        print("   and update the WEIGHTS path at the top of this script.")
        return

    print("=" * 58)
    print("  DETECTOR VERIFICATION")
    print("=" * 58)
    print(f"  Weights: {WEIGHTS}")

    model = YOLO(WEIGHTS)
    print(f"\n  Classes it knows ({len(model.names)}):")
    for i, n in model.names.items():
        print(f"    {i}: {n}")

    images = gather_images(arg)
    if not images:
        print("\n  ⚠️  No test images found. Pass a folder/image path, e.g.:")
        print("     python src\\11_verify_detector.py path\\to\\images")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n  Testing on {len(images)} image(s)...\n")

    for img_path in images:
        results = model(img_path, conf=0.25, verbose=False)
        r = results[0]
        name = os.path.basename(img_path)
        print(f"  {name}: {len(r.boxes)} detections")
        # tally classes found
        from collections import Counter
        found = Counter(model.names[int(b.cls)] for b in r.boxes)
        for cls, cnt in found.most_common():
            print(f"     {cls:26s} x{cnt}")
        # save annotated
        out = os.path.join(OUT_DIR, "det_" + name)
        r.save(filename=out)

    print(f"\n  ✅ Annotated images saved to: {OUT_DIR}\\")
    print("  Open them to SEE the detector working on Dhaka vehicles.")
    print("  If the boxes look right, your detector is good -> build tracking next.")


if __name__ == "__main__":
    main()