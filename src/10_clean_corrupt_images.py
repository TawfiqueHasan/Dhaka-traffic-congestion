"""
============================================================
10_clean_corrupt_images.py  —  Dhaka Traffic Project
============================================================
Finds corrupt / truncated images in the FINAL dataset and:
  - tries to RE-SAVE them cleanly (keeps image + label), else
  - REMOVES the image and its matching label (last resort).

⚠️ RUN THIS ONLY WHEN TRAINING IS NOT RUNNING. Modifying the
dataset while YOLO reads it can disrupt an active run.

Makes a backup list of everything it changes.

Run from project root:
    python src/10_clean_corrupt_images.py
============================================================
"""

import os
import warnings
from PIL import Image, ImageFile

# Make PIL raise on truncated files so we can detect them
ImageFile.LOAD_TRUNCATED_IMAGES = False
warnings.simplefilter("error", Image.DecompressionBombWarning)

FINAL = os.path.join("data", "processed", "unified", "final")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

resaved = []
removed = []
ok_count = 0


def label_path_for(img_path, split):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(FINAL, split, "labels", stem + ".txt")


def process(split):
    global ok_count
    img_dir = os.path.join(FINAL, split, "images")
    if not os.path.isdir(img_dir):
        return
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
    print(f"  [{split}] checking {len(files)} images...")
    for fn in files:
        path = os.path.join(img_dir, fn)
        # 1) verify integrity
        try:
            with Image.open(path) as im:
                im.verify()                       # checks structure
            with Image.open(path) as im:
                im.load()                         # forces full decode
            ok_count += 1
            continue                              # image is fine
        except Exception:
            pass  # corrupt -> try to salvage below

        # 2) attempt clean re-save (allow truncated load just for repair)
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as im:
                rgb = im.convert("RGB")
            rgb.save(path, "JPEG", quality=95)
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            resaved.append(path)
        except Exception:
            # 3) unsalvageable -> remove image + its label
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            try:
                os.remove(path)
            except OSError:
                pass
            lp = label_path_for(path, split)
            if os.path.exists(lp):
                try:
                    os.remove(lp)
                except OSError:
                    pass
            removed.append(path)


def main():
    print("=" * 58)
    print("  CLEAN CORRUPT IMAGES (run only when NOT training)")
    print("=" * 58)
    for split in ("train", "valid"):
        process(split)

    print("\n" + "=" * 58)
    print("  RESULT")
    print("=" * 58)
    print(f"  Images OK            : {ok_count}")
    print(f"  Re-saved (repaired) : {len(resaved)}")
    print(f"  Removed (unfixable) : {len(removed)}")

    if resaved:
        print("\n  Repaired files:")
        for p in resaved[:20]:
            print(f"    {os.path.basename(p)}")
        if len(resaved) > 20:
            print(f"    ... and {len(resaved) - 20} more")
    if removed:
        print("\n  Removed files (image + label):")
        for p in removed[:20]:
            print(f"    {os.path.basename(p)}")
        if len(removed) > 20:
            print(f"    ... and {len(removed) - 20} more")

    # write a log for the record
    with open(os.path.join(FINAL, "corrupt_cleanup_log.txt"), "w", encoding="utf-8") as f:
        f.write("RESAVED:\n" + "\n".join(resaved) + "\n\nREMOVED:\n" + "\n".join(removed))
    print(f"\n  Log written: {os.path.join(FINAL, 'corrupt_cleanup_log.txt')}")
    print("  Done. Re-run 08_validate_labels.py if you removed any files.")


if __name__ == "__main__":
    main()