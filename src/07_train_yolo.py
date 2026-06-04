"""
============================================================
07_train_yolo.py  —  Dhaka Traffic Project (Phase 4)
============================================================
Fine-tunes YOLOv8m on the unified 10-class dataset.

Settings (as requested):
  - model      : yolov8m (medium - best accuracy/feasible on 12GB)
  - epochs     : 300  (upper bound)
  - patience   : 50   (early stopping if no val improvement for 50 epochs)
  - best.pt    : Ultralytics ALWAYS saves best.pt + last.pt automatically
  - batch      : 4    (explicit & safe for imgsz=1280 on 12GB)
  - imgsz      : 1280 (high-res: better on small/distant vehicles)
  - device     : 0    (your RTX 3060)

⚠️ imgsz=1280 is memory-heavy. If you get a CUDA "out of memory"
error in the first epoch, lower BATCH below from 4 -> 2 (or 2 -> 1)
and re-run. This is the ONLY thing to change for OOM.

⚠️ This is a LONG run (likely several hours to overnight). It's
safe to leave running. Progress prints each epoch. If interrupted,
you can resume from the 'last.pt' weights (ask Claude how).

Outputs land in:  runs/detect/dhaka_yolov8m/
    weights/best.pt   <- use THIS for inference/tracking
    weights/last.pt
    results.png, confusion_matrix.png, etc.

Run from project root:
    python src/07_train_yolo.py
============================================================
"""

import os
import torch
from ultralytics import YOLO

DATA = os.path.join("data", "processed", "unified", "final", "data.yaml")

# If you hit CUDA out-of-memory at imgsz=1280, lower this: 4 -> 2 -> 1
BATCH = 2
IMGSZ = 1280
WORKERS = 1     # 16GB RAM -> single worker, max stability

def main():
    # Sanity checks before committing hours of compute
    if not os.path.exists(DATA):
        print(f"❌ Cannot find dataset yaml at: {DATA}")
        print("   Run 06_build_unified_dataset.py first.")
        return

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available — training would run on CPU and take DAYS.")
        print("   Aborting. Fix the GPU setup before training.")
        return

    print("=" * 58)
    print("  YOLOv8m TRAINING — Dhaka 10-class detector")
    print("=" * 58)
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  Dataset  : {DATA}")
    print("  Model    : yolov8m.pt (pretrained -> fine-tuned)")
    print("  Epochs   : 300 (early stop patience=50)")
    print(f"  Batch    : {BATCH}   imgsz: {IMGSZ}")
    print("\n  ⚠️  High-res long run. If 'CUDA out of memory' appears in")
    print("      epoch 1, lower BATCH in the script (4->2->1) and re-run.\n")

    # Load pretrained medium model (auto-downloads ~50MB on first run)
    model = YOLO("yolov8m.pt")

    model.train(
        data=DATA,
        epochs=300,
        patience=50,        # early stopping
        batch=BATCH,        # explicit (safe for imgsz=1280 on 12GB)
        imgsz=IMGSZ,        # 1280 high-res
        device=0,
        workers=WORKERS,
        cache=False,        # never cache images in RAM           # RTX 3060
        project="runs/detect",
        name="dhaka_yolov8m",
        exist_ok=False,     # don't overwrite a previous run silently
        save=True,          # save checkpoints (best.pt + last.pt)
        plots=True,         # save result curves + confusion matrix
        # light, sensible augmentation for dense traffic scenes:
        degrees=5.0,        # small rotation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,         # horizontal flip OK for traffic
        mosaic=0.0,         # mosaic aug helps small-object density
        # patience for the loader / misc left at defaults
    )

    print("\n" + "=" * 58)
    print("  ✅ TRAINING FINISHED.")
    print("  Best weights: runs/detect/dhaka_yolov8m/weights/best.pt")
    print("  Review: runs/detect/dhaka_yolov8m/results.png")
    print("          runs/detect/dhaka_yolov8m/confusion_matrix.png")
    print("=" * 58)


if __name__ == "__main__":
    main()