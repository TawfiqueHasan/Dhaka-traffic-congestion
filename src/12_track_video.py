"""
============================================================
12_track_video.py  —  Dhaka Traffic Project (Phase 5)
============================================================
Runs your trained detector + ByteTrack over a video, follows
each vehicle across frames with a persistent ID, and exports a
CSV of trajectories. This CSV is the RAW MATERIAL your Phase 7
wrangling turns into the congestion table.

Output CSV columns (one row per vehicle per frame):
  frame, time_sec, track_id, class_id, class_name,
  x_center, y_center, width, height, conf
(coordinates are in PIXELS, matching the video resolution)

USAGE (from project root):
  # process a video:
  python src/12_track_video.py data\\raw_videos\\my_video.mp4

  # TEST first on just the first 30 seconds (recommended before full run):
  python src/12_track_video.py data\\raw_videos\\my_video.mp4 --seconds 30

  # also save an annotated preview video (slower, optional):
  python src/12_track_video.py data\\raw_videos\\my_video.mp4 --save-video

NOTE: processing a long video frame-by-frame takes time. Always
test on a short --seconds clip first to confirm it works.
============================================================
"""

import os
import sys
import csv
import argparse
import cv2
from ultralytics import YOLO

WEIGHTS = os.path.join("runs", "detect", "runs", "detect",
                       "dhaka_yolov8m-8", "weights", "best.pt")
OUT_DIR = os.path.join("data", "interim", "tracks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to the input video file")
    ap.add_argument("--seconds", type=float, default=None,
                    help="only process the first N seconds (for testing)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection confidence threshold")
    ap.add_argument("--save-video", action="store_true",
                    help="also save an annotated preview video")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (1 = every frame)")
    args = ap.parse_args()

    if not os.path.exists(WEIGHTS):
        print(f"❌ Trained weights not found at:\n   {WEIGHTS}")
        print("   Update the WEIGHTS path at the top of this script.")
        return
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out_csv = os.path.join(OUT_DIR, f"{video_name}_tracks.csv")

    # video info
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    max_frames = int(args.seconds * fps) if args.seconds else total_frames

    print("=" * 58)
    print("  TRACKING — detector + ByteTrack")
    print("=" * 58)
    print(f"  Video      : {args.video}")
    print(f"  Resolution : {width}x{height}  |  FPS: {fps:.1f}")
    print(f"  Frames     : processing {max_frames} of {total_frames}")
    if args.seconds:
        print(f"  (TEST MODE: first {args.seconds} s only)")
    print(f"  Output CSV : {out_csv}\n")

    model = YOLO(WEIGHTS)
    names = model.names

    # video writer (optional)
    writer = None
    if args.save_video:
        out_vid = os.path.join(OUT_DIR, f"{video_name}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_vid, fourcc, fps, (width, height))

    rows = []
    frame_idx = 0
    unique_ids = set()

    # stream=True yields per-frame results without loading all into memory.
    # persist=True keeps track IDs consistent across frames.
    results = model.track(
        source=args.video,
        tracker="bytetrack.yaml",
        conf=args.conf,
        persist=True,
        stream=True,
        verbose=False,
        vid_stride=args.stride,
    )

    for r in results:
        if frame_idx >= max_frames:
            break
        time_sec = frame_idx / fps

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            xywh = r.boxes.xywh.cpu().tolist()  # center x, center y, w, h (pixels)
            for tid, c, cf, (xc, yc, w, h) in zip(ids, clss, confs, xywh):
                unique_ids.add(tid)
                rows.append([
                    frame_idx, round(time_sec, 3), tid, c, names[c],
                    round(xc, 2), round(yc, 2), round(w, 2), round(h, 2),
                    round(cf, 3),
                ])

        if writer is not None:
            writer.write(r.plot())

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  ...processed {frame_idx}/{max_frames} frames, "
                  f"{len(unique_ids)} unique vehicles so far")

    if writer is not None:
        writer.release()

    # write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "track_id", "class_id", "class_name",
                    "x_center", "y_center", "width", "height", "conf"])
        w.writerows(rows)

    print("\n" + "=" * 58)
    print("  ✅ TRACKING DONE")
    print("=" * 58)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Unique vehicles  : {len(unique_ids)}")
    print(f"  Total detections : {len(rows)} (rows in CSV)")
    print(f"  Saved to         : {out_csv}")
    if args.save_video:
        print(f"  Preview video    : {os.path.join(OUT_DIR, video_name + '_annotated.mp4')}")
    print("\n  This CSV is the raw trajectory data for Phase 7 (the congestion table).")


if __name__ == "__main__":
    main()