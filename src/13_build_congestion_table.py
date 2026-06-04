"""
============================================================
13_build_congestion_table.py  —  Dhaka Traffic Project (Phase 7)
============================================================
THE HEART OF THE PROJECT (graded core, part 2).

Takes the raw trajectory CSV from tracking (Phase 5) and:
  STAGE A - cleans the tracks:
     * computes per-vehicle speed (pixels/sec) between frames
     * drops impossibly large jumps (tracking errors)
     * removes tracks that are too short (noise)
  STAGE B - aggregates into the tidy congestion table:
     * divides the frame into a GRID x GRID grid of cells
     * bins time into WINDOW-second windows
     * per (cell, window): vehicle count, per-class counts,
       density, mean & std of speed
     * computes the Congestion Index (0-100) from your Phase 1 formula
     * adds the FUTURE congestion index (the prediction target)

Output: data/processed/congestion_table.csv  <- your tidy deliverable

USAGE (from project root):
  python src/13_build_congestion_table.py data\\interim\\tracks\\my_video_tracks.csv

Requires the video resolution to map pixels -> grid cells. The script
reads it from the tracks (max x/y seen) OR you can set it manually below.
============================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# ---------------- TUNABLE PARAMETERS (change & re-run freely) ----------------
GRID = 3              # GRID x GRID spatial cells (3 = 9 cells). Try 4 if cells are dense.
WINDOW_SEC = 15       # time window length in seconds
PREDICT_AHEAD_SEC = 600   # how far ahead to predict (600 s = 10 min). Set 900 for 15 min.

# Congestion Index weights (from Phase 1):  CI = 100 * (w_d*density_norm + w_s*slowness)
W_DENSITY = 0.5
W_SPEED = 0.5

# Track cleaning thresholds
MAX_SPEED_PX_PER_SEC = 2000   # drop jumps faster than this (tracking errors)
MIN_TRACK_FRAMES = 5          # ignore tracks seen in fewer frames than this
# -----------------------------------------------------------------------------

CLASSES = ["car", "bus_truck", "rickshaw", "cng", "motorbike",
           "bicycle", "human_hauler", "pedestrian",
           "paddlefoot_transportation", "auto_transportation"]

OUT = os.path.join("data", "processed", "congestion_table.csv")


def load_and_clean(csv_path):
    """STAGE A: load tracks, compute speeds, clean."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} detection rows, "
          f"{df['track_id'].nunique()} unique tracks.")

    # remove tracks that are too short (likely noise / false positives)
    counts = df.groupby("track_id").size()
    keep_ids = counts[counts >= MIN_TRACK_FRAMES].index
    before = df["track_id"].nunique()
    df = df[df["track_id"].isin(keep_ids)].copy()
    print(f"  Dropped {before - df['track_id'].nunique()} short tracks "
          f"(< {MIN_TRACK_FRAMES} frames).")

    # sort so consecutive rows of a track are adjacent in time
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    # per-vehicle speed: distance moved / time elapsed, between consecutive frames
    df["prev_x"] = df.groupby("track_id")["x_center"].shift(1)
    df["prev_y"] = df.groupby("track_id")["y_center"].shift(1)
    df["prev_t"] = df.groupby("track_id")["time_sec"].shift(1)

    dx = df["x_center"] - df["prev_x"]
    dy = df["y_center"] - df["prev_y"]
    dt = (df["time_sec"] - df["prev_t"]).replace(0, np.nan)
    df["speed_px_s"] = np.sqrt(dx**2 + dy**2) / dt

    # drop impossible speeds (tracking glitches / ID swaps)
    glitches = (df["speed_px_s"] > MAX_SPEED_PX_PER_SEC).sum()
    df.loc[df["speed_px_s"] > MAX_SPEED_PX_PER_SEC, "speed_px_s"] = np.nan
    print(f"  Nulled {glitches} impossible-speed rows (> {MAX_SPEED_PX_PER_SEC} px/s).")

    return df


def assign_cells_and_windows(df, frame_w, frame_h):
    """Assign each detection to a spatial cell and a time window."""
    # spatial cell (col, row) from pixel position
    col = np.clip((df["x_center"] / frame_w * GRID).astype(int), 0, GRID - 1)
    row = np.clip((df["y_center"] / frame_h * GRID).astype(int), 0, GRID - 1)
    df["cell"] = row * GRID + col          # cell id 0 .. GRID*GRID-1
    # time window index
    df["window"] = (df["time_sec"] // WINDOW_SEC).astype(int)
    return df


def build_table(df):
    """STAGE B: aggregate into (cell, window) rows with features + CI."""
    records = []
    grouped = df.groupby(["window", "cell"])
    for (win, cell), g in grouped:
        rec = {
            "window": win,
            "time_start_sec": win * WINDOW_SEC,
            "cell": cell,
            "cell_row": cell // GRID,
            "cell_col": cell % GRID,
            "vehicle_count": g["track_id"].nunique(),
            "mean_speed_px_s": g["speed_px_s"].mean(),
            "median_speed_px_s": g["speed_px_s"].median(),
            "std_speed_px_s": g["speed_px_s"].std(),
        }
        # per-class counts
        for ci, cname in enumerate(CLASSES):
            rec[f"count_{cname}"] = g[g["class_id"] == ci]["track_id"].nunique()
        records.append(rec)

    table = pd.DataFrame(records).fillna(0)
    if table.empty:
        return table

    # ---- Congestion Index (0-100) ----
    # density_norm: vehicle_count scaled by the max observed (per dataset)
    max_count = table["vehicle_count"].max()
    table["density_norm"] = table["vehicle_count"] / max_count if max_count > 0 else 0

    # slowness: 1 - (speed / max_speed). Slow traffic -> high slowness -> high CI.
    max_speed = table["mean_speed_px_s"].replace(0, np.nan).max()
    if max_speed and max_speed > 0:
        speed_norm = (table["mean_speed_px_s"] / max_speed).clip(0, 1).fillna(0)
    else:
        speed_norm = 0
    table["slowness"] = 1 - speed_norm

    table["congestion_index"] = (
        100 * (W_DENSITY * table["density_norm"] + W_SPEED * table["slowness"])
    ).round(2)

    # ---- FUTURE congestion index (the prediction target) ----
    # shift CI forward by PREDICT_AHEAD_SEC, per cell
    windows_ahead = max(1, int(round(PREDICT_AHEAD_SEC / WINDOW_SEC)))
    table = table.sort_values(["cell", "window"]).reset_index(drop=True)
    table["future_congestion_index"] = (
        table.groupby("cell")["congestion_index"].shift(-windows_ahead)
    )

    return table, windows_ahead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks_csv", help="path to the *_tracks.csv from tracking")
    ap.add_argument("--width", type=int, default=None, help="video width in px (else inferred)")
    ap.add_argument("--height", type=int, default=None, help="video height in px (else inferred)")
    args = ap.parse_args()

    if not os.path.exists(args.tracks_csv):
        print(f"❌ Tracks CSV not found: {args.tracks_csv}")
        return

    print("=" * 60)
    print("  BUILD CONGESTION TABLE  (Phase 7 - the graded core)")
    print("=" * 60)
    print(f"  Grid: {GRID}x{GRID} ({GRID*GRID} cells) | window: {WINDOW_SEC}s | "
          f"predict ahead: {PREDICT_AHEAD_SEC}s\n")

    # STAGE A
    print("STAGE A - cleaning tracks:")
    df = load_and_clean(args.tracks_csv)

    # infer frame size if not given (max coordinate seen + box half-extent)
    fw = args.width or int(np.ceil((df["x_center"] + df["width"] / 2).max()))
    fh = args.height or int(np.ceil((df["y_center"] + df["height"] / 2).max()))
    print(f"  Frame size used: {fw} x {fh} px")

    df = assign_cells_and_windows(df, fw, fh)

    # STAGE B
    print("\nSTAGE B - aggregating into congestion table...")
    result = build_table(df)
    if isinstance(result, tuple):
        table, windows_ahead = result
    else:
        table = result
        windows_ahead = 0

    if table.empty:
        print("  ⚠️  No rows produced (empty/too-short tracks?). Check the input.")
        return

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    table.to_csv(OUT, index=False)

    # summary
    n_with_target = table["future_congestion_index"].notna().sum()
    print("\n" + "=" * 60)
    print("  ✅ CONGESTION TABLE BUILT")
    print("=" * 60)
    print(f"  Rows (cell x window) : {len(table)}")
    print(f"  Columns (attributes) : {len(table.columns)}")
    print(f"  Rows with a future target (usable for training): {n_with_target}")
    print(f"  Mean congestion index: {table['congestion_index'].mean():.1f}")
    print(f"  CI range             : {table['congestion_index'].min():.1f} "
          f"- {table['congestion_index'].max():.1f}")
    print(f"  Saved to             : {OUT}")
    print("\n  Columns:")
    for c in table.columns:
        print(f"     {c}")
    print("\n  This tidy table is your graded deliverable + the input to the forecaster.")


if __name__ == "__main__":
    main()