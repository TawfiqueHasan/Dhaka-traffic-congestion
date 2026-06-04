"""
============================================================
15_eda_congestion.py  —  Dhaka Traffic Project (Phase 8)
============================================================
Explores the congestion table and saves charts. Two purposes:
  1. Sanity-check your wrangled data is meaningful.
  2. Produce figures for the paper.

Charts saved to reports/eda/:
  - congestion_over_time.png   (CI trend, averaged across cells)
  - class_distribution.png     (which vehicle types dominate)
  - density_vs_speed.png       (the core relationship behind CI)
  - congestion_by_cell.png     (which parts of the road jam most)
  - ci_histogram.png           (spread of congestion values)

Input : data/processed/congestion_table.csv
Output: reports/eda/  + printed summary

USAGE (from project root):
  python src/15_eda_congestion.py
============================================================
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLE = os.path.join("data", "processed", "congestion_table.csv")
OUT_DIR = os.path.join("reports", "eda")

CLASSES = ["car", "bus_truck", "rickshaw", "cng", "motorbike",
           "bicycle", "human_hauler", "pedestrian",
           "paddlefoot_transportation", "auto_transportation"]


def main():
    if not os.path.exists(TABLE):
        print(f"❌ Congestion table not found: {TABLE}")
        print("   Build it first with 13_build_congestion_table.py")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(TABLE)

    print("=" * 60)
    print("  EDA — exploring the congestion table (Phase 8)")
    print("=" * 60)
    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")
    print(f"  Time span: {df['time_start_sec'].min():.0f}s - {df['time_start_sec'].max():.0f}s")
    print(f"  Cells: {df['cell'].nunique()}  |  Time windows: {df['window'].nunique()}")
    print(f"\n  Congestion index: mean {df['congestion_index'].mean():.1f}, "
          f"min {df['congestion_index'].min():.1f}, max {df['congestion_index'].max():.1f}")
    print(f"  Avg vehicles per cell-window: {df['vehicle_count'].mean():.1f}")

    # --- 1) congestion over time (averaged across cells) ---
    over_time = df.groupby("time_start_sec")["congestion_index"].mean()
    plt.figure(figsize=(10, 4))
    plt.plot(over_time.index, over_time.values, marker="o", ms=3)
    plt.title("Average congestion index over time")
    plt.xlabel("time (seconds)"); plt.ylabel("congestion index (0-100)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "congestion_over_time.png"), dpi=120); plt.close()

    # --- 2) class distribution (total counts) ---
    class_cols = [f"count_{c}" for c in CLASSES if f"count_{c}" in df.columns]
    totals = df[class_cols].sum().sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh([c.replace("count_", "") for c in totals.index], totals.values)
    plt.title("Vehicle class distribution (total across all cell-windows)")
    plt.xlabel("count"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=120); plt.close()

    # --- 3) density vs speed (the heart of the CI) ---
    plt.figure(figsize=(7, 5))
    plt.scatter(df["vehicle_count"], df["mean_speed_px_s"],
                c=df["congestion_index"], cmap="RdYlGn_r", s=20, alpha=0.7)
    plt.colorbar(label="congestion index")
    plt.title("Density vs speed (colored by congestion)")
    plt.xlabel("vehicle count (density)"); plt.ylabel("mean speed (px/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "density_vs_speed.png"), dpi=120); plt.close()

    # --- 4) congestion by cell (which road areas jam) ---
    by_cell = df.groupby("cell")["congestion_index"].mean()
    plt.figure(figsize=(7, 4))
    plt.bar(by_cell.index.astype(str), by_cell.values)
    plt.title("Average congestion by spatial cell")
    plt.xlabel("cell id"); plt.ylabel("avg congestion index")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "congestion_by_cell.png"), dpi=120); plt.close()

    # --- 5) CI histogram ---
    plt.figure(figsize=(7, 4))
    plt.hist(df["congestion_index"], bins=20, edgecolor="black")
    plt.title("Distribution of congestion index values")
    plt.xlabel("congestion index (0-100)"); plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ci_histogram.png"), dpi=120); plt.close()

    print(f"\n  ✅ 5 charts saved to: {OUT_DIR}\\")
    for f in ["congestion_over_time", "class_distribution", "density_vs_speed",
              "congestion_by_cell", "ci_histogram"]:
        print(f"     {f}.png")
    print("\n  These are sanity checks AND paper figures. Open them to explore.")


if __name__ == "__main__":
    main()