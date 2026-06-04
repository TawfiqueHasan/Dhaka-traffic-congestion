"""
============================================================
14_train_forecaster.py  —  Dhaka Traffic Project (Phase 9 + 10)
============================================================
Trains models to PREDICT future congestion from the congestion
table, and evaluates them. This is the "advanced" payoff that
shows your wrangled features are good enough to forecast jams.

It does:
  1. NAIVE BASELINE  ("future CI = current CI")  -> the bar to beat
  2. RANDOM FOREST    on the engineered features -> main model
  3. Evaluation: RMSE & MAE for both, side by side
  4. Feature importance (proves which wrangled features matter)
  5. Saves predictions + a feature-importance chart

Input : data/processed/congestion_table.csv   (from Phase 7)
Output: reports/forecaster/  (metrics, predictions, importance plot)

USAGE (from project root):
  python src/14_train_forecaster.py
============================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLE = os.path.join("data", "processed", "congestion_table.csv")
OUT_DIR = os.path.join("reports", "forecaster")

# Columns that are NOT features (identifiers or the target itself)
NON_FEATURES = {
    "window", "time_start_sec", "cell", "cell_row", "cell_col",
    "future_congestion_index",
}


def main():
    if not os.path.exists(TABLE):
        print(f"❌ Congestion table not found: {TABLE}")
        print("   Build it first with 13_build_congestion_table.py")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(TABLE)
    print("=" * 60)
    print("  FORECASTER — predict future congestion (Phase 9 + 10)")
    print("=" * 60)
    print(f"  Table rows: {len(df)}")

    # keep only rows that HAVE a future target (the last windows per cell don't)
    df = df.dropna(subset=["future_congestion_index"]).reset_index(drop=True)
    print(f"  Rows with a future target (usable): {len(df)}")
    if len(df) < 20:
        print("\n  ⚠️  Very few usable rows. You likely need a LONGER video")
        print("     (the prediction horizon eats the last part of each cell's series).")
        print("     Proceeding anyway for a dry run.")
    if len(df) < 5:
        print("  ❌ Not enough data to train. Need a longer video.")
        return

    # ---- features / target ----
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feature_cols].values
    y = df["future_congestion_index"].values

    # ---- time-aware split (don't shuffle a time series) ----
    # train on earlier windows, test on later ones
    df_sorted = df.sort_values("window")
    split = int(len(df_sorted) * 0.8)
    train_idx = df_sorted.index[:split]
    test_idx = df_sorted.index[split:]
    Xtr, Xte = df.loc[train_idx, feature_cols].values, df.loc[test_idx, feature_cols].values
    ytr, yte = df.loc[train_idx, "future_congestion_index"].values, df.loc[test_idx, "future_congestion_index"].values
    print(f"  Train rows: {len(Xtr)}  |  Test rows: {len(Xte)}")

    # ---- 1) NAIVE BASELINE: predict future CI = current CI ----
    baseline_pred = df.loc[test_idx, "congestion_index"].values
    base_rmse = np.sqrt(mean_squared_error(yte, baseline_pred))
    base_mae = mean_absolute_error(yte, baseline_pred)

    # ---- 2) RANDOM FOREST ----
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    rf_rmse = np.sqrt(mean_squared_error(yte, rf_pred))
    rf_mae = mean_absolute_error(yte, rf_pred)

    # ---- 3) Results ----
    print("\n" + "=" * 60)
    print("  RESULTS (lower = better)")
    print("=" * 60)
    print(f"  {'Model':<22}{'RMSE':>10}{'MAE':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Naive baseline':<22}{base_rmse:>10.2f}{base_mae:>10.2f}")
    print(f"  {'Random Forest':<22}{rf_rmse:>10.2f}{rf_mae:>10.2f}")
    improved = base_rmse - rf_rmse
    if improved > 0:
        print(f"\n  ✅ Random Forest beats the baseline by {improved:.2f} RMSE "
              f"({100*improved/base_rmse:.1f}% better).")
    else:
        print(f"\n  ⚠️  RF did not beat the baseline. With more data / longer video,")
        print("     or tuned features, it usually will. Worth reporting honestly.")

    # ---- 4) Feature importance ----
    importances = sorted(zip(feature_cols, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print("\n  Top features (what drives the prediction):")
    for name, imp in importances[:10]:
        print(f"     {name:28s} {imp:.3f}")

    # importance chart
    top = importances[:12][::-1]
    plt.figure(figsize=(8, 6))
    plt.barh([t[0] for t in top], [t[1] for t in top])
    plt.title("Feature importance — predicting future congestion")
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=120)
    plt.close()

    # ---- 5) Save predictions + metrics ----
    pred_df = df.loc[test_idx, ["window", "cell", "congestion_index", "future_congestion_index"]].copy()
    pred_df["rf_prediction"] = rf_pred
    pred_df["baseline_prediction"] = baseline_pred
    pred_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

    with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Congestion forecasting results\n")
        f.write(f"Usable rows: {len(df)} (train {len(Xtr)}, test {len(Xte)})\n\n")
        f.write(f"Naive baseline  RMSE {base_rmse:.2f}  MAE {base_mae:.2f}\n")
        f.write(f"Random Forest   RMSE {rf_rmse:.2f}  MAE {rf_mae:.2f}\n\n")
        f.write("Top features:\n")
        for name, imp in importances[:10]:
            f.write(f"  {name}: {imp:.3f}\n")

    print(f"\n  Saved: {os.path.join(OUT_DIR, 'metrics.txt')}")
    print(f"         {os.path.join(OUT_DIR, 'predictions.csv')}")
    print(f"         {os.path.join(OUT_DIR, 'feature_importance.png')}")
    print("\n  (LSTM is an optional next step once Random Forest works on real data.)")


if __name__ == "__main__":
    main()