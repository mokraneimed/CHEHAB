"""
Run from RL/fhe_rl:
  python3 check_signal.py ../veclang_runner/temp/pilot_results.csv > check_out.txt 2>&1
No new measurement: uses the pilot rows (G=0) and the current memory.py.
Answers: (1) did the estimator drift vs the one used for the fit?
         (2) within a fixed (N, L), does est rank the measured RSS?
         (3) per-tier ratio rss/est and rss/pred (bias of the final model)
"""
import sys
import numpy as np
import pandas as pd
import memory

df = pd.read_csv(sys.argv[1])
df = df[df["max_rss_MB_median"].notna() & (df["est_G"] == 0)].copy()

raw, pred, N, L = [], [], [], []
for e in df["expr"]:
    try:
        b = memory.peak_ram_breakdown(e)
        raw.append(b["raw_net_mb"]); pred.append(b["mb"]); N.append(b["N"]); L.append(b["L"])
    except Exception:
        raw.append(np.nan); pred.append(np.nan); N.append(np.nan); L.append(np.nan)
df["raw_now"], df["pred"], df["N"], df["L"] = raw, pred, N, L
df = df.dropna(subset=["raw_now"])

# (1) drift check: current raw estimate vs the one stored when the calibration was fitted
rel = (df["raw_now"] - df["est_net_MB"]).abs() / df["est_net_MB"]
print(f"[drift] rows={len(df)}  max rel diff={rel.max():.4f}  p90={rel.quantile(.9):.4f}")
print("  -> if max > ~0.01 the estimator changed since the fit: refit (python3 fit_calib.py ...) before trusting anything\n")

# (2)+(3) per tier
print(f"{'N':>6} {'L':>3} {'n':>4} {'spearman(est,rss)':>18} {'med rss':>8} {'med pred':>9} {'rss/est':>8} {'rss/pred':>9}")
for (n, l), g in df.groupby(["N", "L"]):
    if len(g) < 5:
        print(f"{int(n):>6} {int(l):>3} {len(g):>4}   (too few rows)")
        continue
    sp = g["raw_now"].rank().corr(g["max_rss_MB_median"].rank())
    print(f"{int(n):>6} {int(l):>3} {len(g):>4} {sp:>18.2f} {g['max_rss_MB_median'].median():>8.0f} "
          f"{g['pred'].median():>9.0f} {(g['max_rss_MB_median']/g['raw_now']).median():>8.2f} "
          f"{(g['max_rss_MB_median']/g['pred']).median():>9.2f}")

# (4) monotonicity of the model across the N boundary (the lever the agent will exploit)
print("\n[floors] model floor per (N,L,K) -> must increase with depth, also across the N change:")
for (n, l) in [(16384, 4), (16384, 5), (16384, 6), (16384, 7), (16384, 8),
               (32768, 9), (32768, 10), (32768, 14)]:
    print(f"  N={n} L={l}: {memory.floor_mb(n, l, 2):.0f} MB")