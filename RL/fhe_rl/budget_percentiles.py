"""
Run from RL/fhe_rl:
  python3 budget_percentiles.py datasets/final_llm_dataset.txt ../veclang_runner/temp/pilot_results.csv > budget_out.txt 2>&1

For each size class (N=16384 shallow circuits, N=32768 deep circuits) it prints:
  - percentiles of the calibrated peak RAM WITHOUT the safety margin (best guess of the real RSS)
    over the whole training dataset
  - percentiles of the REAL measured RSS on the pilot (297 rows), for comparison
  - for a few candidate budgets: share of dataset circuits above the budget
"""
import contextlib, io, multiprocessing as mp, sys
import numpy as np
import pandas as pd
import memory

QS = [5, 25, 50, 75, 90, 95, 99]


def one(e):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            b = memory.peak_ram_breakdown(e, margin=False)
        return (b["mb"], b["group"], b["G"])
    except Exception:
        return None


if __name__ == "__main__":
    mp.set_start_method("fork")
    lines = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    with mp.Pool() as pool:
        res = pool.map(one, lines, chunksize=200)
    ok = [r for r in res if r is not None]
    print(f"dataset: {len(lines)} lines, {len(ok)} estimated, {len(lines) - len(ok)} failed to parse")
    print(f"circuits with rotations (G>0): {sum(1 for r in ok if r[2] > 0)}\n")
    d = pd.DataFrame(ok, columns=["mb", "group", "G"])

    pilot = pd.read_csv(sys.argv[2])
    pilot = pilot[pilot["max_rss_MB_median"].notna() & (pilot["est_G"] == 0)].copy()
    pilot["group"] = np.where(pilot["est_N"] <= 16384, "16384", "32768")

    # ---- whole dataset, one set of budgets for everything
    GQ = [25, 50, 75, 85, 90, 92.5, 95, 97.5, 99, 100]
    allv = d["mb"].values
    print("=== WHOLE DATASET (one set of budgets for all circuits) ===")
    print("  ".join(f"p{q:<5}" for q in GQ))
    print("  ".join(f"{np.percentile(allv, q):6.0f}" for q in GQ))
    print(f"share of circuits in the N=16384 class: {100 * np.mean(d['group'] == '16384'):.1f}%  "
          f"(max there: {d[d['group'] == '16384']['mb'].max():.0f} MB)")
    print("candidate budgets (MB) -> % of dataset circuits ABOVE it (already violating at the start):")
    for b in (100, 150, 250, 400, 600, 800, 1000, 1250, 1500, 2000, 2500, 3000):
        print(f"  {b:5d} MB -> {100 * np.mean(allv > b):5.1f}%")
    print()

    hdr = "  ".join(f"p{q:<4d}" for q in QS)
    for g in ("16384", "32768"):
        sub = d[d["group"] == g]["mb"].values
        real = pilot[pilot["group"] == g]["max_rss_MB_median"].values
        print(f"=== size class N={g} ===")
        print(f"{'':34s}{hdr}")
        print(f"{'dataset, calibrated, no margin (n=' + str(len(sub)) + ')':34s}"
              + "  ".join(f"{np.percentile(sub, q):5.0f} " for q in QS))
        print(f"{'pilot, REAL measured (n=' + str(len(real)) + ')':34s}"
              + "  ".join(f"{np.percentile(real, q):5.0f} " for q in QS))
        cands = [int(round(np.percentile(sub, q))) for q in (25, 50, 75)]
        print("candidate budgets from the dataset percentiles (25/50/75): "
              + ", ".join(f"{c} MB -> {100 * np.mean(sub > c):.0f}% of circuits above" for c in cands))
        print()
    print("note: the pilot is a selected sample, not a random one; its percentiles are a sanity check, not a target.")