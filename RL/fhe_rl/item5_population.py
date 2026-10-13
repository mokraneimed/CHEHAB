"""Run the calibrated model on the whole training set and propose budget tiers.
   python3 item5_population.py RL/fhe_rl/datasets/final_llm_dataset.txt  [out.csv]
(run from RL/fhe_rl so that memory.py is importable, or add it to PYTHONPATH)"""
import csv, sys, os, time, io, contextlib, traceback, collections
import numpy as np
import multiprocessing
from multiprocessing import Pool
import memory

def work(line):
    line = line.strip()
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # the parser prints debug output on bad lines
            b = memory.peak_ram_breakdown(line, margin=True)
            c = memory.peak_ram_breakdown(line, margin=False)["mb"]
        return (b["mb"], c, b["N"], b["L"], b["K"], b["G"], b["raw_net_mb"], "")
    except BaseException as e:
        tb = traceback.format_exc().strip().splitlines()
        return (np.nan, np.nan, 0, 0, 0, 0, np.nan, type(e).__name__ + ": " + str(e)[:120] + " || " + tb[-3].strip()[:160])

if __name__ == "__main__":
    try: multiprocessing.set_start_method("fork", force=True)   # Python 3.14 defaults to forkserver on Linux
    except (ValueError, RuntimeError): pass
    path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "population_estimates.csv"
    lim = int(os.environ.get("LIMIT", "0"))
    lines = [l for l in open(path) if l.strip()]
    if lim: lines = lines[:lim]
    t = time.time()
    with Pool() as p: res = p.map(work, lines, chunksize=64)
    print("estimated %d expressions in %.0fs" % (len(lines), time.time() - t))
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["line", "mb_margin", "mb_central", "N", "L", "K", "G", "raw_net_mb", "error"])
        for i, r in enumerate(res): w.writerow([i, *r])
    ok = [r for r in res if r[7] == ""]
    if len(ok) < 0.5 * len(res):
        print("\n*** MOST EXPRESSIONS FAILED. Top errors:")
        for msg, n in collections.Counter(r[7] for r in res if r[7]).most_common(5): print(n, msg)
        print("\nFirst expression, run directly (full traceback):")
        try: memory.peak_ram_breakdown(lines[0].strip())
        except Exception: traceback.print_exc()
        sys.exit(1)
    print("failed: %d" % (len(res) - len(ok)))
    mb = np.array([r[0] for r in ok]); N = np.array([r[2] for r in ok]); G = np.array([r[5] for r in ok])
    pc = [0, 5, 25, 50, 75, 90, 95, 100]
    print("\nCalibrated peak RAM (MB, with margin), percentiles %s" % pc)
    print(" all     n=%d :" % len(mb), np.percentile(mb, pc).round(0))
    for n in sorted(set(N)):
        print(" N=%-6d n=%d :" % (n, (N == n).sum()), np.percentile(mb[N == n], pc).round(0))
    print(" G>0 (unvalidated) n=%d" % (G > 0).sum())
    print("\nCandidate tiers (quantile -> budget, share of circuits ABOVE it):")
    tiers = [float(np.percentile(mb, q)) for q in (25, 50, 75)]
    floors = {}
    for r in ok: floors[(r[2], r[3], r[4])] = memory.floor_mb(r[2], r[3], r[4])
    for q, b in zip((25, 50, 75), tiers):
        infeas = sum(1 for r in ok if floors[(r[2], r[3], r[4])] > b) / len(ok) * 100
        print(" p%d  budget %.0f MB: violated by %.0f%% | floor already above budget for %.1f%%" % (q, b, (mb > b).mean() * 100, infeas))
    print("\nFloor per (N,L,K) [MB] and count:")
    cnt = {}
    for r in ok: cnt[(r[2], r[3], r[4])] = cnt.get((r[2], r[3], r[4]), 0) + 1
    for k in sorted(floors): print(" ", k, "%.0f MB" % floors[k], cnt[k], "exprs")