#!/usr/bin/env python3
"""Step 5 - how much real peak RSS does each extra rotation step (Galois key) cost?

Builds circuits with TOP-LEVEL rotations (the form that really generates Galois keys; rotations inside the lanes of a
Vec are dropped by the compiler, see packing.py) and measures them with the existing pilot pipeline
(run_pilot.measure_one + build_and_run, 3 runs each, median RSS).

  shallow family: no multiplication  -> N=16384, L=4
  deep family   : chain of 7 VecMul  -> N=32768, L=9      (multiplicative depth 7)
  each with g = 0, 4, 16, 32 distinct rotation steps of the SAME vector (so the number of ciphertexts stays equal).

Output: measured RSS, real go_G / rotation_keys_size_MB reported by the Go program, the marginal MB per key, and what the
current model predicts. Nothing is modified. If the pipeline cannot run an expression the failure text is printed.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "RL", "fhe_rl"))
import memory  # noqa: E402  (also puts RL/pytrs first on sys.path)
sys.path.insert(0, os.path.join(ROOT, "RL", "veclang_runner"))
import run_pilot  # noqa: E402

SLOTS = 64
TIMEOUT = 600
STEPS = [0, 4, 16, 32]


def vec(prefix, n=SLOTS):
    return "(Vec " + " ".join(f"{prefix}{i}" for i in range(n)) + ")"


def build(family, g):
    if family == "shallow":
        acc = vec("a")
    else:                                   # deep: depth-7 multiplication chain
        acc = vec("m1_")
        for i in range(2, 9):
            acc = f"(VecMul {acc} {vec(f'm{i}_')})"
    rot_src = vec("r")                      # the SAME vector rotated g times -> same label after packing
    for j in range(1, g + 1):
        acc = f"(VecAdd {acc} (<< {rot_src} {j}))"
    return acc


env_info = run_pilot.go_env_info()
results = []
for family in ("shallow", "deep"):
    for g in STEPS:
        expr = build(family, g)
        row = {}
        try:
            with tempfile.TemporaryDirectory(prefix="rotprobe_") as td:
                wd = Path(td)
                row = run_pilot.measure_one(expr, wd, env_info, TIMEOUT)
                if row.get("status") == "ok":
                    row = run_pilot.build_and_run(row, wd, env_info, TIMEOUT)
        except Exception as e:  # keep going, report
            row = {"status": "exception", "status_detail": f"{type(e).__name__}: {e}"[:200]}
        try:
            d = memory.peak_ram_breakdown(expr, margin=False)
            est = dict(mb=d["mb"], G=d["G"], gal=d["galois_mb"], N=d["N"], L=d["L"])
        except Exception as e:
            est = dict(mb=float("nan"), G=-1, gal=float("nan"), N=0, L=0)
        results.append((family, g, row, est))
        print(f"[{family} g={g}] status={row.get('status')} {str(row.get('status_detail',''))[:100]}", flush=True)

print("\nfamily   g_req | est(N,L) est_G est_galMB est_total | go(N,L) go_G rotKeys_MB | measured_RSS_MB (runs)")
for family, g, row, est in results:
    print("%-8s %4d | (%d,%d) %3d %9.1f %9.1f | (%s,%s) %s %s | %s (%s)" % (
        family, g, est["N"], est["L"], est["G"], est["gal"], est["mb"],
        row.get("go_N", "?"), row.get("go_L", "?"), row.get("go_G", "?"), row.get("rotation_keys_size_MB", "?"),
        row.get("max_rss_MB_median", "-"), row.get("max_rss_MB_runs", "-")))

print("\nMARGINAL COST PER ROTATION KEY (measured RSS(g) - RSS(0)) / g   vs   estimator's key term / G")
for family in ("shallow", "deep"):
    base = [r for f, g, r, e in results if f == family and g == 0]
    if not base or base[0].get("status") != "ok":
        print(family, ": baseline g=0 failed -> no marginal figure")
        continue
    b0 = float(base[0]["max_rss_MB_median"])
    for f, g, r, e in results:
        if f != family or g == 0:
            continue
        if r.get("status") != "ok":
            print(f"{family} g={g}: not measured ({r.get('status')})")
            continue
        m = float(r["max_rss_MB_median"])
        gg = int(float(r.get("go_G") or 0))
        per_meas = (m - b0) / max(gg, 1)
        per_est = e["gal"] / e["G"] if e["G"] else float("nan")
        print("%-8s g_req=%2d go_G=%2d : measured %+7.1f MB total = %6.2f MB/key | estimator %6.2f MB/key | ratio meas/est = %.2f"
              % (family, g, gg, m - b0, per_meas, per_est, per_meas / per_est if per_est == per_est and per_est else float('nan')))
print("\n(The current model multiplies the estimator's key term by GALOIS_KEY_FACTOR=%.2f; the 'ratio meas/est' above is the figure to replace it.)"
      % memory.GALOIS_KEY_FACTOR)