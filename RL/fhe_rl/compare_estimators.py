"""
Run from RL/fhe_rl (inside the repo):
  python3 compare_estimators.py ../veclang_runner/temp/pilot_results.csv > compare_out.txt 2>&1
  (optional env vars: ORIG=24c2e98   STEP=10)

Part A  current estimator  vs  the original file in git ORIG (the senior's version)
Part B  accuracy against the MEASURED peak RSS of the pilot (G=0 rows):
        raw estimator / calibrated / calibrated + margin
No new measurement, nothing in the repo is modified (a temp copy of the old file is deleted at the end).
"""
import contextlib, importlib, io, json, os, subprocess, sys
import numpy as np
import pandas as pd

ORIG = os.environ.get("ORIG", "24c2e98")
STEP = int(os.environ.get("STEP", "10"))
HERE = os.path.dirname(os.path.abspath(__file__))
PYTRS = os.path.abspath(os.path.join(HERE, "..", "pytrs"))
sys.path.insert(0, PYTRS)

import memory  # noqa: E402  (also puts pytrs on the path)
import ckks_memory_estimator as cur  # noqa: E402

# ---------------------------------------------------------------- Part A
top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip()
old_src = subprocess.check_output(
    ["git", "show", f"{ORIG}:RL/pytrs/ckks_memory_estimator.py"], cwd=top, text=True)
tmp = os.path.join(PYTRS, "_orig_ckks_estimator_tmp.py")
open(tmp, "w", newline="\n").write(old_src.replace("\r\n", "\n"))
try:
    orig = importlib.import_module("_orig_ckks_estimator_tmp")
    lines = [l.strip() for l in open(os.path.join(HERE, "datasets", "final_llm_dataset.txt")) if l.strip()]
    lines = lines[::STEP]
    n_ok = n_both_fail = n_one_fail = 0
    max_rel = 0.0
    bad_params = bad_rot = bad_live = 0
    examples = []
    for e in lines:
        res = []
        for mod in (orig, cur):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    res.append(mod.estimate_expression_memory(e, "cpu"))
            except Exception as ex:
                res.append(type(ex).__name__)
        a, b = res
        if isinstance(a, str) and isinstance(b, str):
            n_both_fail += 1
        elif isinstance(a, str) or isinstance(b, str):
            n_one_fail += 1
            if len(examples) < 3:
                examples.append((a if isinstance(a, str) else "ok", b if isinstance(b, str) else "ok"))
        else:
            n_ok += 1
            max_rel = max(max_rel, abs(a.total_mb - b.total_mb) / max(a.total_mb, 1e-9))
            bad_params += (a.params.N, a.params.L, a.params.K) != (b.params.N, b.params.L, b.params.K)
            bad_rot += a.rotation_steps != b.rotation_steps
            bad_live += a.peak_live_ciphertexts != b.peak_live_ciphertexts
finally:
    os.remove(tmp)

print(f"[A] original ({ORIG}) vs current estimator, every {STEP}th line: {len(lines)} lines")
print(f"    both ok: {n_ok} | both fail: {n_both_fail} | only one fails: {n_one_fail} {examples}")
print(f"    max relative diff of total MB : {max_rel:.6f}")
print(f"    different (N,L,K): {bad_params} | different #rotations: {bad_rot} | "
      f"different peak_live_ciphertexts (cosmetic field): {bad_live}")
print("    -> if max diff ~0 and params equal, your edit did not change any memory number\n")

# ---------------------------------------------------------------- Part B
df = pd.read_csv(sys.argv[1])
df = df[df["max_rss_MB_median"].notna() & (df["est_G"] == 0)]
rows = []
for e, rss in zip(df["expr"], df["max_rss_MB_median"]):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            nm = memory.peak_ram_breakdown(e, margin=False)
            wm = memory.peak_ram_breakdown(e, margin=True)
        rows.append(dict(g=nm["group"], rss=rss, raw=nm["raw_net_mb"], cal=nm["mb"], calm=wm["mb"]))
    except Exception:
        pass
d = pd.DataFrame(rows)


def stats(pred, rss):
    err = (pred - rss) / rss * 100
    return (f"p50|err|={np.percentile(abs(err), 50):6.1f}%  p90|err|={np.percentile(abs(err), 90):7.1f}%  "
            f"median signed={np.median(err):+7.1f}%  under-predicted={100 * np.mean(pred < rss):5.1f}%")


print("[B] prediction vs MEASURED peak RSS (pilot, G=0). 'signed' >0 = over-estimate, <0 = under-estimate")
print("    (calibrated rows are in-sample here; held-out numbers are printed from memory_calib.json below)")
for g, sub in d.groupby("g"):
    print(f"  N-group {g}  (n={len(sub)})")
    for name in ("raw", "cal", "calm"):
        label = {"raw": "raw estimator (senior)", "cal": "calibrated", "calm": "calibrated+margin"}[name]
        print(f"    {label:24s} {stats(sub[name].values, sub['rss'].values)}")
calib = json.load(open(memory.CALIB_PATH))["groups"]
print("\n  held-out (grouped CV by expression, from fit_calib):")
for g, c in calib.items():
    print(f"    N-group {g}: model={c['model']}  p50|err|={c['heldout_p50_abs_pct']:.1f}%  "
          f"p90|err|={c['heldout_p90_abs_pct']:.1f}%  margin={c['margin']:.3f}")