#!/usr/bin/env python3
"""Step 6 - final consistency check of the ACTIVE model (memory.py + memory_calib.json) and of the edits.
Prints PASS/FAIL lines. Takes about one minute (one pass over the dataset, no multiprocessing)."""
import contextlib
import io
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FHE = os.path.join(ROOT, "RL", "fhe_rl")
sys.path.insert(0, FHE)
import memory  # noqa: E402
from parser import parse_sexpr  # noqa: E402  (RL/pytrs/parser.py, put first on sys.path by memory.py)

allpass = True


def check(name, cond, detail=""):
    global allpass
    allpass &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ((" : " + detail) if detail else ""))


print("version:", memory.ESTIMATOR_VERSION, "| calibration:", memory.CALIB_PATH)
cal = json.load(open(memory.CALIB_PATH))
check("estimator version is calib-v3-packed", memory.ESTIMATOR_VERSION == "calib-v3-packed")
check("active calibration is the packed one (N=32768 ratio ~1.76)",
      abs(cal["groups"]["32768"].get("ratio", 0) - 1.7605) < 0.01, str(cal["groups"]["32768"].get("ratio")))
print("cap_ratio active:", cal["groups"]["16384"].get("cap_ratio", "no"))

# floors
seq = [(16384, L) for L in range(4, 9)] + [(32768, L) for L in range(9, 15)]
fl = [memory.floor_mb(N, L, 2, margin=False) for N, L in seq]
print("floors:", [round(x, 1) for x in fl])
check("floors non-decreasing across the whole (N,L) sequence", all(b >= a for a, b in zip(fl, fl[1:])))

# edits
env = open(os.path.join(FHE, "env.py")).read()
m = re.search(r"DEFAULT_BUDGET_OPTIONS\s*=\s*(\[[^\]]*\])", env)
print("env budgets:", m.group(1) if m else "NOT FOUND")
check("env budgets are 1500/2000/2600/unconstrained", bool(m) and "1500, 2000, 2600" in m.group(1))
wr = open(os.path.join(FHE, "wrappers.py")).read()
check("wrappers: lambda is capped", "LAMBDA_MAX" in wr and "min(self.lambda_penalty + 0.01 * delta, LAMBDA_MAX)" in wr)

# dataset pass (clean lines only = lines parse_sexpr accepts, like load_expressions)
lines = [l.strip() for l in open(os.path.join(FHE, "datasets", "final_llm_dataset.txt")) if l.strip()]
mb, N = [], []
for e in lines:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            parse_sexpr(e)
            d = memory.peak_ram_breakdown(e, margin=False)
    except Exception:
        continue
    mb.append(d["mb"])
    N.append(d["N"])
n = len(mb)
deep = [v for v, k in zip(mb, N) if k > 16384]
print("clean circuits:", n, "| deep (N>16384):", len(deep))
check("clean circuit count is 42674 (same as before)", n == 42674, str(n))
exp = {1500: (5.0, 35.3), 2000: (2.4, 17.1), 2600: (0.9, 6.4)}
tol = 1.5
tot_all = tot_deep = 0.0
print("\nbudget | start-above overall | start-above among deep | (expected from the v3 fit)")
for b, (eo, ed) in exp.items():
    so = 100.0 * sum(1 for v in mb if v > b) / n
    sd = 100.0 * sum(1 for v in deep if v > b) / len(deep)
    tot_all += so
    tot_deep += sd
    print("%5d | %6.1f%% | %6.1f%% | (%.1f%% / %.1f%%)" % (b, so, sd, eo, ed))
    check("share above %d within %.1f pp of expected" % (b, tol), abs(so - eo) <= tol and abs(sd - ed) <= tol)
print("average start-infeasible share over the 4 budget options (incl. unconstrained): overall %.1f%% | deep circuits %.1f%%"
      % (tot_all / 4, tot_deep / 4))
print("\nFINAL:", "ALL PASS" if allpass else "SOMETHING FAILED (see FAIL lines)")
sys.exit(0 if allpass else 1)