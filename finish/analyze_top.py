#!/usr/bin/env python3
"""Step 3b - compare log-log and capped predictions with the REAL peak RSS of the 10 large circuits
(measured by run_pilot.py into RL/veclang_runner/temp/top16384_measured.csv).
Writes cap_ratio into memory_calib.json ONLY if the capped model is clearly better (rule printed below);
a backup memory_calib_pre_cap.json is written first.
"""
import csv
import json
import os
import shutil
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FHE = os.path.join(ROOT, "RL", "fhe_rl")
sys.path.insert(0, FHE)
import memory  # noqa: E402

CSV = os.path.join(ROOT, "RL", "veclang_runner", "temp", "top16384_measured.csv")
if not os.path.exists(CSV):
    print("NO MEASUREMENT FILE:", CSV)
    sys.exit(1)

calib_path = memory.CALIB_PATH
calib = json.load(open(calib_path))
cap_ratio_candidate = calib["groups"]["32768"]["ratio"]   # plateau = ratio observed on the deep group
had_cap = "cap_ratio" in calib["groups"]["16384"]
print("calibration file:", calib_path, "| cap already active:", had_cap)
print("candidate cap_ratio (= N=32768 ratio):", round(cap_ratio_candidate, 4))

rows = []
for r in csv.DictReader(open(CSV)):
    if r.get("status") != "ok" or not r.get("max_rss_MB_median"):
        print("skipped (status=%s): %s" % (r.get("status"), r.get("status_detail", "")[:120]))
        continue
    expr = r["expr"]
    d = memory.peak_ram_breakdown(expr, margin=False)
    meas = float(r["max_rss_MB_median"])
    net = d["raw_net_mb"]
    pred_now = d["mb"]
    # model without cap and with cap (independent of whether a cap is already stored)
    g = calib["groups"]["16384"]
    import math
    loglog = math.exp(g["a"] + g["b"] * math.log(max(net - d["galois_mb"], 1e-6)) + g["cL"] * d["L"])
    cap = min(loglog, cap_ratio_candidate * net)
    rows.append(dict(id=r.get("pilot_id"), net=net, L=d["L"], N=d["N"], meas=meas, loglog=loglog, cap=cap,
                     spread=r.get("max_rss_spread", "")))

if not rows:
    print("no usable rows")
    sys.exit(1)

print("\n  id      N    L   net_MB  measured  loglog  err%    capped  err%   spread")
for x in rows:
    e1 = 100 * (x["loglog"] - x["meas"]) / x["meas"]
    e2 = 100 * (x["cap"] - x["meas"]) / x["meas"]
    print("%4s %6d %3d %8.1f %9.1f %8.1f %+6.1f %8.1f %+6.1f   %s" %
          (x["id"], x["N"], x["L"], x["net"], x["meas"], x["loglog"], e1, x["cap"], e2, x["spread"]))

e_log = [100 * (x["loglog"] - x["meas"]) / x["meas"] for x in rows]
e_cap = [100 * (x["cap"] - x["meas"]) / x["meas"] for x in rows]
big = [(a, b) for a, b, x in zip(e_log, e_cap, rows) if x["net"] > 274]   # beyond the pilot training range
print("\nlog-log : median |err| %.1f%%  median signed %+.1f%%  worst under %+.1f%%  worst over %+.1f%%" %
      (st.median(abs(v) for v in e_log), st.median(e_log), min(e_log), max(e_log)))
print("capped  : median |err| %.1f%%  median signed %+.1f%%  worst under %+.1f%%  worst over %+.1f%%" %
      (st.median(abs(v) for v in e_cap), st.median(e_cap), min(e_cap), max(e_cap)))
print("rows beyond the training range (net>274 MB): %d" % len(big))
for a, b in big:
    print("   loglog %+.1f%%   capped %+.1f%%" % (a, b))

# decision rule (printed, deterministic):
#   adopt the cap if (1) its median |err| is lower than log-log's, AND (2) it never under-predicts by more than 10%,
#   AND (3) at least 3 rows are beyond the training range (otherwise the evidence is too thin to change the model)
better = st.median(abs(v) for v in e_cap) < st.median(abs(v) for v in e_log)
safe = min(e_cap) >= -10.0
enough = len(big) >= 3
print("\nRULE: median|err| capped < loglog: %s | no under-prediction beyond -10%%: %s | >=3 rows beyond range: %s" %
      (better, safe, enough))
if better and safe and enough and not had_cap:
    shutil.copy2(calib_path, os.path.join(FHE, "memory_calib_pre_cap.json"))
    calib["groups"]["16384"]["cap_ratio"] = cap_ratio_candidate
    json.dump(calib, open(calib_path, "w"), indent=2)
    print("DECISION: ADOPT CAP -> cap_ratio=%.4f written into %s (backup memory_calib_pre_cap.json)" %
          (cap_ratio_candidate, calib_path))
elif had_cap:
    print("DECISION: cap already active, nothing changed")
else:
    print("DECISION: KEEP PURE LOG-LOG (calibration file unchanged)")