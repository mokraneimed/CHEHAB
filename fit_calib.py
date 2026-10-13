"""Fit memory_calib.json from the pilot.  Run from the repo root:
     python3 fit_calib.py RL/veclang_runner/temp/pilot_results.csv RL/fhe_rl/memory_calib.json
Only G=0 rows are used (G>0 rows are not valid ground truth)."""
import csv, json, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
rows = []
for r in csv.DictReader(open(src)):
    if r["status"] != "ok" or not r.get("max_rss_MB_median") or not r.get("est_net_MB"): continue
    if int(float(r.get("est_G") or 0)) != 0: continue
    rows.append(dict(expr=r["expr"], N=int(r["est_N"]), L=int(r["est_L"]),
                     est=float(r["est_net_MB"]), rss=float(r["max_rss_MB_median"])))

def folds(rs, k=5):
    g = sorted({r["expr"] for r in rs}); np.random.default_rng(0).shuffle(g)
    f = {e: i % k for i, e in enumerate(g)}
    return np.array([f[r["expr"]] for r in rs])

def fit(model, rs):
    e = np.array([r["est"] for r in rs]); y = np.array([r["rss"] for r in rs])
    L = np.array([r["L"] for r in rs], float)
    if model == "ratio":
        return {"model": "ratio", "ratio": float(np.median(y / e))}
    A = np.c_[np.ones(len(rs)), np.log(e), L]
    c, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    return {"model": "loglog", "a": float(c[0]), "b": float(c[1]), "cL": float(c[2])}

def predict(c, rs):
    e = np.array([r["est"] for r in rs]); L = np.array([r["L"] for r in rs], float)
    return c["ratio"] * e if c["model"] == "ratio" else np.exp(c["a"] + c["b"] * np.log(e) + c["cL"] * L)

plan = {"16384": ("loglog", lambda n: n <= 16384), "32768": ("ratio", lambda n: n > 16384)}
out = {"groups": {}, "n_rows": len(rows), "note": "fit on G=0 pilot rows only"}
for g, (model, sel) in plan.items():
    rs = [r for r in rows if sel(r["N"])]
    fo = folds(rs); y = np.array([r["rss"] for r in rs]); oof = np.zeros(len(rs))
    for k in range(5):
        c = fit(model, [r for r, f in zip(rs, fo) if f != k])
        oof[fo == k] = predict(c, [r for r, f in zip(rs, fo) if f == k])
    c = fit(model, rs)                      # final fit on all rows
    c["margin"] = float(max(1.0, np.percentile(y / oof, 90)))
    c["n"] = len(rs)
    absp = np.abs(oof - y) / y * 100
    c["heldout_p50_abs_pct"] = float(np.percentile(absp, 50)); c["heldout_p90_abs_pct"] = float(np.percentile(absp, 90))
    out["groups"][g] = c
    print(g, c)
json.dump(out, open(dst, "w"), indent=2)
print("wrote", dst)