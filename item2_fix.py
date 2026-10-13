# item2_fix2.py  -- python3 item2_fix2.py > item2_out2.txt 2>&1
import csv, numpy as np
from scipy.optimize import nnls

CSV = "RL/veclang_runner/temp/pilot_results.csv"
rows = []
for r in csv.DictReader(open(CSV)):
    if r["status"] != "ok" or not r.get("max_rss_MB_median") or not r.get("est_net_MB"): continue
    if int(float(r.get("est_G") or 0)) != 0: continue
    rows.append(dict(expr=r["expr"], depth=int(r["depth"]), N=int(r["est_N"]), L=int(r["est_L"]),
                     est=float(r["est_net_MB"]), rss=float(r["max_rss_MB_median"])))

def cv_groups(rs, k=5, seed=0):
    g = sorted({r["expr"] for r in rs}); np.random.default_rng(seed).shuffle(g)
    f = {e: i % k for i, e in enumerate(g)}
    return np.array([f[r["expr"]] for r in rs])

def fit_pred(name, tr, te):
    est_tr = np.array([r["est"] for r in tr]); y_tr = np.array([r["rss"] for r in tr])
    est_te = np.array([r["est"] for r in te])
    if name == "ratio":                       # rss = c * est, c = median train ratio
        return est_te * np.median(y_tr / est_tr)
    if name == "affine_nn":                   # rss = a + b*est, a,b >= 0
        c, _ = nnls(np.c_[np.ones(len(tr)), est_tr], y_tr)
        return c[0] + c[1] * est_te
    if name == "loglog":                      # log rss = a + b*log est (+ c*L)
        L_tr = np.array([r["L"] for r in tr]); L_te = np.array([r["L"] for r in te])
        A = np.c_[np.ones(len(tr)), np.log(est_tr), L_tr]
        c, *_ = np.linalg.lstsq(A, np.log(y_tr), rcond=None)
        return np.exp(np.c_[np.ones(len(te)), np.log(est_te), L_te] @ c)

for n in sorted({r["N"] for r in rows}):
    rs = [r for r in rows if r["N"] == n]; y = np.array([r["rss"] for r in rs]); fold = cv_groups(rs)
    print("\n=== N=%d  n=%d  distinct exprs=%d ===" % (n, len(rs), len({r['expr'] for r in rs})))
    print("raw estimator: median rss/est = %.2f | p10 %.2f | p90 %.2f"
          % tuple(np.percentile([r["rss"]/r["est"] for r in rs], q) for q in (50, 10, 90)))
    for name in ("ratio", "affine_nn", "loglog"):
        oof = np.zeros(len(rs))
        for k in range(5):
            tr = [r for r, f in zip(rs, fold) if f != k]; te = [r for r, f in zip(rs, fold) if f == k]
            oof[fold == k] = fit_pred(name, tr, te)
        absp = np.abs(oof - y) / y * 100; sg = (oof - y) / y * 100
        ratio = y / oof; m90 = max(1.0, np.percentile(ratio, 90)); m95 = max(1.0, np.percentile(ratio, 95))
        cov = ((oof * m90) >= y).mean() * 100
        print(" %-10s p50 abs %5.1f%% | p90 abs %6.1f%% | median signed %6.1f%% | under-rate %3.0f%% "
              "| margin p90 %.2f (p95 %.2f), coverage %.0f%% | worst under %.1f%%"
              % (name, np.percentile(absp, 50), np.percentile(absp, 90), np.median(sg),
                 100*(sg < 0).mean(), m90, m95, cov, sg.min()))
    # worst rows for the best-looking simple model
    est = np.array([r["est"] for r in rs]); ratio_all = y / est
    idx = np.argsort(np.abs(np.log(ratio_all)))[-5:]
    print(" most extreme rss/est rows (est, rss, depth, len):",
          [(round(est[i]), round(y[i]), rs[i]["depth"], len(rs[i]["expr"])) for i in idx])