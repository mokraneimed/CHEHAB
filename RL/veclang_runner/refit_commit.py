#!/usr/bin/env python3
"""Item-1 re-fit: CKKS peak-RAM estimator vs pilot MaxRSS (ok rows).

Per-(N,L,K) tier base + per-tier slopes on free physical features
(n_ct_in, ops, est_net). Grouped 5-fold CV (group=expr). Reports p50/p90
abs error, per-depth-bucket bias, G=0 ordering. Winner -> temp/calib.json.
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = "/home/maroua/chehab-unified"
sys.path.insert(0, os.path.join(REPO, "RL/pytrs"))
from parser import parse_sexpr  # noqa: E402
from expr import Op, Var        # noqa: E402

CSV = os.path.join(REPO, "RL/veclang_runner/temp/pilot_results.csv")


def expr_features(expr):
    try:
        t = parse_sexpr(expr)
    except Exception:
        return 0.0, 0.0
    names, nops = set(), [0]

    def rec(n):
        if isinstance(n, Var):
            names.add(n.name)
        elif isinstance(n, Op):
            nops[0] += 1
            for a in n.args:
                rec(a)
    rec(t)
    return float(len(names)), float(nops[0])


def load():
    rows = []
    for r in csv.DictReader(open(CSV)):
        if r["status"] != "ok" or not r.get("max_rss_MB_median"):
            continue
        try:
            rows.append({
                "id": r["pilot_id"], "kind": r["kind"], "expr": r["expr"],
                "depth": int(r["depth"]), "N": int(r["est_N"]),
                "L": int(r["est_L"]), "K": int(r["est_K"]),
                "G": int(r["est_G"] or 0),
                "est_net": float(r["est_net_MB"]),
                "rss": float(r["max_rss_MB_median"]),
            })
        except (ValueError, KeyError):
            continue
    for r in rows:
        r["n_ct_in"], r["ops"] = expr_features(r["expr"])
    return rows


def design(rows, tiers, feats, per_tier_slope):
    T = len(tiers)
    ti = {t: i for i, t in enumerate(tiers)}
    bonus = (T if per_tier_slope else 1) * len(feats)
    X = np.zeros((len(rows), T + bonus))
    y = np.empty(len(rows))
    for i, r in enumerate(rows):
        k = ti[(r["N"], r["L"], r["K"])]
        X[i, k] = 1.0
        for j, f in enumerate(feats):
            col = T + (j * T + k if per_tier_slope else j)
            X[i, col] = r[f]
        y[i] = r["rss"]
    return X, y


def ridge(X, y, lam, n_inter):
    A = X.T @ X + lam * np.eye(X.shape[1])
    A[:n_inter, :n_inter] -= lam * np.eye(n_inter)
    return np.linalg.solve(A, X.T @ y)


def groups(rows):
    seen, g = {}, []
    for r in rows:
        seen.setdefault(r["expr"], len(seen))
        g.append(seen[r["expr"]])
    return np.array(g)


def kfold(g, k=5):
    ug = np.unique(g)
    order = {u: i for i, u in enumerate(ug)}
    fold_of = np.array([order[x] % k for x in g])
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0])
            for f in range(k)]


def bucket(d):
    return ("d01-02" if d <= 2 else "d03-04" if d <= 4 else
            "d05-06" if d <= 6 else "d07-08" if d <= 8 else "d09+")


def report(name, pred, y, rows, tag=""):
    a = np.abs(pred - y) / np.maximum(y, 1e-9) * 100
    b = (pred - y) / np.maximum(y, 1e-9) * 100
    p50, p90 = np.percentile(a, 50), np.percentile(a, 90)
    bb = defaultdict(list)
    for r, bi in zip(rows, b):
        bb[bucket(r["depth"])].append(bi)
    wk = max((abs(np.mean(v)), k, np.mean(v)) for k, v in bb.items())
    print(f"{tag}{name}: p50 {p50:.1f}%  p90 {p90:.1f}%  "
          f"| worst-bucket {wk[1]} {wk[2]:+.1f}%  "
          f"| frac<=25% {np.mean(a<=25)*100:.0f}")
    return p50, p90, {k: float(np.mean(v)) for k, v in bb.items()}


def order_acc(rows, pred):
    sub = [i for i in range(len(rows)) if rows[i]["G"] == 0]
    ag = tot = 0
    order = sorted(sub, key=lambda i: pred[i])
    for ai, bi in zip(order[:-1], order[1:]):
        if rows[ai]["depth"] != rows[bi]["depth"]:
            continue
        dp = pred[bi] - pred[ai]
        dm = rows[bi]["rss"] - rows[ai]["rss"]
        if abs(dm) < 1e-6:
            continue
        tot += 1
        if dp * dm > 0:
            ag += 1
    return (100.0 * ag / tot) if tot else float("nan"), tot


def main():
    allrows = load()
    rows = [r for r in allrows if r["G"] == 0]
    y_all = np.array([r["rss"] for r in rows])
    print(f"ok rows: {len(allrows)}  G>0 excluded: "
          f"{len(allrows)-len(rows)}  | G=0 calibration: {len(rows)}")
    tiers = sorted({(r["N"], r["L"], r["K"]) for r in rows})
    tc = Counter((r["N"], r["L"], r["K"]) for r in rows)
    print("tiers:", {t: tc[t] for t in tiers})
    g = groups(rows)
    folds = kfold(g, 5)

    variants = [
        ("base-only", [], False),
        ("base+net", ["est_net"], False),
        ("base+net(pt)", ["est_net"], True),
        ("base+ctin+ops", ["n_ct_in", "ops"], False),
        ("base+ctin+ops(pt)", ["n_ct_in", "ops"], True),
        ("base+net+ctin+ops(pt)", ["est_net", "n_ct_in", "ops"], True),
    ]
    results = {}
    for name, feats, pts in variants:
        X, _ = design(rows, tiers, feats, pts)
        oof = np.empty(len(rows))
        for tr, te in folds:
            b = ridge(X[tr], y_all[tr], 1.0, len(tiers))
            oof[te] = X[te] @ b
        results[name] = (oof, feats, pts)
        report(name, oof, y_all, rows, tag="[CV] ")

    best = min(results, key=lambda k: np.percentile(
        np.abs(results[k][0] - y_all) / np.maximum(y_all, 1e-9), 90))
    oof, feats, pts = results[best]
    print(f"\n== WINNER: {best} ==")
    report(best, oof, y_all, rows)
    # Per-tier decomposition (in-sample on full data, for diagnosis).
    X, _ = design(rows, tiers, feats, pts)
    bfull = ridge(X, y_all, 1.0, len(tiers))
    pfull = X @ bfull
    bt = defaultdict(list)
    for r, p, yy in zip(rows, pfull, y_all):
        bt[(r["N"], r["L"], r["K"])].append((p - yy) / max(yy, 1e-9) * 100)
    print("  per-tier (in-sample):")
    for t in sorted(bt):
        v = bt[t]
        print(f"    {t}: n={len(v):3d}  bias {np.mean(v):+6.1f}%  "
              f"abs-p90 {np.percentile(np.abs(v),90):6.1f}%")
    oa, tot = order_acc(rows, oof)
    print(f"\nordering (G=0, same-depth, {tot} comparisons): {oa:.1f}%")

    X, _ = design(rows, tiers, feats, pts)
    b = ridge(X, y_all, 1.0, len(tiers))
    coef = {"calibrated_on": "G=0", "tiers": [list(t) for t in tiers],
            "feats": feats, "per_tier_slope": pts, "n_inter": len(tiers),
            "coef": b.tolist()}
    out = os.path.join(REPO, "RL/veclang_runner/temp/calib.json")
    with open(out, "w") as f:
        json.dump(coef, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
