#!/usr/bin/env python3
"""Final item-1 model selection: per-(N,L,K)-tier base + per-tier slope on the
best available feature, grouped 5-fold CV. One run, pick winner by held-out p90.
No new measurements; features derived from pilot CSV + expr parse only."""
import csv, sys, math, random
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/home/maroua/chehab-unified/RL/pytrs")
from parser import parse_sexpr
from expr import Op, Var, Const

CSV = "/home/maroua/chehab-unified/RL/veclang_runner/temp/pilot_results.csv"


def ct_bytes(N, L, K):
    return (L + 1) * N * 8 / (1024.0 * 1024.0)


def stats(expr):
    try:
        tree = parse_sexpr(expr)
    except Exception:
        return None
    n_in = n_ops = n_vec = 0
    slots = 1
    def walk(n):
        nonlocal n_in, n_ops, n_vec
        if isinstance(n, Var):
            n_in += 1
        elif isinstance(n, Op):
            n_ops += 1
            if n.op == "Vec":
                n_vec += 1
            for a in n.args:
                walk(a)
    walk(tree)
    if isinstance(tree, Op) and tree.op == "Vec":
        slots = len(tree.args)
    return dict(n_in=n_in, n_ops=n_ops, n_vec=n_vec, slots=max(slots, 1))


def bucket(d):
    d = int(d)
    return "d01-02" if d <= 2 else "d03-04" if d <= 4 else "d05-06" if d <= 6 \
        else "d07-08" if d <= 8 else "d09+"


def load():
    rows = []
    for r in csv.DictReader(open(CSV)):
        if r["status"] != "ok" or not r.get("est_net_MB") or not r.get("max_rss_MB_median"):
            continue
        st = stats(r["expr"])
        if st is None:
            continue
        N, L, K = int(r.get("N") or r["est_N"]), int(r.get("L") or r["est_L"]), int(r.get("K") or r["est_K"])
        c = ct_bytes(N, L, K)
        packed_in = math.ceil(st["n_in"] / st["slots"]) * c
        rows.append(dict(
            expr=r["expr"], kind=r["kind"], tier=(N, L, K), N=N, L=L, K=K,
            rss=float(r["max_rss_MB_median"]),
            depth=int(r.get("depth") or 0),
            est_net=float(r["est_net_MB"]),
            packed_in_MB=packed_in, input_ct_MB=st["n_in"] * c,
            op_MB=st["n_ops"] * c * 0.05,
        ))
    return rows


def design(rows, tiers, feat, log=False):
    tl = sorted(tiers); ti = {t: i for i, t in enumerate(tl)}; T = len(tl)
    ncol = T + T + 1  # T intercepts + T slopes + nothing else
    X = np.zeros((len(rows), ncol)); y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        t = ti[r["tier"]]
        X[i, t] = 1.0
        fv = r[feat]
        if log:
            fv = math.log1p(fv)
        X[i, T + t] = fv
        y[i] = math.log(r["rss"]) if log else r["rss"]
    return X, y, T


def folds(rows, k=5, seed=0):
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g[r["expr"]].append(i)
    keys = sorted(g); random.Random(seed).shuffle(keys)
    fs = [[] for _ in range(k)]
    for j, key in enumerate(keys):
        fs[j % k].extend(g[key])
    return fs


def fit(X, y, lam, n_inter):
    A = X.T @ X + lam * np.eye(X.shape[1])
    A[:n_inter, :n_inter] -= lam * np.eye(n_inter)
    return np.linalg.solve(A, X.T @ y)


def cv(rows, feat, log, lam=3.0, k=5):
    tiers = {r["tier"] for r in rows}
    X, y, T = design(rows, tiers, feat, log)
    fs = folds(rows, k)
    pred = np.zeros(len(rows))
    for f in range(k):
        te = np.array(sorted(fs[f]))
        tr = np.array(sorted(i for g in range(k) if g != f for i in fs[g]))
        w = fit(X[tr], y[tr], lam, T)
        pred[te] = X[te] @ w
    if log:
        pred = np.exp(pred)
        yr = np.array([r["rss"] for r in rows])
    else:
        yr = y
    ae = np.abs(pred - yr) / np.maximum(yr, 1e-9)
    bias = (pred - yr) / np.maximum(yr, 1e-9)
    return pred, ae, bias


def pct(a, p):
    a = np.sort(a); return a[min(len(a) - 1, int(round(p / 100 * (len(a) - 1))))]


def main():
    rows = load()
    print(f"rows {len(rows)}")
    tiers = {r["tier"] for r in rows}
    best = None
    for feat in ["est_net", "packed_in_MB", "input_ct_MB", "op_MB"]:
        for log in (False, True):
            pred, ae, bias = cv(rows, feat, log)
            p50, p90 = 100 * pct(ae, 50), 100 * pct(ae, 90)
            byb = defaultdict(list)
            for r, b in zip(rows, bias):
                byb[bucket(r["depth"])].append(b)
            wb = max(abs(np.mean(byb[b])) * 100 for b in byb if byb[b])
            print(f"feat={feat:12} log={int(log)}: p50 {p50:5.1f}% p90 {p90:6.1f}% "
                  f"bias {100*bias.mean():+5.1f}% worst-bucket {wb:+5.1f}%")
            if best is None or (p90 < best[0]):
                best = (p90, p50, feat, log, pred, ae, bias)
    _, p50, feat, log, pred, ae, bias = best
    print(f"\n== BEST feat={feat} log={int(log)}: held-out p50 {p50:.1f}% "
          f"p90 {best[0]:.1f}% max {100*ae.max():.1f}% ==")
    byb = defaultdict(list)
    for r, b in zip(rows, bias):
        byb[bucket(r["depth"])].append(b)
    for b in ["d01-02", "d03-04", "d05-06", "d07-08", "d09+"]:
        if byb[b]:
            print(f"  {b}: bias {100*np.mean(byb[b]):+5.1f}% abs-p50 "
                  f"{100*pct(np.abs(byb[b]),50):5.1f}% n={len(byb[b])}")
    from scipy.stats import spearmanr
    print(f"Spearman(pred,actual) {spearmanr(pred,[r['rss'] for r in rows]).correlation:.3f}")
    # per-tier table
    byt = defaultdict(list)
    for r, p, a in zip(rows, pred, ae):
        byt[r["tier"]].append(a)
    print("per-tier abs-p50/p90:")
    for t in sorted(byt):
        print(f"  {t}: p50 {100*pct(byt[t],50):.1f}% p90 {100*pct(byt[t],90):.1f}% n={len(byt[t])}")


if __name__ == "__main__":
    main()
