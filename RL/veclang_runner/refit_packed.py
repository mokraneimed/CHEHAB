#!/usr/bin/env python3
"""Item-1 refit: CKKS peak-RAM estimator vs MaxRSS, grouped 5-fold CV.

Physically-correct packed features:
  * base term  : one coefficient per (N,L,K) tier  (params+runtime floor)
  * input_ct_MB: distinct input vars  x ct_bytes(N,L)   (packed inputs)
  * peak_ct_MB : dependency peak of {Var, Vec-node} cts (each Vec = 1 ct)
  * op_MB      : n_ops x per-op working scalar (temp/plaintext ref bytes)
Per-tier slopes are optional (off by default; tiers are sparse).

Grouped by expression string so a pair's orig/rw land in the same fold.
"""
import csv
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/home/maroua/chehab-unified/RL/pytrs")
from parser import parse_sexpr          # noqa: E402
from expr import Op, Var, Const         # noqa: E402

CSV = "/home/maroua/chehab-unified/RL/veclang_runner/temp/pilot_results.csv"
SEED = 0


def ct_bytes(N, L, K):
    # ciphertext at given level: (level+1) ring polys of N coeffs, 8 bytes
    polys = L + 1
    return polys * N * 8 / (1024.0 * 1024.0)


def expr_stats(expr):
    """Packed physical features. Returns dict or None if unparsable."""
    try:
        tree = parse_sexpr(expr)
    except Exception:
        return None
    n_inputs = 0
    n_vec = 0
    n_ops = 0

    def walk(n):
        nonlocal n_inputs, n_vec, n_ops
        if isinstance(n, Var):
            n_inputs += 1
        elif isinstance(n, Op):
            n_ops += 1
            if n.op == "Vec":
                n_vec += 1
            for a in n.args:
                walk(a)

    walk(tree)

    # dependency peak: each Var and each Vec subtree = 1 ciphertext.
    # count references: a node's ct is live from first demand until last use.
    order = []
    uses = defaultdict(list)   # node id -> list of positions it is consumed

    # assign a stable index to each node via id()
    def assign(n):
        if isinstance(n, (Var, Const)):
            return
        if isinstance(n, Op):
            for a in n.args:
                assign(a)
            if n.op == "Vec":
                order.append(id(n))
    # We compute peak liveness of {Var, Vec} nodes over a post-order schedule.
    # Represent each producer node; children consumed by parents.
    nodes = []          # list of node objects that are "ct-producing"
    parent = {}         # child node -> producer parent
    def collect(n):
        if isinstance(n, Op):
            for a in n.args:
                parent[id(a)] = id(n)
            collect(n)
    # Simpler: peak number of simultaneously-live produced cts.
    # A ct is produced once and consumed once (tree DAG, each node one parent).
    # live(node) = its children's live + 1 at the moment it completes.
    from functools import lru_cache
    def peak(n):
        # returns (peak, height) for subtree rooted at n
        if isinstance(n, Var):
            return 1, 1
        if isinstance(n, Const):
            return 0, 0
        child_pk, child_h = 0, 0
        for a in n.args:
            pk, h = peak(a)
            child_pk = max(child_pk, pk)
            child_h = max(child_h, h)
        is_ct = isinstance(n, Op) and (n.op == "Vec" or n.op == "<<")
        add = 1 if is_ct else 0
        # this node's ct lives alongside the taller child subtree
        pk = child_pk + add
        h = child_h + add
        return pk, h

    pk_cts, _ = peak(tree)
    return dict(n_inputs=n_inputs, n_vec=n_vec, n_ops=n_ops, peak_cts=pk_cts)


def design(rows, tiers, per_tier_slope=False):
    """Global tier set -> fixed column count (no per-fold dimension drift)."""
    tier_list = sorted(tiers)
    tindex = {t: i for i, t in enumerate(tier_list)}
    T = len(tier_list)
    nslope = T if per_tier_slope else 1
    ncol = T + (1 + nslope)  # intercepts + input_ct(+ slopes) + ... below
    # columns: [T one-hot] [input_ct shared] [peak_ct shared] [op shared]
    #         [per-tier input_ct slope?] [per-tier peak_ct slope?]
    base_cols = T + 3
    if per_tier_slope:
        ncol = base_cols + 2 * T
    else:
        ncol = base_cols
    X = np.zeros((len(rows), ncol))
    y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        ti = tindex[r["tier"]]
        X[i, ti] = 1.0
        c = ct_bytes(int(r["N"]), int(r["L"]), int(r["K"]))
        X[i, T + 0] = r["input_ct_MB"]
        X[i, T + 1] = r["peak_ct_MB"]
        X[i, T + 2] = r["op_MB"]
        if per_tier_slope:
            X[i, base_cols + ti] = r["input_ct_MB"]
            X[i, base_cols + T + ti] = r["peak_ct_MB"]
        y[i] = r["rss"]
    return X, y, tier_list, ncol


def grouped_folds(rows, k=5, seed=SEED):
    import random
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["expr"]].append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    folds = [[] for _ in range(k)]
    for j, key in enumerate(keys):
        folds[j % k].extend(groups[key])
    return folds


def ridge_fit(X, y, lam, n_inter):
    A = X.T @ X + lam * np.eye(X.shape[1])
    A[:n_inter, :n_inter] -= lam * np.eye(n_inter)  # don't penalize intercepts
    b = X.T @ y
    return np.linalg.solve(A, b)


def cv_eval(rows, per_tier_slope, lam, k=5):
    tiers = {(int(r["N"]), int(r["L"]), int(r["K"])) for r in rows}
    X, y, tier_list, ncol = design(rows, tiers, per_tier_slope)
    n_inter = len(tier_list)
    folds = grouped_folds(rows, k)
    preds = np.zeros(len(rows))
    for f in range(k):
        te = np.array(sorted(folds[f]))
        tr = np.array(sorted(i for g in range(k) if g != f for i in folds[g]))
        w = ridge_fit(X[tr], y[tr], lam, n_inter)
        preds[te] = X[te] @ w
    abs_err = np.abs(preds - y) / np.maximum(y, 1e-9)
    bias = (preds - y) / np.maximum(y, 1e-9)
    return preds, abs_err, bias, tier_list


def depth_bucket(d):
    d = int(d)
    return ("d01-02" if d <= 2 else "d03-04" if d <= 4 else
            "d05-06" if d <= 6 else "d07-08" if d <= 8 else "d09+")


def pct(a, p):
    a = np.sort(a)
    idx = min(len(a) - 1, int(round(p / 100.0 * (len(a) - 1))))
    return a[idx]


def main():
    rows = []
    for r in csv.DictReader(open(CSV)):
        if r["status"] != "ok" or not r.get("est_net_MB") or \
                not r.get("max_rss_MB_median"):
            continue
        st = expr_stats(r["expr"])
        if st is None:
            continue
        N, L, K = int(r["N"]) if r.get("N") else int(r["est_N"]), \
            int(r["L"]) if r.get("L") else int(r["est_L"]), \
            int(r["K"]) if r.get("K") else int(r["est_K"])
        c = ct_bytes(N, L, K)
        rows.append(dict(
            expr=r["expr"], kind=r["kind"], N=N, L=L, K=K,
            tier=(N, L, K), rss=float(r["max_rss_MB_median"]),
            depth=int(r["depth"]) if r.get("depth") else 0,
            input_ct_MB=st["n_inputs"] * c,
            peak_ct_MB=st["peak_cts"] * c,
            op_MB=st["n_ops"] * c * 0.05,
        ))
    print(f"ok+parsable rows: {len(rows)}")

    best = None
    for pts in (False, True):
        for lam in (0.3, 1.0, 3.0):
            preds, ae, bias, tiers = cv_eval(rows, pts, lam)
            p50, p90 = 100 * pct(ae, 50), 100 * pct(ae, 90)
            print(f"per_tier_slope={pts} lam={lam}: p50 {p50:.1f}% "
                  f"p90 {p90:.1f}%  bias {100*bias.mean():+.1f}%")
            if best is None or p90 < best[0]:
                best = (p90, pts, lam, preds, ae, bias, tiers)

    _, pts, lam, preds, ae, bias, tiers = best
    print(f"\n== BEST per_tier_slope={pts} lam={lam} ==")
    print(f"held-out p50 {100*pct(ae,50):.1f}%  p90 {100*pct(ae,90):.1f}%  "
          f"max {100*ae.max():.1f}%")
    print(f"overall signed bias {100*bias.mean():+.1f}%")

    byb = defaultdict(list)
    for r, b in zip(rows, bias):
        byb[depth_bucket(r["depth"])].append(b)
    print("\nper-depth-bucket signed bias:")
    for b in ["d01-02", "d03-04", "d05-06", "d07-08", "d09+"]:
        if byb[b]:
            print(f"  {b}: {100*np.mean(byb[b]):+.1f}%  "
                  f"abs-p50 {100*pct(np.abs(byb[b]),50):.1f}%  n={len(byb[b])}")

    # ordering accuracy within G=0 rows (sign of RSS difference across
    # pairs, which here is approximated by within-tier rank correlation)
    from scipy.stats import spearmanr
    g0 = [i for i, r in enumerate(rows)]
    rho = spearmanr(preds[g0], [rows[i]["rss"] for i in g0]).correlation
    print(f"\nSpearman(pred, actual) over all ok rows: {rho:.3f}")


if __name__ == "__main__":
    main()

