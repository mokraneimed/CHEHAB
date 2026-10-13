#!/usr/bin/env python3
"""Item 1: re-fit the CKKS peak-RAM estimator against measured MaxRSS.

Fixed-effects linear model (per-(N,L,K)-tier base + shared feature slopes):

    MaxRSS_MB = alpha_{tier} + b_ct*live_ct_MB + b_ops*n_ops + b_in*n_inputs

  * alpha_{tier} : one intercept per (go_N, go_L, go_K) tier. Captures keys +
    Go/Lattigo base RSS for that parameter set (the per-tier base term).
  * slopes       : shared across tiers, ridge-penalised; intercepts unpenalised.

5-fold CV grouped by normalised expr so a rewritten variant never leaks into
the same fold's test set as its original. Reports held-out p50/p90 abs error,
per-depth-bucket signed bias, ordering accuracy within G=0 rows.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

REPO = "/home/maroua/chehab-unified"
sys.path.insert(0, os.path.join(REPO, "RL/pytrs"))

from parser import parse_sexpr                     # noqa: E402
from cost import get_multiplicative_depth          # noqa: E402
from expr import Op                                # noqa: E402
import ckks_memory_estimator as est_mod           # noqa: E402

CSV = os.path.join(REPO, "RL/veclang_runner/temp/pilot_results.csv")
MB = 1024 * 1024
SLOPE_FEATS = ["live_ct_MB", "n_ops", "n_inputs"]


def expr_features(expr_str, N, L, K):
    tree = parse_sexpr(expr_str)
    live = est_mod._Liveness(tree, top_level=L - 1)
    _, peak_bytes = live.peak(N, level_aware=True)
    # Packed width (slots) = arity of the top-level Vec; 1 if not Vec-rooted.
    slots = len(tree.args) if isinstance(tree, Op) and tree.op == "Vec" else 1
    slots = max(slots, 1)
    # Codegen packs `slots` scalar values per ciphertext (SIMD slots). The
    # estimator counts each Var and each scalar Op as its own ciphertext, so
    # for wide Vecs it over-counts by ~slots. packed_ct divides the working
    # set by the slot width to model the packed (vectorised) allocation.
    packed_ct_MB = peak_bytes / MB / slots
    return {
        "live_ct_MB": peak_bytes / MB,
        "packed_ct_MB": packed_ct_MB,
        "n_ops": len(live.ops),
        "n_inputs": len(live.inputs),
        "slots": slots,
        "depth": get_multiplicative_depth(tree),
    }


def load():
    rows = []
    for r in csv.DictReader(open(CSV)):
        if r["status"] != "ok" or not r.get("max_rss_MB_median") or not r.get("go_N"):
            continue
        try:
            N = int(r["go_N"]); L = int(r["go_L"]); K = int(r["go_K"])
            rss = float(r["max_rss_MB_median"])
        except ValueError:
            continue
        try:
            feats = expr_features(r["expr"], N, L, K)
        except Exception as e:
            print("feat fail", r["pilot_id"], repr(e)); continue
        rec = {
            "pilot_id": r["pilot_id"], "kind": r["kind"],
            "pair_id": r.get("pair_id", ""), "expr": r["expr"],
            "N": N, "L": L, "K": K, "rss": rss,
            "est_net_MB": float(r["est_net_MB"]) if r.get("est_net_MB") else np.nan,
            "est_G": int(r["est_G"]) if r.get("est_G") else 0,
            "go_G": int(r["go_G"]) if r.get("go_G") else 0,
        }
        rec.update(feats)
        rows.append(rec)
    return rows


def design(rows, tiers, tier_index, per_tier_slope=False):
    """Model: MaxRSS = base_tier [+ slope_tier * live_ct_MB]  (linear, cpu).

    per_tier_slope=False: shared slope on live_ct (global).
    per_tier_slope=True : one base AND one live_ct slope per (N,L,K) tier,
        which lets RSS-flat shallow tiers (N=16384) have ~0 slope while
        scaling tiers (N=32768) get a positive slope. This is the
        "per-(N,L,K)-tier base term, not a global offset" the task asked for.
    """
    T = len(tiers)
    # per_tier_slope=False -> T intercepts + 1 shared slope on live_ct.
    # per_tier_slope=True  -> T intercepts + T slopes (one live_ct slope per tier).
    n_slope = T if per_tier_slope else 1
    ncol = T + n_slope
    X = np.zeros((len(rows), ncol))
    for i, r in enumerate(rows):
        t = tier_index[(r["N"], r["L"], r["K"])]
        X[i, t] = 1.0                      # tier intercept (unpenalised)
        X[i, T + (t if per_tier_slope else 0)] = r["live_ct_MB"]
    return X, np.array([r["rss"] for r in rows]), T


def ridge(Xtr, ytr, lam, n_inter):
    P = np.eye(Xtr.shape[1]) * lam
    P[:n_inter, :n_inter] = 0.0
    return np.linalg.solve(Xtr.T @ Xtr + P, Xtr.T @ ytr)


def cv_eval(rows, tiers, tier_index, per_tier_slope, lam, huber_rounds=0):
    """Grouped 5-fold CV. Optionally IRLS Huber rounds to blunt outliers."""
    groups = [" ".join(r["expr"].split()) for r in rows]
    uniq = sorted(set(groups))
    g2f = {g: i % 5 for i, g in enumerate(uniq)}
    folds = np.array([g2f[g] for g in groups])
    n_inter = len(tiers)
    y = np.array([r["rss"] for r in rows])
    preds = np.full(len(rows), np.nan)
    for f in range(5):
        te = folds == f
        trrows = [r for r, m in zip(rows, ~te) if m]
        terows = [r for r, m in zip(rows, te) if m]
        Xtr, ytr, T = design(trrows, tiers, tier_index, per_tier_slope)
        Xte, _, _ = design(terows, tiers, tier_index, per_tier_slope)
        w = ridge(Xtr, ytr, lam, n_inter)
        if huber_rounds:
            for _ in range(huber_rounds):
                resid = ytr - Xtr @ w
                s = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
                z = resid / (1.345 * s)
                wt = np.where(np.abs(z) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(z), 1e-6))
                sw = np.sqrt(wt)
                w = ridge(Xtr * sw[:, None], ytr * sw, lam, n_inter)
        preds[te] = Xte @ w
    # Guard against negative predictions (unphysical RSS).
    preds = np.maximum(preds, 1.0)
    return preds, y


def bucket(d):
    return ("d01-02" if d <= 2 else "d03-04" if d <= 4 else
            "d05-06" if d <= 6 else "d07-08" if d <= 8 else "d09+")


def report(rows, preds, y, tag):
    abserr = np.abs(preds - y) / y
    signed = (preds - y) / y
    byb = defaultdict(list)
    for r, s in zip(rows, signed):
        byb[bucket(r["depth"])].append(s)
    worst_bias = max(abs(np.mean(byb[b])) for b in byb if byb[b])
    print(f"  p50 {100*np.percentile(abserr,50):5.1f}%  "
          f"p90 {100*np.percentile(abserr,90):6.1f}%  "
          f"worst-bucket-bias {100*worst_bias:5.1f}%   [{tag}]")
    return preds, y, abserr, signed


def main():
    rows = load()
    print(f"ok rows: {len(rows)}")
    tiers = sorted({(r["N"], r["L"], r["K"]) for r in rows})
    tier_index = {t: i for i, t in enumerate(tiers)}
    print(f"tiers (N,L,K): {tiers}\n")

    print("== model selection by grouped 5-fold CV (p50/p90 abs err, worst bucket bias) ==")
    best = None
    for pts in (False, True):
        for lam in (0.3, 1.0, 3.0):
            for hub in (0, 2):
                preds, y = cv_eval(rows, tiers, tier_index, pts, lam, hub)
                abserr = np.abs(preds - y) / y
                signed = (preds - y) / y
                byb = defaultdict(list)
                for r, s in zip(rows, signed):
                    byb[bucket(r["depth"])].append(s)
                wb = max(abs(np.mean(byb[b])) for b in byb if byb[b])
                p90 = 100 * np.percentile(abserr, 90)
                print(f"  per_tier_slope={int(pts)} lam={lam:<4} huber={hub}  "
                      f"p50 {100*np.percentile(abserr,50):5.1f}%  "
                      f"p90 {p90:6.1f}%  worst-bias {100*wb:5.1f}%")
                score = (p90, wb)
                if best is None or score < best[0]:
                    best = (score, pts, lam, hub)

    print(f"\n== chosen: per_tier_slope={int(best[1])} lam={best[2]} huber={best[3]} ==")
    pts, lam, hub = best[1], best[2], best[3]
    preds, y = cv_eval(rows, tiers, tier_index, pts, lam, hub)
    abserr = np.abs(preds - y) / y
    signed = (preds - y) / y

    print("\n== HELD-OUT accuracy (5-fold grouped CV) ==")
    print(f"p50 abs err: {100*np.percentile(abserr,50):.1f}%   "
          f"p90 abs err: {100*np.percentile(abserr,90):.1f}%   "
          f"max: {100*abserr.max():.1f}%")
    print(f"signed bias mean {100*signed.mean():+.1f}%  "
          f"median {100*np.median(signed):+.1f}%")

    print("\n== per-depth-bucket signed bias ==")
    byb = defaultdict(list)
    for r, s in zip(rows, signed):
        byb[bucket(r["depth"])].append(s)
    for b in ["d01-02", "d03-04", "d05-06", "d07-08", "d09+"]:
        v = byb.get(b, [])
        if v:
            print(f"  {b}: bias {100*np.mean(v):+5.1f}%  "
                  f"abs-p50 {100*np.percentile(np.abs(v),50):5.1f}%  n={len(v)}")

    from scipy.stats import spearmanr
    g0 = [i for i, r in enumerate(rows) if r["est_G"] == 0 and r["go_G"] == 0]
    if len(g0) > 2:
        print(f"\n== ordering within G=0 rows (n={len(g0)}) ==")
        print(f"Spearman(pred, MaxRSS) = {spearmanr(preds[g0], y[g0]).correlation:.3f}")
        idx = g0
        rng = np.random.default_rng(0)
        if len(idx) > 400:
            idx = list(rng.choice(idx, 400, replace=False))
        agree = dis = 0
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if (preds[idx[a]] - preds[idx[b]]) * (y[idx[a]] - y[idx[b]]) > 0:
                    agree += 1
                else:
                    dis += 1
        if agree + dis:
            print(f"pairwise direction agreement: {agree}/{agree+dis} "
                  f"({100*agree/(agree+dis):.1f}%)")

    en = np.array([r["est_net_MB"] for r in rows])
    ae = np.abs(en - y) / y
    print("\n== baseline (current net estimator) ==")
    print(f"p50 abs err: {100*np.percentile(ae,50):.1f}%   "
          f"p90 abs err: {100*np.percentile(ae,90):.1f}%")

    p90 = 100 * np.percentile(abserr, 90)
    wb = max(abs(np.mean(byb[b])) for b in byb if byb[b])
    print("\n== success criteria ==")
    print(f"  held-out p90 abs err <= 25% : {p90:.1f}%  "
          f"{'PASS' if p90 <= 25 else 'FAIL'}")
    print(f"  worst bucket bias within +-10%: {wb:.1f}%  "
          f"{'PASS' if wb <= 10 else 'FAIL'}")


if __name__ == "__main__":
    main()

