#!/usr/bin/env python3
"""Phase-0 pilot sampler: stratified 200 expressions from final_llm_dataset.txt.

Stratifies by multiplicative depth x rotation presence (G==0 vs G>0,
oversampling rotations), deduplicates, and writes pilot_expressions.txt
(one expression per line) plus pilot_expressions_meta.csv with the
stratification columns. No measurement, no full run -- selection only.

Usage (from RL/):
    python3 pytrs/select_pilot_expressions.py \
        --dataset fhe_rl/datasets/final_llm_dataset.txt \
        --out veclang_runner/temp/pilot_expressions.txt \
        --meta veclang_runner/temp/pilot_expressions_meta.csv \
        --n 200 --seed 0
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parser import parse_sexpr  # noqa: E402
from cost import get_multiplicative_depth  # noqa: E402
from expr import Expr, Op  # noqa: E402


def count_rotations(expr: Expr) -> int:
    """Distinct rotation offsets (mirrors estimator liveness G)."""
    steps = set()

    def dfs(node: Expr):
        if isinstance(node, Op):
            if node.op == "<<" and len(node.args) >= 2:
                off = node.args[1]
                try:
                    from expr import Const
                    if isinstance(off, Const):
                        steps.add(off.value)
                except Exception:
                    pass
            for ch in node.args:
                dfs(ch)

    dfs(expr)
    return len(steps)


def depth_bucket(d: int) -> str:
    if d <= 2:
        return "d01-02"
    if d <= 4:
        return "d03-04"
    if d <= 6:
        return "d05-06"
    if d <= 8:
        return "d07-08"
    return "d09+"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen: set = set()
    buckets: dict = defaultdict(list)

    with open(args.dataset) as f:
        for lineno, line in enumerate(f):
            exp = line.split(":")[0].strip() if ":" in line else line.strip()
            if not exp or exp in seen:
                continue
            try:
                tree = parse_sexpr(exp)
                d = get_multiplicative_depth(tree)
                g = count_rotations(tree)
            except Exception:
                continue
            seen.add(exp)
            key = (depth_bucket(d), "G>0" if g > 0 else "G==0")
            buckets[key].append({
                "expr": exp, "depth": d, "G": g, "lineno": lineno,
            })

    print("Bucket counts (depth x rotation):")
    for k in sorted(buckets):
        print(f"  {k}: {len(buckets[k])}")

    # Allocate: half the pilot to G>0 rows (oversample rotations), spread
    # evenly across depth buckets that have rows.
    n = args.n
    n_rot = n // 2
    n_plain = n - n_rot
    rot_keys = sorted(k for k in buckets if k[1] == "G>0")
    plain_keys = sorted(k for k in buckets if k[1] == "G==0")

    def take(keys, total):
        per = max(1, total // max(len(keys), 1))
        picked = []
        for k in keys:
            pool = buckets[k][:]
            rng.shuffle(pool)
            picked.extend(pool[:per])
        rng.shuffle(picked)
        return picked[:total]

    selected = take(rot_keys, n_rot) + take(plain_keys, n_plain)
    # Backfill from any bucket if a stratum ran short.
    if len(selected) < n:
        rest = [r for k, v in buckets.items() for r in v
                if r not in selected]
        rng.shuffle(rest)
        selected.extend(rest[:n - len(selected)])
    rng.shuffle(selected)
    selected = selected[:n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in selected:
            f.write(r["expr"] + "\n")

    with open(args.meta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pilot_id", "depth", "G",
                                          "depth_bucket", "src_lineno"])
        w.writeheader()
        for i, r in enumerate(selected):
            w.writerow({"pilot_id": i, "depth": r["depth"], "G": r["G"],
                        "depth_bucket": depth_bucket(r["depth"]),
                        "src_lineno": r["lineno"]})
    n_g = sum(1 for r in selected if r["G"] > 0)
    print(f"Wrote {len(selected)} expressions to {out} ({n_g} with G>0)")


if __name__ == "__main__":
    main()
