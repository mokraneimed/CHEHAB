#!/usr/bin/env python3
"""Phase-0 pilot rotation-variant synthesizer: (original, rewritten) pairs.

The veclang veclang_runner C++ codegen expects the flat ``(Vec ...)`` dialect
(see RL/pytrs/expr.py and src/fheco/dsl/compiler.cpp). Rotations there are the
binary operator ``(<< vector int-Const)`` -- NOT the ``(VecMul (VecAdd ...))``
nested vectorization IR, which the codegen cannot parse (it spins forever).

This builds each rewritten variant by wrapping selected top-level Vec elements
in ``(<< elem step)`` while keeping the ``(Vec ...)`` root intact, so:
  * parse_sexpr / validate_expression accept it (checked per row),
  * the estimator counts G>0 (Lattigo Galois keys),
  * the C++ codegen and Go backend build and emit galois_keys.

Output: pilot_rotation_pairs.csv with columns
   pair_id, orig_expr, rewritten_expr, orig_depth, rewritten_depth,
   orig_G, rewritten_G, rule, status
Only rows with rewritten_G > 0 and a validated, re-parsable rewrite are kept.

Usage:
    python3 pytrs/synthesize_rotation_pairs.py \
        --in veclang_runner/temp/pilot_expressions.txt \
        --out veclang_runner/temp/pilot_rotation_pairs.csv \
        --n 100 --seed 1
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parser import parse_sexpr              # noqa: E402
from serializer import expr_to_str          # noqa: E402
from cost import get_multiplicative_depth   # noqa: E402
from expr import Expr, Op, Const            # noqa: E402

STEPS = (1, 2, 4)


def count_rotations(tree: Expr) -> int:
    """Distinct rotation step constants (liveness-relevant Galois keys)."""
    steps = set()

    def dfs(node: Expr):
        if isinstance(node, Op):
            if node.op == "<<" and len(node.args) >= 2 and \
                    isinstance(node.args[1], Const):
                steps.add(node.args[1].value)
            for ch in node.args:
                dfs(ch)

    dfs(tree)
    return len(steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.inp) as f:
        raw = [l.strip() for l in f if l.strip()]
    rng.shuffle(raw)

    pairs = []
    for idx, exp in enumerate(raw):
        if len(pairs) >= args.n:
            break
        try:
            tree = parse_sexpr(exp)
        except Exception:
            continue
        if not (isinstance(tree, Op) and tree.op == "Vec"):
            continue
        k = len(tree.args)
        if k < 2:
            continue
        # Wrap 1-3 distinct positions with distinct power-of-two steps so the
        # variant carries G in {1,2,3}. Deterministic per row via rng.
        n_wrap = rng.randint(1, min(3, k))
        positions = rng.sample(range(k), n_wrap)
        steps = rng.sample(STEPS, n_wrap)
        new_args = list(tree.args)
        for pos, step in zip(positions, steps):
            new_args[pos] = Op("<<", [tree.args[pos], Const(step)])
        newroot = Op("Vec", new_args)

        try:
            rw = expr_to_str(newroot)
            # Round-trip both directions; reject anything that does not
            # re-parse cleanly or violates structural validation.
            reparsed = parse_sexpr(rw)
            if not reparsed.validate_expression():
                continue
            g = count_rotations(reparsed)
        except Exception:
            continue

        if g == 0 or rw == exp:
            continue
        pairs.append({
            "orig_expr": exp,
            "rewritten_expr": rw,
            "orig_depth": get_multiplicative_depth(tree),
            "rewritten_depth": get_multiplicative_depth(parse_sexpr(rw)),
            "orig_G": 0,
            "rewritten_G": g,
            "rule": "rot_elem_wrap(<<)",
            "status": "ok",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "orig_expr",
                                          "rewritten_expr", "orig_depth",
                                          "rewritten_depth", "orig_G",
                                          "rewritten_G", "rule", "status"])
        w.writeheader()
        for i, p in enumerate(pairs):
            w.writerow({"pair_id": i, **p})
    gd = {}
    for p in pairs:
        gd[p["rewritten_G"]] = gd.get(p["rewritten_G"], 0) + 1
    print(f"Wrote {len(pairs)} rotation pairs to {out} (rewritten_G dist "
          f"{dict(sorted(gd.items()))})", flush=True)


if __name__ == "__main__":
    main()

