"""
Run from RL/fhe_rl  (no code generation, takes seconds):
  python3 check_shape_rule.py > shape_out.txt 2>&1
Reads ../veclang_runner/temp/input_ct_truth.csv (written by check_lane_packing.py) and tests the
structure-grouping rule (lanes.py) against the REAL input-ciphertext and operation counts.
"""
import csv, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pytrs"))
sys.setrecursionlimit(50000)
from expr import Const, Op, Var  # noqa: E402
from parser import parse_sexpr  # noqa: E402
import lanes  # noqa: E402


def names(t, acc=None):
    acc = set() if acc is None else acc
    if isinstance(t, Var):
        acc.add(t.name)
    elif isinstance(t, Op):
        for a in t.args:
            names(a, acc)
    return acc


def count_ops(tree):
    seen = set()

    def visit(n):
        if isinstance(n, Const):
            return False
        if isinstance(n, Var):
            return True
        if n.op == "Vec" and all(not isinstance(a, Op) for a in n.args):
            return not all(isinstance(a, Const) for a in n.args)
        r = [visit(a) for a in n.args]
        if not any(r):
            return False
        seen.add(str(n)); return True

    visit(tree)
    return len(seen)


rows = list(csv.DictReader(open(HERE.parent / "veclang_runner" / "temp" / "input_ct_truth.csv")))
print(f"{len(rows)} rows with real counts\n")
print(f"{'#':>3} {'N':>6} {'L':>3} | {'REAL_in':>7} {'packed':>6} {'old':>5} | {'REAL_ops':>8} {'packed':>6} {'diff':>5}")
ok_in = ok_old = 0
dops = Counter()
bad = []
for i, r in enumerate(rows):
    real_in, real_ops, old = int(r["real_in"]), int(r["real_ops"]), int(r["orig_in"])
    try:
        t = parse_sexpr(lanes.pack_expr_str(r["expr"], parse_sexpr))
        p_in, p_ops = len(names(t)), count_ops(t)
    except Exception as e:
        print(f"{i:>3} failed: {type(e).__name__}: {str(e)[:70]}"); continue
    ok_in += real_in == p_in; ok_old += real_in == old
    dops[p_ops - real_ops] += 1
    print(f"{i:>3} {r['N']:>6} {r['L']:>3} | {real_in:>7} {p_in:>6} {old:>5} | {real_ops:>8} {p_ops:>6} {p_ops - real_ops:>5}")
    if real_in != p_in and len(bad) < 6:
        bad.append((i, real_in, p_in, r["expr"][:200]))
n = len(rows)
print(f"\nSUMMARY: input ciphertexts  packed rule == REAL in {ok_in}/{n}   (old rule: {ok_old}/{n})")
print(f"         ops difference (packed - REAL), count per value: {sorted(dops.items())}")
for i, a, b, e in bad:
    print(f"\n  MISMATCH row {i}: real={a} packed={b}\n  expr: {e}...")