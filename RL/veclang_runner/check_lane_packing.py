"""
Tests the lane-packing hypothesis. Code generation only (no Go run).
Put in RL/veclang_runner/ (next to run_pilot.py) and run from there:
  python3 check_lane_packing.py > lanes_out.txt 2>&1
Options (env vars): N_SMALL=60 N_LARGE=20 SEED=1

Hypothesis H1: the generated program packs the SAME variable of every lane into ONE ciphertext
(labels c0, c1, ...), so  a_0_0, a_0_1, a_0_2 ... -> one ciphertext "a_0".
  => real input ciphertexts = number of distinct variable names after removing the LAST "_<digits>"
  => operations are done once per packed vector, not once per lane
This script compares, for every sampled expression:
  inputs : REAL (count of encryptedInputs[...] in the Go) vs H1 vs original rule vs agent rule
  ops    : REAL (AddNew/SubNew/MulNew/MulRelinNew/RotateNew/NegNew calls in the Go)
           vs ops of the raw tree vs ops of the lane-normalised tree
and writes input_ct_truth.csv (so nothing has to be generated twice).
"""
import csv, os, random, re, sys, tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pytrs"))
sys.setrecursionlimit(50000)

from expr import Const, Op, Var  # noqa: E402

LABEL_RE = re.compile(r'encryptedInputs\["([^"]+)"\]\s*=')
OPS_RE = re.compile(r'\beval\.(AddNew|SubNew|MulNew|MulRelinNew|RotateNew|NegNew)\(')
LANE_RE = re.compile(r'([A-Za-z][A-Za-z0-9_]*)_(\d+)(?=[\s()]|$)')


def lane_normalize(expr_str):
    """a_0_3 -> a_0 ; x1_1 -> x1 : remove the lane index (last _digits) of every variable."""
    return LANE_RE.sub(r'\1', expr_str)


def count_original(tree):
    nodes, n = set(), [0]

    def visit(node):
        if isinstance(node, Const):
            return False
        key = str(node)
        if key in nodes:
            return True
        if isinstance(node, Var):
            nodes.add(key); n[0] += 1
            return True
        if node.op == "Vec" and all(not isinstance(a, Op) for a in node.args):
            if all(isinstance(a, Const) for a in node.args):
                return False
            nodes.add(key); n[0] += 1
            return True
        res = [visit(a) for a in node.args]
        if not any(res):
            return False
        nodes.add(key)
        return True

    visit(tree)
    return n[0]


def count_vars(tree):
    names = set()

    def collect(node):
        if isinstance(node, Var):
            names.add(node.name)
        elif isinstance(node, Op):
            for a in node.args:
                collect(a)

    collect(tree)
    return names


def count_ops(tree):
    seen = set()

    def visit(node):
        if isinstance(node, Const):
            return False
        if isinstance(node, Var):
            return True
        if node.op == "Vec" and all(not isinstance(a, Op) for a in node.args):
            return not all(isinstance(a, Const) for a in node.args)
        key = str(node)
        res = [visit(a) for a in node.args]
        if not any(res):
            return False
        seen.add(key)
        return True

    visit(tree)
    return len(seen)


def sample_rows(csv_path, n_small, n_large, seed):
    rows = list(csv.DictReader(open(csv_path)))
    ok = [r for r in rows if r.get("status") == "ok" and r.get("est_G") in ("0", "0.0")]
    small = sorted([r for r in ok if int(float(r["est_N"])) <= 16384], key=lambda r: len(r["expr"]))
    large = sorted([r for r in ok if int(float(r["est_N"])) > 16384], key=lambda r: len(r["expr"]))
    rnd = random.Random(seed)

    def spread(lst, k):
        if len(lst) <= k:
            return lst
        step = len(lst) / k
        return [lst[min(int(i * step + rnd.random() * step), len(lst) - 1)] for i in range(k)]

    return spread(small, n_small) + spread(large, n_large)


def main():
    import run_pilot
    from parser import parse_sexpr

    n_small = int(os.environ.get("N_SMALL", "60"))
    n_large = int(os.environ.get("N_LARGE", "20"))
    picks = sample_rows(HERE / "temp" / "pilot_results.csv", n_small, n_large, int(os.environ.get("SEED", "1")))
    print(f"sampled {len(picks)} pilot expressions\n")
    print(f"{'#':>3} {'N':>6} {'L':>3} {'len':>6} | {'REAL':>5} {'H1':>5} {'orig':>5} {'agent':>5} |"
          f" {'rOPS':>5} {'raw':>5} {'norm':>5}")

    out, bad_h1, d_norm, d_raw = [], [], Counter(), []
    for i, r in enumerate(picks):
        expr = r["expr"]
        try:
            with tempfile.TemporaryDirectory(prefix="lane_") as td:
                row = run_pilot.measure_one(expr, Path(td), {}, 300)
            go_src = row.get("_go_src")
            if go_src is None:
                raise RuntimeError(row.get("status_detail", "no go source"))
            labels = LABEL_RE.findall(go_src)
            real_ops = len(OPS_RE.findall(go_src))
            tree = parse_sexpr(expr)
            tree_n = parse_sexpr(lane_normalize(expr))
            real_in, h1 = len(labels), len(count_vars(tree_n))
            orig, agent = count_original(tree), len(count_vars(tree))
            ops_raw, ops_norm = count_ops(tree), count_ops(tree_n)
        except Exception as e:
            print(f"{i:>3} failed: {type(e).__name__}: {str(e)[:80]}")
            continue
        print(f"{i:>3} {r['est_N']:>6} {r['est_L']:>3} {len(expr):>6} | {real_in:>5} {h1:>5} {orig:>5} {agent:>5} |"
              f" {real_ops:>5} {ops_raw:>5} {ops_norm:>5}")
        d_norm[ops_norm - real_ops] += 1
        d_raw.append(ops_raw - real_ops)
        if real_in != h1 and len(bad_h1) < 4:
            names = sorted(count_vars(tree))[:12]
            bad_h1.append((expr[:140], real_in, h1, names, labels[:6]))
        out.append(dict(expr=expr, N=r["est_N"], L=r["est_L"], real_in=real_in, h1_in=h1, orig_in=orig,
                        agent_in=agent, real_ops=real_ops, ops_raw=ops_raw, ops_norm=ops_norm))

    n = len(out)
    print(f"\nSUMMARY over {n} expressions")
    print(f"  inputs: H1 == REAL in {sum(o['real_in'] == o['h1_in'] for o in out)}/{n} | "
          f"orig == REAL in {sum(o['real_in'] == o['orig_in'] for o in out)}/{n} | "
          f"agent == REAL in {sum(o['real_in'] == o['agent_in'] for o in out)}/{n}")
    print(f"  ops   : (ops of lane-normalised tree - REAL ops) distribution: {sorted(d_norm.items())[:8]}")
    if d_raw:
        d_raw.sort()
        print(f"  ops   : raw tree minus REAL ops: median {d_raw[len(d_raw) // 2]}, max {d_raw[-1]}")
    for e, ri, h1, names, labs in bad_h1:
        print(f"\n  H1 MISMATCH real={ri} h1={h1}\n  expr: {e}...\n  first var names: {names}\n  first labels: {labs}")
    with open(HERE / "temp" / "input_ct_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nwrote {HERE / 'temp' / 'input_ct_truth.csv'}")


if __name__ == "__main__":
    main()