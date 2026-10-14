"""
Ground truth for the "how many input ciphertexts" question. NO Go run, only code generation.

Put this file in RL/veclang_runner/ (next to run_pilot.py) and run from there:
  python3 check_input_cts.py > cts_out.txt 2>&1
Options (env vars): N_SMALL=30 N_LARGE=10 SEED=0

For each sampled pilot expression it
  1. generates the Go program with the same code path as the pilot (run_pilot.measure_one),
  2. counts the input ciphertexts the Go program really creates:  encryptedInputs["label"] = ct
  3. computes what the two estimator rules would count:
       ORIGINAL rule : a Vec made only of variables/constants = 1 ciphertext,
                       any other variable = 1 per distinct variable
       AGENT rule    : 1 per distinct variable name in the whole expression
  4. says which rule matches the real count.
"""
import csv, os, random, re, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pytrs"))
sys.setrecursionlimit(50000)

from expr import Const, Op, Var  # noqa: E402

LABEL_RE = re.compile(r'encryptedInputs\["([^"]+)"\]\s*=')


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


def count_agent(tree):
    names = set()

    def collect(node):
        if isinstance(node, Var):
            names.add(node.name)
        elif isinstance(node, Op):
            for a in node.args:
                collect(a)

    collect(tree)
    return len(names)


def real_labels(go_src):
    return LABEL_RE.findall(go_src)


def sample_rows(csv_path, n_small, n_large, seed):
    rows = list(csv.DictReader(open(csv_path)))
    ok = [r for r in rows if r.get("status") == "ok" and r.get("est_G") in ("0", "0.0")]
    small = sorted([r for r in ok if int(float(r["est_N"])) <= 16384], key=lambda r: len(r["expr"]))
    large = sorted([r for r in ok if int(float(r["est_N"])) > 16384], key=lambda r: len(r["expr"]))
    rnd = random.Random(seed)

    def spread(lst, k):  # evenly along expression length, so short AND long ones are tested
        if len(lst) <= k:
            return lst
        step = len(lst) / k
        return [lst[min(int(i * step + rnd.random() * step), len(lst) - 1)] for i in range(k)]

    return spread(small, n_small) + spread(large, n_large)


def main():
    import run_pilot
    from parser import parse_sexpr

    n_small = int(os.environ.get("N_SMALL", "30"))
    n_large = int(os.environ.get("N_LARGE", "10"))
    seed = int(os.environ.get("SEED", "0"))
    csv_path = HERE / "temp" / "pilot_results.csv"
    picks = sample_rows(csv_path, n_small, n_large, seed)
    print(f"sampled {len(picks)} pilot expressions (G=0) from {csv_path}\n")
    print(f"{'#':>3} {'N':>6} {'L':>3} {'len':>6} {'REAL':>5} {'orig':>5} {'agent':>5}  match")

    tally = dict(orig=0, agent=0, both=0, neither=0, fail=0)
    neither = []
    for i, r in enumerate(picks):
        expr = r["expr"]
        try:
            with tempfile.TemporaryDirectory(prefix="cts_") as td:
                row = run_pilot.measure_one(expr, Path(td), {}, 300)
            go_src = row.get("_go_src")
            if go_src is None:
                raise RuntimeError(row.get("status_detail", "no go source"))
            labels = real_labels(go_src)
            tree = parse_sexpr(expr)
            real, o, a = len(labels), count_original(tree), count_agent(tree)
        except Exception as e:
            tally["fail"] += 1
            print(f"{i:>3} {r['est_N']:>6} {r['est_L']:>3} {len(expr):>6}  failed: {type(e).__name__}: {str(e)[:80]}")
            continue
        m = "orig" if (real == o and real != a) else "agent" if (real == a and real != o) \
            else "both" if real == o == a else "NEITHER"
        tally["neither" if m == "NEITHER" else m] += 1
        if m == "NEITHER" and len(neither) < 5:
            neither.append((expr[:150], labels[:6], real, o, a))
        print(f"{i:>3} {r['est_N']:>6} {r['est_L']:>3} {len(expr):>6} {real:>5} {o:>5} {a:>5}  {m}")

    print("\nSUMMARY:", tally)
    print("  orig   = only the original baseline rule matches the real count")
    print("  revised = only the revised rule matches")
    print("  both   = the two rules agree and match (nothing to decide on those)")
    print("  NEITHER= real count follows another rule; examples below")
    for e, labs, real, o, a in neither:
        print(f"\n  expr: {e}...\n  real={real} orig={o} agent={a}  first labels: {labs}")


if __name__ == "__main__":
    main()