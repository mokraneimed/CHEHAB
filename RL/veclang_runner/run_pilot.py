#!/usr/bin/env python3
"""Phase-0 pilot measurement harness (Lattigo/CKKS only, MaxRSS primary).

Part 1: imports, constants, Go env, output parsing helpers.
"""
import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent.parent
PYTRS_DIR = RL_DIR / "pytrs"
sys.path.insert(0, str(PYTRS_DIR))

from parser import parse_sexpr  # noqa: E402
from cost import get_multiplicative_depth  # noqa: E402
from ckks_memory_estimator import (  # noqa: E402
    estimate_expression_memory, chehab_params,
)

LATTIGO_BACKEND_DIR = Path("/home/maroua/chehab-unified/lattigo_backend")
VECLANG_BIN = Path("/home/maroua/chehab-unified/build/RL/veclang_runner/veclang_runner")

GO_ENV = {
    "GOGC": os.environ.get("GOGC", "100"),
    "GOMAXPROCS": os.environ.get("GOMAXPROCS", ""),
}
PROVISIONAL_TIMEOUT_S = int(os.environ.get("PILOT_TIMEOUT_S", "1200"))

FIELDNAMES = [
    "pilot_id", "kind", "pair_id", "expr",
    "depth", "est_N", "est_L", "est_K", "est_G", "est_galois_keys",
    "est_peak_live_cts", "est_net_MB",
    "go_N", "go_L", "go_K", "go_MaxLevel", "go_G", "go_galois_keys",
    "param_match", "param_mismatch_detail",
    "go_precision_checked", "go_has_expected_outputs", "go_has_ref_arrays",
    "go_neg_patched",
    "status", "status_detail",
    "max_rss_MB_median", "max_rss_MB_runs", "max_rss_spread",
    "peak_polled_MB_median",
    "keygen_ms", "compute_ms", "total_ms", "encrypt_ms", "decrypt_ms",
    "rotation_keys_size_MB",
    "compile_s", "run_s_total",
    "GOGC", "GOMAXPROCS", "go_version", "lattigo_version",
    "GOOS", "GOARCH",
]


def run(cmd, cwd, timeout, env=None):
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        env=env,
    )


def go_env_info():
    def out(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=30)
            return r.stdout.strip()
        except Exception:
            return ""
    lattigo_ver = ""
    try:
        gosum = (LATTIGO_BACKEND_DIR / "go.sum").read_text()
        m = re.findall(r"lattigo/v5 (v[\d.]+)", gosum)
        lattigo_ver = m[-1] if m else ""
    except Exception:
        pass
    return {
        "GOGC": GO_ENV["GOGC"],
        "GOMAXPROCS": out(["go", "env", "GOMAXPROCS"]) or GO_ENV["GOMAXPROCS"],
        "go_version": out(["go", "version"]),
        "lattigo_version": lattigo_ver,
        "GOOS": out(["go", "env", "GOOS"]),
        "GOARCH": out(["go", "env", "GOARCH"]),
    }


def parse_go_params(go_src):
    d = {}
    m = re.search(r"LogN:\s*(\d+)", go_src)
    if m:
        d["go_N"] = 1 << int(m.group(1))
    m = re.search(r"LogQ:\s*\[\]int\{([^}]*)\}", go_src)
    if m:
        d["go_L"] = len([t for t in m.group(1).split(",") if t.strip()])
    m = re.search(r"LogP:\s*\[\]int\{([^}]*)\}", go_src)
    if m:
        d["go_K"] = len([t for t in m.group(1).split(",") if t.strip()])
    m = re.search(r"MaxLevel=(\d+)", go_src)
    if m:
        d["go_MaxLevel"] = int(m.group(1))
    m = re.search(r"return \[\]int\{([^}]*)\}", go_src)
    if m:
        steps = [t.strip() for t in m.group(1).split(",") if t.strip()]
        try:
            steps = [int(s) for s in steps]
        except ValueError:
            pass
        d["go_G"] = len(steps)
    d["go_precision_checked"] = "precisionChecked = true" in go_src
    d["go_has_expected_outputs"] = "expectedOutputs" in go_src
    d["go_has_ref_arrays"] = bool(re.search(r"\bref_\d+\b", go_src))
    return d


def patch_neg_new(go_src: str) -> str:
    """Replace eval.NegNew (absent in Lattigo v5.0.7) with Sub-based negate.

    Mirrors RL/demo/demo_challenging_benchmark.py::_patch_generated_go.
    Records whether the patch applied via row['go_neg_patched'] upstream.
    """
    import re
    def _replace_neg(m):
        dst, src = m.group(1), m.group(2)
        return (f"{dst}, _ = eval.SubNew({src}, {src})\n"
                f"\t{dst}, _ = eval.SubNew({dst}, {src})")
    return re.sub(r'(\w+),\s*_\s*=\s*eval\.NegNew\((\w+)\)',
                  _replace_neg, go_src)


def parse_go_stdout(stdout):
    d = {}
    m = re.search(r"BENCHMARK_METRICS\s+(\{[^\n]+\})", stdout)
    if m:
        try:
            mm = json.loads(m.group(1))
            for k in ["encrypt_ms", "compute_ms", "decrypt_ms",
                      "galois_keys", "bootstrap_count", "level_start",
                      "level_end", "precision_checked"]:
                if k in mm:
                    d[k] = mm[k]
        except ValueError:
            pass
    for line in stdout.splitlines():
        if "max_rss_(MB):" in line:
            try:
                d["max_rss_MB"] = float(line.split()[1])
            except (IndexError, ValueError):
                pass
        elif "peak_ram_memory_(MB):" in line:
            try:
                d["peak_ram_polled_MB"] = float(line.split()[1])
            except (IndexError, ValueError):
                pass
        elif "rotation_keys_size_(MB):" in line:
            try:
                d["rotation_keys_size_MB"] = float(line.split()[1])
            except (IndexError, ValueError):
                pass
        elif "galois_keys_generation_time_(ms):" in line:
            try:
                d["keygen_ms"] = float(line.split()[1])
            except (IndexError, ValueError):
                pass
        elif "circuit_execution_time_(ms):" in line:
            try:
                d["compute_ms_go"] = float(line.split()[1])
            except (IndexError, ValueError):
                pass
        elif "total_execution_time_(ms):" in line:
            try:
                d["total_ms_go"] = float(line.split()[1])
            except (IndexError, ValueError):
                pass
    return d


def measure_one(expr, workdir, env_info, timeout_s):
    """Full pipeline for one expression. Always returns a row."""
    row = {"expr": expr, "status": "ok", "status_detail": ""}
    try:
        tree = parse_sexpr(expr)
        depth = get_multiplicative_depth(tree)
        row["depth"] = depth
        params = chehab_params(depth)
        est = estimate_expression_memory(tree, backend="cpu", params=params)
        row.update({
            "est_N": params.N, "est_L": params.L, "est_K": params.K,
            "est_G": est.rotation_steps,
            "est_galois_keys": est.galois_keys,
            "est_peak_live_cts": est.peak_live_ciphertexts,
            "est_net_MB": round(est.total_mb, 3),
        })
    except Exception as e:
        row["status"] = "estimator_fail"
        row["status_detail"] = f"{type(e).__name__}: {e}"[:300]
        return row

    try:
        from expr import Var, Op, Const
        tree0 = parse_sexpr(expr)
        # Inline vec-size calc (mirrors fhe_rl.utils.calc_vec_sizes, which
        # needs torch via fhe_rl.utils import chain -- unavailable here).
        vec_sizes = []
        def _rec(node):
            if isinstance(node, (Const, Var)):
                return
            if isinstance(node, Op):
                if node.op == "Vec":
                    vec_sizes.append(len(node.args))
                else:
                    for a in node.args:
                        _rec(a)
        _rec(tree0)
        slot_count = vec_sizes[0] if vec_sizes else 1
        metadata = f"{slot_count} {slot_count} 1 0"
        (workdir / "vectorized_code.txt").write_text(
            expr + f"\n{metadata}")
        names = set()
        def collect(n):
            if isinstance(n, Var):
                names.add(n.name)
            elif isinstance(n, Op):
                for a in n.args:
                    collect(a)
        collect(tree0)
        names = sorted(names)
        (workdir / "inputs.txt").write_text(
            " ".join(names) + "\n" + " ".join(["1"] * len(names)) + "\n")
        import random as _rand
        with open(workdir / "fhe_io_example.txt", "w") as _f:
            _f.write(f"1 {len(names)} 1\n")
            for inp in names:
                _f.write(f"{inp} 1 1 {_rand.randint(0, 10)}\n")
        build_dir = Path("/home/maroua/chehab-unified/build/RL/veclang_runner")
        for fname in ["vectorized_code.txt", "inputs.txt",
                      "fhe_io_example.txt"]:
            shutil.copy2(workdir / fname, build_dir / fname)
            if fname in ("vectorized_code.txt", "inputs.txt"):
                shutil.copy2(workdir / fname, build_dir.parent / fname)
        r = run(["./veclang_runner", "1", "1", "1"], cwd=build_dir,
                timeout=300)
        if "lattigo_output: generated_fhe.go" not in r.stdout:
            row["status"] = "compile_fail"
            # Tail of stderr is where the real error lives (head is
            # just the file-size diagnostics).
            row["status_detail"] = (
                "codegen: " + ((r.stderr or r.stdout).strip()[-400:]))
            return row
        go_src = (build_dir / "generated_fhe.go").read_text()
        row["go_neg_patched"] = "NegNew" in go_src
        if "NegNew" in go_src:
            go_src = patch_neg_new(go_src)
    except subprocess.TimeoutExpired:
        row["status"] = "compile_fail"
        row["status_detail"] = "codegen timeout"
        return row
    except Exception as e:
        row["status"] = "compile_fail"
        row["status_detail"] = f"{type(e).__name__}: {e}"[:300]
        return row

    gp = parse_go_params(go_src)
    for k in ["go_N", "go_L", "go_K", "go_MaxLevel", "go_G",
              "go_precision_checked", "go_has_expected_outputs",
              "go_has_ref_arrays"]:
        row[k] = gp.get(k)
    mism = []
    for ek, gk in [("est_N", "go_N"), ("est_L", "go_L"), ("est_K", "go_K")]:
        ev, gv = row.get(ek), gp.get(gk)
        if ev is not None and gv is not None and ev != gv:
            mism.append(f"{ek}={ev} vs {gk}={gv}")
    if row.get("est_G") is not None and gp.get("go_G") is not None:
        if row["est_G"] != gp["go_G"]:
            mism.append(f"est_G={row['est_G']} vs go_G={gp['go_G']}")
    row["param_match"] = len(mism) == 0
    row["param_mismatch_detail"] = "; ".join(mism)
    row["_go_src"] = go_src
    return row


def build_and_run(row, workdir, env_info, timeout_s):
    """Build once + 3 fresh-process runs. Mutates row, always returns it."""
    go_src = row.pop("_go_src", None)
    if go_src is None:
        return row
    try:
        dest = workdir / "pilot_go"
        dest.mkdir(exist_ok=True)
        (dest / "main.go").write_text(go_src)
        gosum_src = LATTIGO_BACKEND_DIR / "go.sum"
        if gosum_src.exists():
            shutil.copy2(gosum_src, dest / "go.sum")
        (dest / "go.mod").write_text(
            "module pilot\n\ngo 1.21\n\nrequire github.com/tuneinsight/lattigo/v5 v5.0.7\n")
        env = dict(os.environ)
        if GO_ENV["GOGC"]:
            env["GOGC"] = GO_ENV["GOGC"]
        t1 = time.time()
        run(["go", "mod", "tidy"], cwd=dest, timeout=600, env=env)
        r = run(["go", "build", "-o", "pilot_bin", "."], cwd=dest,
                timeout=900, env=env)
        if r.returncode != 0:
            row["status"] = "compile_fail"
            row["status_detail"] = ("go build: " +
                                    (r.stderr or r.stdout))[:300]
            row["compile_s"] = round(time.time() - t1, 1)
            return row
        row["compile_s"] = round(time.time() - t1, 1)
        rss, polled, keygen, compute = [], [], [], []
        rot_size, galois_n = None, None
        t2 = time.time()
        for _ in range(3):
            try:
                r = run(["./pilot_bin", "-repetitions=1"], cwd=dest,
                        timeout=timeout_s, env=env)
            except subprocess.TimeoutExpired:
                row["status"] = "timeout"
                row["status_detail"] = f"run timeout ({timeout_s}s)"
                return row
            if r.returncode != 0:
                row["status"] = "run_fail"
                row["status_detail"] = ("exit %d: %s" % (
                    r.returncode, (r.stderr or r.stdout)))[:300]
                return row
            p = parse_go_stdout(r.stdout)
            if "max_rss_MB" not in p:
                row["status"] = "run_fail"
                row["status_detail"] = "max_rss_(MB) missing from output"
                return row
            rss.append(p["max_rss_MB"])
            if "peak_ram_polled_MB" in p:
                polled.append(p["peak_ram_polled_MB"])
            if "keygen_ms" in p:
                keygen.append(p["keygen_ms"])
            if "compute_ms_go" in p:
                compute.append(p["compute_ms_go"])
            elif "compute_ms" in p:
                compute.append(p["compute_ms"])
            if "rotation_keys_size_MB" in p:
                rot_size = p["rotation_keys_size_MB"]
            if "galois_keys" in p:
                galois_n = p["galois_keys"]
        row["max_rss_MB_median"] = round(statistics.median(rss), 2)
        row["max_rss_MB_runs"] = ";".join(f"{v:.2f}" for v in rss)
        row["max_rss_spread"] = round(max(rss) - min(rss), 2)
        if polled:
            row["peak_polled_MB_median"] = round(statistics.median(polled), 2)
        if keygen:
            row["keygen_ms"] = round(statistics.median(keygen), 1)
        if compute:
            row["compute_ms"] = round(statistics.median(compute), 1)
        row["rotation_keys_size_MB"] = rot_size
        row["go_galois_keys"] = galois_n
        row["run_s_total"] = round(time.time() - t2, 1)
    except Exception as e:
        import traceback
        if row.get("status", "ok") == "ok":
            row["status"] = "run_fail"
            row["status_detail"] = traceback.format_exc()[-600:]
    row.update(env_info)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=PROVISIONAL_TIMEOUT_S)
    ap.add_argument("--resume", action="store_true",
                    help="reuse measurements already present in the --out CSV")
    ap.add_argument("--out",
                    default="/home/maroua/chehab-unified/RL/veclang_runner/temp/pilot_results.csv")
    ap.add_argument("--pairs",
                    default="/home/maroua/chehab-unified/RL/veclang_runner/temp/pilot_rotation_pairs.csv")
    ap.add_argument("--exprs",
                    default="/home/maroua/chehab-unified/RL/veclang_runner/temp/pilot_expressions.txt")
    args = ap.parse_args()

    env_info = go_env_info()
    print(f"Go env: {env_info}", flush=True)
    print(f"Timeout per run: {args.timeout}s (provisional)", flush=True)

    items = []
    with open(args.exprs) as f:
        for i, line in enumerate(f):
            e = line.strip()
            if e:
                items.append({"pilot_id": i, "kind": "raw", "pair_id": "",
                              "expr": e})
    try:
        with open(args.pairs) as f:
            for r in csv.DictReader(f):
                items.append({"pilot_id": f"pair{r['pair_id']}_orig",
                              "kind": "pair_orig", "pair_id": r["pair_id"],
                              "expr": r["orig_expr"]})
                items.append({"pilot_id": f"pair{r['pair_id']}_rw",
                              "kind": "pair_rw", "pair_id": r["pair_id"],
                              "expr": r["rewritten_expr"]})
    except FileNotFoundError:
        print("WARNING: pairs file missing, raw only", flush=True)
    if args.limit:
        items = items[:args.limit]
    print(f"Measuring {len(items)} expressions...", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    measured_cache = {}
    # Resume: seed the cache with terminal rows already written to --out so a
    # previous (possibly interrupted) run is not re-measured. Pair originals
    # reuse the raw pool's rows; broken-pair rows from an older pairs CSV are
    # naturally absent because their rewritten_expr changed.
    if args.resume and out.exists():
        with open(out) as ef:
            for r in csv.DictReader(ef):
                expr = r.get("expr", "")
                if expr and r.get("status"):
                    measured_cache[expr] = dict(r)
        print(f"Resume: seeded {len(measured_cache)} cached rows from {out}",
              flush=True)
    # Keep only items whose expr still needs measuring, but preserve the full
    # ordering/metadata so the output CSV is complete and consistent.
    pending = []
    for it in items:
        expr = it["expr"]
        cached = measured_cache.get(expr)
        if cached is not None and cached.get("status"):
            measured_cache[expr] = dict(cached)
        else:
            pending.append(it)
    print(f"Measuring {len(pending)} pending / {len(items)} total "
          f"expressions...", flush=True)
    pending_exprs = {p["expr"] for p in pending}
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        # Emit cached rows in canonical item order first (no recompute).
        for it in items:
            cached = measured_cache.get(it["expr"])
            if cached is not None and it["expr"] not in pending_exprs:
                row = dict(cached)
                row["pilot_id"] = it["pilot_id"]
                row["kind"] = it["kind"]
                row["pair_id"] = it["pair_id"]
                w.writerow({k: row.get(k, "") for k in FIELDNAMES})
        f.flush()
        for j, it in enumerate(pending):
            print(f"[{j+1}/{len(pending)}] {it['kind']} {it['pilot_id']}...",
                  flush=True)
            with tempfile.TemporaryDirectory(prefix="pilot_") as td:
                wd = Path(td)
                row = measure_one(it["expr"], wd, env_info, args.timeout)
                if row.get("status") == "ok":
                    row = build_and_run(row, wd, env_info, args.timeout)
                else:
                    row.update({k: "" for k in env_info})
            measured_cache[it["expr"]] = dict(row)
            row["pilot_id"] = it["pilot_id"]
            row["kind"] = it["kind"]
            row["pair_id"] = it["pair_id"]
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})
            f.flush()
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
