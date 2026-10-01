#!/usr/bin/env python3
"""Run the CHEHAB DNN benchmark suite and report encrypted FHE metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
LATTIGO_DIR = REPO_ROOT / "lattigo_backend"
RESULTS_DIR = REPO_ROOT / "results"
TARGET = "dnn_tensor_benchmarks_runner"


@dataclass(frozen=True)
class Benchmark:
    name: str

    @property
    def generated_go(self) -> Path:
        return BUILD_DIR / "benchmarks" / "dnn_tensor" / "he" / f"tensor_{self.name}_auto.go"


BENCHMARKS = {name: Benchmark(name) for name in ("conv2d", "full_dnn", "cryptonets", "mlp", "lola", "alexnet")}
CSV_FIELDS = [
    "benchmark", "backend", "repetitions", "dsl_generation_ms", "encrypt_ms", "compute_ms",
    "decrypt_ms", "total_measured_ms", "rotation_keys", "bootstrap_count", "level_start",
    "level_end", "levels_consumed", "max_abs_error", "precision_checked", "run_status",
    "generated_file", "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CHEHAB encrypted DNN benchmark runner")
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS),
                        help="comma-separated: conv2d,full_dnn,cryptonets,mlp,lola,alexnet")
    parser.add_argument("--repetitions", type=int, default=3,
                        help="encrypted inference repetitions per benchmark")
    parser.add_argument("--optimizer", choices=("trs", "none"), default="trs")
    parser.add_argument("--bsgs-base", default="global", help="global, 0, or a positive BSGS base")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--timeout", type=int, default=180, help="per-command timeout in seconds")
    return parser.parse_args()


def run(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)


def first_lines(output: str, limit: int = 3) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " ".join(lines[:limit])


def dsl_time(output: str) -> float | None:
    match = re.search(r"DSL generation time:\s*([0-9.]+)\s*ms", output, re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_go_metrics(output: str) -> dict[str, Any] | None:
    match = re.search(r"BENCHMARK_METRICS\s+(\{[^\n]+\})", output)
    if not match:
        return None
    try:
        metrics = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    required = {"encrypt_ms", "compute_ms", "decrypt_ms", "galois_keys", "bootstrap_count",
                "level_start", "level_end", "max_abs_error", "precision_checked"}
    return metrics if required.issubset(metrics) else None


def select_benchmarks(raw_names: str) -> list[Benchmark]:
    selected: list[Benchmark] = []
    for name in (item.strip() for item in raw_names.split(",")):
        if not name:
            continue
        try:
            selected.append(BENCHMARKS[name])
        except KeyError as error:
            raise ValueError(f"Unknown benchmark: {name}") from error
    if not selected:
        raise ValueError("No benchmarks selected")
    return selected


def build_target(timeout: int) -> tuple[bool, str]:
    if not (BUILD_DIR / "CMakeCache.txt").is_file():
        configured = run(["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR)], REPO_ROOT, timeout)
        if configured.returncode:
            return False, configured.stdout + configured.stderr
    built = run(["cmake", "--build", str(BUILD_DIR), "--target", TARGET, "--parallel", "2"],
                REPO_ROOT, timeout)
    return built.returncode == 0, built.stdout + built.stderr


def executable_path() -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return BUILD_DIR / "benchmarks" / "dnn_tensor" / f"{TARGET}{suffix}"


def empty_row(benchmark: Benchmark, repetitions: int) -> dict[str, Any]:
    return {
        "benchmark": benchmark.name, "backend": "lattigo", "repetitions": repetitions,
        "dsl_generation_ms": "", "encrypt_ms": "", "compute_ms": "", "decrypt_ms": "",
        "total_measured_ms": "", "rotation_keys": "", "bootstrap_count": "", "level_start": "",
        "level_end": "", "levels_consumed": "", "max_abs_error": "", "precision_checked": False,
        "run_status": "PENDING", "generated_file": "", "error_message": "",
    }


def execute_benchmark(benchmark: Benchmark, executable: Path, args: argparse.Namespace) -> dict[str, Any]:
    row = empty_row(benchmark, args.repetitions)
    dsl = run([str(executable), "--benchmark", benchmark.name, "--optimizer", args.optimizer,
               "--bsgs-base", args.bsgs_base], executable.parent, args.timeout)
    dsl_output = dsl.stdout + dsl.stderr
    row["dsl_generation_ms"] = dsl_time(dsl_output) or ""
    row["generated_file"] = str(benchmark.generated_go)
    if dsl.returncode:
        row.update(run_status=f"DSL_FAILED_{dsl.returncode}", error_message=first_lines(dsl_output))
        return row
    if not benchmark.generated_go.is_file():
        row.update(run_status="GENERATE_FAILED", error_message="Generated Go program was not created")
        return row

    go = run(["go", "run", str(benchmark.generated_go), "-repetitions", str(args.repetitions)],
             LATTIGO_DIR, args.timeout)
    go_output = go.stdout + go.stderr
    if go.returncode:
        row.update(run_status="LATTIGO_FAILED", error_message=first_lines(go_output))
        return row
    metrics = parse_go_metrics(go_output)
    if metrics is None:
        row.update(run_status="METRICS_MISSING", error_message="Generated Go program did not emit BENCHMARK_METRICS")
        return row

    encrypt = float(metrics["encrypt_ms"])
    compute = float(metrics["compute_ms"])
    decrypt = float(metrics["decrypt_ms"])
    level_start = int(metrics["level_start"])
    level_end = int(metrics["level_end"])
    row.update(
        repetitions=int(metrics.get("repetitions", args.repetitions)),
        encrypt_ms=encrypt,
        compute_ms=compute,
        decrypt_ms=decrypt,
        total_measured_ms=round(encrypt + compute + decrypt, 3),
        rotation_keys=int(metrics["galois_keys"]),
        bootstrap_count=int(metrics["bootstrap_count"]),
        level_start=level_start,
        level_end=level_end,
        levels_consumed=level_start - level_end if level_end >= 0 else "",
        max_abs_error="" if metrics["max_abs_error"] is None else float(metrics["max_abs_error"]),
        precision_checked=bool(metrics["precision_checked"]),
        run_status="SUCCESS",
    )
    return row


def write_csv(rows: Iterable[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "all_benchmarks_fhe_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def print_summary(rows: Iterable[dict[str, Any]]) -> None:
    print("\nFHE BENCHMARK METRICS SUMMARY")
    print("| Benchmark | Status | DSL ms | Encrypt ms | Compute ms | Decrypt ms | Total ms | Keys | Levels | Max Error |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print("| {benchmark} | {run_status} | {dsl_generation_ms} | {encrypt_ms} | {compute_ms} | "
              "{decrypt_ms} | {total_measured_ms} | {rotation_keys} | {levels_consumed} | {max_abs_error} |".format(**row))


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0 or args.timeout <= 0:
        raise ValueError("--repetitions and --timeout must be positive")
    benchmarks = select_benchmarks(args.benchmarks)
    if not (LATTIGO_DIR / "go.mod").is_file():
        raise RuntimeError(f"Missing Lattigo module: {LATTIGO_DIR / 'go.mod'}")

    if not args.skip_build:
        print(f"Building {TARGET}...")
        built, output = build_target(args.timeout)
        if not built:
            print(output, file=sys.stderr)
            return 1

    executable = executable_path()
    if not executable.is_file():
        raise RuntimeError(f"Benchmark executable not found: {executable}")

    rows: list[dict[str, Any]] = []
    for index, benchmark in enumerate(benchmarks, start=1):
        print(f"[{index}/{len(benchmarks)}] {benchmark.name}...")
        row = execute_benchmark(benchmark, executable, args)
        rows.append(row)
        print(f"  {row['run_status']}")

    csv_path = write_csv(rows)
    print_summary(rows)
    print(f"\nResults saved to: {csv_path}")
    return 0 if all(row["run_status"] == "SUCCESS" for row in rows) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        raise SystemExit(2)
