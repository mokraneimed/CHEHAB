#!/usr/bin/env python3
"""Phase-0 pilot analysis: failure rate, runtime, repeat spread,
estimator-vs-MaxRSS error, param mismatches, pair ordering accuracy.

Usage: python3 veclang_runner/analyze_pilot.py [results.csv]
"""
import csv
import statistics
import sys
from collections import Counter, defaultdict


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def f(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def main():
    path = (sys.argv[1] if len(sys.argv) > 1 else
            "/home/maroua/chehab-unified/RL/veclang_runner/temp/pilot_results.csv")
    rows = list(csv.DictReader(open(path)))
    n = len(rows)
    out = []
    out.append("# Phase-0 Pilot Report (200-expr selection + rotation pairs)")
    out.append(f"\nSource: `{path}`  rows={n}")

    # 1. Failure rate -------------------------------------------------------
    st = Counter(r["status"] for r in rows)
    ok = st.get("ok", 0)
    fails = n - ok
    out.append("\n## 1. Failure rate")
    out.append(f"- ok: {ok}/{n} ({100.0*ok/n:.1f}%)")
    for k, v in sorted(st.items()):
        if k != "ok":
            out.append(f"- {k}: {v} ({100.0*v/n:.1f}%)")
    if fails:
        out.append("\nFailure details:")
        for r in rows:
            if r["status"] != "ok":
                out.append(f"- [{r['pilot_id']}|{r['kind']}|d={r['depth']}] "
                           f"{r['status']}: {r['status_detail'][:160]}")

    # 2. Runtime per expression --------------------------------------------
    out.append("\n## 2. Runtime per expression (wall clock, build once + 3 runs)")
    comp = [float(r["compile_s"]) for r in rows if r.get("compile_s")]
    runt = [float(r["run_s_total"]) for r in rows if r.get("run_s_total")]
    if runt:
        tot = [c + t for c, t in zip(comp, runt)] if len(comp) == len(runt) else runt
        out.append(f"- compile:   p50 {f(pct(comp,50))}s  p90 {f(pct(comp,90))}s")
        out.append(f"- 3 runs:    p50 {f(pct(runt,50))}s  p90 {f(pct(runt,90))}s  "
                   f"max {f(max(runt))}s")
        out.append(f"- per-run:   p50 {f(pct(runt,50)/3)}s")
        out.append(f"- TOTAL/expr mean {f(statistics.mean(tot))}s  "
                   f"p50 {f(pct(tot,50))}s  p90 {f(pct(tot,90))}s")
    km = [float(r["keygen_ms"]) for r in rows if r.get("keygen_ms")]
    cm = [float(r["compute_ms"]) for r in rows if r.get("compute_ms")]
    if km:
        out.append(f"- keygen_ms: p50 {f(pct(km,50))}  p90 {f(pct(km,90))}  "
                   f"max {f(max(km))}")
    if cm:
        out.append(f"- compute_ms: p50 {f(pct(cm,50))}  p90 {f(pct(cm,90))}  "
                   f"max {f(max(cm))}")
    out.append("- PROPOSED full-run timeout (from these timings): "
               f"{max(400, int(4*pct(comp,90)+4*pct(runt,90)+200))}s per expr")

    # 3. Repeat spread ------------------------------------------------------
    out.append("\n## 3. Repeat spread across 3 fresh-process runs")
    spr = [float(r["max_rss_spread"]) for r in rows if r.get("max_rss_spread")]
    spr_rel = []
    for r in rows:
        if r.get("max_rss_spread") and r.get("max_rss_MB_median"):
            m = float(r["max_rss_MB_median"])
            if m > 0:
                spr_rel.append(100.0 * float(r["max_rss_spread"]) / m)
    if spr:
        out.append(f"- absolute spread (max-min): p50 {f(pct(spr,50))} MB  "
                   f"p90 {f(pct(spr,90))} MB  max {f(max(spr))} MB")
        out.append(f"- relative spread: p50 {f(pct(spr_rel,50))}%  "
                   f"p90 {f(pct(spr_rel,90))}%")
        small = sum(1 for x in spr_rel if x <= 2.0)
        share = 100.0 * small / max(1, len(spr_rel))
        out.append(f"- rows with spread <=2%: {small}/{len(spr_rel)} "
                   f"({share:.0f}%) -> "
                   + ("1-2 repeats acceptable for full run" if share >= 80
                      else "keep 3 repeats"))
    # 4. Estimator vs MaxRSS error -----------------------------------------
    out.append("\n## 4. Estimator (net, cpu) vs MaxRSS (primary target)")
    errs, abserrs, gaps = [], [], []
    for r in rows:
        if r["status"] != "ok" or not r.get("est_net_MB") or not r.get("max_rss_MB_median"):
            continue
        e = float(r["est_net_MB"])
        m = float(r["max_rss_MB_median"])
        if m <= 0:
            continue
        err = 100.0 * (e - m) / m
        errs.append(err)
        abserrs.append(abs(err))
        gaps.append(m - e)
    if errs:
        bias = statistics.mean(errs)
        out.append(f"- n={len(errs)}")
        out.append(f"- signed bias (est-MaxRSS)/MaxRSS: mean {f(bias)}%  "
                   f"median {f(statistics.median(errs))}%")
        out.append(f"- absolute error: p50 {f(pct(abserrs,50))}%  "
                   f"p90 {f(pct(abserrs,90))}%  max {f(max(abserrs))}%")
        out.append(f"- est underestimates on: "
                   f"{sum(1 for e in errs if e < 0)}/{len(errs)} rows")
        out.append(f"- gross gap (MaxRSS - est): p50 {f(pct(gaps,50))} MB  "
                   f"p10 {f(pct(gaps,10))} MB  p90 {f(pct(gaps,90))} MB")
    byd = defaultdict(list)
    for r in rows:
        if r["status"] != "ok" or not r.get("est_net_MB") or not r.get("max_rss_MB_median"):
            continue
        e, m = float(r["est_net_MB"]), float(r["max_rss_MB_median"])
        if m <= 0:
            continue
        d = int(r["depth"]) if r.get("depth") else 0
        b = ("d01-02" if d <= 2 else "d03-04" if d <= 4 else
             "d05-06" if d <= 6 else "d07-08" if d <= 8 else "d09+")
        byd[b].append(100.0 * (e - m) / m)
    out.append("\nBy depth bucket (bias / abs-p50 / n):")
    for b in ["d01-02", "d03-04", "d05-06", "d07-08", "d09+"]:
        v = byd.get(b, [])
        if v:
            out.append(f"- {b}: {f(statistics.mean(v))}% / "
                       f"{f(pct([abs(x) for x in v],50))}% / n={len(v)}")

    # 5. Param mismatches ---------------------------------------------------
    out.append("\n## 5. Parameter check (estimator vs generated Go)")
    mm = [r for r in rows if r.get("param_match") == "False"]
    out.append(f"- matches: {sum(1 for r in rows if r.get('param_match')=='True')}/{n}")
    out.append(f"- mismatches: {len(mm)}")
    for r in mm[:20]:
        out.append(f"- [{r['pilot_id']}|d={r['depth']}] "
                   f"{r['param_mismatch_detail']}")

    # 6. Pair ordering accuracy --------------------------------------------
    out.append("\n## 6. Ordering accuracy on (original, rewritten) pairs")
    by_pair = defaultdict(dict)
    for r in rows:
        if r.get("pair_id"):
            by_pair[r["pair_id"]][r["kind"]] = r
    agree = disagree = tie = skipped = 0
    for pid, d in by_pair.items():
        o, w = d.get("pair_orig"), d.get("pair_rw")
        if not o or not w or o["status"] != "ok" or w["status"] != "ok":
            skipped += 1
            continue
        try:
            de = float(w["est_net_MB"]) - float(o["est_net_MB"])
            dm = float(w["max_rss_MB_median"]) - float(o["max_rss_MB_median"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        if abs(de) < 1e-9 and abs(dm) < 1e-9:
            tie += 1
        elif de * dm > 0:
            agree += 1
        else:
            disagree += 1
    tot_pairs = agree + disagree
    if tot_pairs:
        out.append(f"- comparable pairs: {tot_pairs} (ties {tie}, skipped {skipped})")
        out.append(f"- same-direction sign agreement: {agree}/{tot_pairs} "
                   f"({100.0*agree/tot_pairs:.1f}%)")
    else:
        out.append(f"- no comparable pairs (skipped {skipped})")

    # 7. Environment --------------------------------------------------------
    out.append("\n## 7. Environment (measured = deploy settings)")
    seen = set()
    for r in rows:
        for k in ["GOGC", "GOMAXPROCS", "go_version", "lattigo_version",
                  "GOOS", "GOARCH"]:
            if r.get(k) and (k, r[k]) not in seen:
                seen.add((k, r[k]))
                out.append(f"- {k}: {r[k]}")
    ng = sum(1 for r in rows if r.get("go_neg_patched") == "True")
    out.append(f"- rows needing NegNew patch (0 expected post-fix): {ng}")

    report = "\n".join(out)
    print(report)
    rpath = "/home/maroua/chehab-unified/RL/veclang_runner/temp/pilot_report.md"
    with open(rpath, "w") as fh:
        fh.write(report + "\n")
    print(f"\n[report written to {rpath}]")


if __name__ == "__main__":
    main()

