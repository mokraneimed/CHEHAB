#!/usr/bin/env python3
"""Step 4 - Evaluation of whether circuit rewrites bring peak RAM under target budgets.
Uses the MEASURED pilot pairs (original vs rewritten) in RL/veclang_runner/temp/pilot_results.csv.
Reads only; modifies nothing.
"""
import csv
import os
import statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.path.join(ROOT, "RL", "veclang_runner", "temp", "pilot_results.csv")
BUDGETS = [1200, 1500, 2000, 2600]

orig, rw = {}, {}
for r in csv.DictReader(open(P)):
    if r.get("status") != "ok" or not r.get("max_rss_MB_median"):
        continue
    v = float(r["max_rss_MB_median"])
    if r.get("kind") == "pair_orig":
        orig[r["pair_id"]] = (v, r.get("go_N", ""))
    elif r.get("kind") == "pair_rw":
        rw[r["pair_id"]] = (v, r.get("go_N", ""))

pairs = [(k, orig[k], rw[k]) for k in orig if k in rw]
print("measured pairs with both sides ok: %d" % len(pairs))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


for label, sel in (("N=16384 originals", lambda n: n and int(float(n)) <= 16384),
                   ("N=32768 originals", lambda n: n and int(float(n)) > 16384)):
    sub = [(k, o[0], w[0]) for k, o, w in pairs if sel(o[1])]
    if not sub:
        print(label, ": none")
        continue
    ratios = [w / o for _, o, w in sub]
    print("\n%s: n=%d | rewritten/original RSS ratio: p10 %.2f  p50 %.2f  p90 %.2f | reduced by >=10%%: %d/%d | by >=25%%: %d/%d"
          % (label, len(sub), pct(ratios, .1), st.median(ratios), pct(ratios, .9),
             sum(1 for x in ratios if x <= 0.9), len(sub), sum(1 for x in ratios if x <= 0.75), len(sub)))
    for b in BUDGETS:
        above = [(o, w) for _, o, w in sub if o > b]
        if above:
            fixed = sum(1 for o, w in above if w <= b)
            print("   budget %4d MB: originals above = %2d ; of these the rewritten version is under the budget: %d/%d"
                  % (b, len(above), fixed, len(above)))
print("\nNOTE: these are the pilot's own rewrites (not RL-chosen), so they show what is achievable, not what the agent will do.")