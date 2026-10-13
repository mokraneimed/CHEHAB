"""
Calibrated peak-RAM model for the CHEHAB RL environment (Lattigo / CPU).

Single entry point used by env.py:   peak_ram_mb(expr) -> float   (MB of process peak RSS)

How it works
  1. CKKSMemoryEstimator terms (Mounir) give the analytic "net" memory.
  2. A per-N-group calibration maps net -> measured MaxRSS (fitted on the pilot, see fit_calib.py).
  3. A multiplicative safety margin (held-out p90 of measured/predicted) is applied by default,
     so the constraint rarely under-estimates.
  4. Circuits WITH rotations (G>0) are NOT validated against the pipeline: the calibration is
     applied to the non-key terms only, and Galois-key bytes are scaled by GALOIS_KEY_FACTOR
     (from Mounir's synthetic Lattigo rows: measured/estimated p90 = 1.35, N=32768 p90 = 1.13).
"""
import json, math, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pytrs"))
from ckks_memory_estimator import (estimate_expression_memory, estimate_peak_memory,  # noqa: E402
                                   ciphertext_bytes)

BACKEND = "cpu"                     # only Lattigo/CPU is supported for now
MB = 1024.0 * 1024.0
GALOIS_KEY_FACTOR = 1.20            # G>0 only; NOT validated on the pipeline
CALIB_PATH = os.environ.get("PEAK_RAM_CALIB", os.path.join(_HERE, "memory_calib.json"))
ESTIMATOR_VERSION = "calib-v1"

_calib = None
_cache = {}
_CACHE_MAX = 300_000


def _load():
    global _calib
    if _calib is None:
        if not os.path.exists(CALIB_PATH):
            raise FileNotFoundError(
                f"{CALIB_PATH} not found. Run fit_calib.py first (or set PEAK_RAM_CALIB).")
        _calib = json.load(open(CALIB_PATH))
    return _calib


def group_of(N):
    return "16384" if N <= 16384 else "32768"


def _apply(group, net_mb, L, margin):
    c = _load()["groups"][group]
    if c["model"] == "ratio":
        pred = c["ratio"] * net_mb
    else:  # loglog:  log rss = a + b*log(net) + cL*L
        pred = math.exp(c["a"] + c["b"] * math.log(max(net_mb, 1e-6)) + c["cL"] * L)
    return pred * (c["margin"] if margin else 1.0)


def peak_ram_breakdown(expr, margin=True):
    est = estimate_expression_memory(expr, BACKEND)
    p = est.params
    gal_mb = est.terms.get("galois_keys", 0.0) / MB
    non_key_mb = est.total_mb - gal_mb
    g = group_of(p.N)
    mult = _load()["groups"][g]["margin"] if margin else 1.0
    pred = _apply(g, non_key_mb, p.L, margin) + gal_mb * GALOIS_KEY_FACTOR * mult
    return dict(mb=pred, N=p.N, L=p.L, K=p.K, G=est.rotation_steps, group=g,
                raw_net_mb=est.total_mb, galois_mb=gal_mb, live_cts=est.peak_live_ciphertexts,
                validated=(est.rotation_steps == 0))


def peak_ram_mb(expr, margin=True):
    """Calibrated peak RSS estimate in MB. Cached by expression string."""
    key = (expr, margin)
    v = _cache.get(key)
    if v is None:
        if len(_cache) > _CACHE_MAX:
            _cache.clear()
        v = peak_ram_breakdown(expr, margin)["mb"]
        _cache[key] = v
    return v


def floor_mb(N, L, K, margin=True):
    """Memory of an (almost) empty circuit with these params: the smallest value a rewrite can reach."""
    t = estimate_peak_memory(N, L, K, 0, 2 * ciphertext_bytes(N, L - 1), BACKEND)
    return _apply(group_of(N), sum(t.values()) / MB, L, margin)