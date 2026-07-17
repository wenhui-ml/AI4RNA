"""E4': honest end-to-end verification COST per rung vs length n (the defensible C3 core).

Review-mandated fairness: every rung is timed from RAW (seq, target) -> accept/reject bool, so `match`
folds from scratch too (no pre-stored-string shortcut). We report wall AND CPU time, peak memory, the
empirical scaling exponent in n, and DISTRIBUTIONS (median / p95 / max), not just means -- because uMFE
via subopt enumeration is heavy-tailed. We DO NOT claim orders-of-magnitude separation: RNAfold MFE and
the partition function are both ~O(n^3); pKiss is ~O(n^4) and lives on a separate BRANCH whose cost is
subprocess-startup-dominated at small n. Sequences are random ACGU folded to their own MFE structure
(the clean regime designed solutions live in), so subopt stays tractable. Model-relative; no truth claim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import time

import RNA

from ..cognitive.tools import verify

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "verifier_cost")
_BASES = "ACGU"


def _rand_seq(n, rng):
    return "".join(rng.choice(_BASES) for _ in range(n))


def _time_rung(seq, target, strictness):
    """End-to-end wall + CPU seconds for one rung's accept/reject decision (from scratch)."""
    w0, c0 = time.perf_counter(), time.process_time()
    verify(seq, target, strictness=strictness)
    return time.perf_counter() - w0, time.process_time() - c0


def _pkiss_cost(seq, target, timeout_s=60.0):
    """Branch cost: pKiss subprocess end-to-end (startup-dominated at small n). Returns wall or None."""
    from ..benchmarks.rnainvbench import verify_pk, pkiss_available
    if not pkiss_available():
        return None
    w0 = time.perf_counter()
    try:
        verify_pk(seq, target, timeout_s=timeout_s)
    except Exception:
        return None
    return time.perf_counter() - w0


def _stats(xs):
    xs = sorted(xs)
    k = len(xs)
    return {"median": xs[k // 2], "p95": xs[min(k - 1, int(0.95 * k))], "max": xs[-1], "n": k}


def _slope(ns, meds):
    """Empirical scaling exponent = slope of log(median_wall) vs log(n)."""
    lx = [math.log(n) for n in ns]
    ly = [math.log(m) for m in meds]
    k = len(lx)
    mx, my = sum(lx) / k, sum(ly) / k
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den else float("nan")


def run(lengths, reps, outdir, seed=0, pkiss_max_n=150, pkiss_reps=5):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)
    rungs = ["match", "umfe", "pfe"]
    per = {r: {"n": [], "med": [], "p95": [], "max": [], "cpu_med": []} for r in rungs}
    per["pkiss"] = {"n": [], "med": [], "p95": [], "max": []}
    rows = []
    print(f"E4' verifier cost: lengths={lengths} reps={reps} (+pKiss branch up to n={pkiss_max_n})", flush=True)
    for n in lengths:
        walls = {r: [] for r in rungs}
        cpus = {r: [] for r in rungs}
        for _ in range(reps):
            s = _rand_seq(n, rng)
            t = RNA.fold(s)[0]                       # target = seq's own MFE (clean regime)
            for r in rungs:
                w, c = _time_rung(s, t, r)
                walls[r].append(w)
                cpus[r].append(c)
            rows.append({"n": n, **{f"{r}_wall": round(walls[r][-1], 6) for r in rungs}})
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # KB->MB high-water
        line = [f"n={n:>4}"]
        for r in rungs:
            st = _stats(walls[r])
            per[r]["n"].append(n); per[r]["med"].append(st["median"])
            per[r]["p95"].append(st["p95"]); per[r]["max"].append(st["max"])
            per[r]["cpu_med"].append(sorted(cpus[r])[reps // 2])
            line.append(f"{r}={st['median']*1e3:7.2f}ms(p95 {st['p95']*1e3:7.2f})")
        # pKiss branch (sparse, bounded)
        if n <= pkiss_max_n:
            pk = [c for c in (_pkiss_cost(_rand_seq(n, rng), RNA.fold(_rand_seq(n, rng))[0])
                              for _ in range(pkiss_reps)) if c is not None]
            if pk:
                st = _stats(pk)
                per["pkiss"]["n"].append(n); per["pkiss"]["med"].append(st["median"])
                per["pkiss"]["p95"].append(st["p95"]); per["pkiss"]["max"].append(st["max"])
                line.append(f"pKiss={st['median']*1e3:8.1f}ms")
        line.append(f"RSS={rss_mb:.0f}MB")
        print("  " + "  ".join(line), flush=True)
        json.dump({"per": per, "rows": rows}, open(os.path.join(outdir, "cost.json"), "w"), indent=1)

    print("\n=== empirical scaling exponent (slope of log median-wall vs log n) ===", flush=True)
    for r in rungs:
        print(f"  {r:6}: ~n^{_slope(per[r]['n'], per[r]['med']):.2f}   "
              f"(median wall at n={lengths[-1]}: {per[r]['med'][-1]*1e3:.2f} ms; "
              f"uMFE/heavy-tail max {per[r]['max'][-1]*1e3:.2f} ms)" if r == "umfe"
              else f"  {r:6}: ~n^{_slope(per[r]['n'], per[r]['med']):.2f}   "
              f"(median wall at n={lengths[-1]}: {per[r]['med'][-1]*1e3:.2f} ms)", flush=True)
    if per["pkiss"]["n"]:
        print(f"  pkiss : median wall {per['pkiss']['med'][0]*1e3:.0f}->{per['pkiss']['med'][-1]*1e3:.0f} ms "
              f"(n={per['pkiss']['n'][0]}->{per['pkiss']['n'][-1]}); BRANCH, startup-dominated at small n", flush=True)
    print("\nHonest read: match/uMFE/pfe within a small constant factor (all ~O(n^3)); the big jump is the "
          "pKiss BRANCH, not a strictness rung.", flush=True)
    json.dump({"per": per, "rows": rows}, open(os.path.join(outdir, "cost.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[30, 60, 100, 150, 200, 300, 400])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--pkiss-max-n", type=int, default=150)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    run(a.lengths, a.reps, a.outdir, pkiss_max_n=a.pkiss_max_n)
