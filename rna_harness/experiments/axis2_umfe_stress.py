"""R3-3.8: stress-test the A_uMFE co-optimal enumeration.

Reviewer: the claim that the narrow-window suboptimal enumeration is 'bounded, avoiding exponential blow-up'
is not guaranteed -- co-optimal (MFE-energy) structures can be numerous, and uniq_ML only removes multiloop
backtracking duplicates. We measure the ACTUAL number of MFE-energy structures fc.subopt(0) returns per
candidate, across designed exact-match solutions and random sequences over a length range, and report the
distribution, max, and any anomalies -- so the cost/boundedness claim is empirical, not asserted."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import RNA

from ..benchmarks.eterna100 import load_eterna100
from .verifier_routing import _solver_seq

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "axis2_umfe_stress")


def _cooptimal_count(seq):
    """# structures at exactly the MFE energy (subopt window 0), with uniq_ML backtracking."""
    fc = RNA.fold_compound(seq)
    RNA.cvar.uniq_ML = 1
    _, mfe_e = fc.mfe()
    sols = fc.subopt(0)                       # 0 dekacal window == MFE-energy co-optimal set
    return len(sols)


def run(n_targets, max_n, seed, outdir):
    import random
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)
    puz = [p for p in sorted(load_eterna100(version="V2"), key=lambda p: len(p["target"]))
           if len(p["target"]) <= max_n][:n_targets]
    rows = []
    for p in puz:
        t = p["target"]; n = len(t)
        sol = _solver_seq(t, rng, budget=4000)          # a designed (near-)exact-match sequence
        rnd = "".join(rng.choice("ACGU") for _ in range(n))
        for kind, s in (("solved", sol), ("random", rnd)):
            try:
                c = _cooptimal_count(s)
                rows.append({"id": p["id"], "n": n, "kind": kind, "cooptimal": c})
            except Exception as e:
                rows.append({"id": p["id"], "n": n, "kind": kind, "cooptimal": None, "err": str(e)[:80]})
    counts = [r["cooptimal"] for r in rows if r["cooptimal"] is not None]
    arr = np.array(counts)
    summ = {"n_seqs": len(counts), "min": int(arr.min()), "median": int(np.median(arr)),
            "mean": round(float(arr.mean()), 2), "p95": int(np.percentile(arr, 95)),
            "max": int(arr.max()), "frac_gt_10": round(float((arr > 10).mean()), 4),
            "frac_gt_100": round(float((arr > 100).mean()), 4),
            "errors": sum(1 for r in rows if r["cooptimal"] is None)}
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(outdir, "umfe_stress.json"), "w"), indent=1)
    print("=== A_uMFE co-optimal enumeration stress test ===", flush=True)
    print(f"  {summ['n_seqs']} sequences (designed + random), n<= {max_n}", flush=True)
    print(f"  co-optimal count: min={summ['min']} median={summ['median']} mean={summ['mean']} "
          f"p95={summ['p95']} MAX={summ['max']}", flush=True)
    print(f"  frac >10: {summ['frac_gt_10']}  frac >100: {summ['frac_gt_100']}  errors: {summ['errors']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=60)
    ap.add_argument("--max-n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    run(a.n_targets, a.max_n, a.seed, a.outdir)
