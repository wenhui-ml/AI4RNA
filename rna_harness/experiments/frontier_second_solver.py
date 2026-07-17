"""Reviewer-hardening control: is the Axis-2 over-acceptance gap a property of the VERIFIER, or an
artifact of our REMC solver? We re-run the frontier with a completely independent solver --- ViennaRNA's
built-in RNAinverse (adaptive-walk inverse folding, a different algorithm family from replica-exchange) ---
and judge its exact-match solutions under the same match/uMFE/pfe ladder. If survival still collapses, the
gap is intrinsic to cheap-proxy verification, not to any one solver."""
from __future__ import annotations
import argparse, json, os, random
import RNA
from ..benchmarks.eterna100 import load_eterna100
from .verifier_frontier import ladder_eval, LEVELS

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "frontier_second_solver")
_BASES = "ACGU"


def rnainverse_solve(target, seed=0, tries=6):
    """Independent solver: ViennaRNA adaptive-walk inverse folding, multi-start. Returns an exact-match
    sequence (RNA.fold==target) or None."""
    rng = random.Random(seed)
    for _ in range(tries):
        start = "".join(rng.choice(_BASES) for _ in range(len(target)))
        seq, dist = RNA.inverse_fold(start, target)   # dist==0 => folds exactly to target
        if dist == 0 and RNA.fold(seq)[0] == target:
            return seq
    return None


def run(ids, outdir, seeds=3, max_n=120):
    os.makedirs(outdir, exist_ok=True)
    puz = {p["id"]: p for p in load_eterna100(version="V2")}
    ids = [i for i in ids if i in puz and len(puz[i]["target"]) <= max_n]
    print(f"second-solver (RNAinverse) frontier on {len(ids)} targets (n<={max_n}) x {seeds} seeds",
          flush=True)
    rows = []
    for pid in ids:
        t = puz[pid]["target"]
        got = 0
        for s in range(seeds):
            seq = rnainverse_solve(t, seed=1000 * s + pid)
            if seq is None:
                continue
            ev = ladder_eval(seq, t)
            rows.append({"pid": pid, "seed": s, "n": len(t), **ev})
            got += 1
        print(f"  pid {pid:>3} n={len(t):>4} | exact-match {got}/{seeds}", flush=True)
        json.dump({"rows": rows}, open(os.path.join(outdir, "second_solver.json"), "w"), indent=1)
    if not rows:
        print("no exact-match solutions", flush=True); return
    n = len(rows)
    surv = {lv: sum(r[lv]["solved"] for r in rows) / n for lv in LEVELS}
    print(f"\n=== SECOND-SOLVER (RNAinverse) FRONTIER over {n} solutions ===", flush=True)
    for lv in LEVELS:
        print(f"  {lv:6} survival={surv[lv]*100:5.1f}%", flush=True)
    print(f"  false-positive mass (match not pfe) = {(surv['match']-surv['pfe'])*100:.1f}%", flush=True)
    json.dump({"agg": {"n": n, "survival": surv}, "rows": rows},
              open(os.path.join(outdir, "second_solver.json"), "w"), indent=1)
    print(f"  wrote {os.path.join(outdir, 'second_solver.json')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-n", type=int, default=120)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    all_ids = sorted(p["id"] for p in load_eterna100(version="V2"))
    run(all_ids[: a.n] if a.n else all_ids, a.outdir, seeds=a.seeds, max_n=a.max_n)
