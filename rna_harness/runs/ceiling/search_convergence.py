"""Stage A of the search-vs-objective test: run the solver at four time limits, keeping the tiers SEPARATE.

WHY THE EXISTING POOL CANNOT ANSWER THIS. `pool_rerank.py` deduplicated candidates ACROSS tiers, in ladder
order, so a sequence found at both 5 s and 60 s is filed under 5 s. What survives in the 60 s tier is
therefore precisely the candidates the 5 s tier did NOT find -- which is the opposite of a fair comparison,
because a solver that converges to the same good optima has all its good repeats stripped out and is left
looking worse the longer it runs. Tiers are kept independent here, deduplicated only within a tier.

THE HYPOTHESIS THIS IS PAIRED WITH. `criterion_vs_tube.py` asks whether our thermodynamic panel predicts the
tube worse than RibonanzaNet does. If it does, then our solver spends its extra seconds optimising a target
that is not the one we are graded on, and a sharp prediction follows:

    more search time should NOT improve RNet crossed-pair F1, and may actively degrade it.

That is a real prediction with a direction, not a convergence check. It can come out wrong, and if it does,
the misalignment story is weakened regardless of what G2 says.

    S1  MEDIAN QUALITY. Per target, median RNet crossed-pair F1 of the 300 s tier minus that of the 5 s tier.
        Wilcoxon over targets, TWO-SIDED, because the interesting outcome here is a decrease and a one-sided
        test would hide it.
    S2  BEST CANDIDATE. The same on the per-tier maximum -- what a perfect chooser could have had.
    S3  MONOTONICITY. Median quality across 5 -> 15 -> 60 -> 300 s, to see whether any trend is monotone or
        whether it turns over.

    READ-OFF fixed in advance:
        rises with time    -> the search is under-converged, more compute is a cheap real gain
        flat               -> the search has converged and the objective is the ceiling
        falls with time    -> the solver is over-optimising an objective we are not graded on, which is the
                              misalignment story in its strongest form

COST: $0 API. Local solver.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from leaderboard_design import targets                                          # noqa: E402


def work(item):
    pid, target, tl, seed = item
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=tl, results_number=20).design(
            DesignTask(task_id=f"conv{tl}", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                      # noqa: BLE001
        return {"puzzle": pid, "tier": tl, "error": repr(e)[:160], "pool": []}
    cands = [c for c in ([res.best_sequence] if res.best_sequence else [])
             + list(getattr(res, "candidates", None) or []) if c]
    seen, pool = set(), []
    for c in cands:                                       # dedup WITHIN the tier only
        if c not in seen and len(c) == len(target):
            seen.add(c)
            pool.append(c)
    return {"puzzle": pid, "tier": tl, "pool": pool, "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="5,15,60,300")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "search_convergence_pools.json"))
    a = ap.parse_args()

    cmd = os.environ.get("DESIRNA_CMD")
    assert cmd and os.path.exists(cmd.split()[-1]), (
        "DESIRNA_CMD is unset or points nowhere; without it every target silently returns nothing and this "
        "run would measure our environment rather than our solver")
    print(f"求解器:{cmd}")

    tiers = [int(x) for x in a.tiers.split(",")]
    T = targets()
    jobs = [(p, v["target"], tl, a.seed) for p, v in sorted(T.items()) for tl in tiers]
    print(f"{len(T)} 个靶标 x {len(tiers)} 档时限 {tiers} = {len(jobs)} 次求解,各档独立不跨档去重\n",
          flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            R.append(f.result())
            json.dump({"seed": a.seed, "tiers": tiers, "targets": T, "rows": R},
                      open(a.out, "w"), ensure_ascii=False)
            if n % 20 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    for tl in tiers:
        sz = [len(r["pool"]) for r in R if r["tier"] == tl]
        print(f"  {tl:4d}s 档:{len(sz)} 个靶标,候选数中位 {sorted(sz)[len(sz)//2] if sz else 0}")
    print(f"\n  wrote {a.out}   (下一步:用 rnet venv 跑 search_convergence_score.py)")


if __name__ == "__main__":
    main()
