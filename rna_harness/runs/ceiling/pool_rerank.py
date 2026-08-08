#!/usr/bin/env python3
"""Keep every candidate the solver produced, so we can ask whether a better CHOOSER moves our placement.

WHAT THIS IS FOR, and it is a capability move rather than another measurement. Yesterday put us on the
OpenKnot table for the first time and we placed 11th of 14, losing to human players, to four AI methods, and
to the natural wild-type sequence. The decomposition said the loss is in the QUALITY of what we deliver
(-0.143) far more than in coverage (-0.057). Separately, our candidate selector was measured against real
molecules and does not beat picking at random at the pool size the product actually uses, while
RibonanzaNet -- now reproduced locally at median base-pair F1 1.000 -- does.

Those two facts together make one concrete question worth answering before anything else is attempted:

    Our answer to each target was chosen from ~131 candidates by a selector that has just been shown not to
    work. Was there a BETTER SEQUENCE ALREADY IN THAT POOL?

If yes, our placement is a selection problem and it is fixable today with a model we already run. If no,
the pool itself is the ceiling, the problem is generation, and every hour spent on selection from here is
wasted. Either answer redirects the project; that is why this runs before any other capability work.

WHAT STAGE A DOES DIFFERENTLY. `leaderboard_design.py` kept only the sequence the product returned. This
keeps EVERY unique candidate across every tier, with the panel's own verdict on each, so the pool can be
re-ranked offline by any rule without re-running the solver.

THE ISOLATION RULE, unchanged and non-negotiable. RibonanzaNet never enters the design loop. The pool is
generated exactly as the product generates it, adjudicated exactly as the product adjudicates it, and only
then, offline, is a different chooser applied. We have paid for breaking this: an operator that used pKiss
in-loop reported 43 of 60 solved against a panel-confirmed 1 of 60.

PRE-REGISTERED, before any pool was scored. Paired within target, Wilcoxon over the 57 puzzles.

    C1  CEILING. The best candidate in the pool by RNet crossed-pair F1 beats the candidate the product
        actually returned. This is an ORACLE bound -- it uses the answer to pick -- so it is not a product
        proposal; it is the largest gain any selector could ever deliver from these pools. If C1's gain is
        small, selection is not where our placement comes from and the axis closes.

    C2  REALISABLE. Re-ranking by RNet alone (no oracle knowledge of the target beyond what RNet predicts,
        which is exactly what the product would have at run time) beats the product's current choice.
        This is the shippable version. PASS threshold: p < 0.025 (two tests) and a median gain >= 0.05 F1,
        because a statistically real gain below that does not move a leaderboard.

    C3  PLACEMENT. With C2's chooser, re-run the leaderboard comparison. Does our position against the
        field median change from 8/57 better, and by how much?

    KILL: C1's median gain < 0.05 -> the pools do not contain better sequences, the ceiling is generation
    rather than selection, and this project stops trying to fix its placement by choosing better.

COST: $0 API. The solver and panel are local binaries; RNet is local CPU.
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

from leaderboard_design import targets                                         # noqa: E402


def work(item):
    """Run the product's own ladder and keep the whole pool, not just the winner.

    TWO THINGS DELIBERATELY NOT DONE HERE, both of which the first version did and both of which doubled
    or quadrupled the cost for nothing:

      no panel adjudication of the pool. Adjudicating every candidate costs 55 s each at 240 nt (ShapeKnots
      is 88% of that) and would dominate the run. The panel grade is only needed by a selector we have
      already measured to be worthless at this pool size, so the pool is collected raw and scored offline.

      no second `certify_design` call. What the product returned for these exact targets, at this exact
      seed and ladder, is already on disk in leaderboard_ours.json; re-deriving it would re-run the whole
      ladder to learn something we already recorded.
    """
    pid, target, tiers, seed = item
    from rna_harness.generators.desirna import DesiRNAGenerator         # same call the product makes
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    seen, pool = set(), []
    try:
        for tl in tiers:
            res = DesiRNAGenerator(replicas=10, timelimit_s=tl, results_number=20).design(
                DesignTask(task_id="pool", target=target, oracle=oc), seed=seed)
            cands = [c for c in ([res.best_sequence] if res.best_sequence else [])
                     + list(getattr(res, "candidates", None) or []) if c]
            for c in cands:
                if c in seen or len(c) != len(target):
                    continue
                seen.add(c)
                pool.append({"seq": c, "tier": tl})
    except Exception as e:                                                     # noqa: BLE001
        return {"puzzle": pid, "error": repr(e)[:200], "pool": pool}
    return {"puzzle": pid, "n": len(target), "pool": pool,
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "pool_rerank_pools.json"))
    a = ap.parse_args()

    cmd = os.environ.get("DESIRNA_CMD")
    assert cmd and os.path.exists(cmd.split()[-1]), "DESIRNA_CMD unset -- see leaderboard_design.py"
    print(f"求解器:{cmd}")

    T = targets()
    tiers = (5, 15, 60)
    print(f"{len(T)} 个靶标,阶梯 {tiers}(与出厂 standard 相同),保留【全部候选】\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(p, v["target"], tiers, a.seed) for p, v in sorted(T.items())]
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            R.append(r)
            # checkpoint every result: the previous run kept 57 targets in memory for hours and a crash
            # in the optional rescue phase would have destroyed all of them
            json.dump({"seed": a.seed, "tiers": list(tiers), "targets": T, "rows": R},
                      open(a.out, "w"), ensure_ascii=False)
            if n % 5 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min  "
                      f"池子中位 {sorted(len(x['pool']) for x in R)[len(R)//2]} 条", flush=True)

    sizes = [len(r["pool"]) for r in R]
    print(f"\n候选池:中位 {sorted(sizes)[len(sizes)//2]} 条,范围 {min(sizes)}-{max(sizes)},"
          f"合计 {sum(sizes)} 条序列")
    print(f"  wrote {a.out}   (下一步:用 rnet venv 跑 pool_rerank_score.py)")


if __name__ == "__main__":
    main()
