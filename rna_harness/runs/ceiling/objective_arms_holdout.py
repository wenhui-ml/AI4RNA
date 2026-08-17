#!/usr/bin/env python3
"""Holdout confirmation on the 19 targets that have never been used, plus the mechanism's own prediction.

WHY A HOLDOUT. `objective_arms` found that adding a structure-agreement term to DesiRNA's objective cuts the
fraction of candidates missing the pseudoknot entirely from 34.5% to 16.0% on the fewest-crossing third, and
`mcc` was picked as the best of four arms on those same targets. Picking the best of four and quoting its
effect on the data that picked it is the error that has cost this project three retractions in two days. The
middle 19 targets were deliberately never touched by any run in this line -- not by the diagnosis, not by the
arms, not by the calibration -- so they are a clean test.

PRE-CHECKED BEFORE RUNNING, because the last version of this experiment was invalid for a reason no result
could reveal. All six passed:
    run integrity      190 rows, 0 errors, 0 empty pools
    arms are distinct  0 of 38 targets identical between any two distinct scoring arms
    the -sf reaches DesiRNA  verified in its own output directory name: SFEd-Epf_0.5_1-MCC_0.5
    groups disjoint    crossing counts 8-18 against 30-88, no overlap; n_cross independently recomputed
    holdout clean      the 19 middle targets appear in no prior run of this line
    reproducible       re-running an arm reproduces the stored pool BYTE-IDENTICALLY, so arm is the only
                       variable and there is no run-to-run noise in the comparison

PRE-REGISTERED, before any holdout target ran. Paired within target, Wilcoxon over the 19.

    H1  CONFIRMATION. Does `mcc` reduce the F1=0 fraction on targets that had no part in choosing it?
        PASS: median reduction > 0 and p < 0.025.
        KILL: no reduction -> the -13.0 points was the best of four arms fitted to 19 targets, and the
        finding does not survive contact with unseen ones.

    H2  THE MECHANISM'S QUANTITATIVE PREDICTION, which can fail even if H1 passes. The mechanism says a
        small pseudoknot is skipped because it contributes too little to the energy objective, so the
        benefit should scale with how small the pseudoknot is. These 19 targets sit BETWEEN the two groups
        already measured (crossing counts 19-29 against 8-18 and 30-88), so the reduction should land
        between the control's -3.5 points and the low group's -13.0.
        PASS: the holdout reduction falls inside that interval.
        A result outside it does not undo H1 -- the knob would still work -- but it refutes the explanation,
        and that gets reported as a refutation rather than smoothed over.

    H3  `mcc_only` carried along as the patch-independent arm: it is a SINGLE-term objective that parsed
        correctly even before I patched DesiRNA, so if it moves with `mcc`, the finding does not depend on
        my patch being right.

COST: $0 API. Local solver, 3 arms x 19 targets at a 30-step budget.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from leaderboard_design import targets                                          # noqa: E402
from generation_lever import pk_pairs, crossing                                 # noqa: E402
from objective_arms import work, ARMS, _assert_scoring_parses                   # noqa: E402

HOLDOUT_ARMS = ["base", "mcc", "mcc_only"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--per-group", type=int, default=19)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "objective_arms_holdout_pools.json"))
    a = ap.parse_args()

    cmd = os.environ.get("DESIRNA_CMD")
    assert cmd and os.path.exists(cmd.split()[-1]), "DESIRNA_CMD unset"
    _assert_scoring_parses()

    T = targets()
    nc = {p: len(crossing(pk_pairs(v["target"]))) for p, v in T.items()}
    order = sorted(T, key=lambda p: nc[p])
    used = set(order[:a.per_group]) | set(order[-a.per_group:])
    mid = [p for p in order if p not in used]
    assert mid, "no middle targets left"
    print(f"求解器:{cmd}(多项目标函数解析已校验)")
    print(f"留出集 {len(mid)} 个靶标,交叉数 {nc[mid[0]]}-{nc[mid[-1]]}"
          f"(已用过的两组是 {nc[order[0]]}-{nc[order[a.per_group-1]]} 和 "
          f"{nc[order[-a.per_group]]}-{nc[order[-1]]})")
    print(f"{len(HOLDOUT_ARMS)} 臂 x {len(mid)} 靶标 = {len(HOLDOUT_ARMS)*len(mid)} 次;步数 {a.steps}\n",
          flush=True)

    jobs = [(p, T[p]["target"], arm, a.seed, a.steps) for p in mid for arm in HOLDOUT_ARMS]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    ckpt = a.out + ".partial"
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            R.append(f.result())
            json.dump({"steps": a.steps, "seed": a.seed, "targets": {p: T[p] for p in mid},
                       "n_cross": {p: nc[p] for p in mid}, "arms": {k: ARMS[k] for k in HOLDOUT_ARMS},
                       "rows": R, "partial": True}, open(ckpt, "w"), ensure_ascii=False)
            if n % 10 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    json.dump({"steps": a.steps, "seed": a.seed, "targets": {p: T[p] for p in mid},
               "n_cross": {p: nc[p] for p in mid}, "arms": {k: ARMS[k] for k in HOLDOUT_ARMS},
               "rows": R}, open(a.out, "w"), ensure_ascii=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    err = [r for r in R if "error" in r]
    print(f"\n  完成 {len(R)},报错 {len(err)}")
    sz = {arm: sorted(len(r["pool"]) for r in R if r["arm"] == arm) for arm in HOLDOUT_ARMS}
    for arm in HOLDOUT_ARMS:
        v = sz[arm]
        print(f"  {arm:10s} 候选数中位 {v[len(v)//2] if v else 0}")
    print(f"\n  wrote {a.out}   (下一步:用 rnet venv 跑 objective_arms_holdout_score.py)")


if __name__ == "__main__":
    main()
