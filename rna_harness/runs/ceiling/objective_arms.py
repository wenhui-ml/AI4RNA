#!/usr/bin/env python3
"""Stage A: does telling the solver to care about the structure stop it ignoring small pseudoknots?

THE DIAGNOSIS THIS TESTS. Our generation gap against novelty-matched peers is worst where the target is
easiest -- fewest-crossing third -0.403, middle -0.186, most-crossing third -0.092, while the peers sit flat
at 0.94-0.97 across all three. And the shape is bimodal rather than shifted: on the fewest-crossing targets
33.4% of our candidates reproduce not a single crossing pair while 17.5% reproduce almost all of them, and
six targets have a pool MEDIAN of exactly 0.000 against a pool maximum of 0.88 to 1.000.

THE MECHANISM. DesiRNA optimises `Ed-Epf:1.0` by default -- the energy of the desired structure against the
partition function -- and that is what our adapter has always used. When the crossing stems are a small part
of the structure they contribute little to that quantity, so the solver can satisfy its objective while
folding into a competing nested structure and leaving the pseudoknot out. When the crossing stems dominate
the target it cannot. That predicts exactly the monotone curve above.

WHAT MAKES IT TESTABLE. DesiRNA's `-sf` takes weighted combinations, and `1-MCC` and `1-recall` score the
predicted structure against the target directly. Both are pseudoknot-aware here: `SimScore` is handed the
full target string with its bracket types, and the prediction goes through `get_pk_struct`, with `pkon`
already set. So a structure-agreement term rewards forming the crossing pairs in a way that the pure energy
term does not. `-nd on` attacks the same failure from the other side by penalising competing folds.

    ARMS. base `Ed-Epf:1.0` (what ships) / mcc `Ed-Epf:0.5,1-MCC:0.5` / recall `Ed-Epf:0.5,1-recall:0.5` /
    mcc_only `1-MCC:1.0` / negdesign `Ed-Epf:1.0` with `-nd on`.

FIRST RUN RETRACTED, 2026-08-08, and with it the claim that the mechanism was refuted. The `mcc` and
`recall` arms produced BYTE-IDENTICAL pools on 38 of 38 targets, which two different objectives cannot do.
The cause is a bug in DesiRNA: `parse_scoring_functions` had its `return` indented inside the parsing loop,
so every multi-term scoring function silently collapsed to its first term.

    -sf Ed-Epf:0.5,1-MCC:0.5      parsed to  [('Ed-Epf', 0.5)]
    -sf Ed-Epf:0.5,1-recall:0.5   parsed to  [('Ed-Epf', 0.5)]

Both arms therefore contained NO structure term whatsoever; they were `Ed-Epf` at half weight, run twice.
So "adding structure agreement does nothing, therefore the mechanism is wrong" was not measured -- the
intervention never happened. The mechanism is untested again, not refuted. Patched locally (the `-sf` help
text advertises exactly this syntax and `get_scoring_function` already loops over the returned list, so the
intent is unambiguous); the patch is documented in that file.

    `mcc_only` uses a SINGLE-term scoring function, which parsed correctly even before the patch. It is
    there so the conclusion does not rest on my patch being right: if the mixed arms and mcc_only disagree
    about whether structure agreement helps, the patch is the first suspect.

The negdesign result is unaffected -- `-nd on` is a separate flag that never goes through this parser -- but
it is being re-run alongside so every arm comes from one comparable batch. Noted for follow-up: negdesign
gave pools identical to base on 9 of 38 targets, so on those the flag changed nothing at all.

PRE-REGISTERED, before any arm ran. Reproducible by construction: step budget plus pinned hash seed, so
every arm sees the same solver machinery and arm is the only thing that varies. Paired within target.

    M1  THE INTERVENTION, on the 19 fewest-crossing targets. Fraction of candidates scoring exactly 0.000
        crossed-pair F1, best arm against base. Wilcoxon over targets, one-sided less.
        PASS: median reduction >= 10 percentage points and p < 0.025.
        KILL: no arm reduces it -> the objective is not why small pseudoknots get dropped, and the
        mechanism is wrong however plausible it reads.

    M2  THE CONTROL THAT CAN FALSIFY THE MECHANISM, not just the intervention. The same arms on the 19
        MOST-crossing targets, where the mechanism says the pseudoknot could not be ignored anyway and the
        intervention therefore has little to fix.
        The mechanism predicts M1's effect to be MUCH LARGER than M2's. If a structure term helps equally
        everywhere, the intervention may still be worth shipping, but the explanation on offer is wrong and
        will be reported as wrong -- an intervention that works for a reason I cannot demonstrate is not a
        mechanism, it is a knob.

    M3  DOES IT COST ANYTHING. Pool median crossed-pair F1 per arm. A structure term could trade energy
        quality for agreement; if the median falls while the zero-fraction falls, that is a trade to report,
        not a win.

COST: $0 API. Local solver.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from leaderboard_design import targets                                          # noqa: E402
from generation_lever import pk_pairs, crossing                                 # noqa: E402

ARMS = {
    "base":      "-sf Ed-Epf:1.0",
    "mcc":       "-sf Ed-Epf:0.5,1-MCC:0.5",
    "recall":    "-sf Ed-Epf:0.5,1-recall:0.5",
    "mcc_only":  "-sf 1-MCC:1.0",          # single term: parses correctly with or without the patch
    "negdesign": "-sf Ed-Epf:1.0 -nd on",
}


def _assert_scoring_parses():
    """Fail loudly if the multi-term objectives collapse, instead of silently testing the same arm twice.

    The first run of this file reported that structure agreement does nothing, on the strength of two arms
    that had silently become identical. A guard is cheaper than the retraction was.
    """
    import subprocess
    probe = ("import sys; sys.path.insert(0, '/data/wenhui/DesiRNA');"
             "from utils.stats_inputs_outputs import parse_scoring_functions as f;"
             "print(len(f('Ed-Epf:0.5,1-MCC:0.5')))")
    py = (os.environ.get("DESIRNA_CMD") or "python").split()[0]
    out = subprocess.run([py, "-c", probe], capture_output=True, text=True).stdout.strip()
    assert out == "2", (
        f"DesiRNA parses a two-term scoring function into {out!r} components, not 2. The multi-term arms "
        f"would collapse to their first term and this run would compare identical arms -- see the "
        f"retraction note in this file's header.")


def work(item):
    pid, target, arm, seed, steps = item
    os.environ["DESIRNA_ARGS"] = f"-t 1800 -s {steps} {ARMS[arm]}"
    os.environ["DESIRNA_SCRAPE_TRACE"] = "0"
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=1800, results_number=20).design(
            DesignTask(task_id="obj", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                     # noqa: BLE001
        return {"puzzle": pid, "arm": arm, "error": repr(e)[:160], "pool": []}
    pool = [c for c in ([res.best_sequence] if res.best_sequence else [])
            + list(getattr(res, "candidates", None) or []) if c and len(c) == len(target)]
    seen, uniq = set(), []
    for c in pool:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return {"puzzle": pid, "arm": arm, "pool": uniq, "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--per-group", type=int, default=19)
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "objective_arms_pools.json"))
    a = ap.parse_args()

    cmd = os.environ.get("DESIRNA_CMD")
    assert cmd and os.path.exists(cmd.split()[-1]), "DESIRNA_CMD unset"
    _assert_scoring_parses()
    print(f"求解器:{cmd}(多项目标函数解析已校验)")

    T = targets()
    nc = {p: len(crossing(pk_pairs(v["target"]))) for p, v in T.items()}
    order = sorted(T, key=lambda p: nc[p])
    low, high = order[:a.per_group], order[-a.per_group:]
    group = {p: "low" for p in low}
    group.update({p: "high" for p in high})
    print(f"最少交叉 {a.per_group} 个(交叉数 {nc[low[0]]}-{nc[low[-1]]}),"
          f"最多交叉 {a.per_group} 个({nc[high[0]]}-{nc[high[-1]]})")
    print(f"{len(ARMS)} 个臂 x {len(group)} 个靶标 = {len(ARMS)*len(group)} 次;步数 {a.steps},"
          f"哈希种子已钉住,臂是唯一变量\n", flush=True)
    for k, v in ARMS.items():
        print(f"  {k:10s} {v}")
    print(flush=True)

    jobs = [(p, T[p]["target"], arm, a.seed, a.steps) for p in sorted(group) for arm in ARMS]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    ckpt = a.out + ".partial"
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            R.append(f.result())
            json.dump({"steps": a.steps, "seed": a.seed, "targets": T, "group": group,
                       "n_cross": nc, "arms": ARMS, "rows": R, "partial": True},
                      open(ckpt, "w"), ensure_ascii=False)
            if n % 10 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    json.dump({"steps": a.steps, "seed": a.seed, "targets": T, "group": group,
               "n_cross": nc, "arms": ARMS, "rows": R}, open(a.out, "w"), ensure_ascii=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    sz = collections.defaultdict(list)
    for r in R:
        sz[r["arm"]].append(len(r["pool"]))
    for arm in ARMS:
        v = sorted(sz[arm])
        print(f"  {arm:10s} 候选数中位 {v[len(v)//2] if v else 0}")
    print(f"\n  wrote {a.out}   (下一步:用 rnet venv 跑 objective_arms_score.py)")


if __name__ == "__main__":
    main()
