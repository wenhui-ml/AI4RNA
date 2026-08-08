#!/usr/bin/env python3
"""A controlled A/B of the trajectory-scrape fix, with the solver held byte-identical between arms.

WHY THE FIRST ATTEMPT DID NOT COUNT. The post-fix leaderboard rerun returned 43 of 57 targets against the
old run's 47, and that difference is uninterpretable: the two arms were separate solver runs, and the solver
was irreproducible. Two invocations at the same seed shared 1 candidate out of 121. The measured effect was
smaller than the noise between two runs of identical code, so no conclusion could be drawn either way.

WHAT MAKES IT ANSWERABLE NOW. The irreproducibility was Python's per-process hash randomisation reaching
DesiRNA's dict and set iteration; pinning PYTHONHASHSEED plus a step budget makes a run byte-identical
(verified: three runs, three targets, identical 200-candidate pools). So both arms here execute the SAME
solver trajectory, and the only thing that differs is which files the adapter scrapes candidates from.
`DESIRNA_SCRAPE_TRACE=1` restores the pre-fix behaviour on the real code path, reading the recorded walk
order rather than an approximation of it, so ordering is preserved too -- which matters, because
`certify_design` selects the FIRST candidate in pool order that the panel accepts.

CALIBRATION OF THE STEP BUDGET, done before this test and reported because it constrains what it can show.
The trace files contribute a roughly constant ten candidates while the answer files grow with the step
count, so the contamination FRACTION depends on how long the solver ran, and the 200-slot cap erases the
difference entirely once the answer files alone overflow it:

    steps= 3    31 vs 41 candidates    trace 24% of the old pool
    steps=10   101 vs 111              trace  9%      <- matches the 8.3% measured on the shipped 60 s tier
    steps=20   200 vs 200              cap reached, the two arms are IDENTICAL

steps=10 is used because it reproduces the shipped contamination fraction. A consequence worth stating
plainly: at the product's longer tiers the answer files can fill the cap on their own, in which case this
fix changes nothing at all, and at the 300 s tier the trace is enormous, in which case it changes much more.
This measures the middle case, which is the one users actually get.

PRE-REGISTERED, before any target ran. Paired within target over 57 targets; the arms differ by one
environment variable and nothing else.

    A1  COVERAGE. Count of targets returning a sequence, new against old. Descriptive, no threshold. The
        uncontrolled rerun suggested the fix COSTS coverage, and this is the test of that.

    A2  DELIVERED QUALITY. RibonanzaNet crossed-pair F1 of what each arm hands back, a target neither arm
        delivers scoring 0.0 in both, a target only one arm delivers scoring 0.0 in the other. Paired,
        Wilcoxon, TWO-SIDED -- I do not know the sign. Removing candidates can only shrink the pool, and
        because selection takes the first acceptable candidate rather than the best one, removing a bad
        early candidate can help and removing the only acceptable one can hurt.
        alpha = 0.05.

    A3  HOW OFTEN DOES ANYTHING CHANGE. Targets whose delivered sequence differs at all between arms. If
        this is near zero the fix is a correctness change with no measurable delivery effect, which is a
        legitimate and reportable outcome.

    READ-OFF fixed here: the fix is justified on CORRECTNESS regardless of A2 -- returning states the
    solver itself discarded is wrong whatever it scores. But if A2 shows a real quality LOSS, that has to
    be reported as the price of the correctness fix, not buried.

COST: $0 API. Local solver, local panel.
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
    pid, target, arm, seed, steps = item
    # Set BEFORE importing/using the generator; each worker is its own process, so this cannot leak
    # across arms running concurrently.
    os.environ["DESIRNA_SCRAPE_TRACE"] = "1" if arm == "old" else "0"
    os.environ["DESIRNA_ARGS"] = f"-t 900 -s {steps}"      # argparse last-wins: overrides the tier's -t
    from rna_harness.cognitive.certify import certify_design
    t0 = time.time()
    try:
        c = certify_design(target, budget="cheap", seed=seed)
    except Exception as e:                                                     # noqa: BLE001
        return {"puzzle": pid, "arm": arm, "error": repr(e)[:200]}
    return {"puzzle": pid, "arm": arm, "verdict": c.verdict, "sequence": c.sequence,
            "n": c.n, "strictness": c.strictness,
            "n_screened": c.detail.get("n_candidates_screened"),
            "wall_s": round(time.time() - t0, 1), "reason": c.reason[:140]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "scrape_ab.json"))
    a = ap.parse_args()

    cmd = os.environ.get("DESIRNA_CMD")
    assert cmd and os.path.exists(cmd.split()[-1]), "DESIRNA_CMD unset"
    print(f"求解器:{cmd}")

    T = targets()
    jobs = [(p, v["target"], arm, a.seed, a.steps)
            for p, v in sorted(T.items()) for arm in ("new", "old")]
    print(f"{len(T)} 个靶标 x 2 臂 = {len(jobs)} 次;步数预算 {a.steps},两臂求解轨迹逐字节相同,"
          f"唯一差别是抓取范围\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    ckpt = a.out + ".partial"
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            R.append(f.result())
            json.dump({"steps": a.steps, "seed": a.seed, "targets": T, "rows": R, "partial": True},
                      open(ckpt, "w"), ensure_ascii=False)
            if n % 10 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    json.dump({"steps": a.steps, "seed": a.seed, "targets": T, "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    got = {arm: {r["puzzle"] for r in R if r.get("arm") == arm and r.get("sequence")}
           for arm in ("new", "old")}
    print(f"\n=== A1 覆盖 ===")
    print(f"  新抓取 {len(got['new'])}/{len(T)}   旧抓取 {len(got['old'])}/{len(T)}")
    print(f"  仅新臂交付 {sorted(got['new'] - got['old'])}")
    print(f"  仅旧臂交付 {sorted(got['old'] - got['new'])}")
    seq = {(r["arm"], r["puzzle"]): r.get("sequence") for r in R if "error" not in r}
    diff = [p for p in T if seq.get(("new", p)) != seq.get(("old", p))]
    print(f"\n=== A3 交付序列发生变化的靶标:{len(diff)}/{len(T)} ===")
    print(f"  {sorted(diff)}")
    print(f"\n  wrote {a.out}   (下一步:用 rnet venv 跑 scrape_ab_score.py 做 A2)")


if __name__ == "__main__":
    main()
