#!/usr/bin/env python3
"""Stage A: are the 14 undelivered targets undesignable, or just refused by a threshold we mis-set?

WHERE THIS COMES FROM. The clean baseline delivers 43 of 57 OpenKnot targets. The other 14 return
`not_found` because no candidate satisfied the shipped criterion: at least one oracle reproducing ALL of
the target's crossing pairs. `panel_operating_point_holdout.py` then showed, on 4,560 synthesised molecules
and under held-out validation, that this criterion is the wrong half of the panel's signal. Accepting on
"the oracles saw at least something of the pseudoknot" (mean crossing recall >= 0.1) accepts 83.0% against
57.8% and separates the tube BETTER out of sample, +3.37 points against +1.93. Strong three-oracle agreement,
past 0.7, stops separating at all.

So the 14 may be targets our threshold refuses rather than targets we cannot design for. This asks whether
they are, and -- more importantly -- whether the looser screen still works when applied to OUR OWN designs
rather than to the field's.

    THE TRANSFER QUESTION IS THE POINT, not the coverage number. Everything behind the loose threshold was
    measured on molecules other people synthesised. Our designs are de novo, out of that distribution, and
    have no wet-lab counterpart and never will. If the screen does not separate quality inside our own
    pools, then a coverage gain bought with it is unjustified no matter how good the field-level evidence
    looks, and this run says so.

PRE-REGISTERED, before any candidate was adjudicated. Pools are re-collected with the fixed scrape and the
pinned hash seed, on the same shipped ladder that produced the 14, so nothing is compared across different
machinery. The panel SCREENS and RibonanzaNet JUDGES -- the isolation rule is intact, RNet never enters the
design loop, and it is the only judge here that the screen cannot have been fitted to.

    N1  COVERAGE. How many of the 14 hold at least one candidate with panel mean recall >= 0.1. Descriptive.

    N2  TRANSFER, and this one can fail. Within our own pools, do candidates passing the screen score higher
        on RNet crossed-pair F1 than those failing it? Paired within target, Wilcoxon, one-sided greater.
        PASS: p < 0.025 and a median difference >= 0.05.
        KILL: no separation inside our own pools -> the field-calibrated screen does not transfer to our
        designs, the coverage gain is withdrawn, and the 14 stay undelivered until generation improves.

    N3  WHAT WOULD ACTUALLY BE HANDED OVER. Among candidates passing the screen, the one the panel ranks
        highest, scored by RNet. Against the current answer for these targets, which is nothing at all.
        Reported per target, because a rescue that hands back 0.2 F1 is not the same product event as one
        that hands back 0.9, and an average would hide that.

COST: $0 API. Local solver, local oracle binaries.
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


def collect(item):
    """Re-run the shipped ladder for one target and keep every unique candidate."""
    pid, target, tiers, seed = item
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    seen, pool = set(), []
    try:
        for tl in tiers:
            res = DesiRNAGenerator(replicas=10, timelimit_s=tl, results_number=20).design(
                DesignTask(task_id="rescue", target=target, oracle=oc), seed=seed)
            for c in ([res.best_sequence] if res.best_sequence else []) + \
                     list(getattr(res, "candidates", None) or []):
                if c and len(c) == len(target) and c not in seen:
                    seen.add(c)
                    pool.append(c)
    except Exception as e:                                                     # noqa: BLE001
        return {"puzzle": pid, "error": repr(e)[:160], "pool": pool}
    return {"puzzle": pid, "pool": pool}


def adjudicate(item):
    """Panel-screen one candidate. Returns the continuous recall the new operating point reads."""
    pid, seq, target = item
    from rna_harness.cognitive.certify import _knot_recall
    try:
        kr = _knot_recall(seq, target)
    except Exception as e:                                                     # noqa: BLE001
        return {"puzzle": pid, "seq": seq, "error": repr(e)[:120]}
    vals = [v for v in kr["per_oracle"].values() if v is not None]
    if not vals:
        return {"puzzle": pid, "seq": seq, "error": "all oracles failed"}
    return {"puzzle": pid, "seq": seq, "panel_mean": sum(vals) / len(vals),
            "panel_k": sum(1 for v in vals if v >= 1.0),
            "confirmed": bool(kr["knot_confirmed"]), "n_cross": kr["n_cross"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=os.path.join(HERE, "leaderboard_clean.json"))
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "rescue_notfound.json"))
    a = ap.parse_args()

    cmd = os.environ.get("DESIRNA_CMD")
    assert cmd and os.path.exists(cmd.split()[-1]), "DESIRNA_CMD unset"
    print(f"求解器:{cmd}")

    S = json.load(open(a.baseline))
    T = S["targets"]
    nf = sorted(r["puzzle"] for r in S["rows"] if not r.get("sequence"))
    print(f"干净基线未交付的 {len(nf)} 个靶标:{nf}")
    tiers = (5, 15, 60)
    print(f"用同一条出厂阶梯 {tiers} 重收候选池(修好的抓取 + 钉住的哈希种子)\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(p, T[p]["target"], tiers, a.seed) for p in nf]
    pools, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as ex:
        for n, f in enumerate(as_completed([ex.submit(collect, j) for j in jobs]), 1):
            r = f.result()
            pools[r["puzzle"]] = r.get("pool", [])
            print(f"    池 {n}/{len(jobs)}  {r['puzzle']} {len(pools[r['puzzle']])} 条  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    jobs2 = [(p, s, T[p]["target"]) for p in nf for s in pools.get(p, [])]
    print(f"\n面板逐条裁决 {len(jobs2)} 条候选(全池,不抽样),{a.workers} 进程\n", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(adjudicate, jobs2, chunksize=4), 1):
            rows.append(r)
            if n % 200 == 0 or n == len(jobs2):
                print(f"    {n}/{len(jobs2)}  {(time.time()-t0)/60:.1f} min", flush=True)

    ok = [r for r in rows if "error" not in r]
    print(f"\n  裁决成功 {len(ok)}/{len(rows)}")
    thr = 0.1
    hit = {p: [r for r in ok if r["puzzle"] == p and r["panel_mean"] >= thr] for p in nf}
    print(f"\n=== N1 用新筛(平均交叉召回 >= {thr})能救回几个 ===")
    for p in nf:
        n_all = sum(1 for r in ok if r["puzzle"] == p)
        print(f"  {p:5s} {T[p]['n']:3d} nt  池 {n_all:4d} 条,过筛 {len(hit[p]):4d} 条"
              f"  最高平均召回 {max([r['panel_mean'] for r in ok if r['puzzle']==p], default=0):.3f}")
    print(f"  -> {sum(1 for p in nf if hit[p])}/{len(nf)} 个靶标至少有一条候选过筛")

    json.dump({"threshold": thr, "targets": {p: T[p] for p in nf}, "pools": pools, "rows": ok},
              open(a.out, "w"), ensure_ascii=False)
    print(f"\n  wrote {a.out}   (下一步:用 rnet venv 跑 rescue_notfound_score.py 做 N2/N3)")


if __name__ == "__main__":
    main()
