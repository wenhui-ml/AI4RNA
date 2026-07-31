#!/usr/bin/env python3
"""Is 51/55 a property of the method, or of one lucky pool? Re-run the solver under independent seeds.

WHAT 51/55 ACTUALLY IS, stated precisely because the phrasing has been loose. kissing_deliver.py does not
call the solver: it re-scores an ARCHIVED candidate pool (agent/pk_desirna.json, 10 replicas / 60 s / 25
candidates kept per target). The oracles are deterministic, so given that archive the number 51/55 has no
run-to-run variance at all -- its only uncertainty is over TARGETS, which the Clopper-Pearson interval
[82.4%, 98.0%] now covers.

That is exactly why it is not enough. The archive is ONE solver run, and this project has measured that
DesiRNA moves by +/-1 between runs on this kind of target and has already had to apply that caveat to a
result that favoured it. So the question a referee actually asks is not "is your scoring stable" -- it is
deterministic -- but "would a DIFFERENT pool give a different answer?" Nothing so far has measured that.

THIS DOES. Independent solver seeds, identical everything else, the same shipped rule, delivery re-scored per
seed. What it reports is the SPREAD, because the spread is the thing in question:

  * spread within a point or two    the number is a property of the method and the archive was not lucky
  * spread of five or more          51/55 was a draw and every downstream sentence needs a range

The absolute level may sit slightly off the archive's, since matching its exact 60 s budget on identical
hardware is not something this run can guarantee; the budget is declared and held constant ACROSS seeds,
which is what the comparison needs. Per-seed delivery, per-target agreement, and the count of targets that
flip between seeds are all reported -- a stable mean hiding heavy per-target churn would be a different
finding from a stable mean with stable targets, and averaging would hide it.
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


def work(job):
    tid, target, timelimit, seed, keep = job
    from rna_harness.cognitive.certify import _knot_recall
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=timelimit, results_number=50).design(
            DesignTask(task_id="dseed", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                    # noqa: BLE001
        return {"task_id": tid, "seed": seed, "error": repr(e)[:160]}
    pool, seen = [], set()
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and s not in seen:
            seen.add(s)
            pool.append(s)
    # SAME BUDGET OF ORACLE CALLS PER TARGET AS THE ARCHIVE (25), so a difference cannot be "we looked at
    # more candidates this time" -- the archive kept 25 and so does this.
    pool = pool[:keep]
    best, n_pass = 0.0, 0
    hit = None
    for s in pool:
        kr = _knot_recall(s, target)
        ran = [v for v in kr["per_oracle"].values() if v is not None]
        if kr.get("knot_confirmed"):
            n_pass += 1
            if hit is None:
                hit = s
        if ran:
            best = max(best, max(ran))
    return {"task_id": tid, "seed": seed, "delivered": hit is not None, "sequence": hit,
            "n_pool": len(pool), "n_pass": n_pass, "best_recall": round(best, 3),
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--timelimit", type=int, default=60, help="the archive used 60 s")
    ap.add_argument("--keep", type=int, default=25, help="the archive kept 25 candidates per target")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--group", default="kissing / chain")
    ap.add_argument("--out", default=os.path.join(HERE, "deliver_seeds.json"))
    a = ap.parse_args()

    top = {r["task_id"]: r for r in json.load(open(os.path.join(HERE, "pk_topology.json")))["rows"]}
    pk = {r["task_id"]: r for r in
          json.load(open(os.path.join(HERE, "agent", "pk_desirna.json")))["rows"]}
    ids = [t for t in top if top[t]["topology"] == a.group and t in pk]
    jobs = [(t, pk[t]["target"], a.timelimit, s, a.keep) for s in range(a.seeds) for t in ids]
    print(f"多种子交付复现:{len(ids)} 条 {a.group} 靶标 × {a.seeds} 个独立种子,"
          f"每条 {a.timelimit}s / 保留 {a.keep} 个候选(与归档一致)\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 25 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    from scipy.stats import beta
    import statistics as st
    per_seed = {}
    for r in R:
        if r.get("error"):
            continue
        per_seed.setdefault(r["seed"], []).append(r)
    print(f"\n=== 逐种子交付率(同一批 {len(ids)} 条靶标) ===")
    rates = []
    for s in sorted(per_seed):
        sub = per_seed[s]
        k = sum(1 for r in sub if r["delivered"])
        lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, len(sub) - k + 1))
        hi = 1.0 if k == len(sub) else float(beta.ppf(0.975, k + 1, len(sub) - k))
        rates.append(k / len(sub))
        print(f"  seed={s}   {k:>3}/{len(sub):<4} = {k/len(sub):6.1%}  [{lo:5.1%}, {hi:5.1%}]")
    print(f"\n  跨种子:均值 {st.mean(rates):.1%}  极差 {max(rates)-min(rates):.1%}"
          + (f"  标准差 {st.stdev(rates):.1%}" if len(rates) > 1 else ""))
    print(f"  归档单次运行给出的是 51/55 = 92.7%")

    # PER-TARGET CHURN, kept separate: a stable mean can hide targets flipping in and out, and averaging
    # would report those two very different situations as the same number.
    by_t = {}
    for r in R:
        if not r.get("error"):
            by_t.setdefault(r["task_id"], []).append(bool(r["delivered"]))
    always = sum(1 for v in by_t.values() if all(v))
    never = sum(1 for v in by_t.values() if not any(v))
    flip = len(by_t) - always - never
    print(f"\n  逐靶标:每个种子都交付 {always},每个种子都失败 {never},**在种子间翻转 {flip}**")
    json.dump({"group": a.group, "seeds": a.seeds, "timelimit_s": a.timelimit, "keep": a.keep,
               "per_seed_rate": rates, "spread": (max(rates) - min(rates)) if rates else None,
               "always": always, "never": never, "flipped": flip,
               "archive_single_run": "51/55", "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
