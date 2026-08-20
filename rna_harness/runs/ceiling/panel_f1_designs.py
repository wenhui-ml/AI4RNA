#!/usr/bin/env python3
"""Re-adjudicate the 4,560 measured OpenKnot designs and record the panel's PRECISION, not just its recall.

WHY THIS RUN EXISTS -- A DEFECT IN A COMPARISON I ALREADY REPORTED. `criterion_vs_tube.py` and everything
built on it compare two rulers that are not the same statistic:

    ours   `_knot_recall` -> per-oracle len(predicted & target_crossing) / len(target_crossing)
           = RECALL of the target's crossing pairs. A design that forms the target's crossing stem AND two
           spurious crossing stems scores 1.000.
    theirs `RNet_F1_crossed_pair` from the release = F1 on crossed pairs, so it is penalised for exactly
           those spurious stems.

Recall is strictly less informative than F1, so any head-to-head on those two columns is loaded against the
panel by construction, and the size of the loading has never been measured. Since competing conformations are
the mechanism this project keeps running into, the discarded precision term is plausibly where the signal is.
The 0-of-5 loss at matched coverage, and the novelty-stratified null of 2026-08-18, both rest on the recall
column, so neither can be read as a statement about the physical panel until this is measured.

WHAT IS COMPUTED. The same deterministic sample (seed 0, 80 per target, replayed from
`novelty_stratified_ruler.replay_sample`), the same three shipped oracles, one fold per sequence:

    recall_product  |predicted & target_crossing| / |target_crossing|            (the SHIPPED column, reproduced)
    recall     |predicted_crossing & target_crossing| / |target_crossing|      (F1-consistent version)
    precision  |predicted_crossing & target_crossing| / |predicted_crossing|   (new)
    f1         harmonic mean of the two, 0 when either side is empty
    n_pred_cross  how many crossing pairs the oracle predicted at all           (the spurious-stem count)

A predicted pair counts as CROSSING when it crosses another pair in the SAME prediction -- the property that
makes a structure a pseudoknot. This is symmetric with how the target's crossing set is defined
(`pk_pairs(T) - pk_pairs(_round_only(T))`), so recall here must reproduce the stored `panel_mean` exactly;
that identity is checked at the end and a mismatch is a failure, not a curiosity.

COST: $0, local folding only. The original adjudication took 23 minutes at 48 workers; this box is shared and
already loaded, so the default is 32. Panel/oracle jobs are ~1 core per worker.
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


def crossing_subset(pairs):
    """The pairs in `pairs` that cross at least one other pair in `pairs`."""
    ps = sorted(pairs)
    out = set()
    for a, (i, j) in enumerate(ps):
        for k, l in ps[a + 1:]:
            if i < k < j < l or k < i < l < j:
                out.add((i, j))
                out.add((k, l))
    return out


def work(item):
    puz, seq, target, f1c, wet, method = item
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from rna_harness.oracles_pk import oracle_pairs
    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    if not cross:
        return {"puzzle": puz, "error": "target has no crossing pairs"}
    try:
        preds = oracle_pairs(seq)
    except Exception as e:                                                     # noqa: BLE001
        return {"puzzle": puz, "error": repr(e)[:120]}
    per = {}
    for name, ps in preds.items():
        if ps is None:
            per[name] = None
            continue
        # TWO recalls, deliberately. `recall_product` is the shipped certificate's own quantity: a target
        # crossing pair counts as recovered if the oracle predicts that PAIR, whether or not the pair still
        # crosses anything in the oracle's own structure. `recall_cross` restricts to the oracle's crossing
        # subset, which is the convention an F1 has to use if precision is to mean anything -- and it is the
        # convention the release's own crossed-pair F1 uses. On the 24-design smoke the two differ on 6, so
        # the distinction is not cosmetic: the shipped recall gives credit for recovering a pair that the
        # oracle did NOT fold into a pseudoknot.
        pc = crossing_subset(ps)
        rec_prod = len(ps & cross) / len(cross)
        hit = len(pc & cross)
        rec = hit / len(cross)
        pre = (hit / len(pc)) if pc else 0.0
        f1 = (2 * rec * pre / (rec + pre)) if (rec + pre) > 0 else 0.0
        per[name] = {"recall_product": rec_prod, "recall": rec, "precision": pre, "f1": f1,
                     "n_pred_cross": len(pc), "n_pred_total": len(ps)}
    vals = [v for v in per.values() if v is not None]
    if not vals:
        return {"puzzle": puz, "error": "all oracles failed to run"}
    mean = lambda k: sum(v[k] for v in vals) / len(vals)                       # noqa: E731
    return {"puzzle": puz, "seq": seq, "method": method, "wet": wet, "f1c": f1c,
            "panel_recall_product": mean("recall_product"),
            "panel_recall": mean("recall"), "panel_precision": mean("precision"),
            "panel_f1": mean("f1"), "panel_f1_max": max(v["f1"] for v in vals),
            "n_pred_cross": mean("n_pred_cross"), "n_oracles": len(vals),
            "per_oracle": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-target", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0, help="smoke mode: only the first N designs")
    ap.add_argument("--third", type=int, default=0,
                    help="third-sample mode: this many per target from designs NEITHER the (--seed, 80) nor "
                         "the 160/target confirmatory sample touched. Disjointness from BOTH is asserted.")
    ap.add_argument("--fifth", type=int, default=0,
                    help="fifth-sample mode: this many per (target, METHOD) CELL from designs none of the "
                         "four prior samples touched. Exists to raise the CELL COUNT, not the cell size: "
                         "after four samples the library cannot supply more than 33 cells at 40, which is "
                         "fewer cells than the fourth sample's 110, so power comes from pooling at 24.")
    ap.add_argument("--fourth", type=int, default=0,
                    help="fourth-sample mode: this many per (target, METHOD) CELL from designs none of the "
                         "three prior samples touched. Per-cell, not per-target, because A13's estimator is "
                         "a within-cell contrast and a per-target draw left the median cell at 8 molecules.")
    ap.add_argument("--fresh", type=int, default=0,
                    help="confirmatory mode: sample this many per target from designs the (--seed, 80) "
                         "sample never touched. Disjointness is asserted, not assumed.")
    ap.add_argument("--out", default=os.path.join(HERE, "panel_f1_designs.json"))
    a = ap.parse_args()

    from leaderboard_design import targets
    from novelty_stratified_ruler import replay_fifth, replay_fresh, replay_sample, replay_third, replay_fourth
    T = targets()
    if a.fifth:
        jobs_raw, wild = replay_fifth(a.seed, a.fifth)
    elif a.fourth:
        jobs_raw, wild = replay_fourth(a.seed, a.fourth)
    elif a.third:
        jobs_raw, wild = replay_third(a.seed, a.third)
    elif a.fresh:
        jobs_raw, wild = replay_fresh(a.seed, a.fresh, old_seed=a.seed, old_per_target=a.per_target)
    else:
        jobs_raw, wild = replay_sample(a.seed, a.per_target)
    jobs = [(p, s, T[p]["target"], f1c, wet, m) for p, s, f1c, wet, m in jobs_raw]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len({j[0] for j in jobs})} 个靶标,{len(jobs)} 条设计,{a.workers} 进程", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, res in enumerate(ex.map(work, jobs, chunksize=4), 1):
            R.append(res)
            if n % 400 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    bad = [r for r in R if "error" in r]
    ok = [r for r in R if "error" not in r]
    print(f"\n  成功 {len(ok)},失败 {len(bad)}"
          + (f" ({collections.Counter(r['error'] for r in bad).most_common(1)[0][0]})" if bad else ""))

    # ---- identity check against the stored recall column -----------------------------------------------
    src = json.load(open(os.path.join(HERE, "criterion_vs_tube.json")))
    n_cmp, n_bad = 0, 0
    if (not a.limit and not a.fresh and not a.third and not a.fourth and not a.fifth
            and len(R) == len(src["designs"])):
        for r, d in zip(R, src["designs"]):
            if "error" in r:
                continue
            n_cmp += 1
            if abs(r["panel_recall_product"] - d["panel_mean"]) > 1e-9:
                n_bad += 1
        print(f"  与已存 panel_mean 的一致性:比对 {n_cmp} 条,不一致 {n_bad} 条"
              + ("  ✓" if n_bad == 0 else "  ✗ 召回的定义没对上,后续分析不可用"))

    json.dump({"n": len(ok), "n_failed": len(bad), "seed": a.seed, "per_target": a.per_target,
               "fresh_per_target": a.fresh, "third_per_target": a.third, "fourth_per_cell": a.fourth,
               "fifth_per_cell": a.fifth,
               "recall_identity_checked": n_cmp, "recall_identity_mismatches": n_bad,
               "designs": ok}, open(a.out, "w"), indent=1)
    print(f"写出 {a.out}")
    return 0 if n_bad == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
