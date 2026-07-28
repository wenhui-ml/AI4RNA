#!/usr/bin/env python3
"""Is "unsolvable" DETECTABLE by a cheap multi-seed probe, when it is not predictable from features?

Today's abstention experiment was a null: on the 28 Eterna targets left unsolved by the t=5 tier, every
learner (kNN / logistic, with and without the cheap tier's bp) caught 0 of the 16 hopeless targets at zero
solves lost, and the best trivial rule caught 3/16 at p = 0.171. All of those re-analyse signals the system
already produces -- a class that is now 0/7 on this branch, against 3/3 for adding an information source.

This adds an information source, at a known price: run the SAME cheap tier again with different seeds.
"Three independent seeds all failed in 5 s" is evidence of a different kind from "one seed failed and the
structure looks long".

TWO things get measured, and they are separate results:
  (a) CAPABILITY. Do extra t=5 seeds solve any of the 28? Every one is a solve bought at $0.00211.
  (b) DETECTION. Does min-bp-over-3-seeds separate the 12 eventually-solvable from the 16 hopeless better
      than single-seed bp did (2/16 at zero loss, p = 0.317)?

MATCHED-COST CONTROL, which is the part that decides whether this is worth shipping. Two extra t=5 seeds cost
about 2 x $0.00211 = $0.00422 per target. For nearly the same money the cascade could simply run the t=15
rung ($0.00341), which is ALREADY MEASURED to solve 5 more targets. So the probe has to beat "just escalate",
not merely beat "do nothing". Stated before looking.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
OUT = os.path.join(HERE, "probe")
TIER = 5


def work(item):
    pid, target, seed = item
    dest = os.path.join(OUT, f"p{pid}_s{seed}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.cost import solver_cost_usd
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    try:
        res = DesiRNAGenerator(replicas=10, timelimit_s=TIER, results_number=20).design(
            DesignTask(task_id=str(pid), target=target,
                       oracle=OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)),
            seed=seed)
        w = time.time() - t0
        rec = {"id": pid, "n": len(target), "t": TIER, "seed": seed, "solved": bool(res.solved),
               "bp": int(res.best_bp_distance), "wall_s": round(w, 1),
               "cost_usd": solver_cost_usd(w, 10), "seq": res.best_sequence or None}
    except Exception as e:                                            # noqa: BLE001
        rec = {"id": pid, "seed": seed, "error": repr(e)[:200]}
    with open(dest, "w") as f:
        json.dump(rec, f)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    base = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(HERE, "sota_curve", "*.json")):
        d = json.load(open(f))
        base[d["t"]][d["id"]] = d
    B = sorted(base)
    ids = sorted(base[TIER])
    U = [i for i in ids if not base[TIER][i].get("solved")]                    # the decision set
    ever = {i: any(base[t][i].get("solved") for t in B if t > TIER) for i in U}
    cost_t = {t: sum(base[t][i].get("cost_usd", 0.0) for i in ids) / len(ids) for t in B}

    from rna_harness.benchmarks.eterna100 import load_eterna100
    tgt = {p["id"]: p["target"] for p in load_eterna100(version="V2")}
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"multi-seed probe at t={TIER}s on the {len(U)} targets the tier-1 probe left unsolved")
    print(f"  {sum(ever.values())} of them eventually solve at t<=180s, {len(U)-sum(ever.values())} never")
    print(f"  extra seeds {seeds}; measured tier costs {dict((k, round(v,5)) for k,v in cost_t.items())}")
    print(f"  matched-cost control: 2 extra t=5 seeds ~ ${2*cost_t[TIER]:.5f} vs the t=15 rung "
          f"${cost_t[15]:.5f}, which already solves 5 more\n", flush=True)

    if not a.analyse_only:
        todo = [(i, tgt[i], s) for s in seeds for i in U
                if not os.path.exists(os.path.join(OUT, f"p{i}_s{s}.json"))]
        print(f"  {len(todo)} cells to run, {a.workers} workers", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for n, _f in enumerate(as_completed(futs), 1):
                if n % 10 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"    {n}/{len(todo)}  {el/60:.1f} min  "
                          f"eta {el/max(n,1)*(len(todo)-n)/60:.1f} min", flush=True)

    got = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(OUT, "*.json")):
        d = json.load(open(f))
        if "error" not in d:
            got[d["id"]][d["seed"]] = d
    have = [i for i in U if len(got.get(i, {})) == len(seeds)]
    print(f"\n  complete probes: {len(have)}/{len(U)}")

    # ---- (a) capability -------------------------------------------------------------------------
    won = [i for i in have if any(got[i][s]["solved"] for s in seeds)]
    probe_cost = sum(got[i][s]["cost_usd"] for i in have for s in seeds)
    print(f"\n=== (a) capability: do extra t={TIER}s seeds solve any of them? ===")
    print(f"  solved by an extra seed: {len(won)}/{len(have)}  {won}")
    print(f"  probe cost ${probe_cost:.4f} for {len(len(won)*[0])} solves"
          f"{'' if not won else f' = ${probe_cost/len(won):.5f}/solve'}")
    print(f"  the t=15 rung on the same {len(have)} targets costs "
          f"${cost_t[15]*len(have):.4f} and solves "
          f"{sum(1 for i in have if base[15][i].get('solved'))}")

    # ---- (b) detection --------------------------------------------------------------------------
    print(f"\n=== (b) detection: does min-bp over {len(seeds)+1} seeds separate solvable from hopeless? ===")
    def zero_loss_catch(score):
        """Abstain on the top-k of `score` (higher = more hopeless). k = catches before the first solvable."""
        order = sorted(have, key=lambda i: -score[i])
        k = 0
        for i in order:
            if ever[i]:
                break
            k += 1
        return k
    H = sum(1 for i in have if not ever[i])
    N = len(have)
    single = {i: base[TIER][i].get("bp") or 0 for i in have}
    multi = {i: min([base[TIER][i].get("bp") or 0] + [got[i][s]["bp"] for s in seeds]) for i in have}
    nfail = {i: sum(1 for s in seeds if not got[i][s]["solved"]) for i in have}
    length = {i: base[TIER][i]["n"] for i in have}
    for lbl, sc in (("single-seed bp (old)", single), ("min bp over all seeds", multi),
                    ("n seeds failed", nfail), ("length (trivial control)", length)):
        k = zero_loss_catch(sc)
        pv = prod((H - j) / (N - j) for j in range(k)) if k else 1.0
        print(f"  {lbl:<26} catches {k:>2}/{H} at zero loss   p = {pv:.3f}  "
              f"{'significant' if pv < 0.05 else 'not distinguishable from random'}")
    print(f"  (H={H} hopeless of n={N}; p = prod_(j<k) (H-j)/(n-j), the same exact null used all day)")

    json.dump({"tier": TIER, "seeds": seeds, "n_decision": len(U), "n_complete": len(have),
               "hopeless": H, "solved_by_extra_seed": won, "probe_cost": probe_cost,
               "catch": {lbl: zero_loss_catch(sc) for lbl, sc in
                         (("single_bp", single), ("multi_bp", multi), ("n_failed", nfail),
                          ("length", length))}},
              open(os.path.join(HERE, "probe.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/probe.json")


if __name__ == "__main__":
    main()
