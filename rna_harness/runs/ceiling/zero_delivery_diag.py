#!/usr/bin/env python3
"""The five targets we hand back at crossed-pair F1 = 0.000 while published methods sit at a MEDIAN of 1.000.

WHY THESE FIVE. `frontier_check.json` asked, for the first time, where the shipped product actually stands
against the published field per target. Answer: we beat every method's per-target median on 1 of 57, and the
median gap runs -0.155 against us. Inside that, five targets are not a gap but a hole -- we deliver 0.000,
meaning the design contains NONE of the target's crossing pairs, on targets where Eterna players, MPNN-fixbb
and MPNN-RFdiff reach 1.000 routinely. FOUR OF THE FIVE carry `knot_confirmed`, the strongest label the
product has, the one whose text tells the user three independent predictors agree.

THE ONE QUESTION, which separates two problems with nothing in common:
    SELECTION FAILURE -- a good candidate is in the pool and we picked past it. Then the panel is emitting
        confident false positives, the fix is in adjudication, and it is cheap.
    GENERATION FAILURE -- the pool holds nothing above 0.5. Then the solver never made the pseudoknot on
        these targets at all, the fix is in generation, and it is the same hole the `mcc` objective addresses.

WHICH WAY THE EVIDENCE ALREADY LEANS, stated before running so it can be wrong. On these five, other
methods' designs earn a MEDIAN of 1.000 from RibonanzaNet, so RNet plainly resolves the pseudoknot here and
"RNet is blind on these targets" is not available as an excuse. That leaves our panel or our solver.

Same ladder, same seed, same pinned hash seed as the shipped run, so nothing is compared across different
machinery. The panel screens and RNet judges; RNet never enters the design loop.

STAGE A of two. Runs in the `rna` venv (ViennaRNA + oracle binaries). Stage B scores the pools with
RibonanzaNet in its own venv, which has no `RNA` module -- the same split the other two-stage runs use.

COST: $0 API. Local solver on 5 targets, local oracle binaries.
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

from rescue_notfound import collect, adjudicate                                # noqa: E402

BAD = ["P02", "P03", "P09", "P10", "P14"]
GOOD = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "leaderboard_screened.json"))
    ap.add_argument("--score", default=os.path.join(HERE, "leaderboard_screened_score.json"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "zero_delivery_diag.json"))
    a = ap.parse_args()

    cmd = os.environ.get("DESIRNA_CMD")
    assert cmd and os.path.exists(cmd.split()[-1]), "DESIRNA_CMD unset"

    S = json.load(open(a.src))
    T = S["targets"]
    D = json.load(open(a.score))["ours"]
    tiers = (5, 15, 60)
    print(f"求解器:{cmd}")
    print(f"重收 {len(BAD)} 个零交付靶标的候选池,阶梯 {tiers},seed {a.seed}\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        pools = list(ex.map(collect, [(p, T[p]["target"], tiers, a.seed) for p in BAD]))
    print(f"  收池完成 {(time.time()-t0)/60:.1f} min", flush=True)

    jobs = [(r["puzzle"], s, T[r["puzzle"]]["target"]) for r in pools for s in r["pool"]]
    with ProcessPoolExecutor(max_workers=min(56, max(1, len(jobs)))) as ex:
        panel = [x for x in ex.map(adjudicate, jobs) if "error" not in x]
    print(f"  面板裁定 {len(panel)}/{len(jobs)} 条", flush=True)
    pan = {}
    for x in panel:
        pan.setdefault(x["puzzle"], []).append(
            {"seq": x["seq"], "panel_mean": x["panel_mean"], "panel_k": x["panel_k"],
             "confirmed": x["confirmed"]})

    json.dump({"targets": BAD, "tiers": list(tiers), "seed": a.seed, "good_threshold": GOOD,
               "verdicts": {p: D[p]["verdict"] for p in BAD},
               "delivered": {p: D[p]["f1c"] for p in BAD},
               "target_struct": {p: T[p]["target"] for p in BAD},
               "pools": {r["puzzle"]: r["pool"] for r in pools},
               "panel": pan}, open(a.out, "w"), ensure_ascii=False)
    for r in pools:
        print(f"  {r['puzzle']:6s} 候选 {len(r['pool']):4d} 条")
    print(f"\n  wrote {a.out}   (下一步:用 rnet venv 跑 zero_delivery_diag_score.py)")


if __name__ == "__main__":
    main()
