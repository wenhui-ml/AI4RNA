#!/usr/bin/env python3
"""How big is the generation gap, and is it a gap in the whole distribution or only in the tail we ship?

RUNS IN THE RibonanzaNet venv (it only reads `gap_score_dump.json`, but that file is produced there).

WHERE THIS SITS. The audit closed the selection story: the pool-rerank headline was a tautology and a better
chooser is worth at most +0.55 wet-lab points. What replaced it is a generation gap, sized at the method
level -- at our own novelty (28.5% identity to the wild type) the field's learned designers score 0.93-0.98
crossed-pair F1 and we ship 0.816. That comparison used one number per method, so it cannot say WHERE the
gap lives. Now that every one of our 15,226 candidates has a score, it can.

    P1  IS THE WHOLE DISTRIBUTION SHIFTED, or do we merely fail to ship our best? Our pool's median against
        each peer method's per-target median, paired within target, Wilcoxon over targets.

    P2  HOW OFTEN IS A RAW CANDIDATE OF OURS ALREADY COMPETITIVE? Per target, the fraction of our pool that
        exceeds the peer group's median. At 50% our generator matches theirs; far below 50% it does not.

    P3  DOES MORE SEARCH TIME HELP? Reported here for completeness and with an explicit caveat that it CANNOT
        be trusted: `pool_rerank.py` deduplicated across tiers in ladder order, so anything the 5 s tier
        already found is filed under 5 s and the later tiers keep only what was new to them. A converging
        solver therefore looks like a degrading one. `search_convergence.py` runs the tiers independently
        and is the version that answers this; the number here is a placeholder for it, not a result.

THE COMPARISON IS TILTED AGAINST US AND THAT IS DELIBERATE. Their per-target median is over designs a human
or a pipeline chose to synthesise; ours is over raw solver output with nothing filtered. If we lose to a
selected population with an unselected one, the gap is real but its size is overstated; if we were to WIN
that way it would be decisive. Stating which direction the tilt runs is the point.

PEER GROUP. The five learned de-novo designers that sit within a few points of our own novelty --
Struct2SeQ, Struct2SeQ-SHAPE, MPNN-RFdiff, gRNAde-no3d, codesign-RFdiff. Struct2SeQ-SHAPE optimises against
RibonanzaNet and so is reported but excluded from the aggregate, as everywhere else in this directory.

COST: $0. Local CPU.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
HOMEFIELD = "Struct2SeQ-SHAPE"
PEERS = ["Struct2SeQ", "MPNN-RFdiff", "gRNAde-no3d", "codesign-RFdiff"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "gap_score_dump.json"))
    ap.add_argument("--pools", default=os.path.join(HERE, "pool_rerank_pools.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "generation_gap.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    T = json.load(open(a.pools))["targets"]
    D = json.load(open(a.dump))["rows"]
    ours = collections.defaultdict(list)
    by_tier = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in D:
        ours[r["puzzle"]].append(r["f1c"])
        by_tier[r["puzzle"]][r["tier"]].append(r["f1c"])
    print(f"我们的候选 {len(D)} 条,覆盖 {len(ours)} 个靶标")

    csv.field_size_limit(10 ** 9)
    theirs = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {k: i for i, k in enumerate(next(r))}
        for x in r:
            if x[idx["SN_filter"]] != "1":
                continue
            p = x[idx["puzzle"]]
            if p not in T or x[idx["target_structure"]] != T[p]["target"]:
                continue
            try:
                theirs[p][x[idx["method"]]].append(float(x[idx["RNet_F1_crossed_pair"]]))
            except (TypeError, ValueError):
                pass

    # ---- P1 ------------------------------------------------------------------------------------------
    print(f"\n=== P1 我们【整池的中位】对各方法【每靶标中位】(都是'该方法的典型产物')===")
    print(f"  {'方法':20s} {'靶标':>4s} {'他们中位':>9s} {'我们池中位':>10s} {'差':>8s} {'我们更好':>9s} {'p':>9s}")
    table = {}
    for m in PEERS + [HOMEFIELD]:
        ts = [p for p in ours if theirs.get(p, {}).get(m)]
        if len(ts) < 10:
            continue
        d = [st.median(ours[p]) - st.median(theirs[p][m]) for p in ts]
        w, pv, _ = wilcoxon(d, "two-sided")
        tag = "  <- 用自己的目标函数打分" if m == HOMEFIELD else ""
        print(f"  {m:20s} {len(ts):4d} {st.mean([st.median(theirs[p][m]) for p in ts]):9.3f} "
              f"{st.mean([st.median(ours[p]) for p in ts]):10.3f} {st.median(d):+8.3f} "
              f"{sum(1 for x in d if x>0):6d}/{len(ts):<3d} "
              f"{'n/a' if pv is None else f'{pv:.2g}':>9s}{tag}")
        table[m] = {"n_targets": len(ts), "median_diff": st.median(d), "p": pv,
                    "wins": sum(1 for x in d if x > 0)}

    ts = [p for p in ours if any(theirs.get(p, {}).get(m) for m in PEERS)]
    d = []
    for p in ts:
        peer = [v for m in PEERS for v in theirs[p].get(m, [])]
        d.append(st.median(ours[p]) - st.median(peer))
    w, pv, _ = wilcoxon(d, "two-sided")
    print(f"\n  合并同新颖度同行(不含 {HOMEFIELD}):{len(ts)} 个靶标,差中位 {st.median(d):+.3f},"
          f"我们更好 {sum(1 for x in d if x>0)}/{len(ts)},p = {'n/a' if pv is None else f'{pv:.3g}'}")

    # ---- P2 ------------------------------------------------------------------------------------------
    fr = []
    for p in ts:
        peer = [v for m in PEERS for v in theirs[p].get(m, [])]
        med = st.median(peer)
        fr.append(sum(1 for v in ours[p] if v > med) / len(ours[p]))
    print(f"\n=== P2 我们池里有多大比例的候选,已经强于同行的典型设计 ===")
    print(f"  {len(fr)} 个靶标:比例中位 {st.median(fr):.1%}(与同行持平应为 50%),"
          f"比例 >=50% 的靶标 {sum(1 for x in fr if x>=0.5)}/{len(fr)}")
    print(f"  比例为 0 的靶标 {sum(1 for x in fr if x==0)} 个 —— 整池没有一条能达到同行中位")

    # ---- P3 ------------------------------------------------------------------------------------------
    print(f"\n=== P3 时限效应(不可信,见文件头:跨档去重使收敛的求解器看起来在退化)===")
    for tl in (5, 15, 60):
        v = [st.median(by_tier[p][tl]) for p in ours if by_tier[p].get(tl)]
        print(f"  {tl:3d}s 档:{len(v):2d} 个靶标有候选,靶标内中位的中位 {st.median(v):.3f}")
    both = [p for p in ours if by_tier[p].get(5) and by_tier[p].get(60)]
    dd = [st.median(by_tier[p][60]) - st.median(by_tier[p][5]) for p in both]
    w, pv, _ = wilcoxon(dd, "two-sided")
    print(f"  60s − 5s 配对差中位 {st.median(dd):+.3f}({len(both)} 个靶标),"
          f"p = {'n/a' if pv is None else f'{pv:.3g}'}  <- 占位,由 search_convergence.py 取代")

    json.dump({"P1_per_method": table, "P1_peer_pooled_median_diff": st.median(d),
               "P1_peer_pooled_p": pv, "P1_n_targets": len(ts),
               "P2_frac_beating_peer_median": st.median(fr),
               "P2_targets_with_zero": sum(1 for x in fr if x == 0),
               "P2_per_target": {p: f for p, f in zip(ts, fr)},
               "peers": PEERS},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
