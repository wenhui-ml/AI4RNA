#!/usr/bin/env python3
"""Stage B: score our designs on the same instrument that scored theirs, and put us on the table.

RUNS IN THE RibonanzaNet venv (/data/wenhui/rnet_env/bin/python), because the harness venv has no torch and
this one has no ViennaRNA. Stage A produced the sequences; nothing here touches the design loop.

WHAT IS BEING COMPARED. For each OpenKnot puzzle, the crossed-pair F1 of a design's RibonanzaNet-predicted
structure against the target -- the same `RNet_F1_crossed_pair` the release publishes for all 34,750 real
designs, which we verified we reproduce (median base-pair F1 1.000 vs their published structures over 300
designs, crossed-pair Spearman +0.988). Theirs are read from the release; ours are computed. That asymmetry
is only defensible BECAUSE of the reproduction check, and it is the reason that check came first.

THREE FAIRNESS PROBLEMS, all declared before any number was computed. Each of them cuts against a different
party and none is allowed to be quietly dropped.

  0. OUR ONE DESIGN IS ALREADY A SELECTED ONE. The product screens ~131 internal candidates and returns
     its best; each of their rows is a single submission that went straight into the tube. So "ours vs
     their per-target median" compares our best-of-131 against their typical submission, which favours US,
     while "ours vs their per-target best" compares our one against their best-of-hundreds, which favours
     THEM. Neither is the answer. Both are reported, and the truth is bracketed between them -- stating one
     alone would be picking the bracket end that flatters whoever is writing.

  1. THEY SUBMITTED HUNDREDS PER TARGET, WE SUBMIT ONE. The product returns a single sequence. Comparing our
     one against their best-of-N measures the size of their submission, not the quality of our method. The
     PRIMARY comparison is therefore against each method's per-target MEDIAN. The best-of-N comparison is
     reported too, labelled as what a public leaderboard actually shows and as unfair to us.

  2. STRUCT2SEQ-SHAPE OPTIMISES AGAINST RIBONANZANET. Scoring it with RibonanzaNet grades it on its own
     objective. It is reported in its own row and excluded from any "all methods" aggregate, and a loss to
     it means strictly less than a loss to Eterna or Rosetta.

  3. THE PROXY IS A PROXY. RNet's crossed-pair F1 correlates with the tube at +0.364 within target -- the
     best number we have, and far from 1.0. Where the real designs' wet-lab OpenKnot Score exists it is
     reported ALONGSIDE, so a reader can see how much of the ordering the proxy preserves. Our designs have
     no wet-lab score and never will without a laboratory; that gap is the honest limit of this whole
     exercise and is stated in the output rather than in a footnote.

PRE-REGISTERED. Paired within target, Wilcoxon signed-rank over the 57 puzzles, one-sided. Four tests,
Bonferroni threshold 0.05/4 = 0.0125. A target where we returned not_found scores 0.0 and stays in the
population -- dropping our own failures is the move that makes leaderboards meaningless.

    L1  ours beats the per-target median over all real designs (excluding Struct2SeQ-SHAPE)
    L2  ours beats Eterna's per-target median            -- the human players
    L3  ours beats Rosetta's per-target median           -- the established automated baseline
    L4  ours beats the per-target median of gRNAde-no3d  -- the strongest non-RNet-trained AI arm by
                                                            wet-lab >80 share (82.9%)

    KILL for the "we belong on this table" claim: L1 fails. That is not a disaster, it is the answer; we
    have never had ANY placement before, and a measured loss is worth more than an unmeasured hope.

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
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
ALPHA = 0.05 / 4
HOMEFIELD = "Struct2SeQ-SHAPE"          # trained against the judge we are using


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=os.path.join(HERE, "leaderboard_ours.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "leaderboard_score.json"))
    a = ap.parse_args()

    from rnet_reproduce import build_model, pairwise_logits, pairs_from_logits, f1
    from m2r_pairwise import wilcoxon

    _OPEN = {"(": ")", "[": "]", "{": "}", "<": ">"}
    _CLOSE = {v: k for k, v in _OPEN.items()}

    def pk_pairs(s):
        stacks, out = {o: [] for o in _OPEN}, set()
        for i, c in enumerate(s):
            if c in _OPEN:
                stacks[c].append(i)
            elif c in _CLOSE and stacks[_CLOSE[c]]:
                out.add((stacks[_CLOSE[c]].pop(), i))
        return out

    def crossing(pairs):
        out = set()
        for (i, j) in pairs:
            for (m, n) in pairs:
                if (m < i < n < j) or (i < m < j < n):
                    out.add((i, j))
                    break
        return out

    ours_raw = json.load(open(a.ours))
    T = ours_raw["targets"]
    best = {}
    for r in ours_raw["rows"]:                    # keep the standard-budget answer; rescue only if primary failed
        p = r["puzzle"]
        if r.get("sequence") and (p not in best or best[p]["budget"] != "standard"):
            best[p] = r
    print(f"我们的交付:{len(best)}/{len(T)} 个靶标拿到序列"
          f"(其中 best_effort 救回的 {sum(1 for v in best.values() if v['budget'] != 'standard')} 个,"
          f"按声明单独标出)\n")

    # Only designs scored against the SAME target string we designed for enter a comparison pool. Six
    # puzzles carry two target variants (four differing in length), so pooling them would compare our
    # design on one structure against theirs on another.
    csv.field_size_limit(10 ** 9)
    theirs = collections.defaultdict(lambda: collections.defaultdict(list))
    wet = collections.defaultdict(lambda: collections.defaultdict(list))
    dropped_variant = 0
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {c: i for i, c in enumerate(next(r))
               if c in ("puzzle", "method", "SN_filter", "RNet_F1_crossed_pair",
                        "target_openknot_score", "target_structure")}
        for x in r:
            if x[idx["SN_filter"]] != "1":
                continue
            p = x[idx["puzzle"]]
            if p in T and x[idx["target_structure"]] != T[p]["target"]:
                dropped_variant += 1
                continue
            try:
                v = float(x[idx["RNet_F1_crossed_pair"]])
                w = float(x[idx["target_openknot_score"]])
            except (TypeError, ValueError):
                continue
            theirs[p][x[idx["method"]]].append(v)
            wet[p][x[idx["method"]]].append(w)
    print(f"  因靶标变体不同而排除的真实设计 {dropped_variant} 条(6 个 puzzle 有两个变体)")

    print("=== 给我们的设计打分(RibonanzaNet,已验证复现)===")
    model = build_model()
    scores, t0 = {}, time.time()
    for n, (p, meta) in enumerate(sorted(T.items()), 1):
        r = best.get(p)
        if not r or not r.get("sequence"):
            scores[p] = {"f1c": 0.0, "delivered": False, "budget": None}
            continue
        lg = pairwise_logits(model, r["sequence"])
        pred = pairs_from_logits(lg)
        tgt = crossing(pk_pairs(meta["target"]))
        scores[p] = {"f1c": f1(crossing(pred), tgt), "delivered": True,
                     "budget": r["budget"], "n": meta["n"], "verdict": r["verdict"]}
        if n % 10 == 0 or n == len(T):
            print(f"    {n}/{len(T)}  {(time.time()-t0)/60:.1f} min", flush=True)

    d = [v["f1c"] for v in scores.values() if v["delivered"]]
    print(f"\n  我们交付的 {len(d)} 条:交叉配对 F1 中位 {st.median(d):.3f}" if d else "\n  我们一条都没交付")
    print(f"  未交付的 {len(scores)-len(d)} 个按 0.0 计入总体")

    methods = sorted({m for p in theirs for m in theirs[p]})
    print(f"\n=== 逐方法对比(每靶标取【中位】,这是公平的比法)===")
    print(f"  {'方法':22s} {'靶标':>4s} {'他们中位':>9s} {'我们':>7s} {'我们更好':>9s} {'p':>10s}")
    table = {}
    for m in methods:
        ts = [p for p in T if theirs.get(p, {}).get(m) and p in scores]
        if len(ts) < 10:
            continue
        diff = [scores[p]["f1c"] - st.median(theirs[p][m]) for p in ts]
        w, pv, nn = wilcoxon(diff, "greater")
        wins = sum(1 for x in diff if x > 0)
        tag = "  <- 用自己的目标函数打分" if m == HOMEFIELD else ""
        print(f"  {m:22s} {len(ts):4d} {st.mean([st.median(theirs[p][m]) for p in ts]):9.3f} "
              f"{st.mean([scores[p]['f1c'] for p in ts]):7.3f} {wins:6d}/{len(ts):<3d} "
              f"{'n/a' if pv is None else f'{pv:.3g}':>10s}{tag}")
        table[m] = {"n_targets": len(ts), "median_diff": st.median(diff), "wins": wins, "p": pv}

    print(f"\n=== 与【全体真实设计】比(已排除 {HOMEFIELD})===")
    allts = []
    for p in T:
        pool = [v for m, vs in theirs.get(p, {}).items() if m != HOMEFIELD for v in vs]
        if pool and p in scores:
            allts.append((p, scores[p]["f1c"] - st.median(pool)))
    w, p1, _ = wilcoxon([x[1] for x in allts], "greater")
    print(f"  L1  {len(allts)} 个靶标,中位差 {st.median([x[1] for x in allts]):+.3f},"
          f"我们更好 {sum(1 for _, x in allts if x > 0)}/{len(allts)},p = {p1:.3g}"
          f"   {'通过' if (p1 is not None and p1 < ALPHA) else '未通过'}")

    print(f"\n=== 与每靶标【最好的一条真实设计】比(排行榜实际展示的,对我们不公平)===")
    bestts = []
    for p in T:
        pool = [v for m, vs in theirs.get(p, {}).items() if m != HOMEFIELD for v in vs]
        if pool and p in scores:
            bestts.append(scores[p]["f1c"] - max(pool))
    print(f"  {len(bestts)} 个靶标,中位差 {st.median(bestts):+.3f},"
          f"我们更好 {sum(1 for x in bestts if x > 0)}/{len(bestts)}")
    print(f"  他们每靶标的提交量中位 "
          f"{st.median([sum(len(v) for m, v in theirs[p].items() if m != HOMEFIELD) for p in theirs]):.0f} 条,我们 1 条")

    print(f"\n=== 代理的局限:同一批真实设计上,代理与湿实验的关系 ===")
    from m2r_pairwise import spearman
    rr = []
    for p in theirs:
        xs = [v for m in theirs[p] for v in theirs[p][m]]
        ys = [v for m in wet[p] for v in wet[p][m]]
        if len(xs) >= 20:
            s = spearman(xs, ys)
            if s is not None:
                rr.append(s)
    print(f"  靶标内 Spearman(RNet 交叉配对 F1, 实测 OpenKnot Score)中位 {st.median(rr):+.3f}"
          f"({len(rr)} 个靶标)")
    print(f"  -> 我们的分数没有、也不可能有实测对照。这是本次比较的诚实上限。")

    json.dump({"ours": scores, "per_method": table, "L1_p": p1,
               "L1_median_diff": st.median([x[1] for x in allts]),
               "vs_best_median_diff": st.median(bestts),
               "proxy_vs_wetlab_median_rho": st.median(rr), "alpha": ALPHA},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
