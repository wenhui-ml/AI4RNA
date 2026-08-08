#!/usr/bin/env python3
"""Audit of yesterday's pool-rerank claim, before anything is built on top of it.

WHAT WAS CLAIMED. "The better sequence was already in the pool": taking the best of ~302 candidates by
RibonanzaNet crossed-pair F1 raised our median from 0.746 to 1.000, a gain of +0.250 on 51 of 57 targets at
p = 2.6e-10, and rescued all 10 targets the product returned not_found for.

WHY IT NEEDS AN AUDIT BEFORE IT IS USED. The selection metric and the reported metric are THE SAME QUANTITY.
max(x_1..x_302) >= median(x_1..x_302) is arithmetic, not evidence, so both the +0.250 and the p-value are
tautological -- they would come out positive on pure noise. This is the identical error that killed the C3
placement flip the same day; it was caught there and missed here, in the headline.

    Decomposition of the shipped number, over the 45 targets we actually delivered:
        shipped   - pool median  = +0.034 on 27/45   <- our current selector, essentially a coin flip
        pool max  - pool median  = +0.211 on 45/45   <- guaranteed positive by construction
        pool max  - shipped      = +0.184            <- the headline, which is the sum of the two

    So the honest question the headline does NOT answer is whether ranking by RNet picks a sequence that is
    better IN REALITY, or merely one that scores well on RNet. Answering it needs a judge that is not RNet.

THE JUDGE USED HERE IS THE WET LAB. OpenKnot publishes, for every real design, both the RNet crossed-pair F1
we rank by and the measured `target_openknot_score` from the tube. So the transfer can be measured directly:
inside a pool of real designs, pick the argmax by RNet exactly as we would, and read what the tube says.

THE REFINEMENT THAT MATTERS. Yesterday's +1.74-point calibration drew pools across ALL methods. Our pool is
302 candidates from ONE solver against ONE target -- far more homogeneous, so the spread of true quality
inside it is much smaller and argmax has much more room to chase noise. `M2` (a mutational series around a
single parent design) is the closest published analogue and is scored separately here. `Eterna` pools are
hundreds of different human players and are much closer to the mixed case.

    V1  HOMOGENEOUS TRANSFER. In single-method pools of the same size, argmax-by-RNet beats the pool's
        median wet-lab score.
        PASS: p < 0.025 and median gain >= 1.0 OpenKnot point.
        KILL: if it fails on M2, the selector's transfer is NOT established at our pool's homogeneity, the
        pool-rerank result may only ever be quoted in F1 units, those units are tautological, and the
        capability claim collapses to "we can raise a number we also chose by".

    V2  TIE MASS. If many candidates tie at the top RNet score, argmax is an arbitrary pick among them and
        the achievable gain is the mean over the tied set, not the max. Reported, not tested.

    V3  SATURATION IN THE FIELD'S OWN DATA. What share of a real homogeneous pool already sits at RNet
        F1 = 1.000? If large, 1.000 does not discriminate and our 45/57 pools reaching it is unremarkable.

CLUSTERING. The independent unit is the (puzzle, method) cell, never the bootstrap draw and never the
individual design. Draws inside a cell are averaged first, then Wilcoxon runs over cells. Pooling draws
would manufacture an arbitrarily small p-value out of resampling the same molecules.

COST: $0. Local CPU, published columns only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
ALPHA = 0.025


def load():
    csv.field_size_limit(10 ** 9)
    cells = collections.defaultdict(list)
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {k: i for i, k in enumerate(next(r))}
        for x in r:
            if x[idx["SN_filter"]] != "1":
                continue
            try:
                f1c = float(x[idx["RNet_F1_crossed_pair"]])
                wet = float(x[idx["target_openknot_score"]])
            except (TypeError, ValueError):
                continue
            cells[(x[idx["puzzle"]], x[idx["method"]])].append((f1c, wet))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=100, help="pool size to emulate")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "rerank_audit.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon
    rng = random.Random(a.seed)

    cells = load()
    usable = {k: v for k, v in cells.items() if len(v) >= a.pool}
    by_method = collections.defaultdict(list)
    for k in usable:
        by_method[k[1]].append(k)
    print(f"池大小 {a.pool},可用 (puzzle, method) 单元 {len(usable)} 个,"
          f"每单元 {a.draws} 次抽样(先在单元内平均,再跨单元检验)\n")

    # ---- V3 saturation, measured before any sampling ------------------------------------------------
    print("=== V3 真实同质池里,RNet 交叉配对 F1 = 1.000 的占比 ===")
    sat = {}
    for m, ks in sorted(by_method.items()):
        fr = [sum(1 for f, _ in cells[k] if f >= 0.999) / len(cells[k]) for k in ks]
        sat[m] = st.median(fr)
        print(f"  {m:10s} {len(ks):3d} 个单元,单元内满分占比中位 {st.median(fr):.1%}"
              f"(范围 {min(fr):.1%}-{max(fr):.1%})")

    # ---- V1 / V2 ------------------------------------------------------------------------------------
    out = {"pool": a.pool, "draws": a.draws, "saturation_frac_at_1.000": sat, "methods": {}}
    print(f"\n=== V1 同质池内,按 RNet 取 argmax,湿实验分数比池中位高多少 ===")
    print(f"  {'方法':10s} {'单元':>4s} {'湿实验增益':>10s} {'更好单元':>9s} {'并列数':>7s} {'p':>10s}")
    for m, ks in sorted(by_method.items()):
        per_cell, ties = [], []
        for k in ks:
            rows = cells[k]
            gs, ts = [], []
            for _ in range(a.draws):
                pool = rng.sample(rows, a.pool)
                top = max(f for f, _ in pool)
                tied = [w for f, w in pool if f >= top - 1e-12]
                # argmax with an arbitrary tie-break is, in expectation, the MEAN over the tied set --
                # quoting max(tied) would be selecting on the outcome a second time.
                gs.append(st.mean(tied) - st.median([w for _, w in pool]))
                ts.append(len(tied))
            per_cell.append(st.mean(gs))
            ties.append(st.mean(ts))
        w, p, n = wilcoxon(per_cell, "greater")
        wins = sum(1 for x in per_cell if x > 0)
        print(f"  {m:10s} {len(per_cell):4d} {st.median(per_cell):+10.2f} {wins:6d}/{len(per_cell):<3d}"
              f" {st.median(ties):7.0f} {'n/a' if p is None else f'{p:.3g}':>10s}")
        out["methods"][m] = {"n_cells": len(per_cell), "median_gain_wetlab": st.median(per_cell),
                             "wins": wins, "p": p, "median_tie_count": st.median(ties)}

    m2 = out["methods"].get("M2")
    if m2:
        ok = bool(m2["p"] is not None and m2["p"] < ALPHA and m2["median_gain_wetlab"] >= 1.0)
        out["V1_pass_on_homogeneous"] = ok
        print(f"\n  M2(单一亲本的突变系列,与我们的池最接近):"
              f"{'通过' if ok else 'KILL —— 在同质池上,按 RNet 挑并不能带来 >=1.0 分的真实增益'}")

    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
