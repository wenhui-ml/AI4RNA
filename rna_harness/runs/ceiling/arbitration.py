#!/usr/bin/env python3
"""When two verifier families disagree about which design is better, which one does the TUBE side with?

WHY THIS IS OBJECTIVE 4's POSITIVE SIDE AND NOT ANOTHER INSTRUMENT FIX. Objective 4 scores the RANGE of
criterion-manufacturing: how many domains where the criterion is missing at design time AND large-scale
physical measurement exists to validate the ruler we build. Two domains are done (pseudoknot panel,
co-transcriptional Kinfold judge). The third has been blocked from both sides -- measurement availability and
model existence are anti-correlated, and the last escape ("the trained model only works in-distribution")
was closed on 2026-08-18: RibonanzaNet's agreement with the tube is FLAT across novelty (+0.360 vs +0.354),
and at our own novelty it halves the tube-failure rate.

Today produced a different kind of gap, and it passes all three conditions:

    the criterion is missing   Two verifier families rank our selection rules in OPPOSITE orders on the same
                               51 targets in one run: RibonanzaNet says `screen_first` is the best realisable
                               rule (0.826 -> 0.955), and IPknot/Knotty/RNAPKplex say it is the worst of
                               thirteen and worse than what ships (-0.037, p = 0.026). There is no accepted
                               rule for deciding which family to believe. The sibling paper establishes that
                               "solved is a choice of verifier" and stops there; nobody publishes the
                               arbitration.
    the measurement exists     34,750 synthesised OpenKnot designs carry a measured tube score, and any
                               verifier can be run on the same molecules.
    no trained predictor       Nobody has trained a model on "which verifier to trust", because the object is
                               a meta-quantity over verifiers rather than over structures.

And it is the gap that actually blocked a product decision: choosing `screen_first` today rested on a
JUDGEMENT about which (selector, judge) pair lacked a shared-error path. This run is the attempt to replace
that judgement with a measured rule.

THE DESIGN, and why pairwise rather than correlational. The two rulers live on different scales and saturate
in different places, so comparing their correlations with the tube conflates scale with ordering. Instead:
inside a (target, method) cell, take every pair of designs the two families order OPPOSITELY, and ask which
one the tube agrees with. That quantity is scale-free, needs no threshold, and controls target difficulty and
pipeline composition by construction -- the two confounds this project has already been burned by (+4.75
became +1.90 once method composition was controlled).

    A1  PRIMARY. P(tube sides with the learned ruler | the two families disagree), over discordant pairs,
        with a cluster bootstrap over cells. Null 0.5.
            > 0.5 significantly -> the learned family should arbitrate, and by how much is the effect size
            < 0.5 significantly -> the thermodynamic family should
            ~ 0.5             -> no arbitration rule exists between these two rulers, and THAT is objective
                                 4's boundary result for this domain: the disagreement is not resolvable from
                                 the rulers alone
    A2  THE TUBE'S OWN NOISE IS NOT OPTIONAL HERE. Replicate measurements of one molecule differ by a median
        of 3.0 OpenKnot points, so a pair whose tube scores differ by less than that is not evidence about
        either ruler. Reported at margin 0 and at margin 3.0, and the margin-3.0 number is the headline.
    A3  IS IT A RULER OR A CONSTANT? If A1 is off 0.5, the arbitration is only a CRITERION if the winner
        depends on something readable at design time. Stratified by crossing-pair count, length, and by
        whether either ruler is saturated at its own ceiling -- the mechanism already measured on both
        (panel high tercile -0.001, RNet high tercile +0.084, 36.1% of designs at f1c exactly 1.000).

DECLARED BEFORE RUNNING, since a stratified search for a favourable cell is exactly the shape of the +0.250
retraction: the primary is A1 at margin 3.0 pooled over all cells. A3 is exploratory and its strata are fixed
here (crossing count terciles, length terciles, saturation of either ruler) rather than chosen after looking.

Struct2SeQ-SHAPE excluded throughout: it optimises against RibonanzaNet, so it is not a fair test of whether
RibonanzaNet is right.

COST: $0. Arithmetic over files already on disk.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

TUBE_NOISE = 3.0          # measured: median spread between replicate measurements of the same molecule
RNET_OPTIMISED = "Struct2SeQ-SHAPE"


def load(panel_key):
    """The 4,560 adjudicated field designs, deduplicated by molecule, with both families' scores."""
    from ruler_gradient import load as rg_load
    raw, rec, spreads = rg_load(os.path.join(HERE, "criterion_vs_tube.json"), 0,
                                os.path.join(HERE, "panel_f1_designs.json"), panel_key)
    return rec, spreads


def discordant(rows, margin):
    """Every pair the two families order oppositely, labelled by which one the tube agrees with.

    Returns (n_learned_right, n_thermo_right, n_pairs_considered).
    """
    a = b = n = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            x, y = rows[i], rows[j]
            dp, dr, dw = x["panel"] - y["panel"], x["f1c"] - y["f1c"], x["wet"] - y["wet"]
            if dp == 0 or dr == 0:                  # a ruler with no opinion is not a disagreement
                continue
            if (dp > 0) == (dr > 0):                # they agree; not this run's question
                continue
            if abs(dw) <= margin:                   # the tube cannot referee inside its own noise
                continue
            n += 1
            if (dr > 0) == (dw > 0):
                a += 1
            else:
                b += 1
    return a, b, n


def boot_ci(cells, margin, draws, rng):
    """Cluster bootstrap over cells: the independent unit is the cell, never the pair."""
    keys = list(cells)
    out = []
    for _ in range(draws):
        a = b = 0
        for k in (keys[rng.randrange(len(keys))] for _ in keys):
            x, y, _ = cells[k]
            a += x
            b += y
        if a + b:
            out.append(a / (a + b))
    out.sort()
    if not out:
        return None, None
    return out[int(0.025 * len(out))], out[min(len(out) - 1, int(0.975 * len(out)))]


def binom_p(a, n):
    """Two-sided exact binomial against 0.5."""
    if n == 0:
        return None
    k = min(a, n - a)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-key", default="panel_f1",
                    choices=["panel_f1", "panel_recall_product", "panel_precision"])
    ap.add_argument("--unit", default="cell", choices=["cell", "target"],
                    help="cell = (target, method), the estimator this project accepts; target = looser")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "arbitration.json"))
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rec, spreads = load(a.panel_key)
    rec = [r for r in rec if r["method"] != RNET_OPTIMISED]
    print(f"{len(rec)} 条分子(去重后,排除 {RNET_OPTIMISED});面板列 = {a.panel_key}")
    print(f"试管复测噪声 中位 {st.median(spreads):.1f} 分({len(spreads)} 条有重复测量)"
          f" —— A2 的边距用 {TUBE_NOISE:.1f}")

    groups = collections.defaultdict(list)
    for r in rec:
        groups[(r["puzzle"], r["method"]) if a.unit == "cell" else r["puzzle"]].append(r)
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    print(f"分组单位 = {a.unit},可用组 {len(groups)}(每组 >=3 条)\n")

    res = {"panel_key": a.panel_key, "unit": a.unit, "n_designs": len(rec),
           "tube_noise_median": st.median(spreads), "points": []}
    for margin in (0.0, TUBE_NOISE):
        cells = {k: discordant(v, margin) for k, v in groups.items()}
        cells = {k: v for k, v in cells.items() if v[2] > 0}
        A = sum(v[0] for v in cells.values())
        B = sum(v[1] for v in cells.values())
        n = A + B
        frac = A / n if n else None
        lo, hi = boot_ci(cells, margin, a.draws, rng)
        p = binom_p(A, n)
        head = "  <- A1 主检验" if margin == TUBE_NOISE else ""
        print(f"边距 {margin:.1f} 分:相悖配对 {n}(来自 {len(cells)} 组)  "
              f"试管站学习型一方 {A},站热力学一方 {B}  = {frac:.1%}"
              f"  聚类 bootstrap 95% CI [{lo:.1%},{hi:.1%}]  二项 p = {p:.3g}{head}")
        res["points"].append({"margin": margin, "n_pairs": n, "n_cells": len(cells),
                              "learned_right": A, "thermo_right": B, "frac_learned": frac,
                              "ci": [lo, hi], "binom_p": p})

    # ---- A3 -------------------------------------------------------------------------------------------
    print(f"\n=== A3 它是一把尺子还是一个常数(分层在跑之前钉死)===")
    def strata(r, kind):
        if kind == "n_cross":
            return "少" if r["n_cross_q"] == 0 else ("中" if r["n_cross_q"] == 1 else "多")
        if kind == "len":
            return "短" if r["len_q"] == 0 else ("中" if r["len_q"] == 1 else "长")
        if kind == "sat":
            return ("RNet 饱和" if r["f1c"] >= 0.999 else
                    ("面板饱和" if r["panel"] >= 0.999 else "都未饱和"))
        raise ValueError(kind)

    # n_cross is a property of the TARGET, not of the design, and neither upstream file carries it
    # (criterion_vs_tube projected it out) -- so it is recomputed from the target structure here.
    from leaderboard_design import targets as _targets
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    T = {p_: v["target"] for p_, v in _targets().items()}
    NX = {p_: len(pk_pairs(t) - pk_pairs(_round_only(t))) for p_, t in T.items()}
    for r in rec:
        r["n_cross"] = NX.get(r["puzzle"], 0)
    ncs = sorted({r["n_cross"] for r in rec})
    lens = sorted({len(r["seq"]) for r in rec})
    def q(v, xs):
        return 0 if v <= xs[len(xs) // 3] else (1 if v <= xs[2 * len(xs) // 3] else 2)
    for r in rec:
        r["n_cross_q"] = q(r["n_cross"], ncs)
        r["len_q"] = q(len(r["seq"]), lens)

    res["A3"] = {}
    print(f"  分层边界:n_cross 三分位 {ncs[len(ncs)//3]} / {ncs[2*len(ncs)//3]}"
          f"(取值 {min(ncs)}–{max(ncs)});长度三分位 {lens[len(lens)//3]} / {lens[2*len(lens)//3]}nt")
    for kind in ("n_cross", "len", "sat"):
        buckets = collections.defaultdict(lambda: collections.defaultdict(list))
        for k, v in groups.items():
            for r in v:
                buckets[strata(r, kind)][k].append(r)
        row = {}
        for name, g in sorted(buckets.items()):
            g = {k: v for k, v in g.items() if len(v) >= 3}
            cells = {k: discordant(v, TUBE_NOISE) for k, v in g.items()}
            cells = {k: v for k, v in cells.items() if v[2] > 0}
            A = sum(v[0] for v in cells.values())
            B = sum(v[1] for v in cells.values())
            n = A + B
            if n < 20:
                print(f"  {kind:8s} {name:10s} 相悖配对 {n} —— 太少,不给数")
                row[name] = {"n_pairs": n}
                continue
            # The binomial p treats every pair as independent, and pairs inside one cell are not: with
            # thousands of pairs from a few hundred cells it is anti-conservative by a wide margin. The
            # cluster bootstrap over cells is the honest interval, and a stratum only counts as a reversal
            # if ITS interval clears 0.5 -- not because a naive p is small.
            lo_, hi_ = boot_ci(cells, TUBE_NOISE, a.draws, rng)
            p = binom_p(A, n)
            clears = (lo_ is not None and (lo_ > 0.5 or hi_ < 0.5))
            print(f"  {kind:8s} {name:10s} 相悖配对 {n:5d}  组 {len(cells):3d}  站学习型 {A/n:.1%}"
                  f"  聚类 CI [{lo_:.1%},{hi_:.1%}]{'  ✓ 越过 0.5' if clears else '  跨过 0.5'}"
                  f"  (朴素二项 p = {p:.3g},偏松)")
            row[name] = {"n_pairs": n, "n_cells": len(cells), "frac_learned": A / n,
                         "cluster_ci": [lo_, hi_], "ci_clears_half": clears, "binom_p_anticonservative": p}
        res["A3"][kind] = row

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
