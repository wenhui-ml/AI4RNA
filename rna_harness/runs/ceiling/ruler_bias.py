#!/usr/bin/env python3
"""Is the ruler that placed us 11th of 14 neutral between the field's designs and ours?

WHY THIS RUNS BEFORE ANY ATTEMPT TO IMPROVE THE PLACEMENT. We are scored by RibonanzaNet. RibonanzaNet was
trained on Ribonanza chemical-mapping data, which is largely Eterna player designs and natural sequences --
that is, on the same population as several of the arms we are being compared against. Our DesiRNA designs
are, by construction, outside that population. If the model is more accurate on sequences resembling its
training distribution, the leaderboard hands the field a head start that has nothing to do with design
quality, and every hour spent chasing the gap is spent chasing an artifact.

The obvious version of this test -- "does RNet rate arm X above what the tube says" -- cannot separate ruler
bias from a genuine in-silico-good / in-vitro-bad gap, so it is not the test used here. This one instead puts
the field and us on ONE measurable axis: distance from the natural sequence.

    Each puzzle has a `Starting sequence` arm: the natural wild type. Every design for that puzzle can be
    scored by its identity to that wild type, and so can ours. If RNet's agreement with the tube DECAYS as
    designs move away from the natural sequence, then accuracy depends on novelty, and where our designs sit
    on that axis tells us how much of the -0.200 gap is the ruler rather than the design.

    R1  NOVELTY GRADIENT. Within target, controlling for the measured wet-lab score, does RNet crossed-pair
        F1 fall as identity to the wild type falls? Partial Spearman on ranks, clustered by puzzle.
        PASS: partial rho > 0 with p < 0.025 -> near-natural sequences are scored more generously than their
        tube behaviour justifies, and the ruler is not neutral.

    R2  WHERE DO WE SIT. Identity-to-wild-type of our 47 delivered designs versus the field's, per puzzle.
        This is descriptive and cannot fail; it sets the size of whatever R1 finds.

    R3  HOW MUCH OF THE GAP IS EXPLAINED. If R1 passes, the fitted gradient times our identity deficit is the
        part of the -0.200 median gap attributable to novelty rather than quality. Reported with the
        explicit caveat that a gradient measured across the field is being extrapolated to us.

    KILL for the whole line of inquiry: R1 fails -> the ruler is novelty-neutral, the 11/14 placement stands
    without qualification, and it is a design gap we have to close on the merits.

CLUSTERING. The independent unit is the puzzle. Correlations are computed within puzzle and combined across
puzzles; designs are never pooled across targets, because target difficulty drives both variables.

COST: $0. Published columns only, plus our own sequences from disk.
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
ALPHA = 0.025


def identity(a, b):
    """Fraction of positions equal, over the shorter length; designs for one puzzle share a length."""
    if not a or not b:
        return None
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=os.path.join(HERE, "leaderboard_ours.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "ruler_bias.json"))
    a = ap.parse_args()

    from m2r_pairwise import spearman, wilcoxon

    csv.field_size_limit(10 ** 9)
    rows = collections.defaultdict(list)
    wild = {}
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {k: i for i, k in enumerate(next(r))}
        for x in r:
            if x[idx["SN_filter"]] != "1":
                continue
            p, m, s = x[idx["puzzle"]], x[idx["method"]], x[idx["design_sequence"]]
            if m == "Starting sequence" and s:
                wild.setdefault(p, s)
            try:
                f1c = float(x[idx["RNet_F1_crossed_pair"]])
                wet = float(x[idx["target_openknot_score"]])
            except (TypeError, ValueError):
                continue
            if s:
                rows[p].append((m, s, f1c, wet))
    print(f"有野生型参照的 puzzle {len(wild)} 个,带打分设计的 puzzle {len(rows)} 个")

    # ---- R1 -----------------------------------------------------------------------------------------
    # Partial Spearman of (identity, RNet F1) controlling for the wet-lab score, done the rank-residual
    # way: regress each variable's ranks on the control's ranks, correlate the residuals.
    def partial(xs, ys, zs):
        rx, ry, rz = (_rank(v) for v in (xs, ys, zs))
        def resid(v):
            b = _slope(rz, v)
            mv, mz = st.mean(v), st.mean(rz)
            return [vi - (mv + b * (zi - mz)) for vi, zi in zip(v, rz)]
        return spearman(resid(rx), resid(ry))

    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    def _slope(z, v):
        mz, mv = st.mean(z), st.mean(v)
        den = sum((zi - mz) ** 2 for zi in z)
        return 0.0 if den == 0 else sum((zi - mz) * (vi - mv) for zi, vi in zip(z, v)) / den

    per_puzzle, raw = [], []
    for p, rs in sorted(rows.items()):
        w = wild.get(p)
        if not w or len(rs) < 30:
            continue
        ident = [identity(s, w) for _, s, _, _ in rs]
        keep = [i for i, v in enumerate(ident) if v is not None]
        if len(keep) < 30:
            continue
        xs = [ident[i] for i in keep]
        ys = [rs[i][2] for i in keep]
        zs = [rs[i][3] for i in keep]
        if len(set(xs)) < 3 or len(set(ys)) < 3:
            continue
        pr = partial(xs, ys, zs)
        sr = spearman(xs, ys)
        if pr is not None:
            per_puzzle.append(pr)
            raw.append(sr if sr is not None else 0.0)

    w, p1, n = wilcoxon(per_puzzle, "greater")
    print(f"\n=== R1 控制湿实验分数后,'越像野生型 -> RNet 打分越高' 的梯度 ===")
    print(f"  {len(per_puzzle)} 个 puzzle,偏 Spearman 中位 {st.median(per_puzzle):+.3f}"
          f"(未控制时 {st.median(raw):+.3f}),为正的 {sum(1 for x in per_puzzle if x>0)}/{len(per_puzzle)},"
          f"p = {'n/a' if p1 is None else f'{p1:.3g}'}")
    r1 = bool(p1 is not None and p1 < ALPHA and st.median(per_puzzle) > 0)
    print(f"  {'通过 —— 尺子不中立:越接近天然序列,RNet 给的分越高于其真实表现' if r1 else 'KILL —— 尺子对新颖度中立,11/14 的名次不能用这个理由打折'}")

    # ---- R2 -----------------------------------------------------------------------------------------
    O = json.load(open(a.ours))
    ours = {r["puzzle"]: r["sequence"] for r in O["rows"] if r.get("sequence")}
    d = []
    for p, s in sorted(ours.items()):
        w = wild.get(p)
        if not w or p not in rows:
            continue
        mine = identity(s, w)
        theirs = [identity(x[1], w) for x in rows[p] if x[0] != "Starting sequence"]
        theirs = [v for v in theirs if v is not None]
        if mine is None or not theirs:
            continue
        d.append((p, mine, st.median(theirs)))
    print(f"\n=== R2 我们的设计离天然序列有多远(与该靶标全体真实设计相比)===")
    print(f"  {len(d)} 个靶标:我们与野生型的相同位点比例中位 {st.median([x[1] for x in d]):.1%},"
          f"他们中位 {st.median([x[2] for x in d]):.1%}")
    dd = [x[1] - x[2] for x in d]
    w2, p2, _ = wilcoxon(dd, "two-sided")
    print(f"  差值中位 {st.median(dd):+.1%},我们更接近天然的靶标 {sum(1 for x in dd if x>0)}/{len(dd)},"
          f"p = {'n/a' if p2 is None else f'{p2:.3g}'}(双侧,描述性)")

    json.dump({"R1_partial_rho_median": st.median(per_puzzle) if per_puzzle else None,
               "R1_p": p1, "R1_pass": r1, "R1_n_puzzles": len(per_puzzle),
               "R1_uncontrolled_rho_median": st.median(raw) if raw else None,
               "R2_ours_identity_median": st.median([x[1] for x in d]) if d else None,
               "R2_theirs_identity_median": st.median([x[2] for x in d]) if d else None,
               "R2_diff_median": st.median(dd) if dd else None, "R2_p": p2,
               "R2_per_target": {x[0]: {"ours": x[1], "theirs_median": x[2]} for x in d}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
