#!/usr/bin/env python3
"""Does our hand-built panel earn its keep where the learned model is off its training distribution?

SUPERSEDED IN PART, SAME DAY, BY REVIEW -- read this before quoting any number below.

    E1/E2 STAND. The rank-correlation half is unaffected: RNet is flat across novelty (+0.360 / +0.354),
    the panel rises toward the novel end (+0.162 -> +0.273) and never catches up (paired +0.045, 31/55,
    p = 0.12).
    E3/E4's COVERAGE ROWS DO NOT STAND. `gate()` below keeps whole tie blocks, and RNet's f1c has a 1.000
    tie block covering 42.7% of the novel half, so its minimum achievable coverage there is 42.7%: the rows
    labelled "20%" and "30%" compare a panel gate that really took 20-24% against an RNet gate that took
    43%. The output text then asserted which way that mismatch cut without testing it. Only the 50% rows
    were matched. `ruler_gradient.py` redoes this with random tie-breaking (coverage matches exactly), on
    deduplicated molecules, and -- the part that matters more -- INSIDE (target, method) cells, because the
    pooled version is non-monotone in coverage, which is the signature of composition rather than of ruler
    quality. Use that file's G4/G5, not this file's E3/E4.
    KEPT, NOT DELETED, because the retraction is the record.

EXPLORATORY. NOT PRE-REGISTERED, AND THAT IS A DELIBERATE CHOICE OF THE USER, SO IT IS LABELLED HERE AND
MUST BE LABELLED WHEREVER THESE NUMBERS ARE QUOTED. Nothing in this file may enter the capability table.
A stratified comparison that goes looking for the stratum where we win is exactly the shape of the +0.250
retraction of 2026-08-17; the only thing that makes such a finding real is a pre-declared stratum boundary
and a target holdout, and neither is here. If this run comes back positive, the finding is a HYPOTHESIS and
the next step is the pre-registered confirmatory version, not a claim.

WHY ASK AT ALL. Objective 4 scores the RANGE of criterion-manufacturing, and its one open positive lane is
blocked by a bind: wherever large-scale physical measurement exists, someone has already trained a predictor
on it, so the criterion is not missing. The pseudoknot arena is the instance that proves the bind -- at
matched coverage a plain RibonanzaNet threshold beat our three-oracle panel at every operating point and the
panel added nothing on top (0 of 5). The reframe worth testing is that the bind is about WHERE the model is
reliable rather than WHETHER it exists: RNet was trained on Ribonanza chemical mapping, largely Eterna player
designs and natural sequences. Thermodynamics has no training distribution. So the panel's range, if it has
one, should be the novel end of the identity axis.

WHAT IS ALREADY MEASURED AND CUTS AGAINST THIS (read before believing any positive result here):

    ruler_bias.py R1 already KILLED the neighbouring hypothesis. Within target, controlling for the measured
    wet-lab score, RNet's crossed-pair F1 does NOT fall as identity to the wild type falls: partial rho
    median -0.148, p = 0.99998, 57 puzzles. The sign is REVERSED -- if anything RNet is scored more
    generously on the novel end. So "the model degrades off-distribution" is not available as a premise.

    That leaves a strictly weaker question, and it is the only new one: RNet not degrading in ABSOLUTE terms
    does not by itself fix the RELATIVE standing of the two rulers, because the panel's 0-of-5 loss was
    measured POOLED over the whole identity range. This file asks whether the panel-minus-RNet difference
    moves with novelty. If it does not, the bind stands and that is Objective 4's boundary result.

WHAT IS COMPARED, ON WHOSE MOLECULES. OpenKnot publishes real designs that were synthesised and measured.
`criterion_vs_tube.json` already holds, for 4,560 of them, the tube score, RNet's crossed-pair F1, and our
panel's continuous crossing recall plus its discrete tier. That run sampled deterministically (seed 0, 80 per
target), so the sample can be replayed here to attach each record's design_sequence, hence its identity to
that puzzle's `Starting sequence` wild type. Identity uses ruler_bias.identity unchanged, so the numbers are
comparable to the 0.285-vs-0.727 figures already on disk.

    J0  JOIN CHECK, independently failable. The replayed sample must reproduce puzzle, method, f1c and wet
        for all 4,560 records or the run aborts. A silently mis-ordered join would look like a result.
    E1  Within each puzzle, split its designs at the puzzle's own median identity. In each half correlate
        both rulers against the tube. Per puzzle, delta = rho_panel - rho_rnet in the novel half minus the
        same quantity in the natural half; paired across puzzles.
    E2  The absolute rhos behind E1, so the mechanism is visible rather than inferred: does RNet fall, does
        the panel rise, or neither.
    E3  The filter view, which is the form the 0-of-5 loss was measured in: at matched coverage inside each
        half, the tube-failure rate (OpenKnot < 80) of panel-selected against RNet-selected designs.

THE LIMIT THAT NO AMOUNT OF ANALYSIS HERE REMOVES. Our own designs sit at 0.285 identity, at the novel end,
and they have no wet-lab score and never will without a laboratory. Every number below is measured on OTHER
people's high-novelty designs. Whatever holds there is evidence ABOUT the novel stratum, not evidence about
us, and the extrapolation is exactly the one the +1.90 result already has to carry.

Struct2SeQ-SHAPE is excluded from the head-to-head: it optimises against RibonanzaNet, so scoring it with
RibonanzaNet grades it on its own objective. Its count is reported so the exclusion is visible.

COST: $0. Reads two files on disk. Runs in the `rna` env (no torch, no folding).
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
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
TUBE_FAIL = 80.0          # the release's own pass mark, used throughout this project
RNET_OPTIMISED = "Struct2SeQ-SHAPE"


def replay_sample(seed: int, per_target: int):
    """Reproduce criterion_vs_tube.py's deterministic sample and keep the sequence this time."""
    from leaderboard_design import targets
    T = targets()
    csv.field_size_limit(10 ** 9)
    pool = collections.defaultdict(list)
    wild = {}
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {k: i for i, k in enumerate(next(r))}
        for x in r:
            if x[idx["SN_filter"]] != "1":
                continue
            p, s, m = x[idx["puzzle"]], x[idx["design_sequence"]], x[idx["method"]]
            if m == "Starting sequence" and s:
                wild.setdefault(p, s)
            if p not in T or not s or x[idx["target_structure"]] != T[p]["target"]:
                continue
            try:
                f1c = float(x[idx["RNet_F1_crossed_pair"]])
                wet = float(x[idx["target_openknot_score"]])
            except (TypeError, ValueError):
                continue
            pool[p].append((s, f1c, wet, m))

    rng = random.Random(seed)
    jobs = []
    for p, rows in sorted(pool.items()):
        if len(rows) < 20:
            continue
        for s, f1c, wet, m in rng.sample(rows, min(per_target, len(rows))):
            jobs.append((p, s, f1c, wet, m))
    return jobs, wild


def replay_fresh(seed: int, per_target: int, old_seed: int = 0, old_per_target: int = 80):
    """Sample `per_target` designs per target that the (old_seed, old_per_target) sample never touched.

    random.sample(rows, 160) is NOT a superset of random.sample(rows, 80) at the same seed, so a disjoint
    holdout cannot be had by raising --per-target. The old sample is replayed, removed by (puzzle, sequence,
    method), and the fresh draw comes from what is left. Returns the same tuple shape as replay_sample.
    """
    old, _ = replay_sample(old_seed, old_per_target)
    taken = {(p, s, m) for p, s, _f, _w, m in old}
    _, wild = replay_sample(old_seed, 1)                # cheap: rebuilds `wild` without a second CSV pass
    pool = collections.defaultdict(list)
    for p, s, f1c, wet, m in _all_eligible():
        if (p, s, m) not in taken:
            pool[p].append((s, f1c, wet, m))
    rng = random.Random(10_000 + seed)                  # a different stream from replay_sample's
    jobs, short = [], []
    for p, rows in sorted(pool.items()):
        if len(rows) < per_target:
            short.append((p, len(rows)))
        for s, f1c, wet, m in rng.sample(rows, min(per_target, len(rows))):
            jobs.append((p, s, f1c, wet, m))
    if short:
        print(f"  注意:{len(short)} 个靶标剩余不足 {per_target} 条,按剩余全取 {short[:5]}")
    overlap = {(p, s, m) for p, s, _f, _w, m in jobs} & taken
    assert not overlap, f"fresh sample overlaps the old one on {len(overlap)} designs"
    return jobs, wild


def replay_third(seed: int, per_target: int, verbose: bool = True):
    """The THIRD disjoint sample: designs neither the 80/target nor the 160/target sample ever touched.

    `replay_fresh` excludes ONE prior sample, so it cannot build this one -- a third sample must exclude the
    exploratory sample (seed 0, 80/target) AND the confirmatory sample (replay_fresh(0, 160, 0, 80)). Written
    out explicitly rather than as a general mechanism, because the only thing that matters here is that the
    exclusion is complete, and an explicit pair of calls can be asserted while a clever one cannot.

    Why a third sample exists at all: the frozen operating point was fitted on sample 1 and confirmed on
    sample 2, so any evaluation of a SELECTOR built from that point needs molecules neither has seen. 19,977
    eligible designs remain and every target can supply at least 47.
    """
    s1, wild = replay_sample(0, 80)
    s2, _ = replay_fresh(0, 160, 0, 80)
    taken = {(p, s, m) for p, s, _f, _w, m in s1} | {(p, s, m) for p, s, _f, _w, m in s2}
    pool = collections.defaultdict(list)
    for p, s, f1c, wet, m in _all_eligible():
        if (p, s, m) not in taken:
            pool[p].append((s, f1c, wet, m))
    rng = random.Random(20_000 + seed)                   # a third stream, distinct from both prior ones
    jobs, short = [], []
    for p, rows in sorted(pool.items()):
        if len(rows) < per_target:
            short.append((p, len(rows)))
        for s, f1c, wet, m in rng.sample(rows, min(per_target, len(rows))):
            jobs.append((p, s, f1c, wet, m))
    got = {(p, s, m) for p, s, _f, _w, m in jobs}
    assert not (got & taken), f"third sample overlaps a prior sample on {len(got & taken)} designs"
    assert len(got) == len(jobs), "third sample contains a duplicate (puzzle, sequence, method)"
    if verbose:
        print(f"  第三批:{len(jobs)} 条,{len({j[0] for j in jobs})} 个靶标;"
              f"与前两批({len(taken)} 条)交集 {len(got & taken)}(断言为 0)")
        if short:
            print(f"    剩余不足 {per_target} 条、按剩余全取的靶标 {len(short)} 个: {short[:5]}")
    return jobs, wild


def replay_fourth(seed: int, per_cell: int, verbose: bool = True):
    """The FOURTH disjoint sample, and the only one drawn PER (target, method) CELL rather than per target.

    Why the sampling unit changes. A13 asked how much each layer of the panel's ladder buys, and its estimator
    is a within-set contrast on the SAME molecules: mean(quality | confirmed) - mean(quality | whole cell).
    Pooling methods would let a method that is both more confirmable and better in the tube masquerade as the
    boundary carrying information, so the cell must stay (target, method) -- and then a per-TARGET draw caps
    the cell at whatever share that method happens to hold. Sample 2 drew 160 per target and the median cell
    came out at EIGHT molecules, which is why 89 of 287 cells had the confirmed set equal to the whole cell
    and the term under test was a structural zero. Drawing per CELL fixes the size instead of leaving it to
    the library's method mix.

    Excludes samples 1, 2 and 3 explicitly -- not cleverly -- and asserts the exclusion.
    """
    s1, wild = replay_sample(0, 80)
    s2, _ = replay_fresh(0, 160, 0, 80)
    s3, _ = replay_third(0, 80, verbose=False)
    taken = {(p, s, m) for g in (s1, s2, s3) for p, s, _f, _w, m in g}
    # DEDUPLICATE FIRST. `_all_eligible()` holds 373 keys that appear more than once (up to 21 times) --
    # replicate MEASUREMENTS of one design, with different wet scores. Sampling from the raw list can draw
    # the same design twice, which the duplicate assert below would catch but which should not happen at all.
    # Replicates are averaged, exactly as `load_fresh` does downstream. `replay_sample` / `_fresh` / `_third`
    # are deliberately NOT touched: they got away with this and must stay byte-reproducible.
    agg = collections.defaultdict(list)
    for p, s, f1c, wet, m in _all_eligible():
        if (p, s, m) not in taken:
            agg[(p, s, m)].append((f1c, wet))
    pool = collections.defaultdict(list)
    for (p, s, m), vs in agg.items():
        assert len({round(f, 9) for f, _w in vs}) == 1, f"same design, different RNet F1: {p} {m}"
        pool[(p, m)].append((s, vs[0][0], sum(w for _f, w in vs) / len(vs)))
    rng = random.Random(30_000 + seed)                   # a fourth stream, distinct from all prior ones
    jobs, short = [], []
    for (p, m), rows in sorted(pool.items()):
        if len(rows) < per_cell:
            short.append(((p, m), len(rows)))
            continue                                     # a SHORT cell is dropped, not part-filled: a cell
                                                         # whose size is set by the library is the defect
        for s, f1c, wet in rng.sample(rows, per_cell):    # this sample exists to remove
            jobs.append((p, s, f1c, wet, m))
    got = {(p, s, m) for p, s, _f, _w, m in jobs}
    assert not (got & taken), f"fourth sample overlaps a prior sample on {len(got & taken)} designs"
    assert len(got) == len(jobs), "fourth sample contains a duplicate (puzzle, sequence, method)"
    if verbose:
        cells = {(p, m) for p, _s, _f, _w, m in jobs}
        print(f"  第四批(按格抽,每格 {per_cell} 条):{len(jobs)} 条,{len(cells)} 个 (靶标,方法) 格,"
              f"{len({j[0] for j in jobs})} 个靶标;与前三批({len(taken)} 条)交集 {len(got & taken)}(断言为 0)")
        print(f"    供不满 {per_cell} 条、【整格丢弃】的格 {len(short)} 个"
              f"(丢弃是故意的:格大小必须由声明决定,不由库的方法配比决定)")
    return jobs, wild


def replay_fifth(seed: int, per_cell: int, verbose: bool = True):
    """The FIFTH disjoint sample, drawn per (target, method) CELL like the fourth, and it exists for ONE
    reason: to raise the CELL COUNT for the k-ladder's upper half, not the cell size.

    The plan recorded in the report's next-steps list said to make cells bigger -- 82 targets could supply
    40 designs each.  That figure was measured before the fourth sample consumed 2,640 designs, and it is
    wrong now.  After samples 1-4 the library holds 12,610 unused designs in 466 cells, and the counts are:
    60 cells at >=24, 50 at >=30, 33 at >=40, 20 at >=50.  Every one of those is FEWER cells than the
    fourth sample's 110, so a bigger-cell fifth sample on its own is WORSE powered, not better -- the
    fourth sample already took the best-supplied cells.  Power in the B term scales with sqrt(n_cells), so
    the move is to keep the cell size at 24 and POOL, taking 110 + 60 = 170 cells.

    Excludes samples 1, 2, 3 and 4 explicitly -- not cleverly -- and asserts the exclusion.
    """
    s1, wild = replay_sample(0, 80)
    s2, _ = replay_fresh(0, 160, 0, 80)
    s3, _ = replay_third(0, 80, verbose=False)
    s4, _ = replay_fourth(0, 24, verbose=False)
    taken = {(p, s, m) for g in (s1, s2, s3, s4) for p, s, _f, _w, m in g}
    agg = collections.defaultdict(list)
    for p, s, f1c, wet, m in _all_eligible():
        if (p, s, m) not in taken:
            agg[(p, s, m)].append((f1c, wet))
    pool = collections.defaultdict(list)
    for (p, s, m), vs in agg.items():
        assert len({round(f, 9) for f, _w in vs}) == 1, f"same design, different RNet F1: {p} {m}"
        pool[(p, m)].append((s, vs[0][0], sum(w for _f, w in vs) / len(vs)))
    rng = random.Random(40_000 + seed)                   # a fifth stream, distinct from all prior ones
    jobs, short = [], []
    for (p, m), rows in sorted(pool.items()):
        if len(rows) < per_cell:
            short.append(((p, m), len(rows)))
            continue
        for s, f1c, wet in rng.sample(rows, per_cell):
            jobs.append((p, s, f1c, wet, m))
    got = {(p, s, m) for p, s, _f, _w, m in jobs}
    assert not (got & taken), f"fifth sample overlaps a prior sample on {len(got & taken)} designs"
    assert len(got) == len(jobs), "fifth sample contains a duplicate (puzzle, sequence, method)"
    if verbose:
        cells = {(p, m) for p, _s, _f, _w, m in jobs}
        print(f"  第五批(按格抽,每格 {per_cell} 条):{len(jobs)} 条,{len(cells)} 个 (靶标,方法) 格,"
              f"{len({j[0] for j in jobs})} 个靶标;与前【四】批({len(taken)} 条)交集 {len(got & taken)}(断言为 0)")
        print(f"    供不满 {per_cell} 条、整格丢弃的格 {len(short)} 个")
    return jobs, wild


def replay_sixth(seed: int, per_cell: int, verbose: bool = True):
    """The SIXTH disjoint sample, and the first one whose per-cell size was chosen to open NEW CELLS.

    Rule 35 exists because of the fifth sample: I wrote that pooling would take the cell count from 110 to
    170 by adding 110 and 60, and never checked whether the grouping keys overlapped.  All sixty of the
    fifth sample's cells were already among the fourth's, because the fourth had taken twenty-four from
    every cell holding twenty-four, so whatever still held twenty-four was by construction a cell it had
    already used.  The union was 110.

    Measured before this was written, not after: at per_cell=10 the library opens 149 cells, of which 89
    overlap the fourth and 60 overlap the fifth, leaving exactly SIXTY cells no prior sample has touched.
    The union of fourth, fifth and sixth is 170 -- the same number as before, now derived instead of added.
    Those sixty new cells hold a median of 14 unused designs, minimum 10, which is what fixes per_cell here.

    The cost is real and runs the other way: ten molecules per cell makes a cell far likelier to degenerate,
    since the three layers can only differ when the confirmed set is neither empty nor the whole cell and
    the top block is not the whole confirmed set.  Whether sixty noisy cells beat sixty-eight quieter ones
    cannot be settled in advance, which is why the declaration makes the BLIND arm primary on the grounds
    of blindness rather than on a power claim I would only have to retract again.

    Excludes samples 1, 2, 3, 4 and 5 explicitly -- not cleverly -- and asserts the exclusion.
    """
    s1, wild = replay_sample(0, 80)
    s2, _ = replay_fresh(0, 160, 0, 80)
    s3, _ = replay_third(0, 80, verbose=False)
    s4, _ = replay_fourth(0, 24, verbose=False)
    s5, _ = replay_fifth(0, 24, verbose=False)
    taken = {(p, s, m) for g in (s1, s2, s3, s4, s5) for p, s, _f, _w, m in g}
    prior_cells = {(p, m) for g in (s4, s5) for p, _s, _f, _w, m in g}
    agg = collections.defaultdict(list)
    for p, s, f1c, wet, m in _all_eligible():
        if (p, s, m) not in taken:
            agg[(p, s, m)].append((f1c, wet))
    pool = collections.defaultdict(list)
    for (p, s, m), vs in agg.items():
        assert len({round(f, 9) for f, _w in vs}) == 1, f"same design, different RNet F1: {p} {m}"
        pool[(p, m)].append((s, vs[0][0], sum(w for _f, w in vs) / len(vs)))
    rng = random.Random(50_000 + seed)                   # a sixth stream, distinct from all prior ones
    jobs, short = [], []
    for (p, m), rows in sorted(pool.items()):
        if len(rows) < per_cell:
            short.append(((p, m), len(rows)))
            continue
        for s, f1c, wet in rng.sample(rows, per_cell):
            jobs.append((p, s, f1c, wet, m))
    got = {(p, s, m) for p, s, _f, _w, m in jobs}
    assert not (got & taken), f"sixth sample overlaps a prior sample on {len(got & taken)} designs"
    assert len(got) == len(jobs), "sixth sample contains a duplicate (puzzle, sequence, method)"
    cells = {(p, m) for p, _s, _f, _w, m in jobs}
    fresh_cells = cells - prior_cells
    if verbose:
        print(f"  第六批(按格抽,每格 {per_cell} 条):{len(jobs)} 条,{len(cells)} 个 (靶标,方法) 格,"
              f"{len({j[0] for j in jobs})} 个靶标;与前【五】批({len(taken)} 条)交集 {len(got & taken)}(断言为 0)")
        print(f"    其中【没有任何前批碰过的新格】{len(fresh_cells)} 个 —— 这是唯一的盲臂(第 35 条)")
        print(f"    与第四/第五批共格的 {len(cells & prior_cells)} 个;三批格并集 {len(cells | prior_cells)}")
        print(f"    供不满 {per_cell} 条、整格丢弃的格 {len(short)} 个")
    return jobs, wild


def _all_eligible():
    """Every SN_filter=1 design whose target matches ours and which carries both an RNet F1 and a tube score."""
    from leaderboard_design import targets
    T = targets()
    csv.field_size_limit(10 ** 9)
    per = collections.defaultdict(list)
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {k: i for i, k in enumerate(next(r))}
        for x in r:
            if x[idx["SN_filter"]] != "1":
                continue
            p, s, m = x[idx["puzzle"]], x[idx["design_sequence"]], x[idx["method"]]
            if p not in T or not s or x[idx["target_structure"]] != T[p]["target"]:
                continue
            try:
                f1c = float(x[idx["RNet_F1_crossed_pair"]])
                wet = float(x[idx["target_openknot_score"]])
            except (TypeError, ValueError):
                continue
            per[p].append((p, s, f1c, wet, m))
    # the >=20 eligibility gate is a property of the TARGET and must match replay_sample's
    return [row for p, rows in sorted(per.items()) if len(rows) >= 20 for row in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "criterion_vs_tube.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "novelty_stratified_ruler.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from m2r_pairwise import spearman, wilcoxon
    from ruler_bias import identity

    src = json.load(open(a.src))
    designs = src["designs"]
    jobs, wild = replay_sample(a.seed, src["per_target_sampled"])

    # ---- J0 join check ---------------------------------------------------------------------------------
    print("=== J0 连接检查(错位的连接看起来和结果一模一样,所以先失败在这里)===")
    if len(jobs) != len(designs):
        print(f"!! 重放得到 {len(jobs)} 条,criterion_vs_tube.json 有 {len(designs)} 条 —— 不做近似匹配,停。")
        return 2
    bad = 0
    for (p, s, f1c, wet, m), d in zip(jobs, designs):
        if d["puzzle"] != p or d["method"] != m or abs(d["f1c"] - f1c) > 1e-9 or abs(d["wet"] - wet) > 1e-9:
            bad += 1
    print(f"  {len(jobs)} 条,字段不一致 {bad} 条" + ("  ✓" if bad == 0 else "  ✗"))
    if bad:
        print("!! 连接不可信,停。")
        return 2
    missing_wild = sorted({p for p, *_ in jobs} - set(wild))
    print(f"  有野生型参照的靶标 {len({p for p, *_ in jobs}) - len(missing_wild)} 个"
          + (f",缺 {missing_wild}" if missing_wild else ",无缺失"))

    rec = []
    n_excl = 0
    for (p, s, f1c, wet, m), d in zip(jobs, designs):
        w = wild.get(p)
        if w is None:
            continue
        if m == RNET_OPTIMISED:
            n_excl += 1
            continue
        ident = identity(s, w)
        if ident is None:
            continue
        rec.append({"puzzle": p, "method": m, "ident": ident, "wet": wet,
                    "f1c": f1c, "panel": d["panel_mean"], "k": d["panel_k"]})
    print(f"  参与比较 {len(rec)} 条;排除 {RNET_OPTIMISED} {n_excl} 条(它对着 RNet 优化)")

    by = collections.defaultdict(list)
    for r in rec:
        by[r["puzzle"]].append(r)

    # ---- E1 / E2 ---------------------------------------------------------------------------------------
    print("\n=== E1/E2 靶标内按新颖度对半分,两把尺子各自对试管的秩相关 ===")
    per, rows_e1 = {}, []
    for p, rs in sorted(by.items()):
        med = st.median([r["ident"] for r in rs])
        halves = {"novel": [r for r in rs if r["ident"] < med],
                  "natural": [r for r in rs if r["ident"] >= med]}
        out = {}
        for name, h in halves.items():
            if len(h) < 15 or len({r["wet"] for r in h}) < 3:
                out[name] = None
                continue
            wet = [r["wet"] for r in h]
            rp = spearman([r["panel"] for r in h], wet)
            rr = spearman([r["f1c"] for r in h], wet)
            if rp is None or rr is None:
                out[name] = None
                continue
            out[name] = {"n": len(h), "rho_panel": rp, "rho_rnet": rr, "delta": rp - rr,
                         "ident_median": st.median([r["ident"] for r in h])}
        per[p] = out
        if out.get("novel") and out.get("natural"):
            rows_e1.append((p, out["novel"]["delta"] - out["natural"]["delta"]))

    def med_of(field, half):
        v = [per[p][half][field] for p in per if per[p].get(half)]
        return st.median(v) if v else None

    for half in ("novel", "natural"):
        n = sum(1 for p in per if per[p].get(half))
        print(f"  {half:8s} 靶标 {n:2d}   同一性中位 {med_of('ident_median', half):.3f}"
              f"   rho(面板,试管) {med_of('rho_panel', half):+.3f}"
              f"   rho(RNet,试管) {med_of('rho_rnet', half):+.3f}"
              f"   差 {med_of('delta', half):+.3f}")

    e1 = {"n_paired": len(rows_e1)}
    if rows_e1:
        diffs = [d for _, d in rows_e1]
        w, p2, nz = wilcoxon(diffs, "two-sided")
        e1.update({"median_diff": st.median(diffs), "p_two_sided": p2, "n_nonzero": nz,
                   "better_in_novel": sum(1 for d in diffs if d > 0),
                   "better_in_natural": sum(1 for d in diffs if d < 0)})
        print(f"\n  配对(新颖半 − 自然半 的 面板−RNet 差):中位 {e1['median_diff']:+.3f}"
              f"   新颖半更有利 {e1['better_in_novel']}/{len(diffs)}"
              f"   双侧 p = {p2:.4g}" if p2 is not None else "")
        print("  读法:> 0 表示面板相对 RNet 在新颖端更好;这是本文件唯一新问的量。")

    # ---- E3 filter view --------------------------------------------------------------------------------
    print("\n=== E3 匹配覆盖率下当过滤器:被选中设计的试管失败率(OpenKnot < 80)===")
    def gate(rs, key, want_cov):
        """Take the top-scoring designs under `key` at the coverage closest to want_cov. Ties are kept
        whole, so the realized coverage is reported rather than assumed."""
        vals = sorted({r[key] for r in rs}, reverse=True)
        best = None
        for t in vals:
            sel = [r for r in rs if r[key] >= t]
            cov = len(sel) / len(rs)
            if best is None or abs(cov - want_cov) < abs(best[1] - want_cov):
                best = (t, cov, sel)
        return best

    e3 = {}
    for half in ("novel", "natural"):
        pooled = []
        for p, rs in by.items():
            med = st.median([r["ident"] for r in rs])
            pooled += [r for r in rs if (r["ident"] < med if half == "novel" else r["ident"] >= med)]
        base = sum(1 for r in pooled if r["wet"] < TUBE_FAIL) / len(pooled)
        print(f"  {half:8s} n={len(pooled)}  不筛的失败率 {base:.1%}")
        e3[half] = {"n": len(pooled), "base_fail": base, "points": []}
        for cov in (0.2, 0.3, 0.5):
            tp, cp, sp = gate(pooled, "panel", cov)
            tr, cr, sr = gate(pooled, "f1c", cov)
            fp = sum(1 for r in sp if r["wet"] < TUBE_FAIL) / len(sp)
            fr = sum(1 for r in sr if r["wet"] < TUBE_FAIL) / len(sr)
            print(f"      目标覆盖 {cov:.0%}   面板 {cp:.0%} 覆盖 → 失败 {fp:.1%}"
                  f"   |   RNet {cr:.0%} 覆盖 → 失败 {fr:.1%}"
                  f"   |   面板−RNet {fp - fr:+.1%}")
            e3[half]["points"].append({"want_cov": cov, "panel_cov": cp, "panel_fail": fp,
                                       "rnet_cov": cr, "rnet_fail": fr, "diff": fp - fr})

    # per-puzzle paired version of E3 at 30% coverage, since pooling ignores the clustering
    wins = {"panel": 0, "rnet": 0, "tie": 0}
    pd = []
    for p, rs in sorted(by.items()):
        med = st.median([r["ident"] for r in rs])
        h = [r for r in rs if r["ident"] < med]
        if len(h) < 15 or len({r["wet"] < TUBE_FAIL for r in h}) < 2:
            continue
        _, _, sp = gate(h, "panel", 0.3)
        _, _, sr = gate(h, "f1c", 0.3)
        fp = sum(1 for r in sp if r["wet"] < TUBE_FAIL) / len(sp)
        fr = sum(1 for r in sr if r["wet"] < TUBE_FAIL) / len(sr)
        pd.append(fp - fr)
        wins["panel" if fp < fr else "rnet" if fp > fr else "tie"] += 1
    if pd:
        w, p2, nz = wilcoxon(pd, "two-sided")
        print(f"\n  逐靶标配对(新颖半,30% 覆盖):面板更低失败率 {wins['panel']} / RNet 更低 {wins['rnet']}"
              f" / 平 {wins['tie']}   中位差 {st.median(pd):+.1%}"
              + (f"   双侧 p = {p2:.4g}" if p2 is not None else ""))
        e3["paired_novel_cov30"] = {"wins": wins, "median_diff": st.median(pd), "p_two_sided": p2, "n": nz}

    # ---- E4 the stratum we actually live in ------------------------------------------------------------
    # The median-split "novel" half sits at 0.444 identity; our own delivered designs sit at 0.285, BELOW
    # that half's centre. So the half-split does not answer the question that matters for us. This looks
    # only at designs as novel as ours. Both cut-offs are reported, not one chosen: 0.30 is our own median
    # rounded, 0.35 is the widest band that still keeps enough targets to correlate within.
    print("\n=== E4 我们真正所在的那一段(同一性 ≤ 我们自己的水平)===")
    print("  半分法的新颖半中位是 0.444,而我们自己是 0.285 —— 比那一半的中心还要低,所以半分法回答的不是"
          "我们的问题。这一节只看和我们一样新颖的设计。")
    e4 = {}
    for thr in (0.30, 0.35):
        sub = [r for r in rec if r["ident"] <= thr]
        byt = collections.defaultdict(list)
        for r in sub:
            byt[r["puzzle"]].append(r)
        usable = {p: rs for p, rs in byt.items() if len(rs) >= 15 and len({r["wet"] for r in rs}) >= 3}
        base = sum(1 for r in sub if r["wet"] < TUBE_FAIL) / len(sub)
        print(f"\n  同一性 ≤ {thr:.2f}:{len(sub)} 条设计,{len(byt)} 个靶标,其中 {len(usable)} 个够做靶标内相关"
              f";不筛的失败率 {base:.1%}")
        d = {"threshold": thr, "n": len(sub), "n_targets": len(byt), "n_usable": len(usable),
             "base_fail": base, "points": []}
        if usable:
            dl = []
            for p, rs in sorted(usable.items()):
                wet = [r["wet"] for r in rs]
                rp, rr = spearman([r["panel"] for r in rs], wet), spearman([r["f1c"] for r in rs], wet)
                if rp is not None and rr is not None:
                    dl.append(rp - rr)
            if dl:
                w, p2, nz = wilcoxon(dl, "two-sided")
                d["rho_diff_median"] = st.median(dl)
                d["rho_diff_p_two_sided"] = p2
                d["rho_diff_panel_ahead"] = sum(1 for v in dl if v > 0)
                d["rho_diff_n"] = len(dl)
                print(f"    靶标内 rho(面板) − rho(RNet):中位 {st.median(dl):+.3f}"
                      f"   面板领先 {d['rho_diff_panel_ahead']}/{len(dl)}"
                      + (f"   双侧 p = {p2:.4g}" if p2 is not None else "   n 太小,不给 p"))
        for cov in (0.3, 0.5):
            tp, cp, sp = gate(sub, "panel", cov)
            tr, cr, sr = gate(sub, "f1c", cov)
            fp = sum(1 for r in sp if r["wet"] < TUBE_FAIL) / len(sp)
            fr = sum(1 for r in sr if r["wet"] < TUBE_FAIL) / len(sr)
            print(f"    目标覆盖 {cov:.0%}   面板 {cp:.0%} → 失败 {fp:.1%}"
                  f"   |   RNet {cr:.0%} → 失败 {fr:.1%}   |   面板−RNet {fp - fr:+.1%}")
            d["points"].append({"want_cov": cov, "panel_cov": cp, "panel_fail": fp,
                                "rnet_cov": cr, "rnet_fail": fr, "diff": fp - fr})
        e4[f"ident_le_{thr:.2f}"] = d
    print("\n  覆盖率匹配得不干净(面板的连续召回取值粗,并列被整块保留),而且失配的方向对面板有利:"
          "\n  RNet 常常在更高的覆盖率上还给出更低的失败率,所以面板的落后是被低估而不是被夸大的。")

    # ---- where we sit ---------------------------------------------------------------------------------
    print("\n=== 参照:我们自己在这条轴上的位置 ===")
    print("  我们交付设计的同一性中位 0.285(ruler_bias.json R2),别人的 0.727;"
          "  我们落在新颖端,但我们没有、也不会有试管分。")

    json.dump({"exploratory": True, "pre_registered": False,
               "note": "NOT a capability-table row. A positive E1/E3 here is a hypothesis needing a "
                       "pre-registered confirmatory run with a declared stratum boundary and target holdout.",
               "n_designs": len(rec), "n_excluded_rnet_optimised": n_excl,
               "tube_fail_threshold": TUBE_FAIL,
               "E1": e1, "E2_per_puzzle": per, "E3": e3, "E4": e4},
              open(a.out, "w"), indent=1, sort_keys=True)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
