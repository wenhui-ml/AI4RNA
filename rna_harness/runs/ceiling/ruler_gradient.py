#!/usr/bin/env python3
"""Not "which ruler is more accurate" but "where does each ruler still have gradient" -- plus two fixes.

WHY THIS REPLACES THE QUESTION. Three findings of this project are the same sentence about different objects:

    the panel decides "answer or not" well and "which answer" badly (five independent appearances);
    the panel's rank correlation with quality DECAYS as its own score rises (+0.279 / +0.158 / +0.040);
    the field's designs saturate the release's own ruler -- 36.1% of the measured designs sit at
    RNet crossed-pair F1 = 1.000 exactly.

A bounded verifier has no gradient left at its own ceiling, and every selection decision is taken there.
So "our hand-built criterion loses to an off-the-shelf model at matched coverage" may be answering the wrong
question: two rulers that saturate in DIFFERENT places are complementary, not competing, and this project
already measured their agreement at only +0.395 -- "they are not measuring the same thing".

TWO DEFECTS IN THE 2026-08-18 NOVELTY RUN, FOUND ON REVIEW, FIXED HERE:

  D2  THE COVERAGE MATCHING WAS NOT MATCHED. `gate()` walked distinct values and kept whole tie blocks, and
      RNet's f1c has a 1.000 tie block covering 42.7% of the novel half. So "20% coverage" compared a panel
      gate that really took 20% against an RNet gate that took 42.7%, and the run's own output then asserted
      a direction for that mismatch without testing it. Fixed by breaking ties at RANDOM, averaged over 200
      draws, which is what a user holding only that ruler would actually have to do; coverage then matches
      exactly at every level and the comparison is defined where it previously was not.
  D3  DUPLICATE DESIGNS. 31 (puzzle, sequence) pairs occur more than once in the sample (96 rows, 2.2%), and
      every one of the 31 carries INCONSISTENT wet-lab scores across its copies. Deduplicated by averaging
      here. The spread between copies of one molecule is also a free estimate of the tube's own replicate
      noise, which the certificate's `< 80` cut has never been read against, so it is reported.

WHAT IS ASKED:

    G1  the duplicate spread = the tube's replicate noise, against the 80 cut the certificate uses.
    G2  saturation profile of BOTH rulers: within target, the rank correlation with the tube computed inside
        terciles of each ruler's OWN value. The panel half is a replication; the RNet half is new.
    G3  THE COMPLEMENTARITY TEST. Inside RNet's saturated block (f1c >= 0.999) does the panel still predict
        the tube? And inside the panel's own confirmed block does RNet? Per target, then paired.
    G4  matched coverage done properly, three gates: panel alone, RNet alone, and RNet-then-panel (the panel
        breaking RNet's ties). The third is the object the 0-of-5 stacking loss was about.

Struct2SeQ-SHAPE excluded throughout (it optimises against RibonanzaNet). Still the RECALL column: the
precision-aware version needs `panel_f1_designs.py`, which is a separate run. That is the remaining known
defect in this comparison and it is not fixed here.

EXPLORATORY, NOT PRE-REGISTERED. Same standing as novelty_stratified_ruler.py: nothing here may enter the
capability table, and a positive G3 is a hypothesis whose confirmatory version needs a declared block
boundary and a target holdout.

COST: $0. Reads two files.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

TUBE_FAIL = 80.0
SAT = 0.999
RNET_OPTIMISED = "Struct2SeQ-SHAPE"


def load(src_path, seed, panel_src=None, panel_key="panel_f1"):
    """Replay the sample, attach sequences, drop the RNet-optimised arm, deduplicate by molecule.

    `panel_src` swaps the panel column for one of `panel_f1_designs.py`'s -- the point of that run being
    that the shipped column is RECALL while RNet's is F1, so the two rulers were never the same statistic.
    The join is by position, which is only sound if that run lost no design; it is asserted, not assumed.
    """
    from novelty_stratified_ruler import replay_sample
    src = json.load(open(src_path))
    jobs, wild = replay_sample(seed, src["per_target_sampled"])
    assert len(jobs) == len(src["designs"]), "sample replay does not line up with the stored adjudication"
    alt = None
    if panel_src:
        pf = json.load(open(panel_src))
        if pf["n_failed"] or len(pf["designs"]) != len(jobs):
            raise SystemExit(f"panel_f1 run lost {pf['n_failed']} designs ({len(pf['designs'])} of "
                             f"{len(jobs)}); a positional join is not sound -- re-run it storing sequences")
        alt = pf["designs"]
        for j, d in zip(jobs, alt):                      # independently failable: same rows, same order
            assert d["puzzle"] == j[0] and d["method"] == j[4] and abs(d["wet"] - j[3]) < 1e-9, \
                "panel_f1 join mismatch"
    raw = []
    for i, ((p, s, f1c, wet, m), d) in enumerate(zip(jobs, src["designs"])):
        assert d["puzzle"] == p and d["method"] == m and abs(d["f1c"] - f1c) < 1e-9, "join mismatch"
        if m == RNET_OPTIMISED:
            continue
        panel = alt[i][panel_key] if alt else d["panel_mean"]
        raw.append({"puzzle": p, "seq": s, "method": m, "wet": wet, "f1c": f1c,
                    "panel": panel, "k": d["panel_k"], "confirmed": d["confirmed"]})
    groups = collections.defaultdict(list)
    for r in raw:
        groups[(r["puzzle"], r["seq"])].append(r)
    dedup, spreads = [], []
    for _, g in groups.items():
        if len(g) > 1:
            spreads.append(max(x["wet"] for x in g) - min(x["wet"] for x in g))
            assert len({round(x["panel"], 9) for x in g}) == 1, "same molecule, different panel score"
        r = dict(g[0])
        r["wet"] = sum(x["wet"] for x in g) / len(g)
        r["n_copies"] = len(g)
        dedup.append(r)
    return raw, dedup, spreads


def gate_fail(rs, key, cov, draws, rng, second=None):
    """Expected tube-failure rate of taking the top `cov` fraction under `key`, ties broken at random.

    A tie block is where a ruler has no opinion, so the honest cost of using that ruler at that coverage is
    the average over the choices it leaves open -- not the whole block's rate, and not a lucky subset.
    `second` optionally breaks ties by another key first, which is what stacking two rulers means.
    """
    k = max(1, round(cov * len(rs)))
    acc = []
    for _ in range(draws):
        order = sorted(rs, key=lambda r: (-r[key], -(r[second] if second else 0), rng.random()))
        sel = order[:k]
        acc.append(sum(1 for r in sel if r["wet"] < TUBE_FAIL) / k)
    return sum(acc) / len(acc), k / len(rs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "criterion_vs_tube.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "ruler_gradient.json"))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--panel-src", default=None,
                    help="panel_f1_designs.json -- swaps the RECALL column for a precision-aware one")
    ap.add_argument("--panel-key", default="panel_f1",
                    choices=["panel_f1", "panel_f1_max", "panel_precision", "panel_recall",
                             "panel_recall_product"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from m2r_pairwise import spearman, wilcoxon
    rng = random.Random(a.seed)
    raw, rec, spreads = load(a.src, a.seed, a.panel_src, a.panel_key)
    print(f"面板列 = {a.panel_key if a.panel_src else 'panel_mean (召回,出厂列)'}")
    print(f"抽样 {len(raw)} 条 → 去重后 {len(rec)} 条分子(排除 {RNET_OPTIMISED})")

    # ---- G1 -------------------------------------------------------------------------------------------
    print("\n=== G1 同一条分子的重复测量:试管自己的噪声,对着证书用的 80 分线读 ===")
    g1 = {"n_duplicated_molecules": len(spreads)}
    if spreads:
        g1.update({"wet_spread_median": st.median(spreads), "wet_spread_max": max(spreads)})
        print(f"  {len(spreads)} 条分子有重复测量:同分子两次读数之差 中位 {st.median(spreads):.1f} 分,"
              f"最大 {max(spreads):.1f} 分")
        near = [r for r in rec if abs(r["wet"] - TUBE_FAIL) <= st.median(spreads)]
        g1["frac_within_noise_of_cut"] = len(near) / len(rec)
        print(f"  离 80 分线不到一个中位噪声的分子占 {len(near)/len(rec):.1%}"
              f" —— 这一段的『通过/失败』标签本身是抛硬币")

    by = collections.defaultdict(list)
    for r in rec:
        by[r["puzzle"]].append(r)

    # ---- G2 -------------------------------------------------------------------------------------------
    print("\n=== G2 两把尺子各自的饱和曲线(靶标内,按尺子自己的取值分三档)===")
    g2 = {}
    for key, label in (("panel", "面板召回"), ("f1c", "RNet F1")):
        prof = {"low": [], "mid": [], "high": []}
        for p, rs in by.items():
            if len(rs) < 24:
                continue
            srt = sorted(rs, key=lambda r: r[key])
            n = len(srt) // 3
            for name, chunk in (("low", srt[:n]), ("mid", srt[n:2 * n]), ("high", srt[2 * n:])):
                if len({r["wet"] for r in chunk}) < 3 or len({r[key] for r in chunk}) < 2:
                    continue
                s = spearman([r[key] for r in chunk], [r["wet"] for r in chunk])
                if s is not None:
                    prof[name].append(s)
        g2[key] = {k: {"n": len(v), "median": st.median(v) if v else None} for k, v in prof.items()}
        print(f"  {label:8s} " + "   ".join(
            f"{k} {g2[key][k]['median']:+.3f} (n={g2[key][k]['n']})" for k in ("low", "mid", "high")))
    print("  读法:高档趋近 0 = 这把尺子在自己的顶端没有梯度了,而所有选择都发生在顶端。")

    # ---- G3 -------------------------------------------------------------------------------------------
    print("\n=== G3 互补性:在对方饱和的那一块里,自己还有没有梯度 ===")
    g3 = {}
    blocks = {
        "inside_rnet_saturated": (lambda rs: [r for r in rs if r["f1c"] >= SAT], "panel", "f1c"),
        "inside_panel_confirmed": (lambda rs: [r for r in rs if r["confirmed"]], "f1c", "panel"),
        "inside_panel_saturated": (lambda rs: [r for r in rs if r["panel"] >= SAT], "f1c", "panel"),
    }
    for name, (sel, probe, owner) in blocks.items():
        rows, tot = [], 0
        for p, rs in sorted(by.items()):
            b = sel(rs)
            tot += len(b)
            if len(b) < 10 or len({r["wet"] for r in b}) < 3 or len({r[probe] for r in b}) < 2:
                continue
            s = spearman([r[probe] for r in b], [r["wet"] for r in b])
            if s is not None:
                rows.append((p, s, len(b)))
        d = {"block_owner": owner, "probe": probe, "n_designs_in_block": tot, "n_targets": len(rows)}
        if rows:
            v = [s for _, s, _ in rows]
            w, p2, nz = wilcoxon(v, "two-sided")
            d.update({"rho_median": st.median(v), "positive": sum(1 for s in v if s > 0),
                      "p_two_sided": p2})
            print(f"  {name:24s} 块内 {tot:5d} 条,{len(rows):2d} 个靶标够测   "
                  f"rho({probe},试管) 中位 {st.median(v):+.3f}   为正 {sum(1 for s in v if s>0)}/{len(v)}"
                  + (f"   双侧 p = {p2:.4g}" if p2 is not None else ""))
        else:
            print(f"  {name:24s} 块内 {tot:5d} 条 —— 没有靶标够测,不给数")
        g3[name] = d

    # ---- G4 -------------------------------------------------------------------------------------------
    print("\n=== G4 真正匹配覆盖率(并列随机打破,200 次平均);第三行是把面板叠在 RNet 上 ===")
    base = sum(1 for r in rec if r["wet"] < TUBE_FAIL) / len(rec)
    print(f"  全体 {len(rec)} 条分子,不筛的失败率 {base:.1%}")
    g4 = {"base_fail": base, "n": len(rec), "points": []}
    for cov in (0.1, 0.2, 0.3, 0.5):
        fp, cp = gate_fail(rec, "panel", cov, a.draws, rng)
        fr, cr = gate_fail(rec, "f1c", cov, a.draws, rng)
        fs, cs = gate_fail(rec, "f1c", cov, a.draws, rng, second="panel")
        print(f"  覆盖 {cov:.0%}(实际 {cp:.1%})   面板 {fp:.1%}   RNet {fr:.1%}   "
              f"RNet+面板打破并列 {fs:.1%}   |   叠加带来 {fs - fr:+.1%}")
        g4["points"].append({"cov": cov, "realized": cp, "panel": fp, "rnet": fr, "stack": fs,
                             "stack_minus_rnet": fs - fr, "panel_minus_rnet": fp - fr})
    # per-target paired version of the stacking question, at 30%
    pd = []
    for p, rs in sorted(by.items()):
        if len(rs) < 24 or len({r["wet"] < TUBE_FAIL for r in rs}) < 2:
            continue
        fr, _ = gate_fail(rs, "f1c", 0.3, 60, rng)
        fs, _ = gate_fail(rs, "f1c", 0.3, 60, rng, second="panel")
        pd.append(fs - fr)
    if pd:
        w, p2, nz = wilcoxon(pd, "two-sided")
        better = sum(1 for v in pd if v < 0)
        print(f"\n  逐靶标配对(30% 覆盖,叠加 − 只用 RNet):叠加更低失败率 {better}/{len(pd)}"
              f"   中位 {st.median(pd):+.1%}" + (f"   双侧 p = {p2:.4g}" if p2 is not None else ""))
        g4["paired_stack_cov30"] = {"n": len(pd), "better": better, "median": st.median(pd),
                                    "p_two_sided": p2}

    # ---- G5 -------------------------------------------------------------------------------------------
    # THE DEFECT G4 EXPOSES IN AN EXISTING RESULT. G4 is POOLED across targets and methods, and its panel
    # column is non-monotone in coverage (17.2% at 10% -> 28.9% at 20%), which is the signature of whole
    # cells entering the selection rather than of a ruler getting worse. `filter_stack.py` -- the script
    # behind the "RNet wins at every operating point, the panel adds nothing at 0 of 5" row -- pools the
    # same way, with no target or method control, on this same file. This project already retracted a
    # +4.75 for exactly that reason (60% of it was method composition, controlled value +1.90) and
    # `panel_operating_point.py` therefore declares the (target, method) cell as the unit. So the
    # comparison is redone here inside cells, which is the only estimator this project accepts for it.
    print("\n=== G5 同一比较放回 (靶标 × 方法) 格子里做,再跨格子配对 ===")
    cells = collections.defaultdict(list)
    for r in rec:
        cells[(r["puzzle"], r["method"])].append(r)
    usable = {c: rs for c, rs in cells.items()
              if len(rs) >= 10 and 0 < sum(1 for r in rs if r["wet"] < TUBE_FAIL) < len(rs)}
    print(f"  {len(cells)} 个格子,其中 {len(usable)} 个有 ≥10 条且两类结局都出现")
    g5 = {"n_cells": len(cells), "n_usable": len(usable), "points": []}
    for cov in (0.3, 0.5):
        dp, ds = [], []
        for c, rs in sorted(usable.items()):
            fp, _ = gate_fail(rs, "panel", cov, 60, rng)
            fr, _ = gate_fail(rs, "f1c", cov, 60, rng)
            fs, _ = gate_fail(rs, "f1c", cov, 60, rng, second="panel")
            dp.append(fp - fr)
            ds.append(fs - fr)
        out = {"cov": cov, "n": len(dp)}
        for name, d in (("panel_minus_rnet", dp), ("stack_minus_rnet", ds)):
            w, p2, nz = wilcoxon(d, "two-sided")
            better = sum(1 for v in d if v < 0)
            out[name] = {"median": st.median(d), "panel_or_stack_better": better,
                         "worse": sum(1 for v in d if v > 0), "p_two_sided": p2}
            print(f"  覆盖 {cov:.0%}  {name:18s} 中位 {st.median(d):+.1%}"
                  f"   更好 {better} / 更差 {sum(1 for v in d if v>0)} / 平 {len(d)-better-sum(1 for v in d if v>0)}"
                  + (f"   双侧 p = {p2:.4g}" if p2 is not None else ""))
        g5["points"].append(out)
    print("  读法:G4 是混着算的,G5 是格内算的。两者不一致时,按本项目自己的口径,G5 才算。")

    json.dump({"exploratory": True, "pre_registered": False,
               "panel_column": (a.panel_key if a.panel_src else "panel_mean_recall"), "G5": g5,
               "note": "Fixes D2 (tie-aware matched coverage) and D3 (duplicate molecules) of "
                       "novelty_stratified_ruler.py. The recall-vs-F1 asymmetry is NOT fixed here; see "
                       "panel_f1_designs.py.",
               "G1": g1, "G2": g2, "G3": g3, "G4": g4},
              open(a.out, "w"), indent=1, sort_keys=True)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
