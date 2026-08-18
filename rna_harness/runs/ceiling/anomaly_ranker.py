#!/usr/bin/env python3
"""The anomaly this project has not explained: the same criterion is a validated CLASSIFIER and a bad RANKER.

TWO FACTS, both measured, both on record, and not yet reconciled.

  F1  As a classifier the panel is externally validated. On 4,560 synthesised designs from 13 methods,
      designs it passes score +1.90 OpenKnot points higher in the tube than designs it rejects, within
      (target, method) cells, 41 of 58, p = 1.2e-4. That is the project's only wet-lab-anchored result.

  F2  As a ranker the panel is worse than trivial. On our own pools, replacing its grade with "take the
      solver's own first candidate clearing a loose screen" raises delivered crossed-pair F1 by +0.122
      (24 of 51 better, 8 worse, held out on 200 splits at +0.126).

These are not logically contradictory -- a test can separate strata and be useless within one -- but the
project has been asserting the reconciliation rather than measuring it. Three candidate resolutions, and
they make DIFFERENT predictions on data already on disk:

  R-RANGE       Range restriction. The panel separates a wide population (3DRNA at 0% confirm against
                Struct2SeQ at 70.6%) and has no resolving power inside one narrow, homogeneous pool.
                PREDICTS: within (target, method) cells -- narrow strata -- the panel should rank wet lab
                POORLY for everyone, not just for us.

  R-GOODHART    Our designs are out of distribution for the panel: DesiRNA optimises an energy objective
                built on the same nearest-neighbour parameters two of the three predictors use, so our
                candidates are pre-adapted to it and its signal is exhausted on them.
                PREDICTS: the panel should rank BETTER inside methods that do NOT share that model class
                (Struct2SeQ, gRNAde, MPNN -- learned) than inside those that do (Rosetta, 3DRNA, Eterna
                players optimising against folding engines).

  R-THRESHOLD   The information is in the continuous margin and we destroy it by thresholding at tau = 1.0
                and counting votes.
                PREDICTS: continuous `panel_mean` should rank wet lab better than discrete `panel_k`,
                inside the same cells.

Independent unit is the (target, method) cell, never the design -- pooling designs from one molecule is the
pseudo-replication that produced this project's retracted p = 2.05e-28. Cells need >= 8 designs and some
spread in the panel score, or a Spearman is undefined or meaningless.

COST: $0. Everything is already on disk.
"""
from __future__ import annotations
import collections, json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from order_signal import spearman
from m2r_pairwise import wilcoxon

# Which model class each arm's designs came from, declared before looking at the result.
LEARNED = {"Struct2SeQ", "Struct2SeQ-SHAPE", "gRNAde", "gRNAde-no3d", "MPNN-fixbb", "MPNN-RFdiff",
           "codesign-RFdiff"}
PHYSICS = {"Rosetta", "Rosetta-LoRes", "3DRNA", "M2", "Eterna", "Starting sequence"}

D = json.load(open(os.path.join(HERE, "criterion_vs_tube.json")))["designs"]
cells = collections.defaultdict(list)
for d in D:
    cells[(d["puzzle"], d["method"])].append(d)

rows = []
for (p, m), ds in sorted(cells.items()):
    if len(ds) < 8:
        continue
    wet = [d["wet"] for d in ds]
    if len(set(wet)) < 3:
        continue
    r = {"puzzle": p, "method": m, "n": len(ds)}
    for name, key in (("mean", "panel_mean"), ("k", "panel_k"), ("f1c", "f1c")):
        v = [d[key] for d in ds]
        r[name] = spearman(v, wet) if len(set(v)) > 1 else None
    rows.append(r)

print(f"{len(rows)} 个 (靶标, 方法) 格子,每格 >= 8 条设计\n")
print("=== R-RANGE:面板在【任何人的】窄格子里排得动吗 ===")
for name, lab in (("mean", "panel 连续边距"), ("k", "panel 离散票数"), ("f1c", "RibonanzaNet F1")):
    v = [r[name] for r in rows if r[name] is not None]
    _, pv, _ = wilcoxon(v, "greater")
    print(f"  {lab:18s} 格内 Spearman 中位 {st.median(v):+.3f}  为正 {sum(1 for x in v if x>0)}/{len(v)}  "
          f"p = {'n/a' if pv is None else f'{pv:.3g}'}")
print("  -> 若三者都接近 0,则窄格子里谁都排不动,R-RANGE 成立,我们池里的失败不特殊")
print("  -> 若 RNet 排得动而 panel 排不动,则 R-RANGE 不足以解释,面板本身有问题\n")

print("=== R-GOODHART:面板在【不共享热力学模型类】的方法里排得更好吗 ===")
out = {}
for fam, names in (("学习型", LEARNED), ("物理型", PHYSICS)):
    v = [r["mean"] for r in rows if r["method"] in names and r["mean"] is not None]
    if v:
        out[fam] = v
        print(f"  {fam:5s} {len(v):3d} 格  panel 连续边距中位 {st.median(v):+.3f}  为正 {sum(1 for x in v if x>0)}/{len(v)}")
if len(out) == 2:
    a, b = out["学习型"], out["物理型"]
    print(f"  差 {st.median(a)-st.median(b):+.3f}  -> "
          f"{'与 R-GOODHART 一致' if st.median(a) > st.median(b) + 0.05 else '不支持 R-GOODHART'}")

print("\n=== R-THRESHOLD:连续边距 vs 离散票数,同格配对 ===")
pair = [(r["mean"], r["k"]) for r in rows if r["mean"] is not None and r["k"] is not None]
d = [a - b for a, b in pair]
_, pv, _ = wilcoxon(d, "greater")
print(f"  {len(pair)} 个格子:连续 − 离散 中位 {st.median(d):+.3f}  更好 {sum(1 for x in d if x>0)}/{len(d)}  "
      f"p = {'n/a' if pv is None else f'{pv:.3g}'}")
print(f"  -> {'阈值化确实丢掉了信息' if st.median(d) > 0 and pv is not None and pv < 0.025 else '不支持 R-THRESHOLD:阈值化不是主因'}")

json.dump({"n_cells": len(rows), "rows": rows}, open(os.path.join(HERE, "anomaly_ranker.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"\n  wrote anomaly_ranker.json")
