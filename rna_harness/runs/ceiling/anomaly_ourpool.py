#!/usr/bin/env python3
"""Apples to apples: does the panel rank OUR designs the way it ranks THEIRS?

RUNS IN THE RibonanzaNet venv.

WHY THIS IS THE DECISIVE ONE. `anomaly_ranker` killed all three of my explanations for why the panel is a
validated classifier and a bad ranker. Inside narrow (target, method) cells on other people's molecules the
panel's continuous margin still ranks the TUBE weakly but really (+0.073, 72/117, p = 0.001), while
RibonanzaNet ranks it at +0.249. Thresholding is not the cause (continuous minus discrete = +0.000,
p = 0.875), and the model-class split is underpowered at 7 learned cells and points the wrong way.

So the failure may be specific to OUR designs -- but that has never been measured against theirs on the
SAME quantity with the SAME judge. Here it is: within-stratum Spearman between the panel's continuous
margin and RibonanzaNet crossed-pair F1, computed on their designs and on ours, side by side. Their
stratum is a (target, method) cell; ours is one target's pool. The solver's own ordering is carried along
as the thing that beat the panel, so all three appear on one scale.

    PREDICTION IF THE ANOMALY IS REAL: panel-vs-RNet is clearly positive on their designs and at or below
    zero on ours, while pool order is negative (earlier is better) on ours.
    KILL: if the panel ranks our pools about as well as theirs, then nothing needs explaining and the
    selector result is only about the argmax rule, not about the criterion.

COST: $0.
"""
from __future__ import annotations
import collections, json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from order_signal import spearman
from m2r_pairwise import wilcoxon
from pool_rerank_score import _score_chunk
from selector_replay import load_dump, KNOT_VERDICTS

# --- theirs -------------------------------------------------------------------------------------------
D = json.load(open(os.path.join(HERE, "criterion_vs_tube.json")))["designs"]
cells = collections.defaultdict(list)
for d in D:
    cells[(d["puzzle"], d["method"])].append(d)
theirs = []
for k, ds in cells.items():
    if len(ds) < 8: continue
    pm = [d["panel_mean"] for d in ds]; f1 = [d["f1c"] for d in ds]
    if len(set(pm)) > 1 and len(set(f1)) > 1:
        theirs.append(spearman(pm, f1))

# --- ours ---------------------------------------------------------------------------------------------
R = json.load(open(os.path.join(HERE, "leaderboard_replay.json")))
T = {p: v["target"] for p, v in R["targets"].items()}
rows = {r["puzzle"]: r for r in R["rows"]}
P = load_dump(os.path.join(HERE, "pool_dump"), T)
elig = [p for p in P if rows.get(p, {}).get("verdict") in KNOT_VERDICTS]
jobs = [((p, i), r["seq"], T[p]) for p in elig for i, r in enumerate(P[p])]
from concurrent.futures import ProcessPoolExecutor
CH = max(1, len(jobs) // 160); sc = {}
print(f"给我们自己的 {len(jobs)} 条候选打分\n", flush=True)
with ProcessPoolExecutor(max_workers=40) as ex:
    for res in ex.map(_score_chunk, [jobs[i:i + CH] for i in range(0, len(jobs), CH)]):
        sc.update(dict(res))

ours_pm, ours_ord, ours_k = [], [], []
for p in elig:
    v = [(i, P[p][i]["mean_recall"], len(P[p][i]["confirmed_by"]), sc[(p, i)])
         for i in range(len(P[p])) if (p, i) in sc]
    if len(v) < 10: continue
    f1 = [x[3] for x in v]
    if len(set(f1)) < 3: continue
    if len(set(x[1] for x in v)) > 1: ours_pm.append(spearman([x[1] for x in v], f1))
    if len(set(x[2] for x in v)) > 1: ours_k.append(spearman([x[2] for x in v], f1))
    ours_ord.append(spearman([x[0] for x in v], f1))

def line(lab, v, alt="greater"):
    _, pv, _ = wilcoxon(v, alt)
    print(f"  {lab:34s} 中位 {st.median(v):+.3f}   为正 {sum(1 for x in v if x>0):3d}/{len(v):<3d}   "
          f"p = {'n/a' if pv is None else f'{pv:.3g}'}")

print(f"=== 同一个裁判(RibonanzaNet F1),同一个量(面板连续边距),两批分子 ===")
line(f"他们的设计,{len(theirs)} 个格子", theirs)
line(f"我们的池,{len(ours_pm)} 个靶标", ours_pm)
print()
line(f"我们的池:面板离散票数", ours_k)
line(f"我们的池:池内位置(求解器排序)", ours_ord, "less")
print(f"\n  位置那一行用的是【单边 less】—— 越靠前越好意味着相关为负,所以负值才是信号。")

diff_note = ("面板在【他们的】分子上排得动,在【我们的】分子上排不动 —— 反常是关于我们的设计"
             if st.median(theirs) > 0.05 and st.median(ours_pm) < 0.05 else
             "两边表现接近 —— 反常不成立,选择器的收益只关于 argmax 规则,不关于判据本身")
print(f"\n=== 判读 ===\n  {diff_note}")
json.dump({"theirs_panel_vs_rnet": st.median(theirs), "n_theirs": len(theirs),
           "ours_panel_vs_rnet": st.median(ours_pm), "n_ours": len(ours_pm),
           "ours_k_vs_rnet": st.median(ours_k), "ours_order_vs_rnet": st.median(ours_ord),
           "theirs_all": theirs, "ours_all": ours_pm},
          open(os.path.join(HERE, "anomaly_ourpool.json"), "w"), ensure_ascii=False, indent=1)
print(f"  wrote anomaly_ourpool.json")
