#!/usr/bin/env python3
"""Audit H2's seven declared void gates for whether they CAN fail, and replace the vacuous ones.

Rule 38 says a gate must come with an input that does NOT fire it. The dual obligation -- an input that DOES
fire it -- is what caught g3's tautology and count_kx's vacuous domain check. Doing it to H2's own gates:

  gate 1 "no target leakage": `assert not (tr & te)` where tr=set(order[:cut]), te=set(order[cut:]). Those are
          slices of one duplicate-free list, so the intersection is empty BY CONSTRUCTION. VACUOUS.
          The risk it was meant to cover is a MOLECULE from a held-out target appearing in training -- that is
          what a wrong key or a cell-level split would cause, and the assert cannot see it. Replaced below.
  gate 6 "bootstrap unit is the target": `nb == len(used)` where boot_ci returns len(vec) and vec is the
          per-target list. VACUOUS (shown in h2_review_diag D3).
  gates 2 and 5 are inspection statements, not asserts -- they cannot fire at runtime at all.
  gates 3(a,b,c), 4, 7 are real: 3 dropped 423 (target,split) pairs and 6 targets, 4 reported 2.35% ties,
          7 checks var>0 and |corr|<0.999 and could fail on a mis-wired column.

WHAT THIS RUNS: the non-vacuous molecule-level leakage check over all 200 splits on both blind populations,
plus a deliberately broken variant (split by CELL instead of by TARGET) that must trigger it.
$0, offline.
"""
from __future__ import annotations
import collections, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h2_review_diag import build

def check(fn, splits=200, seed=0, break_it=False):
    rws, cols, _nB = build(fn)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t); order = list(tg)
    cell_order = list(range(len(rws)))
    rng = random.Random(seed)
    leaks_mol = leaks_tg = 0
    for _s in range(splits):
        if break_it:                       # deliberately wrong: split by CELL, so a target straddles
            rng.shuffle(cell_order)
            cut = max(1, int(0.7 * len(cell_order)))
            TR = [rws[i] for i in cell_order[:cut]]; TE = [rws[i] for i in cell_order[cut:]]
        else:
            rng.shuffle(order)
            cut = max(1, int(0.7 * len(order)))
            tr, te = set(order[:cut]), set(order[cut:])
            TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        mol_tr = {id(r) for _c, g in TR for r in g}
        mol_te = {id(r) for _c, g in TE for r in g}
        tg_tr = {c[0] for c, _g in TR}; tg_te = {c[0] for c, _g in TE}
        if mol_tr & mol_te:
            leaks_mol += 1
        if tg_tr & tg_te:
            leaks_tg += 1
    return leaks_mol, leaks_tg, len(rws), len(tg)

print("=== 非空转的泄漏闸门:分子级 + 靶标级,全 200 次划分 ===")
for nm, fn in (("第六批(主,盲)", "panel_f1_sixth.json"), ("第五批(复现,盲)", "panel_f1_fifth.json")):
    lm, lt, nc, nt = check(fn)
    print(f"  {nm}(格 {nc},靶标 {nt}):分子级泄漏 {lm}/200,靶标级泄漏 {lt}/200"
          f"  {'✓ 干净' if lm == 0 and lt == 0 else '✗'}", flush=True)

print("\n=== 第 38 条的对偶义务:构造一个【必须让它报警】的输入(按格切分,靶标会跨边) ===")
lm, lt, nc, nt = check("panel_f1_sixth.json", splits=50, break_it=True)
print(f"  故意按格切分(第六批,50 次):分子级泄漏 {lm}/50,靶标级泄漏 {lt}/50")
print(f"  {'✓ 闸门确实会报警,不是空转' if lt > 0 else '✗ 连故意打断都测不出来 -> 闸门无用'}")
print(f"  注:分子级 {lm}/50 —— 按格切分不会让同一【分子】两边都出现,"
      f"所以真正抓住错误的是【靶标级】那一半。两半都要有。")
