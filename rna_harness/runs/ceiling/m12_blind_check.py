#!/usr/bin/env python3
"""M12 replication on the two BLIND batches -- the test M7 and M8b both died on.

NOT pre-declared: m12_sequence_intrinsic_declaration.json scoped itself to the four core populations.
This applies the SAME frozen analysis, unchanged, to the fifth and sixth batches, which no experiment
in the S-block line has touched. Nothing here is tuned; the code is imported from m12, not rewritten.

If the increment dies here, `sequence_intrinsic_block_adds` does not generalise and the positive side
does not hold. Offline, $0.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import m12_sequence_intrinsic as M                                             # noqa: E402

BLIND = (("第五批(盲)", "panel_f1_fifth.json"), ("第六批(盲)", "panel_f1_sixth.json"))
SEED, WORKERS = 0, 24

print("=== M12 盲复现:第五批 / 第六批 ===")
print("  **不是预注册**:M12 声明的范围是四个核心人群。本文件把【同一份冻结的分析】原样搬到两个盲批次。")
print("  代码从 m12 导入,一行没改。\n")
OUT = {}
for nm, fn in BLIND:
    try:
        rows = M.build(fn, WORKERS)
    except Exception as exc:                                                    # noqa: BLE001
        print(f"=== {nm} ===  载入失败 {exc!r}\n"); continue
    if not rows:
        print(f"=== {nm} ===  可用分子 0 ⇒ 跳过\n"); continue
    y, X, tg, ci, ncell, groups = M.matrices(rows, M.GC + M.S1 + M.S2)
    nT = len(set(tg.tolist()))
    nb = len(rows[0]["B"])
    base = list(range(nb)) + [nb + i for i in range(len(M.GC))]
    i1 = [nb + len(M.GC) + i for i in range(len(M.S1))]
    i2 = [nb + len(M.GC) + len(M.S1) + i for i in range(len(M.S2))]
    print(f"=== {nm} ===  分子 {len(y)}  格 {ncell}  靶标 {nT}")
    if len(y) < 300 or nT < 12:
        print(f"  分子 {len(y)} / 靶标 {nT} 低于可跑下限 ⇒ 报出,不裁定\n")
        OUT[fn] = {"pop": nm, "n": len(y), "targets": nT, "skipped": True}
        continue
    nf = min(M.NFOLD, nT)
    fold = M.folds(tg, nf, SEED)
    pB = M.cv_pred(y, X, fold, base)
    r2b = 1 - float(((y - pB) ** 2).sum()) / float((y * y).sum())
    print(f"  基线 B+G 留出 R^2 = {r2b:.4f}  (占上限 {r2b/M.CEILING*100:.1f}%)   K折={nf}")
    rec = {"pop": nm, "n": int(len(y)), "cells": int(ncell), "targets": int(nT),
           "r2_base": round(r2b, 5), "arms": {}}
    for k, cols in (("S1", i1), ("S2", i2), ("S1+S2", i1 + i2)):
        p = M.cv_pred(y, X, fold, base + cols)
        d = M.dr2_boot(y, pB, p, tg, SEED)
        cap = M.capture(groups, pB, p, ci, y, SEED)
        cg = "--" if cap is None else f"{cap['gain']:+.4f}"
        rec["arms"][k] = {**d, "capture": cap}
        print(f"  **B+G+{k:<6s} dR2 = {d['dr2']:+.5f} [{d['ci'][0]:+.5f},{d['ci'][1]:+.5f}]**"
              f"  {'**排除0**' if d['excl_pos'] else '含 0 '}   留出 capture@10 增益 {cg}")
    # the same rule-72 permutation, unchanged
    import random
    rng = random.Random(SEED + 202); h = 0
    for _ in range(M.R_NOISE):
        E = X[:, i1 + i2].copy()
        for kk in range(int(ci.max()) + 1):
            m = np.where(ci == kk)[0]
            pm = list(m); rng.shuffle(pm)
            E[m] = X[:, i1 + i2][pm]
        h += M.arm_dr2(y, X, fold, base, E, tg, SEED)["excl_pos"]
    rec["P3"] = h
    print(f"  P3(第 72 条)格内打乱 S:正向排除 0 的 **{h}/{M.R_NOISE}**  {'✓' if h <= 1 else '✗'}")
    OUT[fn] = rec
    print()

live = [v for v in OUT.values() if not v.get("skipped")]
pos = sum(1 for v in live if v["arms"]["S1+S2"]["excl_pos"])
print("=== 盲复现结论 ===")
if not live:
    print("  **两个盲批次都低于可跑下限 ⇒ 盲复现不可得,`sequence_intrinsic_block_adds` 的外推状态是【未测】**")
else:
    print(f"  可裁定的盲人群 {len(live)},其中 S1+S2 正向排除 0 的 **{pos}/{len(live)}**")
    for v in live:
        print(f"    {v['pop']:<13s} dR2 {v['arms']['S1+S2']['dr2']:+.5f} "
              f"{v['arms']['S1+S2']['ci']}  {'排除0' if v['arms']['S1+S2']['excl_pos'] else '含 0'}")
    print(f"  ⇒ {'**盲复现成立**' if pos == len(live) else '**盲复现不成立 / 部分**'}")
json.dump(OUT, open(os.path.join(HERE, "m12_blind_check.json"), "w"), indent=1, ensure_ascii=False)
print(f"\n  -> m12_blind_check.json")
