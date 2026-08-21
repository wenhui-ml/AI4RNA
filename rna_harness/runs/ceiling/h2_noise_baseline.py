#!/usr/bin/env python3
"""How much does ONE USELESS COLUMN cost on the held-out metric? D2 says it is not zero.

H1/H2's K1 asked whether the increment excludes 0. On a held-out metric that is the wrong null: an extra
least-squares parameter fits training noise and DEGRADES test predictions, so a column carrying no
information lands BELOW zero, not at it. The in-sample estimator had a permutation null precisely to subtract
"what feature COUNT buys"; the held-out version never carried that correction across.

This measures the noise-column baseline on three populations so the size of the omission is bounded. K2 (the
paired difference against the MFE placebo) is already count-matched and is NOT affected -- that is the number
the verdict rests on.
"""
from __future__ import annotations
import json, os, random, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2_heldout_cluster_ci as H
from h2_review_diag import build, per_target

POPS = (("第六批(主,盲)", "panel_f1_sixth.json"), ("第五批(复现,盲)", "panel_f1_fifth.json"),
        ("基准4560(非盲)", "panel_f1_designs.json"))

out = {}
for nm, fn in POPS:
    rws, cols, nB = build(fn)
    mols = [r for _c, g in rws for r in g]
    rng = random.Random(12345)
    for r in mols:
        cols[id(r)] = cols[id(r)] + [rng.gauss(0, 1.0)]
    iBC = list(range(nB + 2))
    blocks = (("噪声列(无信息)", iBC + [nB + 4]), ("ed_mfe 安慰剂", iBC + [nB + 3]),
              ("ed_skel 处理组", iBC + [nB + 2]))
    print(f"\n=== {nm} 分子 {len(mols)} 格 {len(rws)} ===", flush=True)
    row = {}
    for tag, idx in blocks:
        per, used, _ns = per_target(rws, cols, iBC, idx, splits=200, seed=0)
        g = [st.fmean(per[t]) for t in used]
        ci, n = H.boot_ci(g, 17)
        row[tag] = {"mean": round(st.fmean(g), 4), "ci": ci, "n": n}
        print(f"  {tag:16s} {st.fmean(g):+.4f}  95% {ci}"
              f"  {'排除 0' if ci[0] > 0 or ci[1] < 0 else '含 0'}", flush=True)
    z = row["噪声列(无信息)"]["mean"]
    print(f"  --- 相对【无信息列】基线 {z:+.4f} 校正后 ---")
    for tag in ("ed_mfe 安慰剂", "ed_skel 处理组"):
        print(f"  {tag:16s} {row[tag]['mean'] - z:+.4f}")
    row["_corrected"] = {t: round(row[t]["mean"] - z, 4) for t in ("ed_mfe 安慰剂", "ed_skel 处理组")}
    out[nm] = row
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "h2_noise_baseline.json"),
                    "w"), indent=1, ensure_ascii=False)
print("\n写出 h2_noise_baseline.json")
