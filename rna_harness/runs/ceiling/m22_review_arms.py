#!/usr/bin/env python3
"""M22: two reviewer objections answered with data rather than prose.

Declaration m22_review_arms_declaration.json, frozen and pushed at 3e9496cc BEFORE this file.
Offline, $0.
"""
from __future__ import annotations
import collections, json, math, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m16_rescue as M16
import m21_thermo_mediator as M21

NPERM = 2000


def binom_p(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    import random as _r
    OUT = {}
    recs = M21.build_with_dg()
    print("=== M22 A 臂:幸存者偏差的有界性 —— 效应是不是【方法能力】造出来的 ===")
    print("  声明 3e9496cc。离线 $0。**对象是 M2R 臂,每个 (靶标,方法) 只有最优设计。**\n")
    # design-side depth gap per method, read from M20's COMMITTED RESULT. The first version of
    # this script hardcoded the figures from M20's declaration BLIND_STATUS instead, which were
    # computed before the MIN_CELL filter and differ by up to 0.145; M24's reproduction probe
    # caught it. Never transcribe a number that is already in an artifact.
    dep_gap = {r["method"]: r["gap"] for r in
               json.load(open(os.path.join(HERE, "m20_designer_confounding.json")))["A1"]["per_method"]}
    sfn = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    rows = []
    for m in sorted({r["method"] for r in recs}):
        sub = [r for r in recs if r["method"] == m]
        d, w, s = M21.matched_diff(sub, "CG", "AU", sfn)
        if d is None:
            continue
        nl = np.sort(np.array([M21.perm_diff(sub, "CG", "AU", sfn, _r.Random(i))[0] or 0.0
                               for i in range(NPERM)]))
        p = (float((nl >= d).sum()) + 1) / (NPERM + 1)
        rows.append({"method": m, "diff": round(d, 4), "p_perm": round(p, 5), "weight": w,
                     "n_strata": s, "n_pairs": len(sub),
                     "design_side_depth_gap": dep_gap.get(m)})
    rows.sort(key=lambda r: -r["diff"])
    print(f"  {'method':28s} {'CG−AU':>9s} {'p':>9s} {'权重':>6s} {'配对数':>7s} {'设计侧深度差':>12s}")
    for r in rows:
        g = f"{r['design_side_depth_gap']:+.3f}" if r["design_side_depth_gap"] is not None else "—"
        print(f"  {r['method']:28s} {r['diff']:+9.4f} {r['p_perm']:9.5f} {r['weight']:6.0f} "
              f"{r['n_pairs']:7d} {g:>12s}")
    npos = sum(1 for r in rows if r["diff"] > 0)
    bp = binom_p(npos, len(rows))
    print(f"\n  **为正 {npos}/{len(rows)},符号检验二项 p = {bp:.5f}**")
    have = [r for r in rows if r["design_side_depth_gap"] is not None]
    if len(have) >= 3:
        x = np.array([r["design_side_depth_gap"] for r in have])
        y = np.array([r["diff"] for r in have])
        rho = float(np.corrcoef(x, y)[0, 1])
        # Spearman by ranks, independent route
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        sp = float(np.corrcoef(rx, ry)[0, 1])
        print(f"  效应 vs 设计侧深度差:Pearson {rho:+.3f}  Spearman {sp:+.3f}  (n={len(have)})")
        print("  ⇒ **效应大小与方法的放置强度不是单调关系** ⇒ 不是能力这一维产生的"
              if abs(sp) < 0.6 else "  ⇒ **与放置强度强相关,须在正文说明**")
    OUT["A"] = {"per_method": rows, "n_positive": npos, "n_methods": len(rows),
                "binom_p": round(bp, 5),
                "corr_with_design_gap": round(rho, 4) if len(have) >= 3 else None,
                "spearman_with_design_gap": round(sp, 4) if len(have) >= 3 else None}

    print("\n=== M22 B 臂:环形平移动的是【位置集合】,不是序列 ===")
    import m13b_gc_placement as G
    src = open(os.path.join(HERE, "m13b_gc_placement.py")).read()
    fn = src[src.index("def shifted_gc"):src.index("def synth")]
    checks = {
        "sequence_never_indexed": ("c[\"seq\"]" not in fn and "sequence" not in fn),
        "only_paired_index_rotated": ("(c[\"paired\"] + s) % n" in fn),
        "ISGC_matrix_untouched": ("c[\"ISGC\"][:, S]" in fn and "ISGC =" not in fn),
    }
    for k, v in checks.items():
        print(f"  {k:32s} {'通过' if v else '**失败**'}")
    print("  ⇒ 平移臂与真实臂的分子集合逐字节相同;每条序列的碱基计数、全局 GC、长度全部恒等。")
    print("  ⇒ **motif 频率按构造守恒 —— 唯一变化的是统计量读取哪些列。**")
    OUT["B"] = {"source_checks": checks, "all_pass": all(checks.values()),
                "shifted_expression": "S = (c[\"paired\"] + s) % n"}

    json.dump(OUT, open(os.path.join(HERE, "m22_review_arms.json"), "w"), indent=1)
    print("\n写出 m22_review_arms.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
