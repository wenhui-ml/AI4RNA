#!/usr/bin/env python3
"""M7: promote the line's strongest positive from a diagnostic to a verdict.

Pre-registered in m7_empirical_invariant_declaration.json.

WHY THIS IS OVERDUE. m5_audit.md section 7 promised exactly this run, and I then spent M5 and M6 on two more
negative-side family searches while the one positive result -- the empirical invariant splitting all four
populations at permutation p from 0.00005 to 0.00085 -- stayed a diagnostic with no declaration, no placebo
arm and no verdict. That is the third costume of the same failure: not writing "refuted" any more, not
discounting positives with post-hoc criteria any more, but letting negative-side elimination keep queue
priority over consolidating the positive.

THE ATTRIBUTION QUESTION THE PLACEBO DECIDES. R6 already showed that on the unmatched construction the
pre-registered placebo ed_mfe reproduces across populations at rho +0.397 against the real column's +0.437,
and that its own split separates. So "the other populations' g_t splits the library" has to be shown not to
be "any ensemble-concentration column's g_t splits the library". Arm P runs the placebo through the identical
pipeline; arm D runs the paired difference, which cancels generic concentration split by split; arm N runs
twelve pure-noise columns, because one control is not a calibration.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block                                      # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402
from m2_invariant_split import half_stats                                         # noqa: E402
from m2b_gt_target_property import fold                                           # noqa: E402
from m2b_review_audit import acc_pair_of_blocks, perm_diff_p                      # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"),
        ("第五批(本线未读)", "panel_f1_fifth.json"), ("第六批(本线未读)", "panel_f1_sixth.json"))
CORE = [f for _n, f in POPS[:4]]
PRIMARY = "panel_f1_fresh.json"
R_NOISE = 12
MIN_HALF = 15


def g_all_arms(keep, M, splits, seed, noise_seed):
    """Per-target increments for every arm off ONE set of splits: ed, mfe, paired, and R_NOISE noise arms."""
    cells, cols = collections.defaultdict(list), {}
    rngn = random.Random(noise_seed)
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = (base_block(r) + m["C"] + [m["ed_skel"], m["ed_mfe"]]
                        + [rngn.gauss(0, 1.0) for _ in range(R_NOISE)])
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    if not rws:
        return {}, 0
    nB = len(base_block(keep[0]))
    iBC = list(range(nB + 2))
    idx = {"ed": iBC + [nB + 2], "mfe": iBC + [nB + 3]}
    for j in range(R_NOISE):
        idx[f"n{j}"] = iBC + [nB + 4 + j]
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {t: [] for t in tg} for k in list(idx) + ["paired"]}
    rng = random.Random(seed); order = list(tg); leak = 0
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            leak += 1
        PA = fit_predict(TR, TE, cols, iBC)
        PX = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx.items()}
        for t in te:
            base = None; vals = {}
            okpairs = True
            for k, P in PX.items():
                aA, aB, nA, _n, _t2, _a, _b = acc_pair_of_blocks(by_t[t], PA, P, "common05")
                if aA is None or aB is None or nA < H.MIN_PAIRS_PER_TARGET:
                    okpairs = False; break
                base = aA; vals[k] = aB - aA
            if not okpairs:
                continue
            for k, v in vals.items():
                per[k][t].append(v)
            per["paired"][t].append(vals["ed"] - vals["mfe"])
    out = {}
    for k in per:
        out[k] = {t: st.fmean(v) for t, v in per[k].items() if len(v) >= H.MIN_SPLITS_PER_TARGET}
    return out, leak


def split_eval(G, arm, fn, others, seed):
    """Select targets by `arm` measured on `others`, evaluate the SAME arm's g on `fn`. Frozen: median."""
    sel = {}
    for t in G[fn][arm]:
        v = [G[o][arm][t] for o in others if arm in G[o] and t in G[o][arm]]
        if len(v) >= 2:
            sel[t] = st.fmean(v)
    if len(sel) < 2 * MIN_HALF:
        return None
    thr = st.median(list(sel.values()))
    hi = [G[fn][arm][t] for t in sel if sel[t] > thr]
    lo = [G[fn][arm][t] for t in sel if sel[t] <= thr]
    if min(len(hi), len(lo)) < MIN_HALF:
        return {"void_half": [len(hi), len(lo)]}
    d, p = perm_diff_p(list(hi), list(lo))
    rb = random.Random(seed + 77); bs = []
    for _ in range(5000):
        A = [hi[rb.randrange(len(hi))] for _ in range(len(hi))]
        B = [lo[rb.randrange(len(lo))] for _ in range(len(lo))]
        bs.append(st.fmean(A) - st.fmean(B))
    bs.sort(); ci = [round(bs[125], 4), round(bs[4874], 4)]
    ex = 0
    for s2 in range(50):
        r2 = random.Random(3300 + 13 * s2); dd = []
        for _ in range(1500):
            A = [hi[r2.randrange(len(hi))] for _ in range(len(hi))]
            B = [lo[r2.randrange(len(lo))] for _ in range(len(lo))]
            dd.append(st.fmean(A) - st.fmean(B))
        dd.sort()
        ex += 1 if (dd[37] > 0 or dd[1462] < 0) else 0
    return {"thr": round(thr, 5), "n_hi": len(hi), "n_lo": len(lo),
            "hi": round(st.fmean(hi), 4), "lo": round(st.fmean(lo), 4), "diff": round(d, 4),
            "ci": ci, "excl": bool(ci[0] > 0 or ci[1] < 0), "perm_p": p, "seed_flip": f"{ex}/50"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m7_empirical_invariant.json"))
    a = ap.parse_args()
    G, R, SETS = {}, {}, {}
    usable = []
    print("=== M7 把经验不变量从【诊断】升为【裁定】;真列 / 安慰剂 / 配对差 / 12 根噪声,同批划分 ===", flush=True)
    for j, (nm, fn) in enumerate(POPS):
        keep, M = fold(fn, a.workers)
        SETS[fn] = {(r["puzzle"], r["method"], r["seq"]) for r in keep}
        G[fn], lk = g_all_arms(keep, M, a.splits, a.seed, 7700 + 131 * j)
        n_ed = len(G[fn].get("ed", {}))
        if n_ed >= H.MIN_TARGETS:
            usable.append(fn)
        print(f"  {nm:16s} 靶标(ed) {n_ed:3d}  泄漏 {lk}/{a.splits} {'✓' if not lk else '✗'}"
              f"  {'可用' if n_ed >= H.MIN_TARGETS else '**靶标不足 cluster 地板,本人群跳过**'}", flush=True)

    print("\n=== 闸门 V1:分子级零重叠 ===")
    worst = 0
    for i, (n1, f1) in enumerate(POPS):
        for n2, f2 in POPS[i + 1:]:
            k = len(SETS[f1] & SETS[f2]); worst = max(worst, k)
    print(f"  十五对最大重叠 {worst}  {'✓ 零重叠' if worst == 0 else '✗ -> 作废'}")
    R["overlap_max"] = worst
    if worst:
        R["verdict"] = "void_overlap"; json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False); return 0

    print("\n=== 切分评估(选择只用其他人群;阈值 = 选择集中位数,看结果前定下)===")
    res = {}
    for arm in ["ed", "mfe", "paired"] + [f"n{j}" for j in range(R_NOISE)]:
        res[arm] = {}
        for nm, fn in POPS:
            if fn not in usable:
                continue
            others = [o for o in usable if o != fn]
            if len(others) < 2:
                continue
            r = split_eval(G, arm, fn, others, a.seed)
            if r:
                res[arm][nm] = r
        if arm in ("ed", "mfe", "paired"):
            lbl = {"ed": "**真列 g_ed(主臂)**", "mfe": "**安慰剂 g_mfe**", "paired": "**配对差 g_ed−g_mfe**"}[arm]
            print(f"\n  {lbl}")
            for nm, r in res[arm].items():
                if "void_half" in r:
                    print(f"    {nm:16s} 半边不足 15 -> 作废 {r['void_half']}"); continue
                print(f"    {nm:16s} 高半 {r['hi']:+.4f}(n {r['n_hi']}) 低半 {r['lo']:+.4f}(n {r['n_lo']})"
                      f"  半间差 {r['diff']:+.4f}  95% {r['ci']}  置换 p {r['perm_p']:.5f}"
                      f"  {'**排除 0**' if r['excl'] else '含 0'}  [种子 {r['seed_flip']}]")
    R["splits"] = res

    print(f"\n=== 噪声臂({R_NOISE} 根,同批划分同一套切分流程;第 53 条只数同侧)===")
    ns = []
    for j in range(R_NOISE):
        rj = res[f"n{j}"].get("样本2(主)")
        if rj and "diff" in rj:
            ns.append(rj)
    pos = sum(1 for r in ns if r["excl"] and r["ci"][0] > 0)
    neg = sum(1 for r in ns if r["excl"] and r["ci"][1] < 0)
    print(f"  主人群上 {len(ns)} 根噪声:半间差中心 {st.fmean([r['diff'] for r in ns]):+.5f}"
          f"  | **同侧(正) {pos}/{len(ns)}**,负 {neg}/{len(ns)}")
    R["noise"] = {"n": len(ns), "same_side": pos, "opposite": neg,
                  "center": round(st.fmean([r["diff"] for r in ns]), 5) if ns else None}

    print("\n=== 判定(冻结的打死条件)===")
    K0 = pos >= 2
    ed = res["ed"]; mfe = res["mfe"]; pr = res["paired"]
    K1 = not ed.get("样本2(主)", {}).get("excl", False)
    rep = sum(1 for nm in ed if ed[nm].get("excl"))
    K2 = rep < 3
    rep_m = sum(1 for nm in mfe if mfe[nm].get("excl"))
    mean_ed = st.fmean([ed[nm]["diff"] for nm in ed if "diff" in ed[nm]])
    mean_mf = st.fmean([mfe[nm]["diff"] for nm in mfe if "diff" in mfe[nm]])
    K3 = (rep_m >= 3) and (abs(mean_mf) >= 0.5 * abs(mean_ed))
    K4 = not pr.get("样本2(主)", {}).get("excl", False)
    print(f"  K0 噪声也切得开(同侧 {pos}/{len(ns)}):{'**FIRES -> 整个实验作废**' if K0 else '不触发 ✓'}")
    if K0:
        R["verdict"] = "void_noise_splits_too"
        json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("  **这个切分流程会无中生有,前面所有基于它的诊断都要重看。**"); return 0
    print(f"  K1 主人群真列切不开:{'FIRES -> unresolved' if K1 else '不触发 ⇒ 排除 0'}")
    print(f"  K2 复现不足(真列排除 0 的人群 {rep}/{len(ed)}):{'FIRES' if K2 else '不触发'}")
    print(f"  K3 安慰剂也做得到(安慰剂 {rep_m}/{len(mfe)} 人群,均值 {mean_mf:+.4f} vs 真列 {mean_ed:+.4f}):"
          f"{'FIRES' if K3 else '不触发'}")
    print(f"  K4 相消后切不开(配对差主人群):{'FIRES' if K4 else '不触发'}")
    if K1:
        v = "unresolved_on_primary"
    elif K3:
        v = "splits_but_generic_concentration_does_too"
    elif K4:
        v = "splits_but_not_skeleton_specific"
    else:
        v = "empirical_invariant_splits_the_library"
    if K2:
        v += "__underreplicated"
    print(f"  裁定:{v}")
    if K3:
        print("  **强制标签口径:这不是『证伪』,是把主张的范围从 ensemble_defect 收到【集合集中度类】。**")
    print("  **口径:区间含 0 一律写 `unresolved`,绝不写 `refuted`。**")
    print("  **边界:本实验判【它成不成立】,不判【它是什么】—— 五个族三十五根列在 |rho| ≳ 0.18 的"
          "分辨力下都没说出它是靶标的什么性质。没有盲人群;第五/六批是『本条线未读』,H2 读过。不判上线。**")
    R["verdict"] = v
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
