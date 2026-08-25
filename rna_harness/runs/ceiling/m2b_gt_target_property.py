#!/usr/bin/env python3
"""M2b: is g_t a property of the TARGET, or is the between-target spread estimation noise?

Pre-registered in m2b_gt_is_it_a_target_property_declaration.json, written after M2's K1 fired and before
anything in this file was computed.

WHY THIS AND NOT "WE FAILED TO FIND AN INVARIANT". M2 screened ten invariants, all of them functions of the
target's predicted pair set -- the same representation family the 20.1% ceiling review already named as the
bottleneck. Its screen had power at rho=0.535 and none at rho=0.25, so what it measured is "no STRONG
structural invariant", not "no invariant". The prior question is whether there is anything there to find:
if g_t reproduces across disjoint design populations, an invariant must exist and the empirical g_t is
already one; if it does not, the spread is noise and the pooled estimator was right all along. Both answers
are results. Neither is "not measurable".

Arm 2 turns the first answer into something usable: select targets by g_t measured on three populations,
evaluate on a fourth the selection never touched.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block, load_pop                            # noqa: E402
from e3_scrambled_reference import work                                           # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402
from m2_invariant_split import spearman, rho_ci, half_stats, R_SEED               # noqa: E402

NPERM = 5000
MIN_HALF = 15
RAND_SPLITS = 200


def fold(fn, workers):
    from leaderboard_design import targets
    T = targets()
    rec, _s, _m = load_pop(fn)
    keep = [r for r in rec if base_block(r) is not None]
    jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
    from concurrent.futures import ProcessPoolExecutor
    M = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for m in ex.map(work, jobs, chunksize=8):
            M.append(m)
    return keep, M


def g_from(keep, M, splits, seed, cell_half=None, half_seed=99):
    """Per-target held-out ensemble_defect gain. cell_half in {None,0,1} keeps only half the cells."""
    cells, cols = collections.defaultdict(list), {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["C"] + [m["ed_skel"]] + [m["ed_mfe"]]
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    if cell_half is not None:
        hr = random.Random(half_seed)
        pick = {}
        by_t0 = collections.defaultdict(list)
        for c, g in rws:
            by_t0[c[0]].append(c)
        for t, cs in by_t0.items():
            cs = sorted(cs); hr.shuffle(cs)
            for i, c in enumerate(cs):
                pick[c] = i % 2
        rws = [(c, g) for c, g in rws if pick[c] == cell_half]
    tg = sorted({c[0] for c, _g in rws})
    nB = len(base_block(keep[0]))
    iBC = list(range(nB + 2)); iED = iBC + [nB + 2]
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    per_t = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in (("BC", iBC), ("BC_ed", iED))}
        for t in te:
            a = {}; nt = None
            for k in P:
                v, n, _ti = H.pair_accuracy_rows(by_t[t], P[k])
                a[k] = v; nt = n
            if nt is None or nt < H.MIN_PAIRS_PER_TARGET or any(v is None for v in a.values()):
                continue
            per_t[t].append(a["BC_ed"] - a["BC"])
    return per_t


def gmean(per_t, floor=None):
    f = H.MIN_SPLITS_PER_TARGET if floor is None else floor
    return {t: st.fmean(v) for t, v in per_t.items() if len(v) >= f}


def halves_mc(per_t, floor=10):
    """Odd/even over the splits that held t out -- Monte-Carlo only, the UPPER bound on reliability."""
    A = {t: st.fmean(v[0::2]) for t, v in per_t.items() if len(v) >= 2 * floor}
    B = {t: st.fmean(v[1::2]) for t, v in per_t.items() if len(v) >= 2 * floor}
    return A, B


def pair_rho(a, b, seed, tag, level=0.95):
    ts = sorted(set(a) & set(b))
    if len(ts) < MIN_HALF:
        return {"n": len(ts), "skipped": True}
    x = [a[t] for t in ts]; y = [b[t] for t in ts]
    r = spearman(x, y)
    ci = rho_ci(x, y, seed, level)
    ex = 0
    for s in range(R_SEED):
        c = rho_ci(x, y, 6100 + 13 * s, level)
        ex += 1 if (c[0] > 0 or c[1] < 0) else 0
    return {"n": len(ts), "rho": round(r, 4), "ci": ci, "excl": bool(ci[0] > 0 or ci[1] < 0),
            "seed_flip": f"{ex}/{R_SEED}", "tag": tag}


def sb(r):
    """Spearman-Brown: a half-length reliability corrected to full length."""
    return 2 * r / (1 + r) if r > -1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m2b_gt_target_property.json"))
    a = ap.parse_args()
    PRIMARY = "panel_f1_fresh.json"
    POPS = (("样本2(主)", PRIMARY), ("基准4560", "panel_f1_designs.json"),
            ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
    res = {"declaration": "m2b_gt_is_it_a_target_property_declaration.json"}

    KE, ME, G, GA, GB, MCA, MCB = {}, {}, {}, {}, {}, {}, {}
    print("=== 折叠 + 三种 g_t(全量 / 格半分A / 格半分B)===", flush=True)
    for nm, fn in POPS:
        keep, M = fold(fn, a.workers)
        KE[fn], ME[fn] = keep, M
        pt = g_from(keep, M, a.splits, a.seed)
        G[fn] = gmean(pt)
        MCA[fn], MCB[fn] = halves_mc(pt)
        GA[fn] = gmean(g_from(keep, M, a.splits, a.seed, cell_half=0), floor=10)
        GB[fn] = gmean(g_from(keep, M, a.splits, a.seed, cell_half=1), floor=10)
        v = list(G[fn].values())
        print(f"  {nm:10s} 靶标 {len(G[fn]):3d}  均值 {st.fmean(v):+.4f}  sd {st.stdev(v):.4f}"
              f"  正号 {sum(1 for x in v if x>0)}/{len(v)}   格半分 A {len(GA[fn])} B {len(GB[fn])}", flush=True)

    # ---- V1: molecule-level overlap between populations must be zero ---------------------------------
    print("\n=== 闸门 V1 分子级重叠(puzzle, method, seq)===")
    sets = {fn: {(r["puzzle"], r["method"], r["seq"]) for r in KE[fn]} for _n, fn in POPS}
    ov, worst = {}, 0
    for i, (n1, f1) in enumerate(POPS):
        for n2, f2 in POPS[i + 1:]:
            k = len(sets[f1] & sets[f2]); ov[f"{n1}|{n2}"] = k; worst = max(worst, k)
            print(f"  {n1} ∩ {n2} = {k}")
    res["V1_overlap"] = ov
    V1 = worst == 0
    print(f"  {'✓ 零重叠,跨人群相关是真的交叉验证' if V1 else '✗ 有重叠 -> 跨人群 rho 作废'}")

    # ---- V3 / reliability -----------------------------------------------------------------------------
    print("\n=== 信度(arm1 的分母)===")
    rel = {}
    for nm, fn in POPS:
        rmc = pair_rho(MCA[fn], MCB[fn], a.seed + 5, "rel_MC")
        rcl = pair_rho(GA[fn], GB[fn], a.seed + 6, "rel_cell")
        rel[fn] = {"rel_MC_halflen": rmc, "rel_cell_halflen": rcl,
                   "rel_MC_full": round(sb(rmc.get("rho", 0)), 4) if "rho" in rmc else None,
                   "rel_cell_full": round(sb(rcl.get("rho", 0)), 4) if "rho" in rcl else None}
        print(f"  {nm:10s} rel_MC(奇偶划分,必须显著) rho {rmc.get('rho')} {rmc.get('ci')}"
              f" {'响 ✓' if rmc.get('excl') else '✗ 不响 -> 无检出力'}"
              f" | S-B 全长 {rel[fn]['rel_MC_full']}")
        print(f"  {'':10s} rel_cell(格半分,设计抽样)  rho {rcl.get('rho')} {rcl.get('ci')}"
              f" {'排除 0' if rcl.get('excl') else '含 0'} | S-B 全长 {rel[fn]['rel_cell_full']}"
              f"  [种子翻转 {rcl.get('seed_flip')}]")
    res["reliability"] = rel
    V3 = all(rel[fn]["rel_MC_halflen"].get("excl") for _n, fn in POPS)
    V2 = all(rel[fn]["rel_cell_halflen"].get("excl") for _n, fn in POPS)
    print(f"  V3(第47条 必响输入 rel_MC):{'✓ 全部响' if V3 else '✗ 有人群不响'}")
    print(f"  V2(信度可测 rel_cell):{'✓' if V2 else '✗ 有人群的信度区间含 0'}")

    # ---- arm1: cross-population rho -------------------------------------------------------------------
    print("\n=== 臂1 跨人群 g_t 的 Spearman rho(6 对)===")
    pairs = []
    for i, (n1, f1) in enumerate(POPS):
        for n2, f2 in POPS[i + 1:]:
            pr = pair_rho(G[f1], G[f2], a.seed + 11, f"{n1}|{n2}")
            den = None
            rc1, rc2 = rel[f1]["rel_cell_full"], rel[f2]["rel_cell_full"]
            if rc1 and rc2 and rc1 > 0 and rc2 > 0:
                den = round(math.sqrt(rc1 * rc2), 4)
            pr["attenuation_ceiling"] = den
            pairs.append(pr)
            print(f"  {n1:10s} vs {n2:10s} n {pr['n']:3d}  rho {pr.get('rho'):+.4f}  95% {pr.get('ci')}"
                  f"  {'排除 0' if pr.get('excl') else '含 0'}  [翻转 {pr.get('seed_flip')}]"
                  f"  衰减上限 {den}")
    res["arm1_pairs"] = pairs
    obs_mean = st.fmean([p["rho"] for p in pairs if "rho" in p])
    # permutation null on the mean of the six rhos
    rngp = random.Random(a.seed + 303)
    common = sorted(set.intersection(*[set(G[fn]) for _n, fn in POPS]))
    vecs = {fn: [G[fn][t] for t in common] for _n, fn in POPS}
    nulls = []
    for _ in range(NPERM):
        pv = {}
        for _n, fn in POPS:
            v = list(vecs[fn]); rngp.shuffle(v); pv[fn] = v
        rr = []
        for i, (_n1, f1) in enumerate(POPS):
            for _n2, f2 in POPS[i + 1:]:
                rr.append(spearman(pv[f1], pv[f2]))
        nulls.append(st.fmean(rr))
    nulls.sort()
    q95 = nulls[int(0.95 * NPERM)]
    print(f"  六对均值 rho {obs_mean:+.4f};置换零分布 95 分位 {q95:+.4f}"
          f"  ⇒ {'落在零点内' if obs_mean <= q95 else '**超出零点**'}")
    res["arm1_mean_rho"] = round(obs_mean, 4); res["arm1_null_q95"] = round(q95, 4)
    K1 = obs_mean <= q95

    # ---- arm2: empirical invariant split, selection never touches the primary --------------------------
    print("\n=== 臂2 经验不变量切分(选择只用另外三个人群)===")
    SEL = [fn for _n, fn in POPS if fn != PRIMARY]
    assert PRIMARY not in SEL, "V5:选择集包含了主人群"
    sel_g = {}
    for t in set().union(*[set(G[fn]) for fn in SEL]):
        vs = [G[fn][t] for fn in SEL if t in G[fn]]
        if len(vs) >= 2:
            sel_g[t] = st.fmean(vs)
    thr = st.median(list(sel_g.values()))
    print(f"  选择集靶标 {len(sel_g)},中位阈值 {thr:+.5f}(在看主人群结果之前定下)")
    gp = G[PRIMARY]
    hi = [gp[t] for t in gp if t in sel_g and sel_g[t] > thr]
    lo = [gp[t] for t in gp if t in sel_g and sel_g[t] <= thr]
    sh, sl = half_stats(hi), half_stats(lo)
    for nmh, s in (("选中的高半(主项)", sh), ("低半(余项)", sl)):
        print(f"  {nmh:16s} n {s['n']:3d}  均值 {s['mean']:+.4f}  sd {s['sd']:.4f}"
              f"  MDE +{s['MDE']:.4f}  效应/MDE {s['eff_over_mde']:.2f}")
    ex = 0
    for s2 in range(R_SEED):
        c, _n = H.boot_ci(hi, 7700 + 13 * s2)
        ex += 1 if (c[0] > 0 or c[1] < 0) else 0
    ci_hi, _n = H.boot_ci(hi, a.seed + 17)
    print(f"  主项均值 95% {ci_hi}  {'排除 0' if (ci_hi[0]>0 or ci_hi[1]<0) else '含 0'}"
          f"  [第51条 种子翻转 {ex}/{R_SEED} {'稳定' if ex in (0, R_SEED) else '**由 RNG 决定**'}]")
    g3 = min(sh["n"], sl["n"]) >= MIN_HALF
    print(f"  闸门3 每半 >= {MIN_HALF}:{'OK' if g3 else '✗ 切分作废'}")
    # random-split null on the same targets
    rngr = random.Random(a.seed + 77)
    allv = hi + lo
    nl = []
    for _ in range(RAND_SPLITS):
        v = list(allv); rngr.shuffle(v)
        A2, B2 = v[:len(hi)], v[len(hi):]
        nl.append(max(half_stats(A2)["eff_over_mde"], half_stats(B2)["eff_over_mde"]))
    nl.sort(); nq = nl[int(0.95 * RAND_SPLITS)]
    print(f"  随机切分零点({RAND_SPLITS} 次)效应/MDE 95 分位 {nq:.2f};真实 {sh['eff_over_mde']:.2f}"
          f"  ⇒ {'落在零点内' if sh['eff_over_mde'] <= nq else '**超出零点**'}")
    res["arm2"] = {"threshold": thr, "n_sel": len(sel_g), "high": sh, "low": sl,
                   "ci_high": ci_hi, "seed_flip": f"{ex}/{R_SEED}", "gate3": g3,
                   "null_q95": round(nq, 3), "sep": round(sh["mean"] - sl["mean"], 4)}

    print("\n=== 判定(冻结的打死条件)===")
    K2 = sh["eff_over_mde"] < 2.0
    K3 = sh["eff_over_mde"] <= nq
    K4 = sh["mean"] < 0
    print(f"  V1 零重叠 {'✓' if V1 else '✗'} | V2 信度可测 {'✓' if V2 else '✗'} | V3 必响输入 {'✓' if V3 else '✗'}")
    print(f"  K1 跨人群 rho 均值({obs_mean:+.4f})落在置换零点({q95:+.4f})以内:{'FIRES' if K1 else '不触发'}")
    print(f"  K2 主项 效应/MDE < 2({sh['eff_over_mde']:.2f}):{'FIRES' if K2 else '不触发'}")
    print(f"  K3 随机切分也能做到:{'FIRES' if K3 else '不触发'}")
    print(f"  K4 方向为负:{'FIRES' if K4 else '不触发'}")
    if K1:
        v = "between_target_spread_is_estimation_noise"
        print(f"  裁定:{v}")
        print("  **强制口径:这是对【汇总估计量】的正向支持 —— 靶标间离散度既然不可复现,"
              "汇总平均本来就是对的估计量,+0.0155 的分母不该被 sd 0.039 吓小。不是又一个『测不出』。**")
    elif K4:
        v = "empirical_split_wrong_direction"
    elif K3:
        v = "empirical_split_matched_by_random"
    elif K2:
        v = "main_part_below_resolution"
    else:
        v = "main_part_found_by_empirical_invariant"
    print(f"  裁定:{v}")
    print("  **口径:区间含 0 一律写 `unresolved`,绝不写 `refuted`。**")
    print("  边界:不改 H2 在盲第六批上的 `does_not_generalise`;主人群非盲;只判有没有主项,不判上线。")
    res["verdict"] = v
    res["g_all"] = {fn: {t: round(x, 5) for t, x in G[fn].items()} for _n, fn in POPS}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
