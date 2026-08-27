#!/usr/bin/env python3
"""M17: strip the folding predictors out of the main arm and see what is left.

Declaration: m17_oracle_free_declaration.json, frozen and pushed at a4d3b939 BEFORE this file
existed (the eighteenth done that way).

The shipped 22-column panel is literally pkiss + probknot + shapeknots (e1_ensemble_marginals.py
:35). M13b used its cross-fitted prognostic score as a 0.2 SD matching caliper. This file turns
that caliper off, so the primary estimate calls no folding predictor at any point: the truth is
the 2A3 chemical-probing readout, the treatment is where the GC sits, and the matching is on the
cell and on global GC only.

Two arms are always printed side by side -- the declaration forbids reporting one alone.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from m13b_gc_placement import (assemble, match, point, cluster_ci, perm_cols, null_dist, pval,
                               qdelta, shifted_gc, synth, POPS, DELTA, MAX_SMD,
                               MIN_PAIRS, MIN_TGT, NSHIFT)                             # noqa: E402

NPERM_MAIN, NPERM_PROBE = 2000, 400
NOISE_DRAWS, LAD_DRAWS = 12, 8
RUNGS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
P_GATE = 0.005
LAD_SELFCHECK_MAX, LAD_TOP_MIN_HIT = 0.15, 0.80
RATIO_K5 = 2.0


def _D(fn, seed):
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    return assemble(C[fn]["rows"], E[fn], seed)


def smds(d, M):
    sg = float(np.std(np.concatenate([c["gcf"] for c in d["cells"]]))) or 1.0
    return abs(float(M["dgcf"].mean())) / sg, abs(float(M["dpB"].mean())) / (d["sp"] or 1.0)


def arm(d, seed, nperm, panel):
    M = match(d, [c["gcp"] for c in d["cells"]], DELTA, panel=panel)
    if M is None:
        return None
    obs = point(M); null = null_dist(d, seed, nperm, panel=panel)
    sg, sp = smds(d, M)
    return {"effect": round(obs, 4), "p_perm": round(pval(obs, null), 5),
            "null_mean": round(float(null.mean()), 4), "null_sd": round(float(null.std()), 4),
            "z_vs_null": round((obs - float(null.mean())) / (float(null.std()) or 1), 2),
            "boot_ci_secondary": cluster_ci(M, seed),
            "npair": M["npair"], "ntgt": M["ntgt"],
            "smd_gcf": round(sg, 4), "smd_pB": round(sp, 4),
            "usable": bool(M["npair"] >= MIN_PAIRS and M["ntgt"] >= MIN_TGT and sg <= MAX_SMD)}


def gcp_pB_corr(d):
    """rule_64 shape_2: the raw within-cell correlation the reader needs to judge SMD(pB)."""
    v = []
    for c in d["cells"]:
        if len(c["gcp"]) > 2 and np.std(c["gcp"]) > 0 and np.std(c["pB"]) > 0:
            v.append(float(np.corrcoef(c["gcp"] - c["gcp"].mean(), c["pB"] - c["pB"].mean())[0, 1]))
    return round(float(np.mean(v)), 4) if v else float("nan"), len(v)


# ------------------------------------------------------------------ probes
def p0_identity(d):
    """treatment replaced by a constant -> no ordered pair can clear delta -> exactly zero."""
    const = [np.zeros(len(c["gcp"])) for c in d["cells"]]
    M = match(d, const, DELTA, panel=False)
    return {"matched_pairs": 0 if M is None else M["npair"],
            "effect": 0.0 if M is None else round(point(M), 10)}


def p1_noise(d, seed):
    hits, effs = 0, []
    for k in range(NOISE_DRAWS):
        rng = np.random.default_rng(9000 + 97 * seed + k)
        cols = synth(d, 0.0, rng); dl = qdelta(d, cols)
        M = match(d, cols, dl, panel=False)
        if M is None or M["npair"] < 30:
            continue
        obs = point(M)
        rng2 = np.random.default_rng(31000 + 97 * seed + k); nl = []
        for _ in range(NPERM_PROBE):
            sc = [c[rng2.permutation(len(c))] for c in cols]
            m = match(d, sc, dl, panel=False)
            nl.append(point(m) if m else 0.0)
        nl = np.sort(np.array(nl))
        p = pval(obs, nl); effs.append(round(obs, 4))
        hits += int(p <= P_GATE)
    return {"draws": NOISE_DRAWS, "hits_at_p<=0.005": hits, "effects": effs}


def p4_shift(d, real_effect, seed):
    rng = np.random.default_rng(4000 + seed)
    eff, cors, leak = [], [], []
    for _ in range(NSHIFT):
        cols, cr = shifted_gc(d, float(rng.uniform(0.10, 0.90)))
        M = match(d, cols, qdelta(d, cols), panel=False)
        if M is None:
            continue
        eff.append(point(M)); cors.append(cr); leak.append(float(M["dgcp"].mean()))
    if not eff:
        return None
    eff = np.array(eff); q95 = float(np.quantile(eff, .95))
    Mr = match(d, [c["gcp"] for c in d["cells"]], DELTA, panel=False)
    return {"n_shift": len(eff), "shift_mean": round(float(eff.mean()), 4),
            "shift_max": round(float(eff.max()), 4), "shift_q95": round(q95, 4),
            "real": real_effect, "pass": bool(real_effect > q95),
            "mean_corr_with_gcp": round(float(np.nanmean(cors)), 3),
            "leaked_gcp_gap": round(float(np.mean(leak)), 5),
            "real_gcp_gap": round(float(Mr["dgcp"].mean()), 5)}


def p2_rung(d, delta_pts, seed):
    """rule 79: build the dial ON THE NULL. Scramble y within cell first (true effect = 0),
    then add delta_pts of wet score to the molecules above that cell's gc_paired median."""
    hits, recov = 0, []
    for k in range(LAD_DRAWS):
        rng = np.random.default_rng(77000 + 997 * seed + 31 * k + int(delta_pts * 100))
        d2 = {**d, "cells": []}
        for c in d["cells"]:
            y = c["y"][rng.permutation(len(c["y"]))]
            bump = (c["gcp"] > np.median(c["gcp"])).astype(float) * delta_pts
            y = y + bump; y = y - y.mean()
            d2["cells"].append({**c, "y": y})
        M = match(d2, [c["gcp"] for c in d2["cells"]], DELTA, panel=False)
        if M is None or M["npair"] < 30:
            continue
        obs = point(M); recov.append(obs)
        rng2 = np.random.default_rng(88000 + 997 * seed + 31 * k + int(delta_pts * 100))
        nl = np.sort(np.array([point(match(d2, perm_cols(d2, rng2), DELTA, panel=False)
                                     or {"sums": {0: 0.0}, "cnts": {0: 1}})
                               for _ in range(NPERM_PROBE)]))
        hits += int(pval(obs, nl) <= P_GATE)
    n = max(1, len(recov))
    return {"delta_pts": delta_pts, "draws": len(recov), "hit_rate": round(hits / n, 3),
            "recovered_mean": round(float(np.mean(recov)) if recov else 0.0, 4)}


# ------------------------------------------------------------------ per-population job
def job(spec):
    fn, seed, what = spec
    d = _D(fn, seed)
    if what == "arms":
        A = arm(d, seed, NPERM_MAIN, panel=False)
        B = arm(d, seed, NPERM_MAIN, panel=True)
        r, nc = gcp_pB_corr(d)
        out = {"A_oracle_free": A, "B_panel_matched": B, "n": d["n"], "cells": len(d["cells"]),
               "corr_gcp_pB": r, "corr_cells": nc,
               "ratio": round(A["effect"] / (B["effect"] or 1e-9), 3) if A and B else None,
               "P0": p0_identity(d), "P1": p1_noise(d, seed)}
        out["P4"] = p4_shift(d, A["effect"], seed) if A else None
        return fn, "arms", out
    return fn, what, p2_rung(d, float(what.split(":")[1]), seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "m17_oracle_free.json"))
    a = ap.parse_args()
    print("=== M17:把三个折叠预测器从主臂里整个拆掉 ===")
    print("  声明 a4d3b939(代码存在之前已推送)。离线 $0。")
    print("  **主臂对我不盲** —— M13b 已量过面板配平版。本轮唯一改动:panel caliper 关掉。\n")

    # ---------------- P5: the declaration's own sentence, recomputed and asserted
    print("--- P5:L2(oracle 入选筛)掉行率,原地复算并断言 ---")
    from e1_ensemble_marginals import base_block, load_pop
    l2 = {}
    for nm, fn in POPS:
        rec, _s, _m = load_pop(fn)
        A = [r for r in rec if r.get("seq") and r.get("wet") is not None]
        B = [r for r in A if base_block(r) is not None]
        drop = 100.0 * (1 - len(B) / max(len(A), 1))
        l2[fn] = {"pop": nm, "with_seq_and_wet": len(A), "panel_ok": len(B), "drop_pct": round(drop, 4)}
        print(f"  {nm:12s} {len(A):5d} -> {len(B):5d}   掉行 {drop:.4f}%")
    bad = [v["pop"] for v in l2.values() if v["drop_pct"] != 0.0]
    assert not bad, f"P5 失败:{bad} 掉行不为 0 —— 声明里那句话是错的,按 K0 作废"
    print("  P5 通过:六人群全部 0.0000%,分子集合不是被 oracle 筛出来的\n")

    specs = [(fn, a.seed, "arms") for _nm, fn in POPS]
    specs += [(fn, a.seed, f"lad:{r}") for _nm, fn in POPS for r in RUNGS]
    from concurrent.futures import ProcessPoolExecutor
    RES = collections.defaultdict(dict)
    print(f"--- 并行 {len(specs)} 个任务,{a.workers} 进程(单线程 BLAS) ---", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fn, what, r in ex.map(job, specs):
            RES[fn][what] = r
            print(f"  完成 {fn:26s} {what}", flush=True)

    print("\n--- 主结果:两臂并排(声明禁止只报一个) ---")
    print(f"  {'人群':12s} {'oracle-free':>12s} {'p':>8s} {'z':>6s} {'配对':>7s} | "
          f"{'面板配平':>10s} {'p':>8s} {'配对':>7s} | {'比值':>6s} {'SMD(pB)':>8s} {'corr':>7s}")
    for nm, fn in POPS:
        v = RES[fn]["arms"]; A, B = v["A_oracle_free"], v["B_panel_matched"]
        print(f"  {nm:12s} {A['effect']:+12.3f} {A['p_perm']:8.5f} {A['z_vs_null']:6.1f} "
              f"{A['npair']:7d} | {B['effect']:+10.3f} {B['p_perm']:8.5f} {B['npair']:7d} | "
              f"{v['ratio']:6.2f} {A['smd_pB']:8.4f} {v['corr_gcp_pB']:+7.3f}")

    print("\n--- 探针 ---")
    for nm, fn in POPS:
        v = RES[fn]["arms"]
        print(f"  {nm:12s} P0 配对{v['P0']['matched_pairs']} 效应{v['P0']['effect']:+.8f} | "
              f"P1 命中 {v['P1']['hits_at_p<=0.005']}/{v['P1']['draws']} | "
              f"P4 真实{v['P4']['real']:+.2f} vs 平移95分位{v['P4']['shift_q95']:+.2f} "
              f"泄漏{v['P4']['leaked_gcp_gap']:+.4f}(真臂{v['P4']['real_gcp_gap']:+.4f}) "
              f"{'通过' if v['P4']['pass'] else '**不通过**'}")

    print("\n--- P2 梯级(建在零上,第 79 条) ---")
    print(f"  {'人群':12s} " + "  ".join(f"δ={r:<4g}" for r in RUNGS))
    for nm, fn in POPS:
        row = [RES[fn][f"lad:{r}"] for r in RUNGS]
        print(f"  {nm:12s} " + "  ".join(f"{x['hit_rate']:5.2f} " for x in row))
        print(f"  {'':12s} " + "  ".join(f"({x['recovered_mean']:+4.2f})" for x in row)
              + "   <- 括号是回收到的效应量")

    # ---------------- verdict
    P0ok = all(RES[fn]["arms"]["P0"]["effect"] == 0.0 for _n, fn in POPS)
    P1ok = sum(RES[fn]["arms"]["P1"]["hits_at_p<=0.005"] for _n, fn in POPS) <= 3
    P4n = sum(1 for _n, fn in POPS if RES[fn]["arms"]["P4"] and RES[fn]["arms"]["P4"]["pass"])
    self0 = max(RES[fn]["lad:0.0"]["hit_rate"] for _n, fn in POPS)
    topmin = min(RES[fn][f"lad:{RUNGS[-1]}"]["hit_rate"] for _n, fn in POPS)
    maxeff = max(RES[fn]["arms"]["A_oracle_free"]["effect"] for _n, fn in POPS)
    ladok = self0 <= LAD_SELFCHECK_MAX and topmin >= LAD_TOP_MIN_HIT
    K4 = RUNGS[-1] < maxeff
    sig = sum(1 for _n, fn in POPS
              if RES[fn]["arms"]["A_oracle_free"]["p_perm"] <= P_GATE
              and RES[fn]["arms"]["A_oracle_free"]["effect"] > 0
              and RES[fn]["arms"]["A_oracle_free"]["usable"])
    ratios = [RES[fn]["arms"]["ratio"] for _n, fn in POPS]
    K5 = max(ratios) > RATIO_K5
    K0 = not (P0ok and P1ok and P4n >= 5 and ladok)
    verdict = ("void_probe_failed" if K0 else K4 and "void_ladder_range_too_narrow"
               or ("oracle_free_placement_holds" if sig >= 5 else
                   "underpowered" if sum(1 for _n, fn in POPS
                                         if RES[fn]["arms"]["A_oracle_free"]["usable"]) < 5 else
                   "placement_needs_panel_matching" if sig <= 2 else "unresolved"))
    smdp = [RES[fn]["arms"]["A_oracle_free"]["smd_pB"] for _n, fn in POPS]
    reading = "(甲) caliper 本来就没在做事" if max(smdp) <= 0.10 else "(乙) 面板能看见一部分"

    print("\n=== 裁定 ===")
    print(f"  P0 {'合格' if P0ok else '不合格'} | P1 合计命中 "
          f"{sum(RES[fn]['arms']['P1']['hits_at_p<=0.005'] for _n,fn in POPS)}/72 上限3 "
          f"{'合格' if P1ok else '不合格'} | P4 {P4n}/6 | "
          f"梯级 δ=0 自检最大 {self0:.2f}(上限{LAD_SELFCHECK_MAX}) 顶档最小 {topmin:.2f} "
          f"{'合格' if ladok else '不合格'}")
    print(f"  K4 顶档 {RUNGS[-1]} vs 主臂最大 {maxeff:+.3f} -> {'触发(作废)' if K4 else '不触发'}")
    print(f"  K1 显著人群 {sig}/6 | K5 比值最大 {max(ratios):.2f}(阈 {RATIO_K5}) "
          f"-> {'触发' if K5 else '不触发'}")
    print(f"  SMD(pB) 范围 [{min(smdp):.4f}, {max(smdp):.4f}] -> 读法 {reading}")
    print(f"  **裁定:{verdict}**")
    if K5:
        print("  **K5 触发:oracle-free 版必须写成【含面板可见成分的上界】,"
              "§4 仍以面板配平版为主臂。**")

    json.dump({"declaration": "m17_oracle_free_declaration.json @ a4d3b939",
               "L2_drop": l2, "pops": {fn: RES[fn] for _n, fn in POPS},
               "verdict": verdict, "K4": bool(K4), "K5": bool(K5), "sig": sig,
               "P0_ok": bool(P0ok), "P1_ok": bool(P1ok), "P4_pass": P4n,
               "ladder_selfcheck_max": self0, "ladder_top_min": topmin,
               "smd_pB_range": [min(smdp), max(smdp)], "reading": reading,
               "ratios": {fn: RES[fn]["arms"]["ratio"] for _n, fn in POPS}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
