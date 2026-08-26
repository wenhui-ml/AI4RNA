#!/usr/bin/env python3
"""M9: the router -- does the PAIRED difference in capture between panel and GC track target length?

Declaration: m9_router_paired_difference_declaration.json, frozen and pushed at 0c277711 BEFORE this
file existed.

M8d's control arm found the 22-column panel's share of available gain RISING with target length on the
pooled blind arm (+0.2696 [+0.0299,+0.4774]) while GC_only's falls. If both hold, "which criterion is
better" is the wrong question and the product is a ROUTER keyed on length.

WHY THE PAIRED DIFFERENCE IS THE WHOLE EXPERIMENT. r1b measured that longer targets have LESS
reproducible wet readings (partial +0.1196 [+0.0030,+0.2485]). That attenuates ANY criterion's capture
on long targets -- a common-mode effect two marginal arms cannot separate from a real difference in
length-dependence, and which the paired difference cancels. M4b already paid for this: its K4 compared
marginal intervals and stayed silent while the paired difference was +1.9519 [+0.5570,+3.2905]. M8d then
reported two margins separately. This run computes the difference.

DISCOVERY AND CONFIRMATION ARE INVERTED. The hypothesis was born on the BLIND populations, so the four
core populations are its confirmation set -- and that is where it looks weakest. This is not "confirm
the router", it is "take a hypothesis born where it should have been confirmed to the least favourable
place available".

m8b's one() is deliberately NOT reused: its V5 hardcodes `neg`, the direction-hardcoding that rule 60
came from. This run declares POSITIVE in advance and counts both sides.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import numpy as np                                                            # noqa: E402
from scipy.stats import pearsonr, spearmanr                                   # noqa: E402
from m1c_skeleton_bookkeeping import build                                    # noqa: E402
from m8b_target_length import CORE, BLIND, run_pop                            # noqa: E402

RUNGS = (0.1, 0.2, 0.3, 0.4, 0.6)
DRAWS, PASS_FRAC, NBOOT = 20, 0.8, 2000


def rho_ci(x, y, seed, nb=NBOOT):
    r = spearmanr(x, y)[0]
    rng = random.Random(seed); n = len(x); B = []
    for _ in range(nb):
        ix = [rng.randrange(n) for _ in range(n)]
        xs = [x[i] for i in ix]; ys = [y[i] for i in ix]
        if len(set(xs)) < 3 or len(set(ys)) < 3:
            continue
        v = spearmanr(xs, ys)[0]
        if not math.isnan(v):
            B.append(v)
    B.sort()
    return r, [B[int(.025 * len(B))], B[int(.975 * len(B))]]


def arm(D, yvec, xmap, seed):
    u = D["used"]
    x = [xmap[t] for t in u]; y = [yvec[t] for t in u]
    if len(set(x)) < 3:
        return None
    r, ci = rho_ci(x, y, seed)
    return {"rho": round(r, 4), "ci": [round(ci[0], 4), round(ci[1], 4)], "n": len(u),
            "pos": bool(ci[0] > 0), "neg": bool(ci[1] < 0)}


def sb(r):
    return 2 * r / (1 + r) if r > -1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m9_router_paired_difference.json"))
    a = ap.parse_args()
    print("=== M9:路由器 —— 配对差 capture^A − capture^GC 是否随靶标长度上升 ===")
    print(f"  声明 0c277711(代码存在之前已推送)。方向预注册=**正**。离线 $0。\n")

    POPS = CORE + BLIND
    CORE_FN = {fn for _n, fn in CORE}
    D, HALF, names = {}, {}, {}
    for nm, fn in POPS:
        print(f"=== {nm} ===", flush=True)
        keep, M = build(fn, a.workers)
        names[fn] = nm
        d = run_pop(nm, keep, M, a.splits, a.seed, 9500)
        if d is None:
            continue
        D[fn] = d
        ok = [i for i, m in enumerate(M) if "error" not in m]
        rng = random.Random(a.seed + 5); rng.shuffle(ok)
        h1, h2 = sorted(ok[: len(ok) // 2]), sorted(ok[len(ok) // 2:])
        d1 = run_pop(nm + "/半1", [keep[i] for i in h1], [M[i] for i in h1], a.splits, a.seed, 9500)
        d2 = run_pop(nm + "/半2", [keep[i] for i in h2], [M[i] for i in h2], a.splits, a.seed, 9500)
        if d1 and d2:
            cm = sorted(set(d1["used"]) & set(d2["used"]))
            if len(cm) >= 20:
                g1 = [d1["vec"]["A"][t] - d1["vec"]["GC_only"][t] for t in cm]
                g2 = [d2["vec"]["A"][t] - d2["vec"]["GC_only"][t] for t in cm]
                r, ci = rho_ci(g1, g2, a.seed + 31)
                HALF[fn] = {"rho": round(r, 4), "n": len(cm)}
                print(f"  信度 对半 {r:+.4f} ⇒ **全样本(SB) {sb(r):+.4f}** (n={len(cm)})")

    if len(D) < 4:
        json.dump({"verdict": "no_data"}, open(a.out, "w"), indent=1, ensure_ascii=False); return 0
    DIF = {fn: {t: D[fn]["vec"]["A"][t] - D[fn]["vec"]["GC_only"][t] for t in D[fn]["used"]} for fn in D}
    CEIL = sb(st.fmean([v["rho"] for v in HALF.values()])) if HALF else None
    print(f"\n  **对半天花板均值 {st.fmean([v['rho'] for v in HALF.values()]):+.4f} ⇒ "
          f"SB 全样本 {CEIL:+.4f}**" if HALF else "\n  天花板不可得")

    print(f"\n=== 构造性梯级(第 67 条,每档 {DRAWS} 次;不沿用任何前一实验的阈值)===")
    LAD, thr = {}, None
    core_fns = [fn for fn in D if fn in CORE_FN]
    for rho in RUNGS:
        hits = []
        for k in range(DRAWS):
            rr = random.Random(a.seed + int(rho * 100) + 7919 * k)
            n_ok = 0
            for fn in core_fns:
                u = D[fn]["used"]
                L = np.array([D[fn]["len"][t] for t in u], float)
                z = (np.argsort(np.argsort(L)).astype(float) - len(u) / 2) / (len(u) / 2)
                y = {t: math.sqrt(rho) * z[i] + math.sqrt(1 - rho) * rr.gauss(0, 1)
                     for i, t in enumerate(u)}
                r = arm(D[fn], y, D[fn]["len"], a.seed + 77)
                n_ok += bool(r and r["pos"])
            hits.append(1 if n_ok >= 3 else 0)
        f = st.fmean(hits)
        LAD[str(rho)] = {"pass_frac": round(f, 3), "raw": hits}
        print(f"  rho={rho:.1f}  确证集(核心 >=3/4)达标比例 **{f:.2f}** ({int(f*DRAWS)}/{DRAWS})")
        if thr is None and f >= PASS_FRAC:
            thr = rho
    print(f"  **检出阈值 = {thr if thr else '> 0.6'}**" + (f"  vs 天花板 {CEIL:.4f}" if CEIL else ""))

    print("\n=== P1 噪声臂(12 根等宽噪声替换长度,正向排除 0 必须 <= 1/12)===")
    nz = 0
    for j in range(12):
        h = sum(1 for fn in core_fns
                if (lambda r: bool(r and r["pos"]))(arm(D[fn], DIF[fn], D[fn]["noise"][f"TN{j}"], a.seed + 77)))
        nz += (h >= 3)
    print(f"  12 根噪声在确证集上达标 **{nz}/12**  {'✓' if nz <= 1 else '✗ 作废'}")

    print("\n=== P2 边际臂(必须复现 M8b/M8d)与 K5 共模检查 ===")
    print(f"  {'人群':<13}{'面板 A':>24}{'GC_only':>24}{'同号?':>7}")
    MARG, same = {}, 0
    for fn in D:
        ra = arm(D[fn], D[fn]["vec"]["A"], D[fn]["len"], a.seed + 17)
        rg = arm(D[fn], D[fn]["vec"]["GC_only"], D[fn]["len"], a.seed + 17)
        MARG[fn] = {"A": ra, "GC": rg}
        s = (ra["rho"] > 0) == (rg["rho"] > 0)
        if fn in CORE_FN:
            same += s
        print(f"  {names[fn]:<13}{ra['rho']:+8.4f} [{ra['ci'][0]:+.3f},{ra['ci'][1]:+.3f}]"
              f"{rg['rho']:+8.4f} [{rg['ci'][0]:+.3f},{rg['ci'][1]:+.3f}]{'  同' if s else '  异':>7}")
    print(f"  ⇒ 核心人群里两臂同号 **{same}/4**(>=3 则共模未消掉,K5 触发)")

    print("\n=== 主臂:配对差 d_t = capture^A − capture^GC 对靶标长度 ===")
    MAIN, cpos, cneg = {}, 0, 0
    for fn in D:
        r = arm(D[fn], DIF[fn], D[fn]["len"], a.seed + 77)
        MAIN[fn] = r
        tag = "**确证集**" if fn in CORE_FN else "盲(复述)"
        m = "**正向排除0**" if r["pos"] else ("负向排除0" if r["neg"] else "含 0")
        if fn in CORE_FN:
            cpos += r["pos"]; cneg += r["neg"]
        print(f"  {names[fn]:<13}{tag:<10} n={r['n']:3d}  rho {r['rho']:+.4f} "
              f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  {m}")
    print(f"  ⇒ **确证集(四核心)正向排除 0 {cpos}/4**,负向 {cneg}/4")

    print("\n=== 第 64 条形状一:是不是【方差】而不是【均值】的长度依赖 ===")
    VAR = {}
    for fn in core_fns:
        rv = arm(D[fn], {t: abs(v) for t, v in DIF[fn].items()}, D[fn]["len"], a.seed + 77)
        VAR[fn] = rv
        print(f"  {names[fn]:<13} |d_t| 对长度 rho {rv['rho']:+.4f} "
              f"[{rv['ci'][0]:+.4f},{rv['ci'][1]:+.4f}]")
    var_stronger = sum(1 for fn in core_fns
                       if abs(VAR[fn]["rho"]) > abs(MAIN[fn]["rho"])) >= 3

    main_mean = st.fmean([MAIN[fn]["rho"] for fn in core_fns])
    K0 = (nz > 1)
    K1 = CEIL is not None and main_mean > CEIL
    K2 = cpos >= 3
    K3 = cpos <= 1
    K4 = (thr is None) or (CEIL is not None and thr > CEIL)
    K5 = same >= 3
    print("\n=== 判决条件 ===")
    for k, v, d in (("K0", K0, "噪声臂响 ⇒ 作废"), ("K1", K1, f"主臂均值 {main_mean:+.4f} > 天花板 ⇒ 作废"),
                    ("K2", K2, f"确证集正向 {cpos}/4 >= 3 ⇒ router_confirmed"),
                    ("K3", K3, "确证集正向 <= 1 ⇒ 未确证"),
                    ("K4", K4, "检出阈值 > 天花板 ⇒ 无裁决力"),
                    ("K5", K5, f"两臂同号 {same}/4 >= 3 ⇒ 共模未消掉")):
        print(f"  {k}: {'**触发**' if v else '未触发'}   {d}")

    if K0 or K1:
        verdict = "void_" + ("noise_fired" if K0 else "above_ceiling")
    elif K4:
        verdict = "no_adjudicating_power"
    elif K2 and not K3:
        verdict = "router_confirmed_on_core"
    elif K3 and not K2:
        verdict = "router_not_confirmed_on_core"
    else:
        verdict = "unresolved"
    if K5:
        verdict += "__common_mode_not_cancelled"
    if var_stronger:
        verdict += "__variance_not_mean"

    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  确证集均值 rho {main_mean:+.4f};盲人群复述:"
          + ", ".join(f"{names[fn]} {MAIN[fn]['rho']:+.4f}" for fn in D if fn not in CORE_FN))
    print(f"\n  **跑前预测照引:『确证集大概率落在 1/4 或 2/4,即 K3 或 unresolved,而不是 K2』。"
          f"实测 {cpos}/4 ⇒ {'预测成立' if cpos <= 2 else '**预测被否 —— 路由器在最不利的集合上被确证了**'}。"
          f"若未确证,正确结论是【这条线索在核心人群上没有得到确证】,不是【路由器不存在】。**")

    json.dump({"verdict": verdict, "main": {names[k]: v for k, v in MAIN.items()},
               "marginals": {names[k]: v for k, v in MARG.items()},
               "ladder": LAD, "threshold": thr, "ceiling_full_SB": round(CEIL, 4) if CEIL else None,
               "half": {names[k]: v for k, v in HALF.items()}, "noise_hits": int(nz),
               "same_sign_core": int(same), "abs_arm": {names[k]: v for k, v in VAR.items()},
               "core_pos": int(cpos), "core_neg": int(cneg), "main_mean": round(main_mean, 4),
               "kill": {k: bool(v) for k, v in (("K0", K0), ("K1", K1), ("K2", K2), ("K3", K3),
                                                ("K4", K4), ("K5", K5))}},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
