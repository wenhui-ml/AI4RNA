#!/usr/bin/env python3
"""M9-pre: does M3's effect sit on the SAME targets across populations, or is it a heavy tail?

Declaration: m9pre_heavytail_vs_concentration_declaration.json, frozen and pushed at 5fd49c1f BEFORE
this file existed.

M3 left the reading undecided in its own audit: mean excludes zero (+0.0140) while the sign test gives
p = 0.689 with 26 of 56 targets negative. Either (a) heavy-tailed noise, making "excludes 0" fragile, or
(b) a real effect concentrated on a subset -- move two's main-term/remainder shape. Move two then ran
five experiments on (b) and never tested (a). r1b's 557 repeat groups now give (a) direct support:
mean/median of |diff| is 1.35 against 1.18 under normality.

ROBUST ESTIMATORS CANNOT DECIDE THIS, which is why they are a report-only arm here. A trimmed mean is
near zero under BOTH readings -- under (a) because the noise is symmetric, under (b) because most
targets have no effect at all. The discriminator is whether the targets carrying the effect CHANGE from
population to population: independent noise draws change, a real subset does not. That is M4c's
machinery, reused.

Everything about the per-target quantity -- exact GC matching, held-out splits, resampling unit --
is M3 arm 1 unchanged. The only new thing is what we do with the per-target vector afterwards.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import numpy as np                                                            # noqa: E402
from scipy.stats import spearmanr                                             # noqa: E402
from e1_ensemble_marginals import base_block, marginals                       # noqa: E402
import h2_heldout_cluster_ci as H                                             # noqa: E402
from m3_gc_cancelled import build, heldout                                    # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
MIN_COMMON, NBOOT = 20, 3000


def per_target(keep, M, seed, subset=None, noise_rng=None):
    """M3 arm 1 (formable + ensemble_defect) verbatim; returns per[arm][target] = [split values].

    subset: optional set of design ids to restrict to (used for the design half-split ceiling).
    noise_rng: if given, the ed column is replaced by width-matched pure noise (P1 probe, M4d's build).
    """
    cells, cols, key = collections.defaultdict(list), {}, {}
    eds = [m["ens_defect_skel"] for m in M if "error" not in m and "ens_defect_skel" in m]
    sd = st.stdev(eds) if len(eds) > 2 else 1.0
    for i, (r, m) in enumerate(zip(keep, M)):
        if "error" in m or "ens_defect_skel" not in m:
            continue
        if subset is not None and i not in subset:
            continue
        rr = dict(r)
        B = base_block(r)
        ext = [noise_rng.gauss(0.0, sd)] if noise_rng is not None else [m["ens_defect_skel"]]
        cols[id(rr)] = B + [m["gc_cross"], m["gc_frac"]] + ext
        key[id(rr)] = (m["gc_cross"], round(m["gc_frac"], 3))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    if not rws:
        return None, None, None
    nB = len(base_block(keep[0]))
    idx = {"A": list(range(nB)), "A_ed": list(range(nB)) + [nB + 2]}
    per, used, s, _ref = heldout(rws, cols, idx, key, seed, "exact")
    return per, used, s


def gvec(per, used, a="A_ed", b="A"):
    return {t: st.fmean(per[a][t]) - st.fmean(per[b][t]) for t in used}


def rho_ci(x, y, seed):
    """Spearman with a bootstrap CI over the common TARGETS (the resampling unit)."""
    if len(x) < 4:
        return None, [None, None]
    r = spearmanr(x, y)[0]
    rng = random.Random(seed); n = len(x); B = []
    for _ in range(NBOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        xs = [x[i] for i in idx]; ys = [y[i] for i in idx]
        if len(set(xs)) < 3 or len(set(ys)) < 3:
            continue
        v = spearmanr(xs, ys)[0]
        if not math.isnan(v):
            B.append(v)
    if not B:
        return r, [None, None]
    B.sort()
    return r, [B[int(0.025 * len(B))], B[int(0.975 * len(B))]]


def pairwise(G, names, seed, label):
    """six population pairs; returns rows and the counts the kill conditions read."""
    rows, pos, neg, short = [], 0, 0, 0
    ks = list(G)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            common = sorted(set(G[a]) & set(G[b]))
            if len(common) < MIN_COMMON:
                short += 1
                rows.append({"pair": [names[a], names[b]], "n": len(common), "void": True})
                continue
            r, ci = rho_ci([G[a][t] for t in common], [G[b][t] for t in common], seed + 7 * i + j)
            ex_p = ci[0] is not None and ci[0] > 0
            ex_n = ci[1] is not None and ci[1] < 0
            pos += int(ex_p); neg += int(ex_n)
            rows.append({"pair": [names[a], names[b]], "n": len(common), "rho": round(r, 4),
                         "ci": [round(ci[0], 4), round(ci[1], 4)], "excl_pos": bool(ex_p),
                         "excl_neg": bool(ex_n)})
    print(f"\n  --- {label} ---")
    for x in rows:
        if x.get("void"):
            print(f"    {x['pair'][0]:<10} × {x['pair'][1]:<10} 共同靶标 {x['n']:3d} < {MIN_COMMON} **作废**")
        else:
            m = "**正向排除0**" if x["excl_pos"] else ("**负向排除0**" if x["excl_neg"] else "含 0")
            print(f"    {x['pair'][0]:<10} × {x['pair'][1]:<10} n={x['n']:3d}  rho {x['rho']:+.4f} "
                  f"[{x['ci'][0]:+.4f},{x['ci'][1]:+.4f}]  {m}")
    print(f"    ⇒ 正向排除 0 **{pos}/6**,负向排除 0 **{neg}/6**,作废 {short}")
    return rows, int(pos), int(neg), int(short)


def shape(g):
    v = sorted(g)
    n = len(v)
    k = int(0.1 * n)
    tri = st.fmean(v[k:n - k]) if n - 2 * k > 0 else float("nan")
    hl = st.median([(v[i] + v[j]) / 2 for i in range(n) for j in range(i, n)])
    posn = sum(1 for x in v if x > 0)
    from math import comb
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(posn, n - posn) + 1)) / 2 ** n)
    m = st.fmean(v); s = st.stdev(v)
    kurt = st.fmean([((x - m) / s) ** 4 for x in v]) if s > 0 else float("nan")
    return {"mean": round(m, 4), "trim10": round(tri, 4), "median": round(st.median(v), 4),
            "hodges_lehmann": round(hl, 4), "pos": int(posn), "n": int(n), "sign_p": round(p, 4),
            "kurtosis": round(kurt, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m9pre_heavytail_vs_concentration.json"))
    a = ap.parse_args()
    print("=== M9-pre:重尾 vs 效应集中 —— 扛效应的靶标换不换人 ===")
    print(f"  声明 5fd49c1f(代码存在之前已推送)。离线 $0。共同靶标地板 {MIN_COMMON}。\n")

    Gmain, Gdiff, Gnoise, Ghalf, names, SH = {}, {}, {}, {}, {}, {}
    for nm, fn in POPS:
        print(f"=== {nm} ===", flush=True)
        keep, M = build(fn, a.workers)
        names[fn] = nm
        per, used, s = per_target(keep, M, a.seed)
        if per is None or len(used) < H.MIN_TARGETS:
            print(f"  靶标 {0 if used is None else len(used)} < {H.MIN_TARGETS} ⇒ 本人群无数据")
            continue
        print(f"  泄漏 分子 {s['leak_mol']}/200 靶标 {s['leak_tg']}/200"
              f"  {'✓' if not s['leak_mol'] and not s['leak_tg'] else '✗ 作废'}  产生 g_t 的靶标 {len(used)}")
        Gmain[fn] = gvec(per, used)
        Gdiff[fn] = {t: st.fmean(per["A"][t]) for t in used}          # P0: 靶标难度本身
        SH[fn] = shape(list(Gmain[fn].values()))
        print(f"  形状 均值 {SH[fn]['mean']:+.4f} 截尾10% {SH[fn]['trim10']:+.4f} "
              f"中位 {SH[fn]['median']:+.4f} HL {SH[fn]['hodges_lehmann']:+.4f} "
              f"正 {SH[fn]['pos']}/{SH[fn]['n']} 符号p {SH[fn]['sign_p']:.3f} 峰度 {SH[fn]['kurtosis']}")

        # P1 probe: width-matched pure noise in place of the ed column
        pn, un, _sn = per_target(keep, M, a.seed, noise_rng=random.Random(a.seed + 991))
        if pn is not None and len(un) >= H.MIN_TARGETS:
            Gnoise[fn] = gvec(pn, un)

        # reliability ceiling: split DESIGNS in half, recompute g_t on each half
        ok = [i for i, m in enumerate(M) if "error" not in m and "ens_defect_skel" in m]
        rng = random.Random(a.seed + 5); rng.shuffle(ok)
        h1, h2 = set(ok[: len(ok) // 2]), set(ok[len(ok) // 2:])
        p1, u1, _ = per_target(keep, M, a.seed, subset=h1)
        p2, u2, _ = per_target(keep, M, a.seed, subset=h2)
        if p1 is not None and p2 is not None:
            g1, g2 = gvec(p1, u1), gvec(p2, u2)
            cm = sorted(set(g1) & set(g2))
            if len(cm) >= MIN_COMMON:
                r, ci = rho_ci([g1[t] for t in cm], [g2[t] for t in cm], a.seed + 31)
                Ghalf[fn] = {"rho": round(r, 4), "ci": [round(ci[0], 4), round(ci[1], 4)], "n": len(cm)}
                print(f"  信度天花板(设计对半) rho **{r:+.4f}** [{ci[0]:+.4f},{ci[1]:+.4f}] (n={len(cm)})")
        json.dump({"partial": True}, open(a.out, "w"))

    if len(Gmain) < 3:
        print("\n**可用人群 < 3 ⇒ 无数据,不出裁定**")
        json.dump({"verdict": "no_data", "n_pops": len(Gmain)}, open(a.out, "w"), indent=1, ensure_ascii=False)
        return 0

    print("\n=== 必响探针 ===")
    p0rows, p0pos, p0neg, p0short = pairwise(Gdiff, names, a.seed, "P0 靶标难度 A 本身(必须 >= 5/6 正向排除 0)")
    p1rows, p1pos, p1neg, p1short = pairwise(Gnoise, names, a.seed, "P1 等宽纯噪声(必须 <= 1/6 正向排除 0)") \
        if len(Gnoise) >= 3 else ([], 0, 0, 6)

    print("\n=== 主臂:g_t = A_ed − A 的跨人群持续性 ===")
    rows, pos, neg, short = pairwise(Gmain, names, a.seed, "主臂")

    ceil_mean = st.fmean([v["rho"] for v in Ghalf.values()]) if Ghalf else None
    main_mean = st.fmean([r["rho"] for r in rows if not r.get("void")])
    p0_mean = st.fmean([r["rho"] for r in p0rows if not r.get("void")]) if p0rows else None

    K0 = not (p0pos >= 5 and p1pos <= 1)
    K1 = ceil_mean is not None and main_mean > ceil_mean
    K2 = pos >= 4
    K3 = pos <= 1
    K4 = short >= 3
    print("\n=== 判决条件 ===")
    for k, v, d in (("K0", K0, "探针未按预期 ⇒ 整轮作废"), ("K1", K1, "超过信度天花板 ⇒ 作废"),
                    ("K2", K2, "正向排除 0 >= 4 ⇒ (b) 效应集中"), ("K3", K3, "正向排除 0 <= 1 ⇒ (a) 重尾"),
                    ("K4", K4, "共同靶标不足的对 >= 3 ⇒ 无数据")):
        print(f"  {k}: {'**触发**' if v else '未触发'}   {d}")

    if K0 or K1 or K4:
        verdict = "void_" + ("probe_failed" if K0 else ("above_reliability_ceiling" if K1 else "no_data"))
    elif K2 and not K3:
        verdict = "concentrated_effect__move_two_framework_stands"
    elif K3 and not K2:
        verdict = "heavy_tailed_noise__move_two_was_chasing_a_tail"
    else:
        verdict = "unresolved_between_heavy_tail_and_concentration"
    if neg >= 3:
        verdict += "__PLUS_anti_persistence"      # rule 64 shape 1
    if K2 and p0_mean is not None and main_mean / max(1e-9, p0_mean) > 0.8:
        verdict += "__confounded_by_target_difficulty"   # rule 64 shape 2

    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  主臂六对均值 rho {main_mean:+.4f}" + (f",信度天花板均值 {ceil_mean:+.4f}"
          f"(占 {100*main_mean/ceil_mean:.0f}%)" if ceil_mean else ""))
    if p0_mean is not None:
        print(f"  P0 靶标难度均值 rho {p0_mean:+.4f} ⇒ 主臂/P0 = {main_mean/p0_mean:.2f}"
              f"(接近 1 则『持续的只是靶标难度』,第 64 条形状二)")
    print("\n  **第 61 条照引:本臂共同靶标给出的半宽只能裁决 |rho| >~ 0.3 的持续性。"
          "M4c 的两个量级作参照:GC_only +0.4898 可检出,面板 +0.1523 检不出。"
          "K3 触发只等于『本臂看不见 0.15』,不等于 (a) 成立 —— "
          f"可检出性的实证上界由 P0 在同一批共同靶标上拿到的 {p0_mean if p0_mean else float('nan'):+.4f} 给出。**")

    print("\n=== 次要臂(只报出,不进判据):g_t 分布形状 ===")
    print(f"  {'人群':<12}{'均值':>9}{'截尾10%':>10}{'中位':>9}{'HL':>9}{'正/n':>9}{'符号p':>8}{'峰度':>8}")
    for fn, v in SH.items():
        print(f"  {names[fn]:<12}{v['mean']:+9.4f}{v['trim10']:+10.4f}{v['median']:+9.4f}"
              f"{v['hodges_lehmann']:+9.4f}{str(v['pos'])+'/'+str(v['n']):>9}{v['sign_p']:8.3f}{v['kurtosis']:8.2f}")

    json.dump({"verdict": verdict, "main": rows, "P0": p0rows, "P1": p1rows, "ceiling": Ghalf,
               "shape": {names[k]: v for k, v in SH.items()},
               "counts": {"main_pos": pos, "main_neg": neg, "P0_pos": p0pos, "P1_pos": p1pos,
                          "short": short},
               "means": {"main": round(main_mean, 4),
                         "ceiling": round(ceil_mean, 4) if ceil_mean else None,
                         "P0": round(p0_mean, 4) if p0_mean else None},
               "kill": {"K0": K0, "K1": K1, "K2": K2, "K3": K3, "K4": K4}},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
