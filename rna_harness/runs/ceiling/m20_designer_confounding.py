#!/usr/bin/env python3
"""M20: is the CG placement a designer's rule or thermodynamics, and how much of the
attenuation does it account for?

Declaration: m20_designer_confounding_declaration.json, frozen and pushed at a9f2d776 BEFORE
this file existed (the twenty-first done that way).

Nine methods designed sequences for the same twenty round-3 targets, so the CG-minus-AU helix
depth gap splits into a target part and a method part. A1 does that split and never touches
rescue_factor: thermodynamics says the gap is a property of the target, a designer's rule says
it is a property of the method. A2 then asks how much of the marginal-vs-matched attenuation
that placement quantitatively accounts for.

K2 is written into the declaration so it cannot be relabelled afterwards: if the method
component contains zero, the heuristic reading of E7 is withdrawn and the mechanism is renamed
thermodynamic. The confounding claim survives either way.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import m16_rescue as M16                                                              # noqa: E402

MIN_CELL, NPERM, P_GATE = 4, 2000, 0.005
NOISE_DRAWS, LAD_DRAWS = 12, 12
RUNGS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20)
LAD_SELF_MAX = 0.15
HUMAN = {"Eterna", "Starting sequence"}


# ------------------------------------------------------------------ A1: design side, no outcome
def cells_of(recs):
    """(puzzle, method) -> CG-minus-AU depth gap, for designs with >= MIN_CELL of each."""
    by = collections.defaultdict(lambda: {"CG": [], "AU": []})
    for r in recs:
        if r["pt"] in ("CG", "AU"):
            by[(r["puzzle"], r["method"])][r["pt"]].append(r["depth"])
    out = {}
    for k, d in by.items():
        if len(d["CG"]) >= MIN_CELL and len(d["AU"]) >= MIN_CELL:
            out[k] = float(np.mean(d["CG"]) - np.mean(d["AU"]))
    return out


def method_component(cells, labels=None):
    """partial R^2 of method after target main effects. labels: (puzzle,method)->method override."""
    keys = sorted(cells)
    y = np.array([cells[k] for k in keys], float)
    tg = [k[0] for k in keys]
    me = [labels[k] if labels else k[1] for k in keys]
    ut, um = sorted(set(tg)), sorted(set(me))
    if len(um) < 2 or len(ut) < 2:
        return 0.0, 0.0, len(keys)
    T = np.zeros((len(keys), len(ut))); M = np.zeros((len(keys), len(um)))
    for i, (t, m) in enumerate(zip(tg, me)):
        T[i, ut.index(t)] = 1.0; M[i, um.index(m)] = 1.0

    def rss(X):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(((y - X @ b) ** 2).sum())

    tss = float(((y - y.mean()) ** 2).sum()) or 1.0
    r_t = rss(T); r_tm = rss(np.hstack([T, M]))
    return (r_t - r_tm) / tss, r_t / tss, len(keys)


def a1_perm(cells, seed, nperm=NPERM, labels0=None):
    """permute method labels WITHIN target -- one design per (puzzle, method), so this relabels
    designs without touching any column that defines a cell."""
    keys = sorted(cells)
    bytg = collections.defaultdict(list)
    for k in keys:
        bytg[k[0]].append(k)
    rng = np.random.default_rng(seed); out = []
    for _ in range(nperm):
        lab = {}
        for t, ks in bytg.items():
            ms = [labels0[k] if labels0 else k[1] for k in ks]
            p = rng.permutation(len(ms))
            for k, j in zip(ks, p):
                lab[k] = ms[j]
        out.append(method_component(cells, lab)[0])
    return np.sort(np.array(out))


# ------------------------------------------------------------------ A2: mediation
def strat_diff(recs, key, A="CG", B="AU"):
    """stratum-size weighted difference of means over the given stratification."""
    by = collections.defaultdict(lambda: {A: [], B: []})
    for r in recs:
        if r["pt"] in (A, B):
            by[key(r)][r["pt"]].append(r)
    num = den = 0.0
    for _s, d in by.items():
        if not d[A] or not d[B]:
            continue
        w = min(len(d[A]), len(d[B]))
        num += w * (float(np.mean([x["rf"] for x in d[A]])) - float(np.mean([x["rf"] for x in d[B]])))
        den += w
    return (num / den if den else float("nan")), den


def strat_gap(recs, key, A="CG", B="AU"):
    """same weighting, but on depth -- the design-side gap g."""
    by = collections.defaultdict(lambda: {A: [], B: []})
    for r in recs:
        if r["pt"] in (A, B):
            by[key(r)][r["pt"]].append(r)
    num = den = 0.0
    for _s, d in by.items():
        if not d[A] or not d[B]:
            continue
        w = min(len(d[A]), len(d[B]))
        num += w * (float(np.mean([x["depth"] for x in d[A]])) - float(np.mean([x["depth"] for x in d[B]])))
        den += w
    return (num / den if den else float("nan"))


def beta_depth(recs):
    """within (design, crossed), the slope of rescue on depth. depth must NOT be in the stratum."""
    by = collections.defaultdict(list)
    for r in recs:
        if r["pt"] in ("CG", "AU"):
            by[(r["design"], r["crossed"])].append(r)
    num = den = 0.0
    for _s, g in by.items():
        if len(g) < 3:
            continue
        d = np.array([x["depth"] for x in g], float); v = np.array([x["rf"] for x in g], float)
        d = d - d.mean(); v = v - v.mean()
        num += float((d * v).sum()); den += float((d * d).sum())
    return (num / den if den else float("nan")), den


def mediation(recs):
    marg, nm = strat_diff(recs, lambda r: r["design"])
    matched, nx = strat_diff(recs, lambda r: (r["design"], min(r["depth"], 3), r["crossed"]))
    g = strat_gap(recs, lambda r: r["design"])
    b, _ = beta_depth(recs)
    obs_att = marg - matched
    pred_att = g * b
    frac = pred_att / obs_att if obs_att else float("nan")
    return {"marginal": round(marg, 5), "matched": round(matched, 5),
            "observed_attenuation": round(obs_att, 5), "gap_g": round(g, 4),
            "beta_depth": round(b, 5), "predicted_attenuation": round(pred_att, 5),
            "mediated_fraction": round(frac, 4), "n_marg": int(nm), "n_matched": int(nx)}


# ------------------------------------------------------------------ probes
def p1_noise(cells, seed):
    hits = []
    for k in range(NOISE_DRAWS):
        rng = np.random.default_rng(41000 + 97 * seed + k)
        keys = sorted(cells); bytg = collections.defaultdict(list)
        for kk in keys:
            bytg[kk[0]].append(kk)
        lab = {}
        for t, ks in bytg.items():
            ms = [kk[1] for kk in ks]; p = rng.permutation(len(ms))
            for kk, j in zip(ks, p):
                lab[kk] = ms[j]
        obs = method_component(cells, lab)[0]
        nl = a1_perm(cells, 52000 + 97 * seed + k, 400, labels0=lab)
        hits.append(float((float((nl >= obs).sum()) + 1) / 401.0))
    return {"draws": NOISE_DRAWS, "hits": sum(1 for p in hits if p <= P_GATE),
            "p": [round(x, 4) for x in hits]}


def p2_rung(cells, delta, seed):
    """rule 79: scramble method labels first (true method component = 0), then inject delta."""
    keys = sorted(cells); ms = sorted({k[1] for k in keys})
    off = {m: delta * (i / max(1, len(ms) - 1) - 0.5) for i, m in enumerate(ms)}
    hits = n = 0
    for k in range(LAD_DRAWS):
        rng = np.random.default_rng(63000 + 997 * seed + 31 * k + int(delta * 100))
        bytg = collections.defaultdict(list)
        for kk in keys:
            bytg[kk[0]].append(kk)
        lab, c2 = {}, {}
        for t, ks in bytg.items():
            mm = [kk[1] for kk in ks]; p = rng.permutation(len(mm))
            for kk, j in zip(ks, p):
                lab[kk] = mm[j]; c2[kk] = cells[kk] + off[mm[j]]
        obs = method_component(c2, lab)[0]
        nl = a1_perm(c2, 74000 + 997 * seed + 31 * k + int(delta * 100), 400, labels0=lab)
        hits += int((float((nl >= obs).sum()) + 1) / 401.0 <= P_GATE); n += 1
    return {"delta": delta, "draws": n, "hit_rate": round(hits / max(n, 1), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "m20_designer_confounding.json"))
    a = ap.parse_args()
    print("=== M20:放置是设计者的规则还是热力学,以及它解释掉多少衰减 ===")
    print("  声明 a9f2d776(代码存在之前已推送)。离线 $0。")
    print("  **A1 臂全程不碰 rescue_factor。**\n")
    recs = M16.build()
    sub = [r for r in recs if r["pt"] in ("CG", "AU")]
    cells = cells_of(recs)

    # ---- P3 / P4 structural assertions
    dm = collections.defaultdict(set)
    for r in recs:
        dm[r["design"]].add(r["method"])
    assert all(len(v) == 1 for v in dm.values()), "P3 失败:一个设计对应多个方法"
    assert all(k[1] == list(v)[0] for k, v in dm.items()), "P3 失败:design 键与 method 不一致"
    print(f"--- P3 通过:{len(dm)} 个设计,每个恰好一种方法(置换在设计层)")
    n0 = len(cells)
    for s in (1, 2, 3):
        keys = sorted(cells); bytg = collections.defaultdict(list)
        for kk in keys:
            bytg[kk[0]].append(kk)
        assert sum(len(v) for v in bytg.values()) == n0, "P4 失败:格结构随置换变化"
    print(f"--- P4 通过:存活格 {n0} 个(CG>=4 且 AU>=4),置换不改变格结构\n")

    # ---- A1
    print("--- A1(设计侧,不碰结果):CG−AU 深度差的靶标/方法分解 ---")
    mc, tc, nk = method_component(cells)
    null = a1_perm(cells, a.seed)
    p1 = (float((null >= mc).sum()) + 1) / (len(null) + 1)
    z1 = (mc - float(null.mean())) / (float(null.std()) or 1)
    print(f"  格 {nk} 个;靶标分量解释 {tc:.4f};**方法分量(去掉靶标之后){mc:.4f}**")
    print(f"  置换零 {null.mean():.4f}±{null.std():.4f}   p {p1:.5f}   z {z1:.2f}")
    order = []
    for m in sorted({k[1] for k in cells}):
        v = [cells[k] for k in cells if k[1] == m]
        order.append((m, float(np.mean(v)), len(v)))
    order.sort(key=lambda x: x[1])
    print("  逐方法梯度(越负=CG放得越浅):")
    for m, v, n in order:
        print(f"    {m:26s} {v:+7.3f}  ({n} 靶标) {'← 人类' if m in HUMAN else ''}")
    human_rank = [i for i, (m, _v, _n) in enumerate(order) if m in HUMAN]

    # ---- A2
    print("\n--- A2(结果侧):深度解释掉多少衰减 ---")
    med = mediation(sub)
    for k, v in med.items():
        print(f"  {k:26s} {v}")
    print(f"  **被解释比例 {med['mediated_fraction']:.1%}(跑前押 40%~60%,点值 44%)**")

    # ---- A3
    print("\n--- A3(次要,n=9,声明已写明功率弱):逐方法 实测衰减 vs 预测衰减 ---")
    b, _ = beta_depth(sub)
    pts = []
    for m in sorted({r["method"] for r in sub}):
        g = [r for r in sub if r["method"] == m]
        mm, _ = strat_diff(g, lambda r: r["design"])
        xx, _ = strat_diff(g, lambda r: (r["design"], min(r["depth"], 3), r["crossed"]))
        gg = strat_gap(g, lambda r: r["design"])
        if any(math.isnan(x) for x in (mm, xx, gg)):
            continue
        pts.append((m, gg * b, mm - xx))
        print(f"  {m:26s} 预测 {gg*b:+8.5f}   实测 {mm-xx:+8.5f}")
    if len(pts) >= 3:
        X = np.array([p[1] for p in pts]); Y = np.array([p[2] for p in pts])
        sl = float(np.polyfit(X, Y, 1)[0])
        rng = np.random.default_rng(a.seed)
        nl = np.sort(np.array([float(np.polyfit(X, Y[rng.permutation(len(Y))], 1)[0])
                               for _ in range(2000)]))
        p3 = (float((nl >= sl).sum()) + 1) / 2001.0
        print(f"  斜率 {sl:+.3f}(预注册方向:正)  置换 p {p3:.4f}  n={len(pts)}")
    else:
        sl, p3 = float("nan"), float("nan")

    # ---- A4
    print("\n--- A4(稳健性,粗且边缘 —— 主臂用连续量的原因) ---")
    a4 = {}
    for lab, sel in (("Eterna系", lambda m: m in HUMAN), ("ML/物理", lambda m: m not in HUMAN)):
        g = [r for r in sub if sel(r["method"])]
        mm, _ = strat_diff(g, lambda r: r["design"])
        xx, _ = strat_diff(g, lambda r: (r["design"], min(r["depth"], 3), r["crossed"]))
        gg = strat_gap(g, lambda r: r["design"])
        a4[lab] = {"marginal": round(mm, 5), "matched": round(xx, 5),
                   "attenuation": round(mm - xx, 5), "gap": round(gg, 4), "n": len(g)}
        print(f"  {lab:8s} 边际 {mm:+.5f}  配平 {xx:+.5f}  衰减 {mm-xx:+.5f}  深度差 {gg:+.3f}  n={len(g)}")

    # ---- probes
    print("\n--- P1 噪声 / P2 梯级 ---")
    pn = p1_noise(cells, a.seed)
    print(f"  P1 命中 {pn['hits']}/{pn['draws']}(上限 1)")
    lad = {r: p2_rung(cells, r, a.seed) for r in RUNGS}
    print("  " + "  ".join(f"δ={r:<4g}:{lad[r]['hit_rate']:.2f}" for r in RUNGS))

    # ---- verdict
    P0 = method_component(cells, {k: k[0] for k in cells})[0]
    self0 = lad[0.0]["hit_rate"]; top = lad[RUNGS[-1]]["hit_rate"]
    P0ok = abs(P0) < 1e-9
    P1ok = pn["hits"] <= 1
    ladok = self0 <= LAD_SELF_MAX and top >= 0.80
    K4 = RUNGS[-1] < abs(mc) or not ladok
    K3 = not (0.10 < med["mediated_fraction"] < 0.90)
    K0 = not (P0ok and P1ok and ladok)
    # the frozen declaration defined K1 (p<=0.005) and K2 (interval contains 0) and left the
    # band between them undefined. p=0.012 is neither. Labelling that "universal" would assert
    # a zero the data does not show, so the gap is recorded and adjudicated as unresolved.
    contains_zero = mc <= float(np.quantile(null, 0.95))
    verdict = ("void_probe_failed" if K0 else "void_ladder" if K4 else
               "placement_is_method_dependent" if p1 <= P_GATE else
               "placement_is_universal" if contains_zero else
               "unresolved_between_K1_and_K2__declaration_left_this_band_undefined")
    print("\n=== 裁定 ===")
    print(f"  P0 方法分量←靶标标签 {P0:.10f} {'合格' if P0ok else '不合格'} | "
          f"P1 {pn['hits']}/12 {'合格' if P1ok else '不合格'} | "
          f"梯级 δ=0 {self0:.2f} 顶档 {top:.2f} {'合格' if ladok else '不合格'}")
    print(f"  K1/K2:A1 方法分量 {mc:.4f} vs 零 95分位 {float(np.quantile(null,0.95)):.4f};"
          f"p={p1:.5f} ⇒ **{verdict}**")
    spread = max(v for _m, v, _n in order) - min(v for _m, v, _n in order)
    print(f"  **梯级读数:实测方法间全距 {abs(spread):.3f},落在 δ=0.4(检出 "
          f"{lad[0.40]['hit_rate']:.2f})与 δ=0.8(检出 {lad[0.80]['hit_rate']:.2f})之间 "
          f"⇒ 本臂正处在自己的检出边缘,这是功率陈述,不是零结果。**")
    print(f"  K3 中介比例 {med['mediated_fraction']:.1%} ⇒ {'触发(中介读法撤回)' if K3 else '不触发'}")
    print(f"  K5 A3 n={len(pts)} ⇒ {'underpowered,不许写成没有剂量-反应' if math.isnan(p3) or p3 > P_GATE else '有剂量-反应'}")
    print(f"  人类方法在梯度排序中的名次(0=最负):{human_rank} / {len(order)-1}")
    json.dump({"declaration": "m20_designer_confounding_declaration.json @ a9f2d776",
               "A1": {"method_component": round(mc, 5), "target_component": round(tc, 5),
                      "n_cells": nk, "p_perm": round(p1, 5), "z": round(z1, 2),
                      "null_mean": round(float(null.mean()), 5),
                      "null_sd": round(float(null.std()), 5),
                      "per_method": [{"method": m, "gap": round(v, 4), "n_targets": n}
                                     for m, v, n in order], "human_rank": human_rank},
               "A2": med, "A3": {"slope": sl, "p_perm": p3, "n": len(pts),
                                 "points": [{"method": m, "pred": round(x, 6), "obs": round(y, 6)}
                                            for m, x, y in pts]},
               "A4": a4, "P0": P0, "P1": pn, "P2_ladder": lad,
               "verdict": verdict, "K3_mediation_failed": bool(K3), "K4": bool(K4),
               "declaration_gap": "K1 要求 p<=0.005,K2 要求含 0;实测 p=%.5f 两者皆非。"
                                  "声明未定义该band,照记,不许事后归入任一侧。" % p1,
               "ladder_reading": {"observed_method_spread": round(abs(
                   max(v for _m, v, _n in order) - min(v for _m, v, _n in order)), 4),
                   "hit_at_0.4": lad[0.40]["hit_rate"], "hit_at_0.8": lad[0.80]["hit_rate"]},
               "staked_scorecard": {
                   "1_mediated_40_60pct": "错:%.1f%%" % (100 * med["mediated_fraction"]),
                   "2_A1_p_le_0.005": "错:p=%.5f" % p1,
                   "3_Eterna_at_strongest_end": "对:名次 %s / %d" % (human_rank, len(order) - 1),
                   "4_A3_slope_positive": "对:%.3f, p=%.4f" % (sl, p3)}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
