#!/usr/bin/env python3
"""M9-pre-c: same question, unmatched construction -- which buys back a clean confirmation set.

Declaration: m9prec_unmatched_heavytail_declaration.json, frozen and pushed at 13f39be7 BEFORE this file
existed.

M9-pre-b settled that the exact-GC-matched construction cannot answer whether M3's per-target effect is
heavy-tailed noise (a) or a concentrated real effect (b): reliability 0.3911 against a detection
threshold of 0.8. This takes m8b's route around the same wall -- drop the matching, keep every design
pair, and the fifth and sixth batches come back.

WHAT THIS COSTS, STATED FIRST. Dropping the matching changes the quantity: M3's +0.0140 was measured
after GC was cancelled by construction, and without matching the scored pair set (and so its difficulty
mix) is different. A BRIDGE ARM measures rho(g_t^matched, g_t^unmatched) within each core population; if
it is low the verdict carries a mandatory __about_a_different_quantity suffix, because then this
experiment is answering something else.

WHAT IT BUYS. The two blind batches have never been used on this question, so for the first time there
is an unseen confirmation set here -- and it exists precisely because exact cancellation was given up.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import h2_heldout_cluster_ci as H                                             # noqa: E402
from e1_ensemble_marginals import base_block                                  # noqa: E402
from m3_gc_cancelled import build, heldout                                    # noqa: E402
from m9pre_heavytail_vs_concentration import MIN_COMMON, rho_ci, shape        # noqa: E402

CORE = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
BLIND = (("第五批(盲)", "panel_f1_fifth.json"), ("第六批(盲)", "panel_f1_sixth.json"))
POPS = CORE + BLIND
BLIND_FN = {fn for _n, fn in BLIND}
RUNGS = (0.1, 0.2, 0.3, 0.4, 0.6)
K5_BAR = 0.3


def per_target_c(keep, M, seed, mode, subset=None, noise_rng=None):
    """M3 arm 1 columns verbatim; `mode` selects exact GC matching or none. A_G included for P2."""
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
    idx = {"A": list(range(nB)), "A_ed": list(range(nB)) + [nB + 2],
           "A_G": list(range(nB)) + [nB, nB + 1]}
    per, used, s, _ref = heldout(rws, cols, idx, key, seed, mode)
    return per, used, s


def gv(per, used, a="A_ed", b="A"):
    return {t: st.fmean(per[a][t]) - st.fmean(per[b][t]) for t in used}


def pairs_tagged(G, names, seed, label, quiet=False):
    """all population pairs, tagged core/blind; returns rows and per-stratum counts."""
    rows = []
    ks = sorted(G)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            nb = (a in BLIND_FN) + (b in BLIND_FN)
            strat = "core" if nb == 0 else ("blind_core" if nb == 1 else "blind_blind")
            common = sorted(set(G[a]) & set(G[b]))
            if len(common) < MIN_COMMON:
                rows.append({"pair": [names[a], names[b]], "n": len(common), "void": True,
                             "strat": strat})
                continue
            r, ci = rho_ci([G[a][t] for t in common], [G[b][t] for t in common], seed + 7 * i + j)
            rows.append({"pair": [names[a], names[b]], "n": len(common), "rho": round(r, 4),
                         "ci": [round(ci[0], 4), round(ci[1], 4)], "strat": strat,
                         "excl_pos": bool(ci[0] is not None and ci[0] > 0),
                         "excl_neg": bool(ci[1] is not None and ci[1] < 0)})

    def cnt(sel):
        v = [x for x in rows if x["strat"] in sel and not x.get("void")]
        return (sum(1 for x in v if x["excl_pos"]), sum(1 for x in v if x["excl_neg"]), len(v))
    core = cnt({"core"}); conf = cnt({"blind_core", "blind_blind"}); bb = cnt({"blind_blind"})
    if not quiet:
        print(f"\n  --- {label} ---")
        for x in rows:
            if x.get("void"):
                print(f"    [{x['strat']:<10}] {x['pair'][0]:<11}× {x['pair'][1]:<11} n={x['n']:3d} **作废**")
            else:
                m = "**正向排除0**" if x["excl_pos"] else ("**负向排除0**" if x["excl_neg"] else "含 0")
                print(f"    [{x['strat']:<10}] {x['pair'][0]:<11}× {x['pair'][1]:<11} n={x['n']:3d}"
                      f"  rho {x['rho']:+.4f} [{x['ci'][0]:+.4f},{x['ci'][1]:+.4f}]  {m}")
        print(f"    ⇒ 发现集(核心×核心) 正向 {core[0]}/{core[2]} | "
              f"**确证集(含盲) 正向 {conf[0]}/{conf[2]}** | 其中盲×盲 {bb[0]}/{bb[2]}")
    return rows, core, conf, bb


def synth(G, rho, seed):
    allt = sorted({t for g in G.values() for t in g})
    ru = random.Random(seed)
    u = {t: ru.gauss(0.0, 1.0) for t in allt}
    out = {}
    for i, (fn, g) in enumerate(sorted(G.items())):
        re_ = random.Random(seed + 1000 * (i + 1))
        out[fn] = {t: math.sqrt(rho) * u[t] + math.sqrt(1 - rho) * re_.gauss(0.0, 1.0) for t in g}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m9prec_unmatched_heavytail.json"))
    a = ap.parse_args()
    print("=== M9-pre-c:不配平构造上的重尾判别 —— 放弃精确相消,换回一个干净的确证集 ===")
    print(f"  声明 13f39be7(代码存在之前已推送)。离线 $0。K5 门槛 {K5_BAR}。\n")

    G, Gn, Ghalf, names, SH, BRIDGE, P2 = {}, {}, {}, {}, {}, {}, {}
    for nm, fn in POPS:
        print(f"=== {nm} ===", flush=True)
        keep, M = build(fn, a.workers)
        names[fn] = nm
        per, used, s = per_target_c(keep, M, a.seed, "none")
        if per is None or len(used) < H.MIN_TARGETS:
            print(f"  靶标不足 ⇒ 本人群无数据"); continue
        print(f"  泄漏 分子 {s['leak_mol']}/200 靶标 {s['leak_tg']}/200"
              f"  {'✓' if not s['leak_mol'] and not s['leak_tg'] else '✗'}  靶标 {len(used)}")
        G[fn] = gv(per, used)
        SH[fn] = shape(list(G[fn].values()))
        gg = [st.fmean(per["A_G"][t]) - st.fmean(per["A"][t]) for t in used]
        ci, _n = H.boot_ci(gg, a.seed + 17)
        P2[fn] = {"mean": round(st.fmean(gg), 4), "ci": ci,
                  "excl": bool(ci[0] > 0 or ci[1] < 0)}
        print(f"  P2 构造断言 A_G−A = {P2[fn]['mean']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]"
              f"  {'**排除0 ✓ 不配平确实生效**' if P2[fn]['excl'] else '✗ 含 0 —— 构造没换成'}")
        print(f"  形状 均值 {SH[fn]['mean']:+.4f} 截尾10% {SH[fn]['trim10']:+.4f} "
              f"正 {SH[fn]['pos']}/{SH[fn]['n']} 峰度 {SH[fn]['kurtosis']}")

        pn, un, _ = per_target_c(keep, M, a.seed, "none", noise_rng=random.Random(a.seed + 991))
        if pn is not None and len(un) >= H.MIN_TARGETS:
            Gn[fn] = gv(pn, un)
        ok = [i for i, m in enumerate(M) if "error" not in m and "ens_defect_skel" in m]
        rng = random.Random(a.seed + 5); rng.shuffle(ok)
        p1, u1, _ = per_target_c(keep, M, a.seed, "none", subset=set(ok[: len(ok) // 2]))
        p2, u2, _ = per_target_c(keep, M, a.seed, "none", subset=set(ok[len(ok) // 2:]))
        if p1 is not None and p2 is not None:
            g1, g2 = gv(p1, u1), gv(p2, u2)
            cm = sorted(set(g1) & set(g2))
            if len(cm) >= MIN_COMMON:
                r, ci = rho_ci([g1[t] for t in cm], [g2[t] for t in cm], a.seed + 31)
                Ghalf[fn] = {"rho": round(r, 4), "ci": [round(ci[0], 4), round(ci[1], 4)], "n": len(cm)}
                print(f"  **信度天花板(不配平,设计对半) rho {r:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] (n={len(cm)})**"
                      f"  跑前预测 0.85 [0.75,0.92]")
        if fn not in BLIND_FN:                                   # bridge arm, core only
            pm, um, _ = per_target_c(keep, M, a.seed, "exact")
            if pm is not None and um is not None and len(um) >= H.MIN_TARGETS:
                gm = gv(pm, um)
                cm = sorted(set(gm) & set(G[fn]))
                if len(cm) >= MIN_COMMON:
                    r, ci = rho_ci([gm[t] for t in cm], [G[fn][t] for t in cm], a.seed + 43)
                    BRIDGE[fn] = {"rho": round(r, 4), "ci": [round(ci[0], 4), round(ci[1], 4)],
                                  "n": len(cm)}
                    print(f"  桥臂 rho(配平 g_t, 不配平 g_t) = **{r:+.4f}** "
                          f"[{ci[0]:+.4f},{ci[1]:+.4f}] (n={len(cm)})")

    if len(G) < 4:
        json.dump({"verdict": "no_data", "n_pops": len(G)}, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n**可用人群 < 4 ⇒ 无数据**"); return 0

    print("\n=== 构造性梯级(第 65/66 条:本构造自己的灵敏度,不沿用 M9-pre-b)===")
    LAD, thr_core, thr_conf = {}, None, None
    for rho in RUNGS:
        Gs = synth(G, rho, a.seed + int(rho * 100))
        _r, c, cf, _bb = pairs_tagged(Gs, names, a.seed, f"P0 构造性 真实 rho={rho:.1f}", quiet=True)
        LAD[str(rho)] = {"core_pos": c[0], "core_n": c[2], "conf_pos": cf[0], "conf_n": cf[2]}
        print(f"  rho={rho:.1f}  发现集 {c[0]}/{c[2]}  **确证集 {cf[0]}/{cf[2]}**")
        if thr_core is None and c[2] and c[0] >= math.ceil(5 / 6 * c[2]):
            thr_core = rho
        if thr_conf is None and cf[2] and cf[0] >= math.ceil(7 / 9 * cf[2]):
            thr_conf = rho
    print(f"  **检出阈值 发现集 {thr_core if thr_core else '>0.6'} | 确证集 {thr_conf if thr_conf else '>0.6'}**")

    _rn, cn, cfn, _b = pairs_tagged(Gn, names, a.seed, "P1 等宽纯噪声(必须 <= 1/6 发现集)") \
        if len(Gn) >= 4 else ([], (9, 0, 0), (9, 0, 0), (0, 0, 0))

    print("\n=== 主臂:g_t = A_ed − A(不配平)的跨人群持续性 ===")
    rows, core, conf, bb = pairs_tagged(G, names, a.seed, "主臂")

    ceil_mean = st.fmean([v["rho"] for v in Ghalf.values()]) if Ghalf else None
    main_mean = st.fmean([r["rho"] for r in rows if not r.get("void")])
    br_mean = st.fmean([v["rho"] for v in BRIDGE.values()]) if BRIDGE else None
    p2_ok = all(v["excl"] for v in P2.values()) and len(P2) >= 4

    K0 = (cn[0] > 1) or (not p2_ok)
    K1 = ceil_mean is not None and main_mean > ceil_mean
    K2 = conf[2] > 0 and conf[0] >= math.ceil(7 / 9 * conf[2])
    K3 = conf[2] > 0 and conf[0] <= math.floor(2 / 9 * conf[2])
    K4 = sum(1 for r in rows if r.get("void")) >= 5
    K5 = (thr_conf is None) or (thr_conf > K5_BAR)
    print("\n=== 判决条件 ===")
    for k, v, d in (("K0", K0, "噪声响 或 P2 构造断言不响 ⇒ 作废"), ("K1", K1, "超过信度天花板 ⇒ 作废"),
                    ("K2", K2, "确证集正向 >= 7/9 ⇒ (b) 效应集中"),
                    ("K3", K3, "确证集正向 <= 2/9 ⇒ (a) 重尾"),
                    ("K4", K4, "作废对 >= 5 ⇒ 无数据"), ("K5", K5, f"确证集检出阈值 > {K5_BAR} ⇒ 无裁决力")):
        print(f"  {k}: {'**触发**' if v else '未触发'}   {d}")

    if K0 or K1 or K4:
        verdict = "void_" + ("probe_or_construction_failed" if K0 else ("above_ceiling" if K1 else "no_data"))
    elif K5:
        verdict = "no_adjudicating_power__unmatched_construction_also_too_noisy"
    elif K2 and not K3:
        verdict = "concentrated_effect__persists_on_blind"
    elif K3 and not K2:
        verdict = "heavy_tailed_noise__no_persistence_on_blind"
    else:
        verdict = "unresolved"
    if conf[1] >= math.ceil(5 / 9 * max(1, conf[2])):
        verdict += "__PLUS_anti_persistence"
    if core[2] and conf[2] and ((core[0] > 0) != (conf[0] > 0)):
        verdict += "__core_blind_disagreement"
    if br_mean is not None and br_mean < 0.4:
        verdict += "__about_a_different_quantity"

    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  主臂 15 对均值 rho {main_mean:+.4f};发现集正向 {core[0]}/{core[2]},"
          f"确证集正向 {conf[0]}/{conf[2]}(盲×盲 {bb[0]}/{bb[2]})")
    if ceil_mean is not None:
        print(f"  **信度天花板实测均值 {ceil_mean:+.4f} —— 跑前预测 0.85 [0.75,0.92]:"
              f"{'预测成立' if 0.75 <= ceil_mean <= 0.92 else '**预测被否**'}"
              f"{',且 <0.6 ⇒ 对数不是噪声主因,我对噪声来源的理解是错的' if ceil_mean < 0.6 else ''}**")
    if br_mean is not None:
        print(f"  桥臂均值 rho(配平, 不配平) = **{br_mean:+.4f}** ⇒ "
              f"{'两个量是同一件事的两种测法,结论搬得回 M3' if br_mean >= 0.4 else '**搬不回 M3,本实验回答的是另一个问题**'}")

    print("\n=== 次要臂(只报出)===")
    for fn, v in SH.items():
        print(f"  {names[fn]:<12} 均值 {v['mean']:+.4f} 截尾10% {v['trim10']:+.4f} "
              f"中位 {v['median']:+.4f} 正 {v['pos']}/{v['n']} 符号p {v['sign_p']:.3f} 峰度 {v['kurtosis']}")

    json.dump({"verdict": verdict, "threshold_core": thr_core, "threshold_conf": thr_conf,
               "ladder": LAD, "main": rows, "ceiling": Ghalf, "bridge": BRIDGE, "P2": P2,
               "shape": {names[k]: v for k, v in SH.items()},
               "counts": {"core": core, "conf": conf, "blind_blind": bb, "noise_core": cn},
               "means": {"main": round(main_mean, 4),
                         "ceiling": round(ceil_mean, 4) if ceil_mean else None,
                         "bridge": round(br_mean, 4) if br_mean else None},
               "kill": {"K0": K0, "K1": K1, "K2": K2, "K3": K3, "K4": K4, "K5": K5}},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
