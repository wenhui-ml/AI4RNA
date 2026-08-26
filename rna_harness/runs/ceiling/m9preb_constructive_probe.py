#!/usr/bin/env python3
"""M9-pre-b: measure what persistence this arm can see AT ALL, then decide whether to read the main arm.

Declaration: m9preb_constructive_probe_declaration.json, frozen and pushed at a6fddaef BEFORE this file
existed. Supersedes M9-pre (void_probe_failed).

M9-pre died because its must-fire probe -- per-target accuracy as a proxy for target difficulty -- was
indistinguishable from pure noise, so it proved no detection power and the main arm could not be read.
The root cause is that per-target ranking accuracy is a property of (target, design pool), not of the
target. m2_audit.md had recorded the identical mistake one day earlier, with the fix: a CONSTRUCTIVE
probe. Rule 65 now: inject a known quantity and require the pipeline to recover it, because an
off-the-shelf observable that stays silent cannot separate "the pipeline is blind" from "that observable
was never related".

Injected: g_synth_p[t] = sqrt(rho)*u[t] + sqrt(1-rho)*e_p[t], u shared across all four populations,
e_p independent per population, so the true cross-population correlation is exactly rho. Rungs are
0.2/0.4/0.6/0.8, and 0.4 is calibrated to M9-pre's measured split-half reliability of +0.3911 -- it
stands for a FULLY persistent target-level effect at the noise level this quantity actually carries.

K5 is what makes a likely negative informative: if the detection threshold exceeds 0.4, the verdict is
no_adjudicating_power -- this construction can never settle the question, and the run says how much
persistence it would take and how many common targets would be needed.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import h2_heldout_cluster_ci as H                                             # noqa: E402
from m3_gc_cancelled import build                                             # noqa: E402
from m9pre_heavytail_vs_concentration import (POPS, MIN_COMMON, per_target,   # noqa: E402
                                              gvec, rho_ci, pairwise, shape)

RUNGS = (0.2, 0.4, 0.6, 0.8)
CALIBRATED_RUNG = 0.4          # = M9-pre's measured reliability ceiling +0.3911


def synth(Gmain, rho, seed):
    """g_synth_p[t] = sqrt(rho)*u[t] + sqrt(1-rho)*e_p[t]; u shared, e_p per population."""
    allt = sorted({t for g in Gmain.values() for t in g})
    ru = random.Random(seed)
    u = {t: ru.gauss(0.0, 1.0) for t in allt}
    out = {}
    for i, (fn, g) in enumerate(sorted(Gmain.items())):
        re_ = random.Random(seed + 1000 * (i + 1))
        out[fn] = {t: math.sqrt(rho) * u[t] + math.sqrt(1 - rho) * re_.gauss(0.0, 1.0) for t in g}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m9preb_constructive_probe.json"))
    a = ap.parse_args()
    print("=== M9-pre-b:先量本臂能看见多大的持续性,再决定主臂读不读 ===")
    print(f"  声明 a6fddaef(代码存在之前已推送)。离线 $0。承 M9-pre 的作废。\n")

    Gmain, Gnoise, Ghalf, names, SH = {}, {}, {}, {}, {}
    for nm, fn in POPS:
        print(f"=== {nm} ===", flush=True)
        keep, M = build(fn, a.workers)
        names[fn] = nm
        per, used, s = per_target(keep, M, a.seed)
        if per is None or len(used) < H.MIN_TARGETS:
            print("  无数据"); continue
        print(f"  泄漏 分子 {s['leak_mol']}/200 靶标 {s['leak_tg']}/200"
              f"  {'✓' if not s['leak_mol'] and not s['leak_tg'] else '✗'}  靶标 {len(used)}")
        Gmain[fn] = gvec(per, used)
        SH[fn] = shape(list(Gmain[fn].values()))
        print(f"  形状 均值 {SH[fn]['mean']:+.4f} 截尾10% {SH[fn]['trim10']:+.4f} "
              f"正 {SH[fn]['pos']}/{SH[fn]['n']} 峰度 {SH[fn]['kurtosis']}")
        pn, un, _ = per_target(keep, M, a.seed, noise_rng=random.Random(a.seed + 991))
        if pn is not None and len(un) >= H.MIN_TARGETS:
            Gnoise[fn] = gvec(pn, un)
        ok = [i for i, m in enumerate(M) if "error" not in m and "ens_defect_skel" in m]
        rng = random.Random(a.seed + 5); rng.shuffle(ok)
        p1, u1, _ = per_target(keep, M, a.seed, subset=set(ok[: len(ok) // 2]))
        p2, u2, _ = per_target(keep, M, a.seed, subset=set(ok[len(ok) // 2:]))
        if p1 is not None and p2 is not None:
            g1, g2 = gvec(p1, u1), gvec(p2, u2)
            cm = sorted(set(g1) & set(g2))
            if len(cm) >= MIN_COMMON:
                r, ci = rho_ci([g1[t] for t in cm], [g2[t] for t in cm], a.seed + 31)
                Ghalf[fn] = {"rho": round(r, 4), "ci": [round(ci[0], 4), round(ci[1], 4)], "n": len(cm)}
                print(f"  信度天花板(设计对半) rho **{r:+.4f}** [{ci[0]:+.4f},{ci[1]:+.4f}] (n={len(cm)})")

    if len(Gmain) < 3:
        json.dump({"verdict": "no_data"}, open(a.out, "w"), indent=1, ensure_ascii=False); return 0

    print("\n=== 构造性必响探针(第 65 条):注入已知的靶标级分量,要求取回 ===")
    LAD, thresh = {}, None
    for rho in RUNGS:
        G = synth(Gmain, rho, a.seed + int(rho * 100))
        tag = "  ← 按实测信度天花板 +0.3911 标定" if abs(rho - CALIBRATED_RUNG) < 1e-9 else ""
        rows, pos, neg, short = pairwise(G, names, a.seed, f"P0 构造性 真实 rho={rho:.1f}{tag}")
        got = st.fmean([r["rho"] for r in rows if not r.get("void")])
        LAD[str(rho)] = {"pos": pos, "neg": neg, "rows": rows, "measured_mean": round(got, 4)}
        print(f"    实测 Spearman 均值 {got:+.4f}(名义 {rho:.1f},偏离 {got - rho:+.4f})")
        if thresh is None and pos >= 5:
            thresh = rho
    print(f"\n  **检出阈值 = {thresh if thresh is not None else '> 0.8(未达到)'}**"
          f"  —— 本臂能可靠看见的最小跨人群持续性")

    rowsN, posN, _nN, _sN = pairwise(Gnoise, names, a.seed, "P1 等宽纯噪声(必须 <= 1/6)") \
        if len(Gnoise) >= 3 else ([], 0, 0, 6)

    print("\n=== 主臂:g_t = A_ed − A 的跨人群持续性 ===")
    print("  **注意:M9-pre 已污染这四个人群(我看过这张表)⇒ 本臂地位是发现,不是确证。**")
    rows, pos, neg, short = pairwise(Gmain, names, a.seed, "主臂")

    ceil_mean = st.fmean([v["rho"] for v in Ghalf.values()]) if Ghalf else None
    main_mean = st.fmean([r["rho"] for r in rows if not r.get("void")])
    ns = [r["n"] for r in rows if not r.get("void")]

    K0 = not (LAD.get("0.8", {}).get("pos", 0) >= 5) or posN > 1
    K1 = ceil_mean is not None and main_mean > ceil_mean
    K2 = pos >= 4
    K3 = pos <= 1
    K4 = short >= 3
    K5 = thresh is None or thresh > CALIBRATED_RUNG
    print("\n=== 判决条件 ===")
    for k, v, d in (("K0", K0, "rho=0.8 仍 <5/6,或噪声响 ⇒ 流程坏了,作废"),
                    ("K1", K1, "主臂超过信度天花板 ⇒ 作废"),
                    ("K2", K2, "主臂正向排除 0 >= 4"), ("K3", K3, "主臂正向排除 0 <= 1"),
                    ("K4", K4, "共同靶标不足 >= 3 ⇒ 无数据"),
                    ("K5", K5, "检出阈值 > 0.4 ⇒ **本臂对本问题无裁决力**")):
        print(f"  {k}: {'**触发**' if v else '未触发'}   {d}")

    if K0 or K1 or K4:
        verdict = "void_" + ("pipeline_broken" if K0 else ("above_ceiling" if K1 else "no_data"))
    elif K5:
        verdict = "no_adjudicating_power__construction_cannot_answer_this"
    elif K2 and not K3:
        verdict = "concentrated_effect__move_two_framework_stands"
    elif K3 and not K2:
        verdict = "heavy_tailed_noise__move_two_was_chasing_a_tail"
    else:
        verdict = "unresolved_between_heavy_tail_and_concentration"
    if neg >= 3:
        verdict += "__PLUS_anti_persistence"

    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  主臂六对均值 rho {main_mean:+.4f},正向排除 0 {pos}/6"
          + (f",信度天花板均值 {ceil_mean:+.4f}" if ceil_mean else ""))
    if K5:
        need = int(math.ceil(3 + (1.96 / CALIBRATED_RUNG) ** 2))
        print(f"\n  **K5 触发 —— 这是一个答案,不是作废:**")
        print(f"  本臂看不见一个在实测噪声水平下【完全持续】的效应(rho≈{CALIBRATED_RUNG}),"
              f"因此主臂的任何空结果都不是关于 (a)/(b) 的证据。")
        print(f"  要让本臂有裁决力,每对需要共同靶标 >= **{need}**(按 1.96/sqrt(n−3) = {CALIBRATED_RUNG} 反解);"
              f"实测共同靶标 {min(ns)}~{max(ns)}。")
        print(f"  **⇒ 这个问题不能在 GC 精确配平构造上回答。承 m8b 的绕行路线:换不配平的量重问。**")

    print("\n=== 次要臂(只报出)===")
    for fn, v in SH.items():
        print(f"  {names[fn]:<12} 均值 {v['mean']:+.4f} 截尾10% {v['trim10']:+.4f} "
              f"中位 {v['median']:+.4f} HL {v['hodges_lehmann']:+.4f} "
              f"正 {v['pos']}/{v['n']} 符号p {v['sign_p']:.3f} 峰度 {v['kurtosis']}")

    json.dump({"verdict": verdict, "detection_threshold": thresh, "ladder": LAD,
               "main": rows, "P1": rowsN, "ceiling": Ghalf,
               "shape": {names[k]: v for k, v in SH.items()},
               "means": {"main": round(main_mean, 4),
                         "ceiling": round(ceil_mean, 4) if ceil_mean else None},
               "counts": {"main_pos": pos, "main_neg": neg, "P1_pos": posN},
               "common_n": ns,
               "kill": {"K0": K0, "K1": K1, "K2": K2, "K3": K3, "K4": K4, "K5": K5}},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
