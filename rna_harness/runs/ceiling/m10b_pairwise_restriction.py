#!/usr/bin/env python3
"""M10b: force two populations onto the same targets, and see whether their length dependence still differs.

Declaration: m10b_pairwise_restriction_declaration.json, frozen and pushed at 15cd2395 BEFORE this file
existed. Supersedes M10 (no_data on a 22-target six-way intersection).

Pairwise intersections are 52-56, more than double the six-way 22. And the two rhos are computed on the
SAME resampled targets, so D = rho_a - rho_b is PAIRED and far narrower than either margin -- the third
application of the lesson M4b paid for with its K4.

The question: target composition or design composition. Forcing both populations onto identical targets
removes composition by construction, so whatever remains in D is the design pool.

P3 is what makes this honest rather than assumed: the paired half-width must actually be NARROWER than
half the sum of the marginal half-widths. No earlier run on this line ever checked that its paired
difference was really paired.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from scipy.stats import spearmanr                                             # noqa: E402
from m1c_skeleton_bookkeeping import build                                    # noqa: E402
from m8b_target_length import CORE, BLIND, run_pop                            # noqa: E402

M9_REF = {"样本2(主)": -0.4584, "基准4560": +0.2609, "样本3": -0.1523,
          "第四批": -0.2000, "第五批(盲)": +0.1162, "第六批(盲)": +0.4187}
GAPS, DRAWS, PASS_FRAC, NBOOT, MIN_C = (0.1, 0.2, 0.4, 0.6, 0.8), 20, 0.8, 3000, 25


def sp(x, y):
    v = spearmanr(x, y)[0]
    return 0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v


def paired_D(La, ya, Lb, yb, seed, nb=NBOOT):
    """rho_a - rho_b, bootstrapping the SAME target indices for both -> genuinely paired."""
    n = len(La)
    d0 = sp(La, ya) - sp(Lb, yb)
    rng = random.Random(seed); B, Ba, Bb = [], [], []
    for _ in range(nb):
        ix = [rng.randrange(n) for _ in range(n)]
        xa = [La[i] for i in ix]; wa = [ya[i] for i in ix]
        xb = [Lb[i] for i in ix]; wb = [yb[i] for i in ix]
        if len(set(xa)) < 3 or len(set(wa)) < 3 or len(set(wb)) < 3:
            continue
        ra, rb = sp(xa, wa), sp(xb, wb)
        B.append(ra - rb); Ba.append(ra); Bb.append(rb)
    if not B:
        return None
    B.sort(); Ba.sort(); Bb.sort()
    q = lambda v: [v[int(.025 * len(v))], v[int(.975 * len(v))]]
    ci = q(B); ca = q(Ba); cb = q(Bb)
    return {"D": round(d0, 4), "ci": [round(ci[0], 4), round(ci[1], 4)],
            "rho_a": round(sp(La, ya), 4), "rho_b": round(sp(Lb, yb), 4),
            "hw_paired": round((ci[1] - ci[0]) / 2, 4),
            "hw_marg_sum_half": round(((ca[1] - ca[0]) / 2 + (cb[1] - cb[0]) / 2) / 2, 4),
            "hw_a": round((ca[1] - ca[0]) / 2, 4), "hw_b": round((cb[1] - cb[0]) / 2, 4),
            "excl": bool(ci[0] > 0 or ci[1] < 0), "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m10b_pairwise_restriction.json"))
    a = ap.parse_args()
    print("=== M10b:把两个人群按到同一批靶标上,长度依赖还差不差 ===")
    print("  声明 15cd2395(代码存在之前已推送)。离线 $0。配对 bootstrap。\n")

    D, names, core_fn = {}, {}, {fn for _n, fn in CORE}
    for nm, fn in CORE + BLIND:
        print(f"=== {nm} ===", flush=True)
        keep, M = build(fn, a.workers)
        d = run_pop(nm, keep, M, a.splits, a.seed, 9500)
        if d is not None:
            D[fn] = d; names[fn] = nm
    if len(D) < 5:
        json.dump({"verdict": "no_data"}, open(a.out, "w"), indent=1, ensure_ascii=False); return 0

    print("\n=== P2 复现 M9(全集 rho)===")
    bad = []
    for fn in D:
        u = D[fn]["used"]
        r = sp([D[fn]["len"][t] for t in u], [D[fn]["vec"]["A"][t] for t in u])
        ref = M9_REF.get(names[fn])
        ok = ref is not None and abs(r - ref) <= 0.02
        if not ok:
            bad.append((names[fn], round(r, 4), ref))
        print(f"  {names[fn]:<13} 本次 {r:+.4f}  M9 {ref:+.4f}  {'✓' if ok else '✗'}")
    print(f"  ⇒ {'✓ 六人群全部复现' if not bad else f'✗ {bad} ⇒ 作废'}")

    def run_pairs(ymap, label, quiet=False):
        rows = []
        ks = list(D)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                fa, fb = ks[i], ks[j]
                C = sorted(set(D[fa]["used"]) & set(D[fb]["used"]))
                strat = "core" if (fa in core_fn and fb in core_fn) else "blind"
                if len(C) < MIN_C:
                    rows.append({"pair": [names[fa], names[fb]], "n": len(C), "void": True,
                                 "strat": strat}); continue
                r = paired_D([D[fa]["len"][t] for t in C], [ymap(fa)[t] for t in C],
                             [D[fb]["len"][t] for t in C], [ymap(fb)[t] for t in C],
                             a.seed + 7 * i + j)
                rows.append({**r, "pair": [names[fa], names[fb]], "strat": strat})
        if not quiet:
            print(f"\n  --- {label} ---")
            for x in rows:
                if x.get("void"):
                    print(f"    [{x['strat']:<5}] {x['pair'][0]:<11}× {x['pair'][1]:<11} n={x['n']:3d} **作废**")
                else:
                    m = "**排除0**" if x["excl"] else "含 0"
                    print(f"    [{x['strat']:<5}] {x['pair'][0]:<11}× {x['pair'][1]:<11} n={x['n']:3d}"
                          f"  rho {x['rho_a']:+.3f} / {x['rho_b']:+.3f}"
                          f"  **D {x['D']:+.4f}** [{x['ci'][0]:+.4f},{x['ci'][1]:+.4f}]  {m}")
        cx = [x for x in rows if x["strat"] == "core" and not x.get("void")]
        return rows, sum(1 for x in cx if x["excl"]), len(cx)

    print("\n=== 主表:限制到逐对共同靶标,配对差 D = rho_a − rho_b ===")
    rows, cpos, cn = run_pairs(lambda fn: D[fn]["vec"]["A"], "主臂")
    print(f"  ⇒ **核心 6 对里 D 排除 0 的 {cpos}/{cn}**")

    print("\n=== P3 配对是否真的更窄(必须 hw_paired < hw_marg_sum_half)===")
    p3 = [x for x in rows if not x.get("void")]
    okp = sum(1 for x in p3 if x["hw_paired"] < x["hw_marg_sum_half"])
    for x in p3[:6]:
        print(f"    {x['pair'][0]:<11}× {x['pair'][1]:<11} 配对半宽 {x['hw_paired']:.4f}"
              f"  vs 边际半宽均值 {x['hw_marg_sum_half']:.4f}"
              f"  {'✓' if x['hw_paired'] < x['hw_marg_sum_half'] else '✗'}")
    print(f"  ⇒ **{okp}/{len(p3)} 对上配对确实更窄**{'  ✓' if okp >= len(p3) * 0.8 else '  ✗ 配对没生效 ⇒ 作废'}")

    print("\n=== P1 噪声臂(12 根等宽噪声替换长度)===")
    nz = 0
    for jj in range(12):
        _r, cp, _cn = run_pairs(lambda fn, j=jj: D[fn]["noise"][f"TN{j}"], "", quiet=True)
        nz += (cp >= 3)
    print(f"  12 根噪声里核心 >=3/6 排除 0 的 **{nz}/12**  {'✓' if nz <= 1 else '✗ 作废'}")

    print(f"\n=== 构造性梯级(第 67 条,每档 {DRAWS} 次)===")
    LAD, thr = {}, None
    ks = [fn for fn in D if fn in core_fn]
    for g in GAPS:
        hits = []
        for k in range(DRAWS):
            rr = random.Random(a.seed + int(g * 100) + 7919 * k)
            syn = {}
            for q, fn in enumerate(D):                      # bug fix: 必须覆盖全部人群
                u = D[fn]["used"]
                Lr = sorted(range(len(u)), key=lambda i: D[fn]["len"][u[i]])
                z = [0.0] * len(u)
                for rank, i in enumerate(Lr):
                    z[i] = (rank - len(u) / 2) / (len(u) / 2)
                sgn = 1.0 if (fn in core_fn and q % 2 == 0) else -1.0
                syn[fn] = {t: sgn * (g / 2) * z[i] + rr.gauss(0, 1) for i, t in enumerate(u)}
            _r, cp, cn2 = run_pairs(lambda fn: syn[fn], "", quiet=True)
            hits.append(1 if cn2 and cp >= 3 else 0)
        f = st.fmean(hits)
        LAD[str(g)] = {"pass_frac": round(f, 3)}
        print(f"  真实 |D|≈{g:.1f}  核心 >=3/6 达标比例 **{f:.2f}** ({int(f*DRAWS)}/{DRAWS})")
        if thr is None and f >= PASS_FRAC:
            thr = g
    print(f"  **检出阈值 = {thr if thr else '> 0.8'}**")

    K0 = bool(bad) or nz > 1 or okp < len(p3) * 0.8
    K3 = sum(1 for x in rows if x.get("void") and x["strat"] == "core") >= 3
    K4 = (thr is None) or thr > 0.4
    K1 = cpos >= 3
    K2 = cpos <= 1
    cx = [x for x in rows if x["strat"] == "core" and not x.get("void") and x["excl"]]
    shared = set(cx[0]["pair"]) if cx else set()
    for x in cx[1:]:
        shared &= set(x["pair"])
    print("\n=== 判决条件 ===")
    for k, v, d in (("K0", K0, "P1/P2/P3 任一不合格 ⇒ 作废"), ("K1", K1, f"核心排除 0 {cpos}/{cn} >= 3 ⇒ 设计组成"),
                    ("K2", K2, "核心排除 0 <= 1 ⇒ 靶标组成"), ("K3", K3, "作废对 >= 3 ⇒ 无数据"),
                    ("K4", K4, f"检出阈值 {thr} > 0.4 ⇒ 无裁决力")):
        print(f"  {k}: {'**触发**' if v else '未触发'}   {d}")

    if K0 or K3:
        verdict = "void_probe_failed" if K0 else "no_data"
    elif K4:
        verdict = "no_adjudicating_power"
    elif K1 and not K2:
        verdict = "design_composition"
    elif K2 and not K1:
        verdict = "target_composition"
    else:
        verdict = "unresolved"
    if cx and shared:
        verdict += "__driven_by_" + list(shared)[0]

    print(f"\n=== 裁定 ===\n  **{verdict}**")
    if cx and shared:
        print(f"  **第 64 条形状三:所有排除 0 的对都含 {list(shared)[0]} ⇒ 去掉它之后剩 "
              f"{sum(1 for x in cx if list(shared)[0] not in x['pair'])} 对**")
    print("\n  **第 61 条照引:样本3 × 第四批 的全集差只有 0.048,远小于任何半宽 —— "
          "该对无论结果如何都不携带信息,已在跑前写明。**")

    json.dump({"verdict": verdict, "rows": rows, "core_excl": int(cpos), "core_n": int(cn),
               "ladder": LAD, "threshold": thr, "noise_hits": int(nz), "P2_bad": bad,
               "P3_ok": int(okp), "P3_n": len(p3), "shared_pop": list(shared),
               "kill": {k: bool(v) for k, v in (("K0", K0), ("K1", K1), ("K2", K2), ("K3", K3), ("K4", K4))}},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
