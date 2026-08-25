#!/usr/bin/env python3
"""M8c: length net of wet spread, and a blind arm whose power was computed before it ran.

M8b left the length lead un-citable, owing three things. This run pays the first two:
  1. rho(len, capture^GC | wet_sd) -- partial, because rho(len, wet_sd) is -0.39..-0.52 on the core
     populations, STRONGER than the length-capture relation it might be casting as a shadow.
  2. A pooled blind arm of 71 targets (40 + 31), rank-standardised within population before pooling.
     Expected half-width 1.96/sqrt(71-3) = 0.238 against a discovery effect of -0.3065, ratio 1.29 > 1.
     Written into the declaration BEFORE running (rule 61, which M8b's failure produced).
The third debt -- the panel pointing the other way -- is measured here and deliberately not explained.

CIRCULARITY, DISCLOSED: wet_sd is computed from wet, and capture's denominator is a functional of the
wet distribution, so the wet_sd arm carries a known circularity and is kept out of the main judgement.
The empirical reason to think the channel is weak: M4d measured pure-noise models at capture +0.0020
and +0.0142, so there is no "spread drives the capture LEVEL" channel visible; noise in y attenuates
correlation rather than creating it. A reason, not a proof.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from m2_invariant_split import spearman, _ranks                                   # noqa: E402
from m8b_target_length import run_pop, CORE, BLIND, POPS, NBOOT                    # noqa: E402
from m1c_skeleton_bookkeeping import build                                        # noqa: E402

M8B_REF = {}                                                                      # filled from m8b json


def resid(y, x):
    """least-squares residual of y on x (both already rank-transformed)."""
    mx, my = st.fmean(x), st.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    b = (sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx) if sxx > 0 else 0.0
    return [c - my - b * (a - mx) for a, c in zip(x, y)]


def partial_spearman(xv, yv, zv):
    rx, ry, rz = _ranks(xv), _ranks(yv), _ranks(zv)
    ex, ey = resid(rx, rz), resid(ry, rz)
    return spearman(ex, ey), ex, ey


def pboot(xv, yv, zv, seed, partial=True):
    rng = random.Random(seed); n = len(xv); vs = []
    for _ in range(NBOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        x = [xv[i] for i in idx]; y = [yv[i] for i in idx]; z = [zv[i] for i in idx]
        vs.append(partial_spearman(x, y, z)[0] if partial else spearman(x, y))
    vs.sort()
    return [round(vs[int(0.025 * NBOOT)], 4), round(vs[int(0.975 * NBOOT)], 4)]


def flip(xv, yv, zv, seed0, partial=True):
    n = 0
    for s in range(50):
        c = pboot(xv, yv, zv, seed0 + 13 * s, partial)
        n += 1 if (c[0] > 0 or c[1] < 0) else 0
    return n


def zrank(v):
    r = _ranks(v); m = st.fmean(r); s = st.stdev(r) if len(r) > 1 else 1.0
    return [(a - m) / s for a in r] if s > 0 else [0.0] * len(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m8c_partial_and_pooled_blind.json"))
    a = ap.parse_args()
    ref = json.load(open(os.path.join(HERE, "m8b_target_length.json")))
    for fn, r in ref["GC"].items():
        M8B_REF[fn] = r["rho"]

    D, NM = {}, {}
    print("=== M8c:扣掉 wet 离散度之后,长度还剩多少;以及一条算过检出力的合并盲臂 ===", flush=True)
    for j, (nm, fn) in enumerate(POPS):
        keep, M = build(fn, a.workers)
        d = run_pop(nm, keep, M, a.splits, a.seed, 9500 + 157 * j)                 # same seeds as M8b
        if d:
            D[fn] = d; NM[fn] = nm
    core = [fn for _n, fn in CORE if fn in D]
    blind = [fn for _n, fn in BLIND if fn in D]

    def vecs(fn, ykey="GC_only"):
        u = D[fn]["used"]
        return ([D[fn]["len"][t] for t in u], [D[fn]["vec"][ykey][t] for t in u],
                [D[fn]["wsd"][t] for t in u])

    print("\n=== 必响探针(第 54 条)===")
    dev = [abs(spearman(*vecs(fn)[:2]) - M8B_REF[fn]) for fn in D]
    p0 = max(dev) < 0.02
    print(f"  P0 复现 M8b 的原始相关  最大差 {max(dev):.4f}  {'✓ 响' if p0 else '✗ 没响'}")
    x, y, z = vecs(core[0]); _r, ex, ey = partial_spearman(x, y, z)
    c1 = abs(spearman(ex, _ranks(z))); c2 = abs(spearman(ey, _ranks(z)))
    p1 = c1 < 0.01 and c2 < 0.01
    print(f"  P1 残差真的正交于 wet_sd  |rho| {c1:.5f} / {c2:.5f}  {'✓ 响' if p1 else '✗ 没响'}")
    PX, PY, PZ, IND = [], [], [], []
    for k, fn in enumerate(blind):
        xx, yy, zz = vecs(fn)
        PX += zrank(xx); PY += zrank(yy); PZ += zrank(zz); IND += [float(k)] * len(xx)
    p2 = abs(spearman(IND, PY)) < 0.05 if blind else False
    print(f"  P2 合并不是水平假象  rho(人群指示, capture 秩) {spearman(IND, PY):+.5f}  {'✓ 响' if p2 else '✗ 没响'}")
    probes = {"P0_dev": round(max(dev), 4), "P1_orth": [round(c1, 5), round(c2, 5)],
              "P2_ind": round(spearman(IND, PY), 5) if blind else None,
              "fired": {"P0": p0, "P1": p1, "P2": p2}}
    if not (p0 and p1 and p2):
        json.dump({"probes": probes, "verdict": "void_probe_did_not_fire"},
                  open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n**必响探针没有全响 ⇒ 本次运行作废(第 54 条)。**"); return 0

    print("\n=== 主表:原始 vs 扣掉 wet 离散度之后(GC_only)===")
    print(f"  {'人群':<16}{'原始 rho':>10}{'偏 rho':>10}{'95%(偏)':>24}{'翻转':>8}{'上界<0':>8}{'标注':>11}")
    R = {}
    for fn in D:
        xx, yy, zz = vecs(fn)
        raw = spearman(xx, yy)
        pr, _e1, _e2 = partial_spearman(xx, yy, zz)
        ci = pboot(xx, yy, zz, a.seed + 17)
        fl = flip(xx, yy, zz, a.seed + 991)
        R[fn] = {"raw": round(raw, 4), "partial": round(pr, 4), "ci": ci, "flip": fl,
                 "neg": bool(ci[1] < 0), "n": len(xx)}
        tag = "discovery" if fn in core else "**BLIND**"
        print(f"  {NM[fn]:<16}{raw:+10.4f}{pr:+10.4f}   [{ci[0]:+.4f},{ci[1]:+.4f}]{fl:>6}/50"
              f"{'✓' if R[fn]['neg'] else '✗':>8}{tag:>11}")

    print("\n=== 合并盲臂(71 靶标,人群内秩标准化;第 61 条:跑前算过检出力 1.29)===")
    pr_pool, _e1, _e2 = partial_spearman(PX, PY, PZ)
    ci_pool = pboot(PX, PY, PZ, a.seed + 17)
    fl_pool = flip(PX, PY, PZ, a.seed + 991)
    raw_pool = spearman(PX, PY)
    print(f"  合并盲臂 n={len(PX)}  原始 {raw_pool:+.4f}  **偏 {pr_pool:+.4f}  95% [{ci_pool[0]:+.4f},{ci_pool[1]:+.4f}]**"
          f"  翻转 {fl_pool}/50  半宽 {(ci_pool[1]-ci_pool[0])/2:.4f}")

    print("\n=== 带循环风险的臂(不进主判据):wet_sd 本身 ===")
    for fn in D:
        xx, yy, zz = vecs(fn)
        print(f"  {NM[fn]:<16}rho(wet_sd, capture^GC) {spearman(zz, yy):+.4f}")

    print("\n=== 对照臂 ===")
    AP, PL, NZ = {}, {}, []
    for fn in D:
        xa, ya, za = vecs(fn, "A")
        AP[fn] = {"partial": round(partial_spearman(xa, ya, za)[0], 4)}
        u = D[fn]["used"]
        pl = [D[fn]["plac"][t] for t in u]
        xg, yg, zg = vecs(fn)                      # 安慰剂:把长度换成靶标间打乱的长度,y/z 不变
        PL[fn] = {"partial": round(partial_spearman(pl, yg, zg)[0], 4)}
    print(f"  {'人群':<16}{'面板偏 rho':>14}{'安慰剂偏 rho':>16}")
    for fn in D:
        print(f"  {NM[fn]:<16}{AP[fn]['partial']:+14.4f}{PL[fn]['partial']:+16.4f}")
    PA = [zrank([D[fn]["vec"]["A"][t] for t in D[fn]["used"]]) for fn in blind]
    PAflat = [v for s in PA for v in s]
    ap_pool = partial_spearman(PX, PAflat, PZ)[0]
    ap_ci = pboot(PX, PAflat, PZ, a.seed + 17)
    PLflat = [v for fn in blind for v in zrank([D[fn]["plac"][t] for t in D[fn]["used"]])]
    pl_pool = partial_spearman(PLflat, PY, PZ)[0]
    pl_ci = pboot(PLflat, PY, PZ, a.seed + 17)
    print(f"  合并盲臂 面板 偏 {ap_pool:+.4f} [{ap_ci[0]:+.4f},{ap_ci[1]:+.4f}]"
          f" | 安慰剂 偏 {pl_pool:+.4f} [{pl_ci[0]:+.4f},{pl_ci[1]:+.4f}]")
    nhit = 0
    for j in range(12):
        NF = [v for fn in blind for v in zrank([D[fn]["noise"][f"TN{j}"][t] for t in D[fn]["used"]])]
        c = pboot(NF, PY, PZ, a.seed + 17)
        NZ.append(1 if c[1] < 0 else 0); nhit += NZ[-1]
    print(f"  12 根噪声在合并盲臂上负向排除 0:{NZ}  **{nhit}/12**")

    core_raw = st.fmean([R[fn]["raw"] for fn in core])
    core_par = st.fmean([R[fn]["partial"] for fn in core])
    K = {"K0": nhit >= 2,
         "K1": not (ci_pool[1] < 0),
         "K2": abs(core_par) <= 0.5 * abs(core_raw),
         "K3": bool(pl_ci[1] < 0),
         "K4": bool(ap_ci[0] > 0)}
    print("\n=== 判决条件 ===")
    for k, v in K.items():
        print(f"  {k}: {'**触发**' if v else '未触发'}")
    if K["K0"]:
        verdict = "void_partial_pipeline_manufactures_negative"
    elif K["K3"]:
        verdict = "void_any_target_column_goes_negative_under_partial"
    else:
        verdict = ("blind_null_on_partial" if K["K1"]
                   else "length_survives_net_of_spread__blind_confirmed")
        if K["K2"]:
            verdict += "__mostly_spread_not_length"
        if K["K4"]:
            verdict += "__panel_goes_the_other_way"
    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  发现样本 原始均值 {core_raw:+.4f} → 偏相关均值 {core_par:+.4f}"
          f"(扣掉了 {(1-abs(core_par)/abs(core_raw))*100:.0f}%)")
    print(f"  **第 61 条照引:跑前算的比值 1.29 是对效应 −0.3065 说的。"
          f"本次偏相关若只有 {abs(core_par):.2f},合并盲臂对那个量级的比值是 "
          f"{abs(core_par)/((ci_pool[1]-ci_pool[0])/2):.2f} —— <1 则本臂对它仍然无裁决力。**")
    json.dump({"names": NM, "per_pop": R, "pooled_blind": {
                   "n": len(PX), "raw": round(raw_pool, 4), "partial": round(pr_pool, 4),
                   "ci": ci_pool, "flip": fl_pool},
               "panel": AP, "panel_pooled": {"partial": round(ap_pool, 4), "ci": ap_ci},
               "placebo": PL, "placebo_pooled": {"partial": round(pl_pool, 4), "ci": pl_ci},
               "noise_hits": nhit, "probes": probes, "kill": K, "verdict": verdict,
               "core_raw": round(core_raw, 4), "core_partial": round(core_par, 4)},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
