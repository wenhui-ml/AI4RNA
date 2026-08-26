#!/usr/bin/env python3
"""M10: is the batch split in the panel's length dependence target composition or design composition?

Declaration: m10_batch_difference_declaration.json, frozen and pushed at e591199c BEFORE this file
existed.

M9 found the panel's length dependence splitting by batch: positive on 基准4560 (+0.2609) and both
blind batches (+0.1162, +0.4187), negative on 样本2 (-0.4584), 样本3 (-0.1523) and 第四批 (-0.2000).

THE OBVIOUS TEST IS USELESS AND THE DECLARATION SAYS SO FIRST. Three populations against three cannot be
tested: six choose three gives twenty arrangements, so an exact two-sided permutation bottoms out at
p = 0.10. This run does not use it.

What IS decidable is a within-population before/after. Restrict every population to the SIX-WAY COMMON
target set and recompute. Spread collapses -> target composition. Spread holds and the sign split
survives on identical targets -> design composition, which is what M9-pre-c/d already showed governs
per-target quantities.

S = sd(rho^common) / sd(rho^full), bootstrapped over POPULATIONS. S carries a known upward bias --
fewer targets widen each rho -- so the ladder measures that bias and the verdict quotes it.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import numpy as np                                                            # noqa: E402
from scipy.stats import spearmanr                                             # noqa: E402
from m1c_skeleton_bookkeeping import build                                    # noqa: E402
from m8b_target_length import CORE, BLIND, run_pop                            # noqa: E402

M9_REF = {"样本2(主)": -0.4584, "基准4560": +0.2609, "样本3": -0.1523,
          "第四批": -0.2000, "第五批(盲)": +0.1162, "第六批(盲)": +0.4187}
SHARES, DRAWS, PASS_FRAC, NBOOT, MIN_COMMON = (0.0, 0.25, 0.5, 0.75, 1.0), 20, 0.8, 2000, 25


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


def rho_on(D, tset, seed):
    u = [t for t in D["used"] if t in tset]
    if len(u) < MIN_COMMON:
        return None
    x = [D["len"][t] for t in u]; y = [D["vec"]["A"][t] for t in u]
    if len(set(x)) < 3:
        return None
    r, ci = rho_ci(x, y, seed)
    return {"rho": round(r, 4), "ci": [round(ci[0], 4), round(ci[1], 4)], "n": len(u),
            "pos": bool(ci[0] > 0), "neg": bool(ci[1] < 0)}


def spread(vals):
    return st.stdev(vals) if len(vals) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m10_batch_difference.json"))
    a = ap.parse_args()
    print("=== M10:批次分裂是【靶标组成】还是【设计组成】 ===")
    print("  声明 e591199c(代码存在之前已推送)。离线 $0。")
    print("  **不做三对三组间检验:6 选 3 只有 20 种排列,双侧最小 p = 0.10。**\n")

    D, names = {}, {}
    for nm, fn in CORE + BLIND:
        print(f"=== {nm} ===", flush=True)
        keep, M = build(fn, a.workers)
        d = run_pop(nm, keep, M, a.splits, a.seed, 9500)
        if d is not None:
            D[fn] = d; names[fn] = nm
    if len(D) < 5:
        json.dump({"verdict": "no_data"}, open(a.out, "w"), indent=1, ensure_ascii=False); return 0

    COMMON = set.intersection(*[set(D[fn]["used"]) for fn in D])
    print(f"\n  **六方共同靶标 {len(COMMON)}**(各人群有效靶标 "
          f"{', '.join(str(len(D[fn]['used'])) for fn in D)})")
    if len(COMMON) < MIN_COMMON:
        print(f"  K0:共同靶标 < {MIN_COMMON} ⇒ **无数据**")
        json.dump({"verdict": "no_data_common_too_small", "n_common": len(COMMON)},
                  open(a.out, "w"), indent=1, ensure_ascii=False); return 0

    print("\n=== 第 64 条形状二:共同靶标集的长度跨度是否被压缩 ===")
    RNG = {}
    for fn in D:
        La = [D[fn]["len"][t] for t in D[fn]["used"]]
        Lc = [D[fn]["len"][t] for t in D[fn]["used"] if t in COMMON]
        ra = max(La) - min(La); rc = max(Lc) - min(Lc)
        RNG[fn] = round(rc / ra, 3)
        print(f"  {names[fn]:<13} 全集跨度 {ra:6.0f} → 共同集 {rc:6.0f}  比 **{rc/ra:.3f}**")
    range_restricted = st.fmean(RNG.values()) < 0.6

    print("\n=== 主表:限制到共同靶标前后 ===")
    FULL, COMM = {}, {}
    print(f"  {'人群':<13}{'M9 参照':>9}{'本次 full':>26}{'common':>26}{'差':>8}")
    p2bad = []
    for fn in D:
        rf = rho_on(D[fn], set(D[fn]["used"]), a.seed + 77)
        rc = rho_on(D[fn], COMMON, a.seed + 77)
        FULL[fn] = rf; COMM[fn] = rc
        ref = M9_REF.get(names[fn])
        if ref is not None and abs(rf["rho"] - ref) > 0.02:
            p2bad.append((names[fn], rf["rho"], ref))
        print(f"  {names[fn]:<13}{ref:+9.4f}{rf['rho']:+9.4f} [{rf['ci'][0]:+.3f},{rf['ci'][1]:+.3f}]"
              f"{rc['rho']:+9.4f} [{rc['ci'][0]:+.3f},{rc['ci'][1]:+.3f}]{rc['rho']-rf['rho']:+8.4f}")
    print(f"  **P2 复现 M9**:{'✓ 六人群全部在 0.02 内' if not p2bad else f'✗ {p2bad} ⇒ 作废'}")

    print("\n=== P1 限制对『本来就几乎全是共同靶标』的人群不该有影响 ===")
    p1bad = []
    for fn in D:
        share = len([t for t in D[fn]["used"] if t in COMMON]) / len(D[fn]["used"])
        if share > 0.90:
            dd = abs(COMM[fn]["rho"] - FULL[fn]["rho"])
            ok = dd < 0.05
            print(f"  {names[fn]:<13} 共同占比 {share:.2f}  |Δrho| {dd:.4f}  {'✓' if ok else '✗'}")
            if not ok:
                p1bad.append(names[fn])
    if not any(len([t for t in D[fn]["used"] if t in COMMON]) / len(D[fn]["used"]) > 0.90 for fn in D):
        print("  (无人群共同占比 > 0.90,本探针无对象 —— 记为不可用,不作为通过)")

    sf = spread([FULL[fn]["rho"] for fn in D]); sc = spread([COMM[fn]["rho"] for fn in D])
    S = sc / sf
    rng = random.Random(a.seed + 5); ks = list(D); B = []
    for _ in range(4000):
        s = [ks[rng.randrange(len(ks))] for _ in range(len(ks))]
        try:
            v = spread([COMM[f]["rho"] for f in s]) / spread([FULL[f]["rho"] for f in s])
            if not math.isnan(v):
                B.append(v)
        except Exception:
            pass
    B.sort(); Sci = [B[int(.025 * len(B))], B[int(.975 * len(B))]]
    print(f"\n=== 主判据 ===")
    print(f"  六个 rho 的标准差:全集 **{sf:.4f}** → 共同集 **{sc:.4f}**")
    print(f"  **S = {S:.4f}  95%(重抽人群) [{Sci[0]:.4f}, {Sci[1]:.4f}]**")

    print(f"\n=== 构造性梯级(第 67 条,每档 {DRAWS} 次):本臂能不能分辨 S 的变化,以及 S 的偏置 ===")
    LAD, thr = {}, None
    nT = {fn: len(D[fn]["used"]) for fn in D}; nC = len(COMMON)
    for sh in SHARES:
        ss = []
        for k in range(DRAWS):
            rr = random.Random(a.seed + int(sh * 100) + 7919 * k)
            u = rr.gauss(0, 1)
            rf_ = [math.sqrt(sh) * u + math.sqrt(1 - sh) * rr.gauss(0, 1) for _ in D]
            noise = lambda n: rr.gauss(0, 1.0 / math.sqrt(max(4, n - 3)))
            a_ = [x + noise(nT[fn]) for x, fn in zip(rf_, D)]
            b_ = [x + noise(nC) for x in rf_]
            try:
                ss.append(spread(b_) / spread(a_))
            except Exception:
                pass
        med = st.median(ss)
        det = st.fmean([1 if v < 0.6 else 0 for v in ss])
        LAD[str(sh)] = {"S_median": round(med, 3), "frac_below_0.6": round(det, 3)}
        print(f"  共享比例={sh:.2f}  S 中位 **{med:.3f}**  低于 0.6 的比例 {det:.2f}")
    noise_S = LAD["0.0"]["S_median"]
    print(f"  **纯人群特异(共享=0)下 S 的中位 = {noise_S:.3f} —— 这就是 S 的偏置,裁定必须扣除它照引。**")

    K0 = len(COMMON) < MIN_COMMON or bool(p2bad) or bool(p1bad)
    K1 = Sci[1] < 0.6
    K2 = Sci[0] > 0.85
    npos = sum(1 for fn in D if COMM[fn]["pos"]); nneg = sum(1 for fn in D if COMM[fn]["neg"])
    K3 = npos >= 2 and nneg >= 2
    K4 = LAD["1.0"]["frac_below_0.6"] < PASS_FRAC
    print("\n=== 判决条件 ===")
    for k, v, d in (("K0", K0, "共同靶标不足 或 P1/P2 不合格 ⇒ 作废"),
                    ("K1", K1, f"S 上界 {Sci[1]:.3f} < 0.6 ⇒ 靶标组成"),
                    ("K2", K2, f"S 下界 {Sci[0]:.3f} > 0.85 ⇒ 设计组成"),
                    ("K3", K3, f"共同靶标上仍 {npos} 正 / {nneg} 负排除 0 ⇒ 设计组成(独立证据)"),
                    ("K4", K4, "梯级在共享=1.0 时都分辨不出 S 收缩 ⇒ 无裁决力")):
        print(f"  {k}: {'**触发**' if v else '未触发'}   {d}")

    if K0:
        verdict = "void_probe_or_no_data"
    elif K4:
        verdict = "no_adjudicating_power"
    elif K1 and not K2:
        verdict = "target_composition"
    elif (K2 or K3) and not K1:
        verdict = "design_composition"
    else:
        verdict = "unresolved"
    if range_restricted:
        verdict += "__range_restriction"
    flips = [names[fn] for fn in D if (FULL[fn]["rho"] > 0) != (COMM[fn]["rho"] > 0)]
    if flips:
        verdict += "__sign_flip_" + "_".join(flips)

    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  限制到同一批 {len(COMMON)} 个共同靶标后:{npos} 个人群正向排除 0,{nneg} 个负向排除 0")
    if flips:
        print(f"  **限制前后变号的人群:{flips} —— 按第 60 条双侧报出,不许只数一侧**")
    print(f"\n  **第 61 条照引:S 有已知的向上偏置(靶标变少 ⇒ 每个 rho 更宽 ⇒ 离散度更大)。"
          f"梯级实测:共享=0 时 S 中位 {noise_S:.3f}。K1 的门槛 0.6 必须对着这个偏置读。**")

    json.dump({"verdict": verdict, "n_common": len(COMMON), "S": round(S, 4),
               "S_ci": [round(x, 4) for x in Sci], "sd_full": round(sf, 4), "sd_common": round(sc, 4),
               "full": {names[k]: v for k, v in FULL.items()},
               "common": {names[k]: v for k, v in COMM.items()},
               "ladder": LAD, "range_ratio": {names[k]: v for k, v in RNG.items()},
               "sign_flips": flips, "P2_mismatch": p2bad,
               "kill": {k: bool(v) for k, v in (("K0", K0), ("K1", K1), ("K2", K2), ("K3", K3), ("K4", K4))}},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
