#!/usr/bin/env python3
"""M3c: re-judge M3's replication count with a per-population noise-calibrated threshold.

`rigor_calibration_0825.md` section 6 item 2 promised exactly this and I did not run it.

What changes versus M3:
  * PRIMARY OBJECT is the paired contrast A_pair = (A_ed - A) - (A_mfe - A), split by split, on M3's own
    exactly-GC-matched pairs. Generic ensemble-concentration cancels by construction; what survives is the
    skeleton-specific part. (`m2_audit.md` section 9 promised the main object move to the paired difference.)
  * REPLICATION is judged against tau_p = the largest SAME-SIDE noise-arm mean in that population, not
    against a bare zero, and the interval verdict must itself be seed-stable (>= 45/50, rule 51).
  * DENOMINATOR is six populations. The fifth and sixth have never been computed on this construction.

DISCLOSURE: the four core populations' numbers are already in m3b_placebo_on_matched.json and I have read
them. For those four this is a re-judgement, not a pre-registration. The threshold has no free parameter
(max of the same-side noise means) so it cannot be tuned to them, and the fifth/sixth batches are the only
genuinely blind test of the new rule. Written into the declaration before running.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block                                      # noqa: E402
from h1_heldout_ensemble_defect import fit_predict                                # noqa: E402
import h2_heldout_cluster_ci as H                                                 # noqa: E402
from m2b_gt_target_property import fold                                           # noqa: E402
from m3b_placebo_on_matched import acc_matched_multi                              # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"),
        ("第五批(本构造未读)", "panel_f1_fifth.json"), ("第六批(本构造未读)", "panel_f1_sixth.json"))
PRIMARY = "panel_f1_fresh.json"
BLIND = ("panel_f1_fifth.json", "panel_f1_sixth.json")
R_NOISE = 12
SPLITS_DEFAULT, MIN_PAIRS, FLIP_MIN = 200, 30, 45


def run_pop(nm, keep, M, splits, seed, noise_seed, placebo="mfe"):
    """placebo: 'mfe' = the real pre-registered placebo; 'copy' = an exact copy of ed_skel (probe P0);
                'noise' = one N(0,1) (probe P1)."""
    cells, cols, key = collections.defaultdict(list), {}, {}
    rngn = random.Random(noise_seed); rngp = random.Random(noise_seed + 7717)
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        if placebo == "mfe":
            pcol = m["ed_mfe"]
        elif placebo == "copy":
            pcol = m["ed_skel"]
        elif placebo == "noise":
            pcol = rngp.gauss(0, 1.0)
        else:
            raise ValueError(placebo)
        cols[id(rr)] = (base_block(r) + m["C"] + [m["ed_skel"], pcol]
                        + [rngn.gauss(0, 1.0) for _ in range(R_NOISE)])
        key[id(rr)] = (m["C"][0], round(m["C"][1], 3))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    if not rws:
        print(f"  {nm:14s} 无可用格 -> 该人群作废", flush=True)
        return {}, 0
    nB = len(base_block(keep[0]))
    iA = list(range(nB))
    idx = {"A": iA, "A_G": iA + [nB, nB + 1], "A_ed": iA + [nB + 2], "A_mfe": iA + [nB + 3]}
    for j in range(R_NOISE):
        idx[f"A_n{j}"] = iA + [nB + 4 + j]
    keys = list(idx)
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    tg = sorted(by_t)
    per = {k: {t: [] for t in tg} for k in keys if k != "A"}
    per["A_pair"] = {t: [] for t in tg}
    rng = random.Random(seed); order = list(tg)
    S = dict(pairs=0, tie_tube=0, skipped=0, leak=0)
    for _s in range(splits):
        rng.shuffle(order)
        cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE:
            continue
        if {id(r) for _c, g in TR for r in g} & {id(r) for _c, g in TE for r in g}:
            S["leak"] += 1                                                        # V2
        P = {k: fit_predict(TR, TE, cols, ix) for k, ix in idx.items()}
        for t in te:
            acc, n, tt2, sk = acc_matched_multi(by_t[t], P, keys, key, "exact")    # V1
            S["pairs"] += n; S["tie_tube"] += tt2; S["skipped"] += sk
            if n < MIN_PAIRS or any(v is None for v in acc.values()):              # V6
                continue
            for k in per:
                if k == "A_pair":
                    per[k][t].append((acc["A_ed"] - acc["A"]) - (acc["A_mfe"] - acc["A"]))
                else:
                    per[k][t].append(acc[k] - acc["A"])
    used = [t for t in tg if len(per["A_ed"][t]) >= H.MIN_SPLITS_PER_TARGET]
    tot = S["pairs"] + S["tie_tube"] + S["skipped"]
    print(f"  {nm:14s} 格 {len(rws)} 靶标 {len(used)} | GC 配平且可判的对 {S['pairs']}"
          f"(被配平剔除 {S['skipped']/max(1,tot)*100:.1f}%) | 分子级泄漏 {S['leak']}/{splits}"
          f" {'✓' if not S['leak'] else '✗'}", flush=True)
    if len(used) < H.MIN_TARGETS:                                                 # V7
        print(f"    ↳ 有效靶标 {len(used)} < {H.MIN_TARGETS} ⇒ **该人群作废,不进分母**", flush=True)
        return {}, len(used)
    out = {}
    for k in per:
        g = [st.fmean(per[k][t]) for t in used]
        ci, nb = H.boot_ci(g, seed + 17)
        assert nb == len(used), "闸门 V5:重抽单位不是靶标"
        mde = 1.96 * st.stdev(g) / math.sqrt(len(g))
        flip = sum(1 for s2 in range(50)
                   if (lambda c: c[0] > 0 or c[1] < 0)(H.boot_ci(g, 9100 + 13 * s2)[0]))
        out[k] = {"mean": round(st.fmean(g), 4), "ci": ci, "n": nb, "MDE": round(mde, 4),
                  "excl": bool(ci[0] > 0 or ci[1] < 0), "pos": sum(1 for x in g if x > 0),
                  "eff_over_mde": round(abs(st.fmean(g)) / mde, 2) if mde > 0 else 0.0,
                  "flip": flip}                                                   # V4
    return out, len(used)


def tau_of(d):
    """rule 53: the threshold counts SAME-SIDE (positive) noise means only."""
    nz = [d[f"A_n{j}"]["mean"] for j in range(R_NOISE)]
    return max(0.0, max(nz)), nz


def replicates(o, tau):
    """the frozen decision line, verbatim from the declaration."""
    return bool(o["mean"] > tau and o["ci"][0] > 0 and o["flip"] >= FLIP_MIN)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=SPLITS_DEFAULT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m3c_noise_calibrated_replication.json"))
    a = ap.parse_args()
    R, VOIDED, FOLDS = {}, [], {}
    print("=== M3c 主臂:六人群,配对差为主对象,精确 GC 配平 ===", flush=True)
    for j, (nm, fn) in enumerate(POPS):
        keep, M = fold(fn, a.workers); FOLDS[fn] = (keep, M)
        R[fn], nu = run_pop(nm, keep, M, a.splits, a.seed, 6000 + 131 * j)
        if not R[fn]:
            VOIDED.append(nm)
        json.dump({"main": R}, open(a.out, "w"), indent=1, ensure_ascii=False)

    live = [(nm, fn) for nm, fn in POPS if R.get(fn)]

    print("\n=== 必响探针(第 54 条:任一不响则整轮作废)===", flush=True)
    keep, M = FOLDS[PRIMARY]
    # 探针必须与主臂同划分数:少于 ~67 划分时靶标数掉到 V7 地板以下,探针会因【没数据】而被读成【没响】,
    # 那是第 54 条要防的事情的镜像 —— 一个流程假象冒充一个实验结论。
    P0, _ = run_pop("P0 拷贝", keep, M, a.splits, a.seed, 6000, placebo="copy")
    P1, _ = run_pop("P1 噪声", keep, M, a.splits, a.seed, 6000, placebo="noise")
    prim = R[PRIMARY]
    p0 = abs(P0["A_pair"]["mean"]) < 1e-9 if P0 else False
    # rule 56: 真列为正、噪声列的已知效应为负或零 ⇒ 差应为正,本探针**单侧**。
    p1 = bool(P1["A_pair"]["ci"][0] > 0) if P1 else False
    # rule 56: 精确配平后 GC 列在对内为常数,无已知非零方向 ⇒ 双侧正确。
    p2 = (not prim["A_G"]["excl"]) if prim else False
    print(f"  P0 相消恒等式  A_pair = "
          + (f"{P0['A_pair']['mean']:+.6f}" if P0 else "该探针人群作废(无数据)")
          + f"  {'✓ 响' if p0 else '✗ 没响'}")
    print(f"  P1 检出力(单侧) A_pair = "
          + (f"{P1['A_pair']['mean']:+.4f} 95% {P1['A_pair']['ci']}" if P1 else "该探针人群作废(无数据)")
          + f"  {'✓ 响' if p1 else '✗ 没响'}")
    print(f"  P2 GC 相消      A_G   = {prim['A_G']['mean']:+.4f} 95% {prim['A_G']['ci']}"
          f"  {'✓ 响' if p2 else '✗ 没响'}")
    probes = {"P0_copy": (P0 or {}).get("A_pair"), "P1_noise": (P1 or {}).get("A_pair"),
              "P2_gc": (prim or {}).get("A_G"), "fired": {"P0": p0, "P1": p1, "P2": p2}}
    if not (p0 and p1 and p2):
        json.dump({"main": R, "probes": probes, "verdict": "void_probe_did_not_fire"},
                  open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n**必响探针没有全响 ⇒ 本次运行作废(第 54 条)。不在本份声明里换探针。**")
        return 0

    print(f"\n=== 逐人群噪声校准阈值 tau(第 53 条:只数同侧){R_NOISE} 根 ===")
    TAU = {}
    for nm, fn in live:
        tau, nz = tau_of(R[fn]); TAU[fn] = tau
        nex = sum(1 for j in range(R_NOISE) if R[fn][f"A_n{j}"]["excl"] and R[fn][f"A_n{j}"]["mean"] > 0)
        print(f"  {nm:14s} 零点中心 {st.fmean(nz):+.5f}  同侧最大 **tau = {tau:+.5f}**"
              f"  同侧排除 0 的根数 {nex}/{R_NOISE}")
    k0_ex = sum(1 for j in range(R_NOISE)
                if prim[f"A_n{j}"]["excl"] and prim[f"A_n{j}"]["mean"] > 0)

    print("\n=== 重判:旧判据 vs 新判据 ===")
    print(f"  {'人群':<16}{'A_pair':<12}{'95%':<22}{'tau':<11}{'翻转':<8}{'旧(排除0)':<12}{'新(校准)':<10}")
    old_rep = new_rep = 0
    JUD = {}
    for nm, fn in live:
        o = R[fn]["A_pair"]; tau = TAU[fn]
        old = bool(o["ci"][0] > 0)
        new = replicates(o, tau)
        old_rep += old; new_rep += new
        JUD[fn] = {"mean": o["mean"], "ci": o["ci"], "tau": round(tau, 5), "flip": o["flip"],
                   "old_rule": old, "new_rule": new}
        print(f"  {nm:<16}{o['mean']:+.4f}     [{o['ci'][0]:+.4f},{o['ci'][1]:+.4f}]   "
              f"{tau:+.5f}   {o['flip']:2d}/50   {'复现' if old else '不复现':<12}"
              f"{'**复现**' if new else '不复现':<10}")
    nlive = len(live)
    print(f"\n  旧判据 {old_rep}/{nlive}   →   **新判据 {new_rep}/{nlive}**"
          + (f"   (作废人群:{', '.join(VOIDED)})" if VOIDED else ""))

    print("\n=== 安慰剂过同一条新判据吗(K3 归属)===")
    pl_rep = 0
    for nm, fn in live:
        o = R[fn]["A_mfe"]; ok = replicates(o, TAU[fn])
        pl_rep += ok
        print(f"  {nm:<16}A_mfe {o['mean']:+.4f} [{o['ci'][0]:+.4f},{o['ci'][1]:+.4f}]"
              f"  翻转 {o['flip']:2d}/50  {'**过**' if ok else '不过'}")
    pm = st.fmean([R[fn]["A_mfe"]["mean"] for _n, fn in live])
    qm = st.fmean([R[fn]["A_pair"]["mean"] for _n, fn in live])
    print(f"  安慰剂 {nlive} 人群均值 {pm:+.4f} vs 配对差 {qm:+.4f}"
          f"  ({pm/qm*100:+.0f}% of 主对象)" if qm else "")

    blind = [fn for fn in BLIND if R.get(fn)]
    blind_rep = sum(1 for fn in blind if replicates(R[fn]["A_pair"], TAU[fn]))
    print(f"\n=== 盲测(第五/第六批,本构造从未读过)===\n  可用 {len(blind)}/2,按新判据复现 **{blind_rep}/{len(blind) if blind else 0}**")
    if not blind:
        print("  **两个盲人群都被 V7 靶标地板作废 ⇒ 这条新判据【一次盲测都没有过】。"
              "裁定串必须带 `blind_unavailable`,不许让它静默消失。**")

    K = {"K0": k0_ex >= 2,
         "K1": not (prim["A_pair"]["ci"][0] > 0),
         "K2": new_rep < 3,
         "K3": bool(pl_rep >= 3 and qm and pm >= 0.5 * qm),
         "K4": bool(blind and blind_rep == 0)}
    print("\n=== 判决条件 ===")
    for k, v in K.items():
        print(f"  {k}: {'**触发**' if v else '未触发'}")
    if K["K0"]:
        verdict = "void_instrument_manufactures_signal"
    elif K["K1"]:
        verdict = "paired_contrast_null_on_primary"
    else:
        verdict = "skeleton_specific_signal_survives_cancellation"
        verdict += "__underreplicated" if K["K2"] else f"__replicates_{new_rep}_of_{nlive}"
        if K["K3"]:
            verdict += "__but_generic_concentration_passes_too"
        if K["K4"]:
            verdict += "__blind_pair_null"
        if not blind:
            verdict += "__blind_unavailable"
    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  复现计数重判:M3 的 K3 记的是 1/4(裸区间);本次在噪声校准阈值 + 种子稳定性下是 {new_rep}/{nlive}。")
    json.dump({"main": R, "probes": probes, "tau": {k: round(v, 6) for k, v in TAU.items()},
               "judgement": JUD, "old_rep": old_rep, "new_rep": new_rep, "n_live": nlive,
               "voided_populations": VOIDED, "placebo_passes": pl_rep,
               "blind_rep": f"{blind_rep}/{len(blind)}", "kill": K, "verdict": verdict},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
