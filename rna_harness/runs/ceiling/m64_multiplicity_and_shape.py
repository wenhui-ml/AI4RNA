#!/usr/bin/env python3
"""M64: apply the correction the manuscript told the reader to apply themselves.

Declaration: m64_multiplicity_and_shape_declaration.json, frozen and pushed BEFORE this file
existed. The 42 family members are named there; this file may not add to or remove from that list.

Limitation (vii) of R1 reads: "a reader assembling a family from these tables should apply their
own correction; we have not applied one on their behalf." Two things are wrong with leaving it
there. The first is that the authors are better placed to define the family than the reader, and
defining it after the fact is exactly the freedom a correction is supposed to remove -- which is
why the list is frozen in the declaration. The second is that the published numbers CANNOT clear
a family-wise bar, for a reason that has nothing to do with the evidence: twelve of the 42 sit at
0.0005 = 1/(2000+1), the smallest number a 2000-draw permutation null can print, and Holm at
0.005 over 42 tests needs 1.19e-4. The bar is below the floor. So this run buys the resolution.

The purchase is only legitimate if the fine null is the SAME null. The vectorised engine here is
an algebraic rewrite of M16/M21's `perm_diff` + `matched_diff`, not a reimplementation of the
idea: within a stratum the label permutation preserves n_A and n_B, so the stratum weight
w = min(n_A, n_B) is CONSTANT under the null, the used-stratum set is invariant, and

    mean_A - mean_B = S_A*(1/n_A + 1/n_B) - T/n_B

with S_A the sum of a uniformly random n_A-subset and T the stratum total. Only S_A is random.
That turns 10^6 permutations into 10^6 random subset-sums per stratum. S4 is the gate on it: at
2000 draws the engine must reproduce the published statistic and the published null SD, and if it
does not, K0 voids the whole purchase.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random as _r, re, sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m16_rescue as M16
import m21_thermo_mediator as M21

PAPER = os.path.join(HERE, "..", "..", "..", "paper", "rna_placement")
ALPHA = 0.005
NPERM_CHECK = 2000            # the published draw count, for the equivalence gate
NPERM_BUY = 1_000_000         # the purchased resolution
NPERM_SHAPE = 200_000
SEED = 20260901
CHUNK = 50_000
TRIM = 0.05


def log(s=""):
    print(s, flush=True)


# ---------------------------------------------------------------- the family
def read_family(decl):
    tex = open(os.path.join(PAPER, "R1_placement.tex")).read()
    num = open(os.path.join(PAPER, "figs", "rna_numbers.tex")).read()
    vals = dict(re.findall(r"\\csname jdg@([A-Za-z0-9]+)\\endcsname\{(.*?)\}\s*$", num, re.M))
    used = set(re.findall(r"\\jdg\{([A-Za-z0-9]+)\}", tex))
    isp = lambda v: bool(re.match(r"^(<)?[0-9.]+(e-?\d+)?$", v or ""))
    fam = set()
    for k in re.findall(r"p\s*(?:=|<|>|\\le|\\ge)?\s*\\jdg\{([A-Za-z0-9]+)\}", tex):
        if isp(vals.get(k, "")):
            fam.add(k)
    for k in used:
        if re.search(r"(P|Binom)$", k) and isp(vals.get(k, "")):
            fam.add(k)
    return sorted(fam), {k: float(vals[k].lstrip("<")) for k in sorted(fam)}


def holm(pv, alpha=ALPHA):
    """Holm-Bonferroni. Valid under arbitrary dependence, which is why the paper quotes it."""
    m = len(pv)
    order = sorted(pv, key=lambda k: pv[k])
    rej, stopped = {}, False
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        if not stopped and pv[k] <= thr:
            rej[k] = True
        else:
            stopped = True
            rej[k] = False
    return rej, {k: alpha / (m - i) for i, k in enumerate(order)}


def bh(pv, q=ALPHA):
    """Benjamini-Hochberg. NOT valid under arbitrary dependence; reported beside Holm, not quoted."""
    m = len(pv)
    order = sorted(pv, key=lambda k: pv[k])
    kmax = 0
    for i, k in enumerate(order, start=1):
        if pv[k] <= i * q / m:
            kmax = i
    return {k: (i < kmax) for i, k in enumerate(order)}


# ---------------------------------------------------------------- the engine
def prep(recs, A, B, sfn, field="rf"):
    """stratum blocks (values with the A-labelled ones first, n_A, n_B), both types required."""
    by = collections.defaultdict(lambda: {A: [], B: []})
    for r in recs:
        if r["pt"] in (A, B):
            by[sfn(r)][r["pt"]].append(float(r[field]))
    S = []
    for st, d in by.items():
        if d[A] and d[B]:
            S.append((np.asarray(d[A] + d[B], float), len(d[A]), len(d[B])))
    return S


def obs(S):
    num = den = 0.0
    for x, na, nb in S:
        w = min(na, nb)
        num += w * (x[:na].mean() - x[na:].mean())
        den += w
    return (num / den if den else None), den, len(S)


def null_vec(S, nperm, seed, chunk=CHUNK):
    """the same within-stratum label permutation, written as random subset-sums."""
    rng = np.random.default_rng(seed)
    W = float(sum(min(na, nb) for _, na, nb in S))
    pre = []
    for x, na, nb in S:
        pre.append((x, na, nb, min(na, nb), 1.0 / na + 1.0 / nb, float(x.sum()) / nb))
    out = np.empty(nperm, float)
    done = 0
    while done < nperm:
        m = min(chunk, nperm - done)
        acc = np.zeros(m)
        for x, na, nb, w, c, tb in pre:
            n = na + nb
            R = rng.random((m, n))
            idx = np.argpartition(R, na - 1, axis=1)[:, :na]
            acc += w * (x[idx].sum(axis=1) * c - tb)
        out[done:done + m] = acc / W
        done += m
    return np.sort(out)


def pval(null, d, nperm):
    return (float((null >= d).sum()) + 1.0) / (nperm + 1.0)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "m64_multiplicity_and_shape.json"))
    ap.add_argument("--nperm", type=int, default=NPERM_BUY)
    a = ap.parse_args()
    OUT = {"alpha": ALPHA, "nperm_buy": a.nperm, "seed": SEED}
    decl = json.load(open(os.path.join(HERE, "m64_multiplicity_and_shape_declaration.json")))

    log("=== M64:把论文让读者自己做的那道校正,自己做了 ===")
    log("  声明 m64_...(代码存在之前已推送)。离线 $0。\n")

    # ---------------- S1 / K4: the family is the frozen one
    log("=== S1 / K4 冻结的族在稿件里还找得到吗 ===")
    names, pub = read_family(decl)
    frozen = decl["family_membership_rule"]["members"]
    OUT["family_found"] = names
    OUT["family_matches_declaration"] = (names == sorted(frozen))
    log(f"  声明冻结 {len(frozen)} 个   运行时找到 {len(names)} 个   "
        f"{'逐字相同' if names == sorted(frozen) else '**不同 ⇒ K4**'}")
    if names != sorted(frozen):
        OUT["verdict"] = "halted_K4"
        OUT["diff"] = {"only_declared": sorted(set(frozen) - set(names)),
                       "only_found": sorted(set(names) - set(frozen))}
        json.dump(OUT, open(a.out, "w"), indent=1)
        log("  K4 触发:族不同。按声明,停机并记 erratum,不改名单。")
        return 0
    floor = sorted(k for k in names if abs(pub[k] - 1.0 / 2001) < 5e-5)
    OUT["published_p"] = pub
    OUT["at_floor"] = floor
    log(f"  其中恰好落在 1/2001 置换地板上的:{len(floor)} 个 -> {', '.join(floor)}\n")

    # ---------------- A: the corrections, on the numbers as published
    log("=== A 已发表数值上的族错误率校正 ===")
    hol, thr = holm(pub)
    bhr = bh(pub)
    OUT["A"] = {"holm_reject": hol, "holm_threshold": thr, "bh_reject": bhr,
                "n_holm": int(sum(hol.values())), "n_bh": int(sum(bhr.values()))}
    log(f"  m = {len(names)}   最严的 Holm 门槛 = {ALPHA}/{len(names)} = {ALPHA/len(names):.3e}")
    log(f"  Holm(alpha={ALPHA}) 拒绝 {sum(hol.values())}/{len(names)};"
        f"  BH(q={ALPHA}) 拒绝 {sum(bhr.values())}/{len(names)}")
    head = ("rescueAUP", "rescueGUP")
    OUT["S2_bh_headline"] = all(bhr[k] for k in head)
    s3_notrej = [k for k in head if not hol[k]]
    OUT["S3"] = {"not_rejected_by_holm": s3_notrej,
                 "and_sits_at_the_floor": all(abs(pub[k] - 1.0 / 2001) < 5e-5 for k in s3_notrej)}
    for k in head:
        log(f"    {k:12s} p={pub[k]:.4f}  Holm {'拒绝' if hol[k] else '**不拒绝**'}  "
            f"BH {'拒绝' if bhr[k] else '不拒绝'}")
    log(f"  S2(BH 拒绝两条头条)     {'成立' if OUT['S2_bh_headline'] else '**不成立**'}")
    log(f"  S3(Holm 至少一条不拒绝,且原因是地板)  "
        f"{'成立' if s3_notrej and OUT['S3']['and_sits_at_the_floor'] else '**不成立**'}\n")

    # ---------------- data
    recs16 = M16.build()
    recs21 = M21.build_with_dg()
    edges = M21.bin_edges(recs21, M21.NBINS)
    for r in recs21:
        r["bin"] = M21.assign_bin(r["dG"], edges)
    log(f"  M16 记录 {len(recs16)}   M21 记录 {len(recs21)}(dG 分箱边界已按 M21 重算)\n")

    m16j = json.load(open(os.path.join(HERE, "m16_rescue.json")))
    m21j = json.load(open(os.path.join(HERE, "m21_thermo_mediator.json")))

    ARMS = [
        ("rescueAUP", recs16, "CG", "AU", M16.stratum,
         m16j["CG_vs_AU"]["matched_diff"], m16j["CG_vs_AU"]["null_sd"]),
        ("rescueGUP", recs16, "CG", "GU", M16.stratum,
         m16j["CG_vs_GU"]["matched_diff"], m16j["CG_vs_GU"]["null_sd"]),
        ("eP", recs21, "CG", "AU", M21.strat_dg,
         m21j["B1"]["cde"], m21j["B1"]["null_sd"]),
        ("eDZeroP", [r for r in recs21 if r["depth"] == 0], "CG", "AU",
         (lambda r: (r["design"], r["crossed"])),
         m21j["B3"]["depth0"]["diff"], None),
        ("eDTwoP", [r for r in recs21 if r["depth"] >= 2], "CG", "AU",
         (lambda r: (r["design"], r["crossed"])),
         m21j["B3"]["depth2plus"]["diff"], None),
    ]

    # ---------------- S4: equivalence at the published draw count
    log("=== S4 等价对拍:2000 抽,向量化引擎必须复现已发表的统计量与零分布宽度 ===")
    eq, ok_all = {}, True
    blocks = {}
    for i, (nm, rr, A, B, sfn, d_pub, sd_pub) in enumerate(ARMS):
        S = prep(rr, A, B, sfn)
        blocks[nm] = S
        d, w, ns = obs(S)
        nl = null_vec(S, NPERM_CHECK, SEED + 100 * i)
        sd = float(nl.std())
        dok = abs(d - d_pub) <= 1e-6 + 5e-5      # published diff is rounded to 4 decimals
        sok = True if sd_pub is None else (0.90 <= sd / sd_pub <= 1.10)
        eq[nm] = {"d_vec": round(d, 6), "d_published": d_pub, "sd_vec": round(sd, 5),
                  "sd_published": sd_pub, "weight": w, "n_strata": ns,
                  "p_at_2000": round(pval(nl, d, NPERM_CHECK), 5), "pass": bool(dok and sok)}
        ok_all &= eq[nm]["pass"]
        log(f"  {nm:11s} d {d:+.6f} vs 已发表 {d_pub:+.4f}   "
            f"零SD {sd:.5f} vs {('%.4f' % sd_pub) if sd_pub is not None else '—':>7s}   "
            f"权重 {w:.0f} 分层 {ns}   {'通过' if eq[nm]['pass'] else '**不通过**'}")
    OUT["S4"] = {"arms": eq, "pass": bool(ok_all)}
    log(f"  S4 {'成立 ⇒ 购买分辨率是合法的' if ok_all else '**不成立 ⇒ K0,B 臂作废**'}\n")

    if not ok_all:
        OUT["verdict"] = "K0_purchase_void"
        json.dump(OUT, open(a.out, "w"), indent=1)
        log("K0 触发:不报任何购买后的 p 值,局限 (vii) 原样保留。")
        return 0

    # ---------------- B: buy the resolution
    log(f"=== B 购买分辨率:同一个零分布,{a.nperm} 抽 ===")
    buy = {}
    for i, (nm, rr, A, B, sfn, d_pub, _sd) in enumerate(ARMS):
        S = blocks[nm]
        d, w, ns = obs(S)
        nl = null_vec(S, a.nperm, SEED + 1000 + 100 * i)
        p = pval(nl, d, a.nperm)
        ge = int((nl >= d).sum())
        buy[nm] = {"d": round(d, 6), "p": p, "n_ge": ge, "nperm": a.nperm,
                   "null_sd": round(float(nl.std()), 6),
                   "z": round(d / float(nl.std()), 3) if nl.std() > 0 else None,
                   "published_p": pub[nm]}
        log(f"  {nm:11s} d {d:+.6f}   零SD {nl.std():.6f}  z {d/nl.std():+.2f}   "
            f"零分布中 >= d 的有 {ge} / {a.nperm}   p = {p:.3e}   (已发表 {pub[nm]})")
    OUT["B"] = buy

    resolved = dict(pub)
    for nm, v in buy.items():
        resolved[nm] = v["p"]
    hol2, thr2 = holm(resolved)
    bhr2 = bh(resolved)
    OUT["B_corrected"] = {"resolved_p": resolved, "holm_reject": hol2, "bh_reject": bhr2,
                          "n_holm": int(sum(hol2.values())), "n_bh": int(sum(bhr2.values()))}
    log(f"\n  购买后重做:Holm 拒绝 {sum(hol2.values())}/{len(names)};"
        f"  BH 拒绝 {sum(bhr2.values())}/{len(names)}")
    for nm in ("rescueAUP", "rescueGUP", "eP", "eDZeroP", "eDTwoP"):
        log(f"    {nm:11s} p={resolved[nm]:.3e}  门槛 {thr2[nm]:.3e}  "
            f"Holm {'拒绝' if hol2[nm] else '**不拒绝**'}")
    OUT["S5"] = bool(hol2["rescueAUP"] and hol2["rescueGUP"])
    OUT["S6"] = bool(hol2["eP"])
    log(f"  S5(两条头条过 Holm)  {'成立' if OUT['S5'] else '**不成立 ⇒ K1**'}")
    log(f"  S6(能量条件下的对比过 Holm)  {'成立' if OUT['S6'] else '**不成立**'}\n")

    # ---------------- C: what the outcome's distribution looks like, and whether it matters
    log("=== C 结果变量 R 的分布形状,以及头条对它的依赖 ===")
    v = np.asarray([r["rf"] for r in recs16], float)
    qs = [1, 5, 25, 50, 75, 95, 99]
    sh = {"n": int(v.size), "mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4),
          "skew": round(float(((v - v.mean()) ** 3).mean() / v.std() ** 3), 3),
          "exkurt": round(float(((v - v.mean()) ** 4).mean() / v.std() ** 4 - 3.0), 3),
          "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
          "q": {str(q): round(float(np.percentile(v, q)), 4) for q in qs},
          "frac_negative": round(float((v < 0).mean()), 4)}
    OUT["C_shape"] = sh
    log(f"  n {sh['n']}  mean {sh['mean']:+.4f}  sd {sh['sd']:.4f}  "
        f"skew {sh['skew']:+.3f}  excess kurtosis {sh['exkurt']:+.3f}")
    log("  分位数 " + "  ".join(f"{q}%:{sh['q'][str(q)]:+.4f}" for q in qs))
    log(f"  min {sh['min']:+.4f}  max {sh['max']:+.4f}  为负的比例 {100*sh['frac_negative']:.1f}%")

    # rank transform, within stratum, on the SAME strata the headline uses
    byst = collections.defaultdict(list)
    for r in recs16:
        byst[M16.stratum(r)].append(r)
    for st, rs in byst.items():
        o = np.argsort(np.argsort([r["rf"] for r in rs]))
        n = len(rs)
        for r, k in zip(rs, o):
            r["rk"] = (float(k) + 0.5) / n
    Sr = prep(recs16, "CG", "AU", M16.stratum, field="rk")
    dr, wr, nsr = obs(Sr)
    nlr = null_vec(Sr, NPERM_SHAPE, SEED + 7)
    pr = pval(nlr, dr, NPERM_SHAPE)
    OUT["C_rank"] = {"diff": round(dr, 5), "p": pr, "weight": wr, "n_strata": nsr,
                     "nperm": NPERM_SHAPE, "null_sd": round(float(nlr.std()), 5)}
    log(f"  层内秩(0..1)上的配平差 {dr:+.5f}  置换 p {pr:.3e}  权重 {wr:.0f}")

    lo, hi = float(np.percentile(v, 100 * TRIM)), float(np.percentile(v, 100 * (1 - TRIM)))
    keep = [r for r in recs16 if lo <= r["rf"] <= hi]
    St = prep(keep, "CG", "AU", M16.stratum)
    dt, wt_, nst = obs(St)
    d_full = obs(prep(recs16, "CG", "AU", M16.stratum))[0]
    OUT["C_trim"] = {"trim": TRIM, "lo": round(lo, 4), "hi": round(hi, 4),
                     "n_kept": len(keep), "diff": round(dt, 5),
                     "retention": round(dt / d_full, 4), "weight": wt_, "n_strata": nst}
    log(f"  两端各裁 {100*TRIM:.0f}%(R 在 [{lo:+.4f}, {hi:+.4f}] 之间,留 {len(keep)}):"
        f"配平差 {dt:+.5f}  = 全量的 {100*dt/d_full:.1f}%")
    OUT["S7"] = bool(dr > 0 and pr <= ALPHA)
    OUT["S8"] = bool(dt / d_full >= 0.50)
    log(f"  S7(秩上同号且过 0.005)  {'成立' if OUT['S7'] else '**不成立 ⇒ K2**'}")
    log(f"  S8(裁尾后留住一半以上)  {'成立' if OUT['S8'] else '**不成立**'}\n")

    # ---------------- D: what the free-energy-stratified estimator runs on
    log("=== D 能量分层估计量到底跑在哪些记录上 ===")
    base = M21.retained(recs21, "CG", "AU", M21.strat_base)
    used = M21.retained(recs21, "CG", "AU", M21.strat_dg)
    # identity, not field equality: M21.retained returns the record objects themselves, and two
    # distinct records can carry the same fields.
    ub = {id(r) for r in used}
    drop = [r for r in base if id(r) not in ub]
    frac = len(used) / len(base)
    smd = {}
    for f, g in (("depth", lambda r: float(r["depth"])), ("crossed", lambda r: float(r["crossed"])),
                 ("rf", lambda r: float(r["rf"])), ("dG", lambda r: float(r["dG"]))):
        x = np.asarray([g(r) for r in used]); y = np.asarray([g(r) for r in drop])
        sp = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0) if y.size > 1 else float("nan")
        smd[f] = round(float((x.mean() - y.mean()) / sp), 4) if sp and sp == sp else None
    comp = {}
    for t in ("CG", "AU"):
        comp[t] = {"used": sum(r["pt"] == t for r in used), "dropped": sum(r["pt"] == t for r in drop)}
    OUT["D"] = {"n_base": len(base), "n_used": len(used), "n_dropped": len(drop),
                "fraction": round(frac, 4), "smd": smd, "composition": comp,
                "max_abs_smd_of_three": round(max(abs(smd[k]) for k in ("depth", "crossed", "rf")), 4)}
    log(f"  基线估计量用 {len(base)} 条;加上能量维之后用 {len(used)} 条 "
        f"= {100*frac:.1f}%;掉出 {len(drop)} 条")
    log("  用到的 vs 掉出的,标准化均值差:" +
        "  ".join(f"{k} {smd[k]:+.4f}" for k in ("depth", "crossed", "rf", "dG")))
    log(f"  组成:CG 用 {comp['CG']['used']} 掉 {comp['CG']['dropped']};"
        f"AU 用 {comp['AU']['used']} 掉 {comp['AU']['dropped']}")
    OUT["S9"] = bool(frac >= 0.25 and OUT["D"]["max_abs_smd_of_three"] <= 0.20)
    log(f"  S9(>=25% 且三个变量上 |SMD| <= 0.20)  "
        f"{'成立' if OUT['S9'] else '**不成立 ⇒ K3**'}\n")

    OUT["stakes"] = {"S1": OUT["family_matches_declaration"], "S2": OUT["S2_bh_headline"],
                     "S3": bool(s3_notrej and OUT["S3"]["and_sits_at_the_floor"]),
                     "S4": OUT["S4"]["pass"], "S5": OUT["S5"], "S6": OUT["S6"],
                     "S7": OUT["S7"], "S8": OUT["S8"], "S9": OUT["S9"]}
    OUT["verdict"] = "complete"
    json.dump(OUT, open(a.out, "w"), indent=1)
    held = sum(bool(x) for x in OUT["stakes"].values())
    log(f"=== 结算 {held}/9 项预测成立 ===")
    for k, x in OUT["stakes"].items():
        log(f"  {k}  {'成立' if x else '**未成立**'}")
    log(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
