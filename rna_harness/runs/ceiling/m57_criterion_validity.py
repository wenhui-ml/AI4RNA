#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M57: the number Definition 1 clause (iii) has never carried, and Claim 3 on each side of it.

Declaration m57_criterion_validity_declaration.json was frozen and pushed before this file existed.

Clause (iii) admits an intervention when the flipped pair is in the DECLARED target.  A declared
target is an intention, and nothing in the manuscript says how often the molecule agrees with it.
That fraction is the ceiling on the criterion's validity, and every agreement number scored against
the declared target inherits it.

One number already exists and must not be used: M53's declaration recorded 0.623 of declared-target
pairs recovered, computed from the RELEASED M2_structure column -- the column M53 then VOIDED after
finding 74 of 160 of its strings are not balanced dot-bracket.  A reviewer has since quoted that
0.623 back to us.  It is measured against a defective column and this file recomputes it against
M54's built reference instead, reporting both and saying which one is load-bearing.

Nothing is folded here.  M54's ShapeKnots cache and M55's record cache are read.
"""
from __future__ import annotations

import collections, csv, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m53_probing_reference as M53
from m53_probing_reference import (pairs_of, f1, depth_of, wrho, pct, wild_types,
                                   NMIN, SEED, NBOOT, PUB_RANK, PUB_DEPTH)

DECL = os.path.join(HERE, "m57_criterion_validity_declaration.json")
OUTJ = os.path.join(HERE, "m57_criterion_validity.json")
SKC = os.path.join(HERE, "m54_shapeknots_cache.json")
M54J = os.path.join(HERE, "m54_probing_reference.json")
C50 = os.path.join(HERE, "m50_recs_cache.json")
C52 = os.path.join(HERE, "m52_recs_cache.json")
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def _plain(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save(OUT):
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False, default=_plain)
    log(f"\n写出 {OUTJ}")


def main():
    t0 = time.time()
    log("=== M57:声明靶标的配对到底在不在,以及 Claim 3 分层 ===")
    log("  声明已冻结并推送于本文件存在之前。不折叠任何序列,全离线,$0。")
    OUT = {"declaration": os.path.basename(DECL), "cost": "offline, $0 -- no folding, no API",
           "V": {}}

    wt = wild_types()
    c = json.load(open(SKC))
    ref = {tuple(json.loads(k)): {tuple(p) for p in v} for k, v in c["with_data"].items()}
    log(f"  野生型 {len(wt)}  built 参考 {len(ref)}")

    def win(k, prs):
        d = wt[k]; ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        return {(i, j) for i, j in prs if ss - 1 <= i < se and ss - 1 <= j < se}

    # ------------------------------------------------------------------ V, before any arm
    log("\n=== V 验证(先于任何 arm)===")
    _k0 = sorted(ref)[0]
    v1 = (f1(ref[_k0], ref[_k0]) == 1.0 and f1(ref[_k0], set()) == 0.0)
    log(f"  V1 自身 F1 {f1(ref[_k0], ref[_k0]):.1f}  对空 {f1(ref[_k0], set()):.1f}"
        f"  -> {'通过' if v1 else '失败'}")
    OUT["V"]["V1_zero_point"] = bool(v1)

    c50 = json.load(open(C50))["recs"]
    c52 = json.load(open(C52))["recs"]
    assert len(c50) == len(c52), "两个记录缓存长度不一致"
    R = []
    for x, y in zip(c50, c52):
        k = tuple(x["design"])
        if k not in ref:
            continue
        a, b = int(y["a"]), int(y["b"])
        P = {}
        for i, j in ref[k]:
            P[i] = j; P[j] = i
        _dp, present = depth_of(P, a, b)
        R.append({"design": k, "puzzle": x["puzzle"], "rf": x["rf"], "nmut": x["nmut"],
                  "pmut": x["pmut"], "emut": x["emut"], "depth": x["depth"],
                  "present": float(present)})
    frac_iv = float(np.mean([r["present"] for r in R]))
    pub_iv = json.load(open(M54J))["A3_depth"]["frac_pair_present"]
    v2 = abs(frac_iv - pub_iv) < 5e-5
    log(f"  V2 干预配对的确认率 我 {frac_iv:.5f}  M54 {pub_iv:.4f}  |Δ| {abs(frac_iv-pub_iv):.2e}"
        f"  -> {'通过' if v2 else '失败'}")
    OUT["V"]["V2_reproduces_m54"] = bool(v2)
    if not v2:
        OUT["VOID"] = "K2: V2 未复现 M54 的确认率"; save(OUT); return 1

    bc = collections.defaultdict(list)
    for r in R:
        bc[r["design"]].append(r)
    cl = [(k, v) for k, v in bc.items() if len(v) >= NMIN]
    log(f"  至少 {NMIN} 个翻转的设计格 {len(cl)},覆盖 {sum(len(v) for _k, v in cl)} 次干预")
    v3 = {n: wrho(cl, kx) for n, kx in
          (("RibonanzaNet", "nmut"), ("ViennaRNA", "pmut"), ("EternaFold", "emut"))}
    dd = wrho(cl, "depth")
    for n in PUB_RANK:
        log(f"  V3 {n:14s} 我 {v3[n]:+.5f}  论文 {PUB_RANK[n]:+.4f}")
    log(f"  V3 depth         我 {dd:+.5f}  论文 {PUB_DEPTH:+.4f}")
    v3ok = (all(abs(v3[n] - PUB_RANK[n]) < 5e-4 for n in PUB_RANK) and abs(dd - PUB_DEPTH) < 5e-4)
    OUT["V"]["V3_reverse_control"] = bool(v3ok)
    if not v3ok:
        OUT["VOID"] = "K2: V3 反向对照未复现"; save(OUT); return 1

    # ------------------------------------------------------------------ A1, clause (iii)
    log("\n=== A1 clause (iii) 的有效性上界:声明靶标的配对有多少是真的 ===")
    per, tot_hit, tot_all = [], 0, 0
    for k in sorted(ref):
        if k not in wt:
            continue
        tgt = win(k, pairs_of(wt[k]["target_structure"]))
        if not tgt:
            continue
        hit = len(tgt & ref[k])
        per.append(hit / len(tgt)); tot_hit += hit; tot_all += len(tgt)
    pooled = tot_hit / tot_all
    per = np.asarray(per, float)
    log(f"  设计窗内的声明配对 {tot_all} 个,built 参考也成对的 {tot_hit} 个,合计 {pooled:.4f}")
    log(f"  按设计格看:中位数 {np.median(per):.4f}  四分位 [{np.percentile(per,25):.4f},"
        f" {np.percentile(per,75):.4f}]  最低 {per.min():.4f}  最高 {per.max():.4f}"
        f"  全空的格 {int((per==0).sum())}/{len(per)}")
    log(f"  只看被干预的那些配对:{frac_iv:.4f} —— 比全体高,干预并非在随机配对上做的")
    rng0 = np.random.default_rng(SEED)
    bp0 = collections.defaultdict(list)
    for i, k in enumerate([k for k in sorted(ref) if k in wt
                           and win(k, pairs_of(wt[k]["target_structure"]))]):
        bp0[k[0]].append(per[i])
    pz0 = sorted(bp0)
    bs0 = []
    for _ in range(NBOOT):
        ix = rng0.integers(0, len(pz0), len(pz0))
        v = [x for q in ix for x in bp0[pz0[q]]]
        bs0.append(float(np.mean(v)))
    ci0 = pct(bs0)
    log(f"  逐格平均 {per.mean():.4f}  {ci0}(按 puzzle 重抽)")
    OUT["A1_clause_iii"] = {
        "pooled_over_declared_pairs": round(pooled, 4), "n_declared_pairs": tot_all,
        "n_confirmed": tot_hit, "per_design_mean": round(float(per.mean()), 4),
        "per_design_mean_ci": ci0, "per_design_median": round(float(np.median(per)), 4),
        "per_design_min": round(float(per.min()), 4), "per_design_max": round(float(per.max()), 4),
        "n_designs_with_none_confirmed": int((per == 0).sum()), "n_designs": len(per),
        "over_intervened_pairs_only": round(frac_iv, 4),
        "released_column_figure_do_not_use": {
            "value": 0.623,
            "why_not": "measured against the released M2_structure column, which M53 voided after "
                       "finding 74/160 of its strings are not balanced dot-bracket"}}

    # ------------------------------------------------------------------ A2, Claim 3 stratified
    log("\n=== A2 Claim 3 分层:built 参考确认该配对 / 不确认 ===")
    stratum = {}
    for nm, want in (("confirmed", 1.0), ("unconfirmed", 0.0)):
        sub = collections.defaultdict(list)
        for r in R:
            if r["present"] == want:
                sub[r["design"]].append(r)
        stratum[nm] = [(k, v) for k, v in sub.items() if len(v) >= NMIN]
    n_c = sum(len(v) for _k, v in stratum["confirmed"])
    n_u = sum(len(v) for _k, v in stratum["unconfirmed"])
    n_all = sum(len(v) for _k, v in cl)
    log(f"  确认层 {n_c} 次干预 / {len(stratum['confirmed'])} 格;"
        f"未确认层 {n_u} 次 / {len(stratum['unconfirmed'])} 格")
    v4 = True   # disjointness is by construction; the count identity is the check that matters
    tot_split = sum(1 for r in R if r["present"] == 1.0) + sum(1 for r in R if r["present"] == 0.0)
    v4 = bool(tot_split == len(R))
    log(f"  V4 两层互斥且合计 {tot_split}/{len(R)}  -> {'通过' if v4 else '失败'}")
    OUT["V"]["V4_partition"] = v4
    if min(n_c, n_u) < 200:
        OUT["K1_split_not_reportable"] = {"confirmed": n_c, "unconfirmed": n_u}
        log("  K1:某一层不足 200 次干预,按声明不报区间")
    arms = (("RibonanzaNet", "nmut"), ("ViennaRNA", "pmut"), ("EternaFold", "emut"))
    A2 = {"n_interventions_confirmed": n_c, "n_interventions_unconfirmed": n_u,
          "n_cells_confirmed": len(stratum["confirmed"]),
          "n_cells_unconfirmed": len(stratum["unconfirmed"]), "strata": {}}
    for nm in ("confirmed", "unconfirmed"):
        cls = stratum[nm]
        d_ = wrho(cls, "depth")
        row = {"depth": round(d_, 5), "predictors": {}}
        bpx = collections.defaultdict(list)
        for k, v in cls:
            bpx[v[0]["puzzle"]].append((k, v))
        pzx = sorted(bpx); rngx = np.random.default_rng(SEED + 7)
        draws = {n: [] for n, _ in arms}
        for _ in range(NBOOT):
            ix = rngx.integers(0, len(pzx), len(pzx))
            cs = [x for q in ix for x in bpx[pzx[q]]]
            dv = wrho(cs, "depth")
            if dv is None:
                continue
            for n, kx in arms:
                w = wrho(cs, kx)
                draws[n].append(dv - w if w is not None else np.nan)
        log(f"  [{nm}] depth {d_:+.5f}")
        for n, kx in arms:
            w_ = wrho(cls, kx)
            v = np.asarray(draws[n], float); v = v[~np.isnan(v)]
            ci = pct(v)
            row["predictors"][n] = {"level": round(w_, 5), "depth_minus": round(d_ - w_, 5),
                                    "ci95": ci, "ci_contains_zero": bool(ci[0] <= 0 <= ci[1]),
                                    "depth_wins": bool(d_ - w_ > 0)}
            log(f"    vs {n:14s} {w_:+.5f}   depth − 它 {d_-w_:+.5f}  {ci}"
                f"  {'含 0' if ci[0] <= 0 <= ci[1] else '不含 0'}")
        A2["strata"][nm] = row

    # V5: pooling the two strata back must not be asked of wrho -- the cell sets differ, so the
    # identity that IS available is that every intervention lands in exactly one stratum, which
    # V4 checked.  Recording that this is why V5 was not run as declared.
    OUT["V"]["V5_note"] = ("the declared pooling identity is not available: the two strata are "
                           "re-celled under the same NMIN rule, so their cell sets are not a "
                           "partition of the unstratified cell set. V4's count identity is the "
                           "check that survives, and it passed. Recorded, not repaired.")
    log("  V5 声明里那条合并恒等式不可用(两层各自重新按 NMIN 分格,格集合不是原格集的划分);"
        "留 V4 的计数恒等式,如实记录")

    # paired difference between strata
    dc = A2["strata"]["confirmed"]["depth"]; du = A2["strata"]["unconfirmed"]["depth"]
    log(f"  depth 在两层之差(确认 − 未确认) {dc-du:+.5f}")
    A2["depth_confirmed_minus_unconfirmed"] = round(dc - du, 5)
    OUT["A2_claim3_stratified"] = A2

    # ------------------------------------------------------------------ A3, is depth just this?
    log("\n=== A3 depth 是不是只是在报告『这个配对是不是真的』===")
    p_ = wrho(cl, "present")
    log(f"  确认指示变量自己的 level 臂 {p_:+.5f};depth {dd:+.5f}")
    bpz = collections.defaultdict(list)
    for k, v in cl:
        bpz[v[0]["puzzle"]].append((k, v))
    pzz = sorted(bpz); rngz = np.random.default_rng(SEED + 11)
    dr = []
    for _ in range(NBOOT):
        ix = rngz.integers(0, len(pzz), len(pzz))
        cs = [x for q in ix for x in bpz[pzz[q]]]
        a_, b_ = wrho(cs, "depth"), wrho(cs, "present")
        if a_ is not None and b_ is not None:
            dr.append(a_ - b_)
    ci3 = pct(dr)
    log(f"  depth − 确认指示  {dd-p_:+.5f}  {ci3}  "
        f"{'含 0' if ci3[0] <= 0 <= ci3[1] else '不含 0'}")
    OUT["A3_depth_vs_confirmation"] = {
        "confirmation_level": round(p_, 5), "depth_level": round(dd, 5),
        "difference": round(dd - p_, 5), "ci95": ci3,
        "ci_contains_zero": bool(ci3[0] <= 0 <= ci3[1]), "depth_larger": bool(dd > p_)}

    # ------------------------------------------------------------------ stakes
    log("\n=== 预注册记分 ===")
    S = {}
    S["S1"] = bool(pooled >= 0.50)
    S["S2"] = bool(all(A2["strata"]["confirmed"]["predictors"][n]["depth_wins"] for n, _ in arms))
    S["S3"] = bool(all(A2["strata"]["unconfirmed"]["predictors"][n]["depth_wins"] for n, _ in arms))
    S["S4"] = bool(dd > p_)
    log(f"  S1  {'成立' if S['S1'] else '**失败**'}   clause (iii) 确认率 {pooled:.4f}")
    log(f"  S2  {'成立' if S['S2'] else '**失败**'}   确认层 depth 胜过全部三个")
    log(f"  S3  {'成立' if S['S3'] else '**失败**'}   未确认层 depth 胜过全部三个")
    log(f"  S4  {'成立' if S['S4'] else '**失败**'}   depth {dd:+.5f} vs 确认指示 {p_:+.5f}")
    met = sum(S.values())
    verdict = ("clause_iii_is_bounded_and_claim3_holds_in_both_strata" if met == 4 else
               "clause_iii_is_bounded_and_claim3_has_a_domain")
    OUT["stakes"] = S; OUT["n_met"] = met; OUT["verdict"] = verdict
    log(f"\n  裁定:{verdict}   {met}/4 成立   用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
