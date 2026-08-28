#!/usr/bin/env python3
"""M30: the selection rule that produced our corpus, and the dynamic range of the score.

Declaration m30_selection_audit_declaration.json, frozen and pushed BEFORE this file existed.
Offline, $0 -- public CSVs on disk plus the ViennaRNA and RibonanzaNet scores already cached by
M26/M26c. No API, no GPU.

The OpenKnotAIDesignData repository says, of the mutate-and-map-rescue experiments specifically,
that the designs were "selected amongst those with RNet_F1 greater than equal to 80", with the
top-scoring design tested anyway when none passed. RNet_F1 is RibonanzaNet's own agreement with
the target structure. Our corpus is conditioned on a neural model's own prediction and no
sentence of the manuscript said so.

This does not argue about it. It reproduces the rule from the public pool (S1), measures how
tightly it binds (S2), runs the control that can decide the question -- the cells in which the
filter did NOT bind (S3) -- asks whether measured skill rises with the tested design's own
RNet_F1 (S4), and reports the dynamic range of the score (D).
"""
from __future__ import annotations

import collections, csv, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

POOL = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
CORPUS = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
NEURAL_CACHE = os.path.join(HERE, "m26c_pneural_cache.npz")

CUT = 0.80              # the repository's "RNet_F1 greater than equal to 80", on the 0-1 scale
NBOOT, NPERM = 2000, 2000
CONTROL_METHOD = "Starting sequence"   # not a designed sequence; not subject to the rule
SAT_HI, SAT_LO = 0.99, 0.08            # the saturation thresholds the outside reading named

LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


# ------------------------------------------------------------------ the two public tables
def read_pool():
    """(puzzle, method, round, openknot score, RNet_F1, design sequence) for every public design."""
    csv.field_size_limit(10 ** 9)
    keep = ("puzzle", "method", "round", "target_openknot_score", "RNet_F1", "design_sequence")
    out = []
    with open(POOL) as f:
        for d in csv.DictReader(f):
            try:
                r = {k: d[k] for k in keep}
                r["round"] = int(float(r["round"]))
                r["RNet_F1"] = float(r["RNet_F1"])
                r["target_openknot_score"] = float(r["target_openknot_score"])
            except (ValueError, KeyError):
                continue
            out.append(r)
    return out


def read_tested():
    """the design actually tested in each M2R cell, cut to its own design window."""
    csv.field_size_limit(10 ** 9)
    out = {}
    with open(CORPUS) as f:
        for d in csv.DictReader(f):
            if d["mutA"] not in ("", "NA", "nan") or d["mutB"] not in ("", "NA", "nan"):
                continue
            ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
            out[(d["puzzle"], d["method"])] = d["sequence"][ss - 1:se]
    return out


def bootstrap_ci(vals, lo=0.025, hi=0.975):
    v = np.sort(np.asarray([x for x in vals if x is not None and np.isfinite(x)], float))
    if len(v) < 20:
        return [None, None]
    return [round(float(v[int(lo * len(v))]), 5), round(float(v[int(hi * len(v))]), 5)]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d else 0.0


def main():
    log("=== M30:产生这份语料的筛选规则,以及分数的动态范围 ===")
    log("  声明已冻结并推送于本文件存在之前。离线 $0。")
    log("  仓库原文(只作用在 mutate-and-map-rescue 上):")
    log('    "The top OpenKnot score design for each method for each target were selected amongst')
    log('     those with RNet_F1 greater than equal to 80. (If no designs passed this secondary')
    log('     structure accuracy cutoff, the top score design was still tested.)"')
    log("  RNet_F1 是 RibonanzaNet 自己与靶标结构的一致度。语料被神经模型自己的预测筛过。\n")
    OUT = {"cutoff": CUT}

    pool = read_pool()
    tested = read_tested()
    log(f"  公开池 {len(pool)} 条设计;M2R 语料 {len(tested)} 个格(靶标 x 方法)")
    OUT["n_pool"] = len(pool)
    OUT["n_cells"] = len(tested)

    # ---------------- linkage: exact design-sequence match, no label map needed
    log("\n=== 连接:按设计序列逐字节匹配(不依赖两个文件互相冲突的方法标签)===")
    by_seq = collections.defaultdict(list)
    for p in pool:
        by_seq[p["design_sequence"]].append(p)
    hit = {k: v for k, v in tested.items() if v in by_seq}
    log(f"  {len(hit)}/{len(tested)} 个被测设计在公开池中找到逐字节相同的序列")
    rounds = collections.Counter(r["round"] for k, v in hit.items() for r in by_seq[v])
    log(f"  它们所在的轮次 {dict(rounds)}")
    OUT["n_linked"] = len(hit)
    OUT["linked_rounds"] = {str(k): v for k, v in rounds.items()}
    if len(hit) < len(tested):
        log("  **未全部连上 ⇒ 下面每一条都只在连上的格上成立,原样写明。**")

    # the tested design's own RNet_F1 -- read directly, not inferred
    f1_of_cell, ok_of_cell, sc_of_cell = {}, {}, {}
    for k, seq in hit.items():
        f1 = max(r["RNet_F1"] for r in by_seq[seq])
        f1_of_cell[k] = f1
        ok_of_cell[k] = bool(f1 >= CUT)
        sc_of_cell[k] = max(r["target_openknot_score"] for r in by_seq[seq])
    npass = sum(ok_of_cell.values())
    log(f"  被测设计自身的 RNet_F1:{npass}/{len(ok_of_cell)} 过线,"
        f"{len(ok_of_cell)-npass} 个是 fallback 格")

    # the empirical method map, derived from the linkage rather than assumed
    mmap = collections.defaultdict(collections.Counter)
    for k, seq in hit.items():
        for r in by_seq[seq]:
            mmap[k[1]][r["method"]] += 1
    method_map = {m: c.most_common(1)[0][0] for m, c in mmap.items()}
    log(f"  由连接反推的方法映射 {method_map}")
    OUT["method_map"] = method_map

    # ---------------- S1: reproduce the stated rule
    log("\n=== S1 用公开池重现仓库写下的那条规则 ===")
    log(f"  对每个非对照格:在同一轮、同一靶标、同一方法的候选里,取 RNet_F1 >= {CUT} 中")
    log("  OpenKnot 分最高者;若无一过线,取全池分最高者。与实际被测的设计比对。")
    rnd = rounds.most_common(1)[0][0] if rounds else None
    by_cell = collections.defaultdict(list)
    for p in pool:
        if p["round"] == rnd:
            by_cell[(p["puzzle"], p["method"])].append(p)
    agree = agreeB = tot = 0
    s1_rows, s1_miss, s1_missB = [], [], []
    for k, seq in sorted(hit.items()):
        if k[1] == CONTROL_METHOD:
            continue
        cand = by_cell.get((k[0], method_map[k[1]]), [])
        if not cand:
            continue
        elig = [c for c in cand if c["RNet_F1"] >= CUT]
        used_fallback = not elig
        pick = max(elig or cand, key=lambda c: c["target_openknot_score"])
        # rule B: the same argmax with the published RNet_F1 filter simply not applied
        pickB = max(cand, key=lambda c: c["target_openknot_score"])
        ok = pick["design_sequence"] == seq
        okB = pickB["design_sequence"] == seq
        srt = sorted(cand, key=lambda c: -c["target_openknot_score"])
        rank = 1 + [i for i, c in enumerate(srt) if c["design_sequence"] == seq][0] \
            if any(c["design_sequence"] == seq for c in cand) else None
        tot += 1; agree += ok; agreeB += okB
        s1_rows.append({"cell": list(k), "n_candidates": len(cand), "n_eligible": len(elig),
                        "rule_used_fallback": used_fallback, "reproduced": bool(ok),
                        "reproduced_no_filter": bool(okB), "rank_on_score": rank})
        if not okB:
            top = srt[0]
            s1_missB.append({"cell": list(k), "rank_on_score": rank,
                             "n_tied_at_top": sum(1 for c in cand
                                 if abs(c["target_openknot_score"]
                                        - top["target_openknot_score"]) < 1e-9),
                             "top_score": round(top["target_openknot_score"], 4),
                             "top_f1": round(top["RNet_F1"], 4),
                             "tested_f1": round(f1_of_cell[k], 4)})
        if not ok:
            s1_miss.append({"cell": list(k), "n_candidates": len(cand), "n_eligible": len(elig),
                            "picked_f1": round(pick["RNet_F1"], 4),
                            "picked_score": round(pick["target_openknot_score"], 4),
                            "tested_f1": round(f1_of_cell[k], 4),
                            "tested_score": round(max(r["target_openknot_score"]
                                                      for r in by_seq[seq]), 4)})
    rate = agree / tot if tot else None
    log(f"  非对照格 {tot};规则重现 {agree} 个,一致率 {rate:.4f}"
        if tot else "  没有可判的格")
    rateB = agreeB / tot if tot else None
    nfb_rule = sum(r["rule_used_fallback"] for r in s1_rows)
    log(f"  其中规则自身走了 fallback 分支的格 {nfb_rule} 个")
    log(f"  同一取最大值、但【不施加】已发表的 RNet_F1 筛子:重现 {agreeB} 个,"
        f"一致率 {rateB:.4f}" if tot else "")
    # the decisive diagnostic: cells where eligible designs existed and were passed over anyway
    passedover = [r for r in s1_rows
                  if (not r["reproduced"]) and r["reproduced_no_filter"] and r["n_eligible"] > 0]
    log(f"  其中 {len(passedover)} 个格:过线的候选存在,却仍然测了不过线的最高分设计")
    log("  ⇒ 已发表的那句规则,在这些格里【没有】起作用。筛选变量是实验 OpenKnot 分,不是 F1。")
    ranks = collections.Counter(r["rank_on_score"] for r in s1_rows)
    log(f"  被测设计在本格 OpenKnot 分上的名次分布 {dict(sorted(ranks.items(), key=lambda x: (x[0] is None, x[0])))}")
    if s1_missB:
        log(f"  『取最高分』也解释不了的 {len(s1_missB)} 个格:")
        for r in s1_missB:
            log(f"    {r['cell']}  名次 {r['rank_on_score']}  顶端并列 {r['n_tied_at_top']}  "
                f"顶端 分={r['top_score']} F1={r['top_f1']}  被测 F1={r['tested_f1']}")
    if s1_miss:
        log(f"  不一致的 {len(s1_miss)} 个格,前五个:")
        for r in s1_miss[:5]:
            log(f"    {r['cell']}  候选 {r['n_candidates']}(过线 {r['n_eligible']})  "
                f"规则选中 F1={r['picked_f1']} 分={r['picked_score']}  "
                f"实测 F1={r['tested_f1']} 分={r['tested_score']}")
    OUT["S1"] = {"n_cells": tot, "n_reproduced": agree,
                 "agreement": round(rate, 4) if rate is not None else None,
                 "n_reproduced_no_filter": agreeB,
                 "agreement_no_filter": round(rateB, 4) if rateB is not None else None,
                 "n_eligible_passed_over": len(passedover),
                 "rank_on_score": {str(k_): v for k_, v in sorted(
                     ranks.items(), key=lambda x: (x[0] is None, x[0]))},
                 "n_rule_fallback": nfb_rule, "round_used": rnd,
                 "mismatches_published_rule": s1_miss[:20],
                 "mismatches_no_filter_rule": s1_missB}
    K0 = bool(rate is not None and rate < 0.70)
    log(f"  **K0(<0.70 则整份审计不可读)= {K0}**")

    # ---------------- S2: how tightly the filter binds
    log("\n=== S2 这道筛子有多紧 ===")
    r3 = [p for p in pool if p["round"] == rnd]
    pool_rate_all = float(np.mean([p["RNet_F1"] >= CUT for p in pool]))
    pool_rate_rnd = float(np.mean([p["RNet_F1"] >= CUT for p in r3])) if r3 else None
    log(f"  全池 {len(pool)} 条中过线 {pool_rate_all:.4f};本轮 {len(r3)} 条中过线 {pool_rate_rnd:.4f}")
    log(f"  被测设计 {npass}/{len(ok_of_cell)} 过线,fallback 格 {len(ok_of_cell)-npass} 个")
    f1v = np.array(list(f1_of_cell.values()))
    log(f"  被测设计 RNet_F1 的分位 "
        f"min {f1v.min():.4f} p10 {np.quantile(f1v,.1):.4f} 中位 {np.median(f1v):.4f} "
        f"p90 {np.quantile(f1v,.9):.4f} max {f1v.max():.4f}")
    OUT["S2"] = {"pool_pass_rate_all_rounds": round(pool_rate_all, 4),
                 "pool_pass_rate_this_round": round(pool_rate_rnd, 4) if pool_rate_rnd else None,
                 "n_pool_this_round": len(r3),
                 "n_tested_pass": npass, "n_tested_fallback": len(ok_of_cell) - npass,
                 "tested_f1_quantiles": {"min": round(float(f1v.min()), 4),
                                         "p10": round(float(np.quantile(f1v, .1)), 4),
                                         "median": round(float(np.median(f1v)), 4),
                                         "p90": round(float(np.quantile(f1v, .9)), 4),
                                         "max": round(float(f1v.max()), 4)}}

    # ---------------- the corpus, with each intervention labelled by its cell's selection status
    log("\n=== 把每一次干预标上它所在格的筛选状态 ===")
    recs, _dropped = M26.build()
    for r in recs:
        r["f1"] = f1_of_cell.get(r["design"])
        r["passed"] = ok_of_cell.get(r["design"])
        r["oks"] = sc_of_cell.get(r["design"])
    unl = [r for r in recs if r["passed"] is None]
    recs = [r for r in recs if r["passed"] is not None]
    keep = [r for r in recs if r["pt"] in ("CG", "AU")]
    npassrec = sum(r["passed"] for r in keep)
    log(f"  M26 纳入集合 {len(recs)+len(unl)};其中格未连上而弃 {len(unl)}")
    log(f"  身份对照用的 CG/AU 干预 {len(keep)}:过线格 {npassrec},fallback 格 {len(keep)-npassrec}")
    K1 = bool(len(keep) - npassrec < 300)
    log(f"  **K1(fallback 干预 <300 则 S3/S4 判为功效不足)= {K1}**")
    OUT["corpus"] = {"n_records": len(recs), "n_unlinked_dropped": len(unl),
                     "n_cgau": len(keep), "n_cgau_pass": npassrec,
                     "n_cgau_fallback": len(keep) - npassrec}

    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    # design is a component of the stratum, and selection status is a function of design alone,
    # so selection status is CONSTANT within every stratum. Verified rather than asserted.
    per = collections.defaultdict(set)
    for r in keep:
        per[sb(r)].add(r["passed"])
    mixed = sum(1 for v in per.values() if len(v) > 1)
    log(f"  分层 {len(per)} 个,其中筛选状态不恒定的 {mixed} 个 "
        f"<- 0 表示筛选状态已被分层完全吸收")
    OUT["strata_mixed_on_selection"] = mixed

    # ---------------- S3: the control that decides
    log("\n=== S3 判定性对照:在筛子没有起作用的那些格里,身份效应还在吗 ===")
    full, wf, uf = M26.matched(keep, "CG", "AU", sb)
    pas = [r for r in keep if r["passed"]]
    fal = [r for r in keep if not r["passed"]]
    dp_, wp, up = M26.matched(pas, "CG", "AU", sb)
    df_, wfb, ufb = M26.matched(fal, "CG", "AU", sb)
    log(f"  全语料   {full:+.5f}  权重 {wp+wfb:.0f} 层 {uf}")
    log(f"  过线格   {dp_:+.5f}  权重 {wp:.0f} 层 {up}")
    log(f"  fallback {df_:+.5f}  权重 {wfb:.0f} 层 {ufb}"
        if df_ is not None else "  fallback 格无可配对的层")
    # puzzle-clustered bootstrap of both arms and of their difference, resampled together
    pz = sorted({r["puzzle"] for r in keep})
    byp = collections.defaultdict(list)
    for r in keep:
        byp[r["puzzle"]].append(r)
    rng = np.random.default_rng(30)
    bf, bp, bd = [], [], []
    for _ in range(NBOOT):
        ix = rng.integers(0, len(pz), len(pz))
        s = [r for q in ix for r in byp[pz[q]]]
        a_, _w, _u = M26.matched([r for r in s if r["passed"]], "CG", "AU", sb)
        b_, _w, _u = M26.matched([r for r in s if not r["passed"]], "CG", "AU", sb)
        if a_ is not None:
            bp.append(a_)
        if b_ is not None:
            bf.append(b_)
        if a_ is not None and b_ is not None:
            bd.append(a_ - b_)
    ci_p, ci_f, ci_d = bootstrap_ci(bp), bootstrap_ci(bf), bootstrap_ci(bd)
    log(f"  过线格 CI {ci_p};fallback 格 CI {ci_f}")
    log(f"  差(过线 - fallback)= {dp_-df_:+.5f}  CI {ci_d}"
        if df_ is not None else "")
    # rule 80: permute the outcome inside the stratification, treatment held fixed
    rngp = np.random.default_rng(31)
    ge = 0
    for _ in range(NPERM):
        v = M26.perm(fal, "CG", "AU", sb, rngp)
        ge += (v is not None and abs(v) >= abs(df_))
    p_fal = (ge + 1) / (NPERM + 1)
    log(f"  fallback 格的置换 p = {p_fal:.5f}(置换结果列,处理列不动 —— 第 80 条)")
    same_sign = bool(df_ is not None and np.sign(df_) == np.sign(full))
    diff_zero = bool(ci_d[0] is not None and ci_d[0] <= 0 <= ci_d[1])
    K2 = bool((not same_sign) or (not diff_zero and (dp_ - df_) > 0))
    log(f"  同号 {same_sign};差的 CI 含 0 {diff_zero}")
    log(f"  **K2(筛选是头条的活解释)= {K2}**")
    OUT["S3"] = {"full": round(full, 5), "pass": round(dp_, 5),
                 "fallback": round(df_, 5) if df_ is not None else None,
                 "diff": round(dp_ - df_, 5) if df_ is not None else None,
                 "ci_pass": ci_p, "ci_fallback": ci_f, "ci_diff": ci_d,
                 "p_perm_fallback": round(p_fal, 5),
                 "weight_pass": wp, "weight_fallback": wfb,
                 "strata_pass": up, "strata_fallback": ufb,
                 "same_sign": same_sign, "diff_ci_contains_zero": diff_zero}

    # ---------------- S4: does measured skill rise with having been selected
    log("\n=== S4 被测出来的『技能』会不会只是被选中的那些设计好预测 ===")
    sl_all, n_all = M26.slope_in(keep, sb, "pmut")
    sl_p, n_p = M26.slope_in(pas, sb, "pmut")
    sl_f, n_f = M26.slope_in(fal, sb, "pmut")
    log(f"  层内 P_model -> rescue 斜率:全语料 {sl_all:+.5f}(n={n_all})  "
        f"过线 {sl_p:+.5f}(n={n_p})  fallback {sl_f:+.5f}(n={n_f})")
    p_sl_f = M26.slope_perm_p(fal, sb, "pmut", sl_f, seed=32, nperm=NPERM)
    log(f"  fallback 格斜率的置换 p = {p_sl_f:.5f}")
    # cell-level skill against the tested design's own RNet_F1
    bycell = collections.defaultdict(list)
    for r in keep:
        bycell[r["design"]].append(r)
    cs, cf, cpz = [], [], []
    for k, rs in bycell.items():
        if len(rs) < 10:
            continue
        s_, _n = M26.slope_in(rs, sb, "pmut")
        if s_ is None or not np.isfinite(s_):
            continue
        cs.append(s_); cf.append(f1_of_cell[k]); cpz.append(k[0])
    rho = spearman(cf, cs) if len(cs) >= 8 else None
    log(f"  可算斜率的格 {len(cs)} 个;格内斜率 vs 该格被测设计的 RNet_F1,Spearman {rho}")
    rb = []
    if rho is not None:
        upz = sorted(set(cpz))
        ixp = collections.defaultdict(list)
        for i, q in enumerate(cpz):
            ixp[q].append(i)
        rr = np.random.default_rng(33)
        for _ in range(NBOOT):
            jj = rr.integers(0, len(upz), len(upz))
            sel = [i for q in jj for i in ixp[upz[q]]]
            if len(set(np.array(cf)[sel])) > 2:
                rb.append(spearman(np.array(cf)[sel], np.array(cs)[sel]))
    ci_rho = bootstrap_ci(rb)
    log(f"  Spearman 的按题聚类自助 CI {ci_rho}")
    OUT["S4"] = {"slope_all": round(sl_all, 5), "slope_pass": round(sl_p, 5),
                 "slope_fallback": round(sl_f, 5), "n_all": n_all, "n_pass": n_p,
                 "n_fallback": n_f, "p_perm_slope_fallback": round(p_sl_f, 5),
                 "n_cells_with_slope": len(cs),
                 "spearman_skill_vs_f1": round(rho, 4) if rho is not None else None,
                 "ci_spearman": ci_rho}

    # ---------------- D: dynamic range
    log("\n=== D 分数的动态范围(外部读法给的判死条件:多数 > 0.99 即饱和)===")
    dd = {}
    pm = np.array([r["pmut"] for r in recs], float)
    series = [("P_model_ViennaRNA", pm)]
    if os.path.exists(NEURAL_CACHE):
        z = np.load(NEURAL_CACHE, allow_pickle=True)
        cache = {k: v for k, v in zip(z["keys"], z["mats"])}
        miss = sum(1 for r in recs if r["mut_seq"] not in cache)
        if miss == 0:
            nm = np.array([float(cache[r["mut_seq"]][min(r["a"], r["b"]), max(r["a"], r["b"])])
                           for r in recs], float)
            series.append(("P_neural_RibonanzaNet", nm))
        else:
            log(f"  神经缓存缺 {miss} 条序列 ⇒ 只报热力学一侧")
        del cache, z
    for nm_, v in series:
        row = {"n": len(v), "min": round(float(v.min()), 5), "max": round(float(v.max()), 5),
               "p10": round(float(np.quantile(v, .1)), 5),
               "median": round(float(np.median(v)), 5),
               "p90": round(float(np.quantile(v, .9)), 5),
               "frac_above_0.99": round(float((v > SAT_HI).mean()), 4),
               "frac_below_0.08": round(float((v < SAT_LO).mean()), 4)}
        row["frac_between"] = round(1.0 - row["frac_above_0.99"] - row["frac_below_0.08"], 4)
        dd[nm_] = row
        log(f"  {nm_:24s} min {row['min']:.4f} p10 {row['p10']:.4f} 中位 {row['median']:.4f} "
            f"p90 {row['p90']:.4f} max {row['max']:.4f}")
        log(f"  {'':24s} >0.99 占 {row['frac_above_0.99']:.4f}   "
            f"<0.08 占 {row['frac_below_0.08']:.4f}   中间 {row['frac_between']:.4f}")
    K3 = bool(dd["P_model_ViennaRNA"]["frac_above_0.99"] > 0.5)
    log(f"  **K3(热力学分数饱和,M26 的标定读法作废)= {K3}**")
    OUT["D"] = dd

    # ---------------- S3b/S4b: the arm S1 forced. Added AFTER the pre-registered arms and
    # marked post-hoc: the declaration aimed S3/S4 at RNet_F1 because the repository names it,
    # and S1 then showed the variable that actually did the selecting is the experimental
    # OpenKnot score. A control aimed at the wrong variable is not a control, so the same two
    # arms are re-run against the right one. Nothing above is altered.
    log("\n=== S3b/S4b(事后追加,由 S1 逼出来)真正做筛选的是实验 OpenKnot 分 ===")
    cells_sc = sorted({(r["design"], r["oks"]) for r in keep}, key=lambda x: x[1])
    med_sc = float(np.median([c[1] for c in cells_sc]))
    hi = [r for r in keep if r["oks"] > med_sc]
    lo = [r for r in keep if r["oks"] <= med_sc]
    log(f"  按格中位 OpenKnot 分 {med_sc:.4f} 一分为二:高 {len(hi)} 条,低 {len(lo)} 条")
    dh, wh, uh = M26.matched(hi, "CG", "AU", sb)
    dl, wl, ul = M26.matched(lo, "CG", "AU", sb)
    log(f"  高分格 {dh:+.5f}(权重 {wh:.0f} 层 {uh});低分格 {dl:+.5f}(权重 {wl:.0f} 层 {ul})")
    rng2 = np.random.default_rng(34)
    bh, bl, bdd = [], [], []
    for _ in range(NBOOT):
        ix = rng2.integers(0, len(pz), len(pz))
        srs = [r for q in ix for r in byp[pz[q]]]
        a_, _w, _u = M26.matched([r for r in srs if r["oks"] > med_sc], "CG", "AU", sb)
        b_, _w, _u = M26.matched([r for r in srs if r["oks"] <= med_sc], "CG", "AU", sb)
        if a_ is not None:
            bh.append(a_)
        if b_ is not None:
            bl.append(b_)
        if a_ is not None and b_ is not None:
            bdd.append(a_ - b_)
    ci_h, ci_l, ci_dd = bootstrap_ci(bh), bootstrap_ci(bl), bootstrap_ci(bdd)
    log(f"  高分格 CI {ci_h};低分格 CI {ci_l};差 {dh-dl:+.5f} CI {ci_dd}")
    sl_h, nh = M26.slope_in(hi, sb, "pmut")
    sl_l, nl = M26.slope_in(lo, sb, "pmut")
    log(f"  层内 P_model -> rescue 斜率:高分格 {sl_h:+.5f}(n={nh})  低分格 {sl_l:+.5f}(n={nl})")
    cs2, cx2, cpz2 = [], [], []
    for k, rs in bycell.items():
        if len(rs) < 10:
            continue
        s_, _n = M26.slope_in(rs, sb, "pmut")
        if s_ is None or not np.isfinite(s_):
            continue
        cs2.append(s_); cx2.append(sc_of_cell[k]); cpz2.append(k[0])
    rho2 = spearman(cx2, cs2) if len(cs2) >= 8 else None
    rb2 = []
    if rho2 is not None:
        upz2 = sorted(set(cpz2))
        ixp2 = collections.defaultdict(list)
        for i, q in enumerate(cpz2):
            ixp2[q].append(i)
        rr2 = np.random.default_rng(35)
        for _ in range(NBOOT):
            jj = rr2.integers(0, len(upz2), len(upz2))
            sel = [i for q in jj for i in ixp2[upz2[q]]]
            if len(set(np.array(cx2)[sel])) > 2:
                rb2.append(spearman(np.array(cx2)[sel], np.array(cs2)[sel]))
    ci_rho2 = bootstrap_ci(rb2)
    log(f"  格内斜率 vs 该格被测设计的 OpenKnot 分,Spearman {rho2}  CI {ci_rho2}")
    dz2 = bool(ci_dd[0] is not None and ci_dd[0] <= 0 <= ci_dd[1])
    rz2 = bool(ci_rho2[0] is not None and ci_rho2[0] <= 0 <= ci_rho2[1])
    log(f"  身份效应两臂差含 0 {dz2};技能与筛选分的相关含 0 {rz2}")
    OUT["S3b_S4b_openknot_score"] = {
        "post_hoc": True, "median_score": round(med_sc, 4),
        "n_hi": len(hi), "n_lo": len(lo),
        "contrast_hi": round(dh, 5), "contrast_lo": round(dl, 5),
        "ci_hi": ci_h, "ci_lo": ci_l, "diff": round(dh - dl, 5), "ci_diff": ci_dd,
        "weight_hi": wh, "weight_lo": wl, "strata_hi": uh, "strata_lo": ul,
        "slope_hi": round(sl_h, 5), "slope_lo": round(sl_l, 5), "n_hi_slope": nh, "n_lo_slope": nl,
        "spearman_skill_vs_score": round(rho2, 4) if rho2 is not None else None,
        "ci_spearman": ci_rho2,
        "diff_ci_contains_zero": dz2, "spearman_ci_contains_zero": rz2}

    # ---------------- S4c: the same question on a scale-free statistic.
    # S4b compares regression slopes across two populations whose P_model dispersion need not
    # match. That is the comparison I made in M28 and retracted (rule 68: put things on the same
    # scale before comparing magnitudes). If the negative association is real it must survive on
    # a rank statistic, which is invariant to the dispersion of either column.
    log("\n=== S4c 同一个问题,换成不随尺度变化的统计量(第 68 条)===")
    sd_hi = float(np.std([r["pmut"] for r in hi]))
    sd_lo = float(np.std([r["pmut"] for r in lo]))
    log(f"  P_model 的离散度:高分格 SD {sd_hi:.4f},低分格 SD {sd_lo:.4f},比 {sd_lo/max(sd_hi,1e-9):.3f}")

    def cell_rank_skill(rs):
        """within-cell Spearman of P_model against rescue, centred inside the finer strata."""
        by = collections.defaultdict(list)
        for r in rs:
            by[sb(r)].append(r)
        xs, ys = [], []
        for _s, g in by.items():
            if len(g) < 2:
                continue
            x = np.array([r["pmut"] for r in g], float)
            y = np.array([r["rf"] for r in g], float)
            xs.append(x - x.mean()); ys.append(y - y.mean())
        if not xs:
            return None
        x = np.concatenate(xs); y = np.concatenate(ys)
        if len(x) < 8 or len(set(x.tolist())) < 3:
            return None
        return spearman(x, y)

    cs3, cx3, cf3, cpz3 = [], [], [], []
    for k, rs in bycell.items():
        if len(rs) < 10:
            continue
        v = cell_rank_skill(rs)
        if v is None or not np.isfinite(v):
            continue
        cs3.append(v); cx3.append(sc_of_cell[k]); cf3.append(f1_of_cell[k]); cpz3.append(k[0])
    log(f"  可算秩技能的格 {len(cs3)} 个")
    res3 = {}
    for nm_, xv in (("openknot_score", cx3), ("RNet_F1", cf3)):
        rho3 = spearman(xv, cs3) if len(cs3) >= 8 else None
        rb3 = []
        if rho3 is not None:
            upz3 = sorted(set(cpz3)); ixp3 = collections.defaultdict(list)
            for i, q in enumerate(cpz3):
                ixp3[q].append(i)
            rr3 = np.random.default_rng(36)
            for _ in range(NBOOT):
                jj = rr3.integers(0, len(upz3), len(upz3))
                sel = [i for q in jj for i in ixp3[upz3[q]]]
                if len(set(np.array(xv)[sel])) > 2:
                    rb3.append(spearman(np.array(xv)[sel], np.array(cs3)[sel]))
        ci3 = bootstrap_ci(rb3)
        zero3 = bool(ci3[0] is not None and ci3[0] <= 0 <= ci3[1])
        res3[nm_] = {"spearman": round(rho3, 4) if rho3 is not None else None,
                     "ci": ci3, "ci_contains_zero": zero3}
        log(f"  格内【秩】技能 vs {nm_}:Spearman {rho3}  CI {ci3}  含 0 {zero3}")
    # and the two arms compared on the rank statistic rather than on slopes
    rk_hi = float(np.median([v for v, x in zip(cs3, cx3) if x > med_sc]))
    rk_lo = float(np.median([v for v, x in zip(cs3, cx3) if x <= med_sc]))
    log(f"  两臂的格内秩技能中位数:高分格 {rk_hi:+.4f},低分格 {rk_lo:+.4f}")
    res3["sd_pmut_hi"] = round(sd_hi, 4)
    res3["sd_pmut_lo"] = round(sd_lo, 4)
    res3["sd_ratio_lo_over_hi"] = round(sd_lo / max(sd_hi, 1e-9), 4)
    res3["median_rank_skill_hi"] = round(rk_hi, 4)
    res3["median_rank_skill_lo"] = round(rk_lo, 4)
    res3["n_cells"] = len(cs3)
    OUT["S4c_scale_free"] = res3

    # ---------------- verdict against what was staked
    log("\n=== 对照声明里预先押下的七条 ===")
    st = {
        "1_rule_reproduces_ge_0.90": bool(rate is not None and rate >= 0.90),
        "1b_no_filter_rule_reproduces_ge_0.90_POSTHOC":
            bool(rateB is not None and rateB >= 0.90),
        "2_pool_pass_rate_ge_0.70": bool(pool_rate_rnd is not None and pool_rate_rnd >= 0.70),
        "3_control_arm_powered": bool((len(ok_of_cell) - npass) >= 20
                                      and (len(keep) - npassrec) >= 300),
        "4_fallback_contrast_same_sign": same_sign,
        "5_diff_ci_contains_zero": diff_zero,
        "6_not_saturated": bool(dd["P_model_ViennaRNA"]["frac_above_0.99"] < 0.5),
        "7_skill_does_not_rise_with_f1": bool(ci_rho[0] is not None and ci_rho[0] <= 0 <= ci_rho[1]),
    }
    for k, v in st.items():
        log(f"  {k:36s} {v}")
    OUT["staked"] = st
    OUT["kill"] = {"K0_rule_not_reproduced": K0, "K1_control_underpowered": K1,
                   "K2_selection_is_live": K2, "K3_score_saturated": K3}
    OUT["verdict"] = ("void_rule_not_reproduced" if K0 else
                      "selection_is_a_live_explanation" if K2 else
                      "selection_conditioned_out_but_control_underpowered" if K1 else
                      "selection_real_disclosed_and_conditioned_out")
    log(f"\n  裁定 {OUT['verdict']}")

    json.dump(OUT, open(os.path.join(HERE, "m30_selection_audit.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m30_selection_audit.log"), "w").write("\n".join(LOG) + "\n")
    log(f"\n写出 m30_selection_audit.json / .log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
