#!/usr/bin/env python3
"""M61: the readout's chemistry, the readout's reliability, and the consequence for a leaderboard.

Declaration: m61_readout_and_consequence_declaration.json, frozen, committed and pushed BEFORE
this file existed. Departures from it are recorded in m61_declaration_erratum.json; the
declaration itself is never edited.

The one structural fact the whole chemistry arm turns on: the compensatory double mutant replaces
(i,j) by its complement, so it preserves base composition exactly (G-C -> C-G, A-U -> U-A). The
numerator rho_ij therefore carries NO composition change. Every composition change in R lives in
the denominator sqrt(rho_i^2 + rho_j^2). Measuring the denominator is measuring the objection.

COST: $0, offline, CPU only.
"""
from __future__ import annotations
import collections, csv, json, math, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m16_rescue as M16
import m21_thermo_mediator as M21
from designaudit import core

SRC = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
SEED = 20260901
NDRAW = 200          # B1 parametric bootstrap draws per record
NBOOT_TIGHT = 400    # D2 cluster bootstrap resamples
THERMO = ("Eterna", "Rosetta")
NEURAL = ("MPNN-fixbb", "MPNN-RFdiff", "gRNAde", "gRNAde-no3d",
          "Shujun_struct2seq", "Shujun_shape_struct2seq")
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


# --------------------------------------------------------------------------- loader
def load_profiles():
    """Every row's reactivity profile AND its released per-nucleotide error."""
    csv.field_size_limit(10 ** 9)
    f = open(SRC)
    rd = csv.DictReader(f)
    rk = sorted(k for k in rd.fieldnames
                if k.startswith("reactivity_") and not k.startswith("reactivity_error"))
    ek = sorted(k for k in rd.fieldnames if k.startswith("reactivity_error"))
    assert len(rk) == len(ek), (len(rk), len(ek))
    num = lambda v: np.nan if v in ("", "NA", "nan") else float(v)
    rows = []
    for d in rd:
        rows.append({"puzzle": d["puzzle"], "method": d["method"], "seq": d["sequence"],
                     "db": d["target_structure"], "mutA": d["mutA"], "mutB": d["mutB"],
                     "rf": d["rescue_factor"], "sn": d["SN_filter"], "ss": d["sub_start"],
                     "et": d["experiment_type"],
                     "react": np.array([num(d[k]) for k in rk], dtype=np.float64),
                     "err": np.array([num(d[k]) for k in ek], dtype=np.float64)})
    f.close()
    return rows


def rmsd(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else np.nan


def build():
    """M16's own record set, plus the three RMSD components and the four profiles."""
    rows = load_profiles()
    et = collections.Counter(r["et"] for r in rows)
    # DEFECT FOUND HERE, in M16 rather than in this file. M16 indexed single mutants by mutA only,
    # so the B-side singles -- released as (mutA empty, mutB set) -- were invisible to it and every
    # record's second single-mutant RMSD was None. M16's declared denominator diagnostic was
    # therefore computed on one side of every pair. There are 4752 of each side; both are indexed
    # here, and M16's number is superseded rather than reproduced.
    wt, singles = {}, collections.defaultdict(dict)
    for d in rows:
        k = (d["puzzle"], d["method"])
        a_set = d["mutA"] not in ("", "NA", "nan")
        b_set = d["mutB"] not in ("", "NA", "nan")
        if not a_set and not b_set:
            wt[k] = d
        elif a_set and not b_set:
            singles[k][int(float(d["mutA"]))] = d
        elif b_set and not a_set:
            singles[k][int(float(d["mutB"]))] = d
    recs = []
    for d in rows:
        if d["rf"] in ("", "NA", "nan") or d["sn"] not in ("1", "1.0"):
            continue
        k = (d["puzzle"], d["method"]); p = wt.get(k)
        if p is None:
            continue
        ss = int(float(d["ss"])); P = M16.parse_pairs(d["db"])
        try:
            ma, mb = int(float(d["mutA"])), int(float(d["mutB"]))
        except ValueError:
            continue
        a, b = ma + ss - 2, mb + ss - 2
        if P.get(a) != b:
            continue
        s = p["seq"]; pt = "".join(sorted({s[a], s[b]}))
        dep = 0
        while P.get(a - dep - 1) == b + dep + 1:
            dep += 1
        up = 0
        while P.get(a + up + 1) == b - up - 1:
            up += 1
        crossed = any(m < a < n < b or a < m < b < n for m, n in P.items() if m < n)
        sa, sb = singles[k].get(ma), singles[k].get(mb)
        if sa is None or sb is None:
            continue
        ri, rj, rij = rmsd(p["react"], sa["react"]), rmsd(p["react"], sb["react"]), \
            rmsd(p["react"], d["react"])
        D = math.sqrt(ri ** 2 + rj ** 2) if np.isfinite(ri) and np.isfinite(rj) else np.nan
        recs.append({"puzzle": d["puzzle"], "method": d["method"], "pt": pt,
                     "rf": float(d["rf"]), "depth": min(dep, up), "crossed": bool(crossed),
                     "design": k, "rho_i": ri, "rho_j": rj, "rho_ij": rij, "D": D,
                     "a_idx": a, "b_idx": b,   # additive, for M61b's off-mutation cut only
                     "R_recomputed": (1 - rij / D) if (np.isfinite(D) and D > 0) else np.nan,
                     "prof": (p["react"], sa["react"], sb["react"], d["react"]),
                     "errs": (p["err"], sa["err"], sb["err"], d["err"])})
    return recs, et


def flat(recs, extra=()):
    """core.Corpus wants flat scalar records."""
    out = []
    for r in recs:
        f = {"design": "%s|%s" % r["design"], "pt": r["pt"], "depth3": min(r["depth"], 3),
             "crossed": r["crossed"], "puzzle": r["puzzle"], "method": r["method"],
             "rf": r["rf"], "D": r["D"], "rho_ij": r["rho_ij"], "rho_i": r["rho_i"],
             "rho_j": r["rho_j"]}
        for k in extra:
            f[k] = r[k]
        out.append(f)
    return out


def ctr(rows, outcome, covs=("depth3", "crossed"), nperm=2000, nboot=400):
    c = core.Corpus.from_records(rows, design="design", treatment="pt", outcome=outcome,
                                 covariates=covs, cluster="puzzle")
    return core.contrast(c, ("CG", "AU"), strata=covs, nperm=nperm, nboot=nboot, seed=SEED)


def show(name, e, scale=""):
    log(f"  {name:<40}{e.effect:>+10.4f}   [{e.ci[0]:+.4f}, {e.ci[1]:+.4f}]{e.p:>9.4f}"
        f"   w={e.weight:.0f}{scale}")
    return {"effect": round(e.effect, 5), "ci": [round(x, 4) for x in e.ci],
            "p": round(e.p, 5), "weight": e.weight, "strata": e.strata}


# --------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    OUT = {"seed": SEED}
    log("=== M61:读数的化学、读数的可靠性,以及排行榜后果 ===")
    log("  声明冻结在代码之前;偏离记在 m61_declaration_erratum.json。全离线,$0。\n")

    recs, et = build()
    log(f"记录 {len(recs)}  设计格 {len({r['design'] for r in recs})}  靶标 {len({r['puzzle'] for r in recs})}")
    rows = flat(recs)

    # ---------------------------------------------------------------- V
    log("\n【V】先复算已发表的数")
    ev = ctr(rows, "rf")
    OUT["V"] = show("rescue 对比 (已发表 +0.05393)", ev)
    ok_v = abs(ev.effect - 0.05393) < 5e-4
    log(f"    -> {'通过' if ok_v else '**不通过,下面不读**'}")
    OUT["V"]["reproduces_published"] = bool(ok_v)

    # agreement of the recomputed R with the released column
    a = np.array([r["rf"] for r in recs]); b = np.array([r["R_recomputed"] for r in recs])
    m = np.isfinite(a) & np.isfinite(b)
    OUT["V"]["recompute_corr"] = round(float(np.corrcoef(a[m], b[m])[0, 1]), 4)
    OUT["V"]["recompute_medabs"] = round(float(np.median(np.abs(a[m] - b[m]))), 4)
    log(f"    从原始反应性重算的 R 与释放列:相关 {OUT['V']['recompute_corr']:+.4f},"
        f" 中位绝对差 {OUT['V']['recompute_medabs']:.4f}  (n={int(m.sum())})")

    # ---------------------------------------------------------------- A
    log("\n【A】化学对照:补偿双突变保持碱基组成不变,所以组成变化全在分母")
    log(f"  A0 语料的试剂:{dict(et)}")
    OUT["A0_experiment_type"] = {k: int(v) for k, v in et.items()}
    log("     单一试剂 2A3(酰化 2'-OH,不是碱基特异的 DMS),所以『DMS 只打 A 和 C』这一版本的")
    log("     反驳在这个语料上不适用;下面测的是它的一般版本。")

    log(f"\n  {'配平后 G-C 减 A-U':<40}{'估计':>10}{'95% CI':>24}{'p':>9}")
    OUT["A1_denominator"] = show("A1 分母 D=sqrt(rho_i^2+rho_j^2)", ctr(rows, "D"))
    OUT["A2_numerator"] = show("A2 分子 rho_ij(组成不变)", ctr(rows, "rho_ij"))
    OUT["A2b_rho_i"] = show("    (参考) rho_i", ctr(rows, "rho_i"))
    OUT["A2c_rho_j"] = show("    (参考) rho_j", ctr(rows, "rho_j"))

    # A3: add a caliper on log D
    ld = np.array([math.log(r["D"]) for r in recs if np.isfinite(r["D"]) and r["D"] > 0])
    qs = np.quantile(ld, [0.2, 0.4, 0.6, 0.8])
    r3 = []
    for r, f in zip(recs, rows):
        if not (np.isfinite(r["D"]) and r["D"] > 0):
            continue
        g = dict(f); g["dq"] = int(np.searchsorted(qs, math.log(r["D"])))
        r3.append(g)
    e3 = ctr(r3, "rf", covs=("depth3", "crossed", "dq"))
    ret = e3.effect / ev.effect
    OUT["A3_conditioned"] = show("A3 rescue,再配平 log D 的五分位", e3, f"  留存 {100*ret:.1f}%")
    OUT["A3_conditioned"]["retention"] = round(float(ret), 4)
    OUT["A3_conditioned"]["ci_excludes_zero"] = bool(e3.ci[0] * e3.ci[1] > 0)

    # A4: decomposition of the R contrast into numerator and denominator movement
    #     dR = -d(rho_ij)/D_bar + rho_ij_bar * d(D)/D_bar^2, at matched means
    Dm = float(np.nanmean([r["D"] for r in recs]))
    Nm = float(np.nanmean([r["rho_ij"] for r in recs]))
    cn = -OUT["A2_numerator"]["effect"] / Dm
    cd = Nm * OUT["A1_denominator"]["effect"] / Dm ** 2
    OUT["A4_decomposition"] = {"D_mean": round(Dm, 4), "rho_ij_mean": round(Nm, 4),
                               "from_numerator": round(cn, 5), "from_denominator": round(cd, 5),
                               "sum": round(cn + cd, 5), "observed": OUT["V"]["effect"],
                               "denominator_share": round(abs(cd) / (abs(cn) + abs(cd)), 4)}
    log(f"  A4 一阶分解:分子贡献 {cn:+.5f},分母贡献 {cd:+.5f},合计 {cn+cd:+.5f}"
        f"(实测 {ev.effect:+.5f});分母占 {100*abs(cd)/(abs(cn)+abs(cd)):.1f}%")

    # A5: M16's declared-but-unreported denominator diagnostic, recomputed on BOTH sides
    dgm = {}
    for t in ("CG", "AU", "GU"):
        v = [x for r in recs if r["pt"] == t for x in (r["rho_i"], r["rho_j"]) if np.isfinite(x)]
        if v:
            dgm[t] = {"mean": round(float(np.mean(v)), 4), "n": len(v)}
    OUT["A5_m16_diagnostic_superseded"] = {
        "recomputed_both_sides": dgm,
        "m16_published_one_side": {"CG": 0.3261, "AU": 0.2881, "GU": 0.3076},
        "note": "M16 indexed single mutants by mutA only; its diagnostic used one side of each pair."}
    log("  A5 M16 声明要求报出、论文从未报出的分母诊断,这里两侧都算:")
    for t, v in dgm.items():
        log(f"     {t}: 单突变 RMSD 均值 {v['mean']:.4f} (n={v['n']}),M16 单侧值 "
            f"{OUT['A5_m16_diagnostic_superseded']['m16_published_one_side'][t]:.4f}")
    log("     方向:分母更大的一类,R=1-rho_ij/D 被抬高。M16 的声明把这写成『保守』,符号是反的。")
    log("     A2 才是干净的判别器:补偿双突变保持碱基组成,所以组成赝像对 rho_ij 预测零差。")

    # ---------------------------------------------------------------- B
    log("\n【B】结果变量自己的可靠性,从释放的逐核苷酸误差传播")
    sds = []
    for r in recs:
        P = np.array(r["prof"]); E = np.array(r["errs"])
        m = np.isfinite(P).all(0) & np.isfinite(E).all(0)
        if m.sum() < 10:
            sds.append(np.nan); continue
        p, e = P[:, m], np.abs(E[:, m])
        z = rng.standard_normal((NDRAW, 4, int(m.sum())))
        d = p[None, :, :] + z * e[None, :, :]
        w = d[:, 0, :]
        ri = np.sqrt(((w - d[:, 1, :]) ** 2).mean(1))
        rj = np.sqrt(((w - d[:, 2, :]) ** 2).mean(1))
        rij = np.sqrt(((w - d[:, 3, :]) ** 2).mean(1))
        DD = np.sqrt(ri ** 2 + rj ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            Rr = 1 - rij / DD
        sds.append(float(np.nanstd(Rr)))
    sds = np.array(sds, float)
    good = np.isfinite(sds)
    q = np.nanquantile(sds[good], [0.25, 0.5, 0.75])
    between = float(np.nanstd([r["rf"] for r in recs]))
    OUT["B1_reliability"] = {
        "n": int(good.sum()), "draws": NDRAW,
        "sd_R_q25": round(float(q[0]), 4), "sd_R_median": round(float(q[1]), 4),
        "sd_R_q75": round(float(q[2]), 4),
        "between_pair_sd_R": round(between, 4),
        "effect_in_measurement_sd": round(ev.effect / float(q[1]), 3),
        "effect_in_between_sd": round(ev.effect / between, 3),
        "icc_like": round(1 - float(np.nanmean(sds[good] ** 2)) / between ** 2, 4)}
    log(f"  B1 每对 R 的测量 SD:中位 {q[1]:.4f}(四分位 {q[0]:.4f}/{q[2]:.4f}),n={int(good.sum())}")
    log(f"     对间 SD(R) {between:.4f};信号方差占比 1 - E[sd^2]/var = {OUT['B1_reliability']['icc_like']:+.4f}")
    log(f"     头条 +{ev.effect:.5f} = 单对测量 SD 的 {OUT['B1_reliability']['effect_in_measurement_sd']:.2f} 倍,"
        f"对间 SD 的 {OUT['B1_reliability']['effect_in_between_sd']:.2f} 倍")

    # B2 denominator census
    Dv = np.array([r["D"] for r in recs], float)
    dq = np.nanquantile(Dv, [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.95])
    keep = [f for r, f in zip(recs, rows) if np.isfinite(r["D"]) and r["D"] > np.nanquantile(Dv, 0.10)]
    e2 = ctr(keep, "rf")
    OUT["B2_denominator"] = {
        "quantiles": {k: round(float(v), 4) for k, v in
                      zip(["q01", "q05", "q10", "q25", "q50", "q75", "q95"], dq)},
        "frac_below_0p05": round(float(np.nanmean(Dv < 0.05)), 4),
        "dropped_decile": show("B2 rescue,去掉分母最小的十分位", e2,
                               f"  留存 {100*e2.effect/ev.effect:.1f}%"),
        "retention": round(float(e2.effect / ev.effect), 4)}
    log(f"     分母分位:{OUT['B2_denominator']['quantiles']}  低于 0.05 的占 "
        f"{100*OUT['B2_denominator']['frac_below_0p05']:.2f}%")

    # B3 split-half of the contrast
    pz = sorted({r["puzzle"] for r in recs})
    halves = []
    for _ in range(200):
        pm = rng.permutation(len(pz)); h1 = {pz[i] for i in pm[:len(pz) // 2]}
        A = [f for f in rows if f["puzzle"] in h1]; B = [f for f in rows if f["puzzle"] not in h1]
        try:
            ea = core.contrast(core.Corpus.from_records(
                A, design="design", treatment="pt", outcome="rf",
                covariates=("depth3", "crossed"), cluster="puzzle"),
                ("CG", "AU"), strata=("depth3", "crossed"), nperm=1, nboot=1, seed=1).effect
            eb = core.contrast(core.Corpus.from_records(
                B, design="design", treatment="pt", outcome="rf",
                covariates=("depth3", "crossed"), cluster="puzzle"),
                ("CG", "AU"), strata=("depth3", "crossed"), nperm=1, nboot=1, seed=1).effect
        except Exception:
            continue
        halves.append((ea, eb))
    H = np.array(halves)
    r_sh = float(np.corrcoef(H[:, 0], H[:, 1])[0, 1])
    both = float(np.mean((H[:, 0] > 0) & (H[:, 1] > 0)))
    OUT["B3_splithalf"] = {"n_splits": len(H), "corr": round(r_sh, 4),
                           "both_positive": round(both, 4),
                           "mean_half_effect": round(float(H.mean()), 5)}
    log(f"  B3 200 次随机对半(按靶标):两半对比的相关 {r_sh:+.4f},两半同为正的比例 {100*both:.1f}%")

    # ---------------------------------------------------------------- C
    log("\n【C】排行榜后果:把 placement 固定住,方法的名次动不动")
    OUT["C"] = leaderboard(recs, rows, rng)

    # ---------------------------------------------------------------- D
    log("\n【D】内部一致性")
    OUT["D"] = consistency(rng)

    # ---------------------------------------------------------------- scoring
    log("\n=== 预注册记分(按冻结声明原文)===")
    S = {}
    S["S1"] = (OUT["A1_denominator"]["ci"][0] * OUT["A1_denominator"]["ci"][1] > 0,
               f"A1 分母对比 {OUT['A1_denominator']['effect']:+.4f} "
               f"[{OUT['A1_denominator']['ci'][0]:+.4f}, {OUT['A1_denominator']['ci'][1]:+.4f}]")
    S["S2"] = (OUT["A3_conditioned"]["retention"] >= 0.60 and OUT["A3_conditioned"]["ci_excludes_zero"],
               f"A3 留存 {100*OUT['A3_conditioned']['retention']:.1f}%,区间"
               f"{'不含' if OUT['A3_conditioned']['ci_excludes_zero'] else '含'} 0")
    S["S3"] = (OUT["B2_denominator"]["retention"] >= 0.70,
               f"B2 留存 {100*OUT['B2_denominator']['retention']:.1f}%")
    S["S4"] = (OUT["B3_splithalf"]["corr"] >= 0.30, f"B3 对半相关 {OUT['B3_splithalf']['corr']:+.4f}")
    S["S5"] = (OUT["C"]["rescue"]["tau"] < 1.0 and OUT["C"]["rescue"]["max_move"] >= 1,
               f"C1 tau {OUT['C']['rescue']['tau']:+.4f},最大名次移动 {OUT['C']['rescue']['max_move']}")
    S["S6"] = (OUT["D"]["D1_depth"]["tight_pct"] < 100.0,
               f"D1 depth 紧界 {OUT['D']['D1_depth']['tight_pct']:.1f}%")
    S["S7"] = (OUT["D"]["D3"]["both_hold"], OUT["D"]["D3"]["summary"])
    npass = 0
    for k in sorted(S):
        ok, txt = S[k]; npass += ok
        log(f"  {k}  {'成立' if ok else '**失败**'}  {txt}")
    kills = []
    if not ok_v:
        kills.append("K0_verification_failed")
    if OUT["A3_conditioned"]["retention"] < 0.40 or not OUT["A3_conditioned"]["ci_excludes_zero"]:
        kills.append("K1_readout_is_chemistry")
    if OUT["B2_denominator"]["retention"] < 0.40 or OUT["B3_splithalf"]["corr"] < 0.10:
        kills.append("K2_outcome_too_noisy")
    if OUT["C"]["rescue"]["tau"] == 1.0 and OUT["C"]["rescue"]["max_move"] == 0:
        kills.append("K3_no_leaderboard_consequence")
    if OUT["D"]["D1_depth"]["tight_pct"] > 100.0:
        kills.append("K4_depth_channel_vacuous")
    if not OUT["D"]["D3"]["both_hold"]:
        kills.append("K5_null_is_method_dependent")
    OUT["scoring"] = {k: {"pass": bool(v[0]), "note": v[1]} for k, v in S.items()}
    OUT["kill_conditions_fired"] = kills
    OUT["n_pass"] = npass
    log(f"\n  裁定:{kills if kills else '无 kill condition 触发'}   {npass}/7 成立")

    json.dump(OUT, open(os.path.join(HERE, "m61_readout_and_consequence.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "m61_readout_and_consequence.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m61_readout_and_consequence.json")
    return 0


# --------------------------------------------------------------------------- C
def _rank(d):
    o = sorted(d, key=lambda k: -d[k])
    return {k: i + 1 for i, k in enumerate(o)}


def _tau(a, b):
    ks = sorted(a); n = len(ks); c = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[ks[i]] - a[ks[j]]) * (b[ks[i]] - b[ks[j]])
            c += s > 0; dis += s < 0
    return (c - dis) / (c + dis) if (c + dis) else float("nan")


def _within(items, key_s, key_m, key_y, nmin=2):
    """Mean of the stratum-centred outcome, per method. Holds every named stratum fixed."""
    by = collections.defaultdict(list)
    for it in items:
        by[key_s(it)].append(it)
    acc = collections.defaultdict(lambda: [0.0, 0])
    used = 0
    for _s, g in by.items():
        if len(g) < nmin or len({key_m(x) for x in g}) < 2:
            continue
        mu = float(np.mean([key_y(x) for x in g])); used += 1
        for x in g:
            a = acc[key_m(x)]; a[0] += key_y(x) - mu; a[1] += 1
    return {k: v[0] / v[1] for k, v in acc.items() if v[1]}, used


def _compare(items, key_m, key_y, raw_name, adj_specs, rng, nboot=300, cluster=None):
    raw = collections.defaultdict(list)
    for it in items:
        raw[key_m(it)].append(key_y(it))
    RAW = {k: float(np.mean(v)) for k, v in raw.items()}
    Rraw = _rank(RAW)
    out = {"n": len(items), "methods": len(RAW),
           "raw": {k: round(v, 5) for k, v in RAW.items()}, "raw_rank": Rraw}
    prev = Rraw
    for name, ks in adj_specs:
        A, used = _within(items, ks, key_m, key_y)
        Ra = _rank(A)
        mv = {k: Rraw[k] - Ra[k] for k in Ra}
        out[name] = {"value": {k: round(v, 5) for k, v in A.items()}, "rank": Ra,
                     "strata_used": used, "tau_vs_raw": round(_tau(Rraw, Ra), 4),
                     "max_move": int(max(abs(v) for v in mv.values())) if mv else 0,
                     "moved": int(sum(v != 0 for v in mv.values())),
                     "move": mv,
                     "tau_vs_prev": round(_tau(prev, Ra), 4)}
        prev = Ra
    # cluster bootstrap on the last adjustment's tau
    if cluster is not None and adj_specs:
        cl = sorted({cluster(it) for it in items})
        byc = collections.defaultdict(list)
        for it in items:
            byc[cluster(it)].append(it)
        taus = []
        name, ks = adj_specs[-1]
        for _ in range(nboot):
            pick = [cl[i] for i in rng.integers(0, len(cl), len(cl))]
            sub = [x for c in pick for x in byc[c]]
            rr = collections.defaultdict(list)
            for it in sub:
                rr[key_m(it)].append(key_y(it))
            if len(rr) < 3:
                continue
            R0 = _rank({k: float(np.mean(v)) for k, v in rr.items()})
            A, _u = _within(sub, ks, key_m, key_y)
            A = {k: v for k, v in A.items() if k in R0}
            if len(A) < 3:
                continue
            taus.append(_tau({k: R0[k] for k in A}, _rank(A)))
        if taus:
            out[adj_specs[-1][0]]["tau_boot_ci"] = [round(float(np.quantile(taus, 0.025)), 4),
                                                    round(float(np.quantile(taus, 0.975)), 4)]
            out[adj_specs[-1][0]]["tau_boot_frac_below_1"] = round(float(np.mean(np.array(taus) < 1.0)), 4)
    return out


def leaderboard(recs, rows, rng):
    OUT = {}
    # -- C1: the pair-formation readout, nine named design methods
    items = [{"m": r["method"], "y": r["rf"], "p": r["puzzle"],
              "d": min(r["depth"], 3), "c": r["crossed"], "t": r["pt"]} for r in recs]
    C = _compare(items, lambda x: x["m"], lambda x: x["y"], "rescue",
                 [("target_only", lambda x: (x["p"],)),
                  ("target_plus_placement", lambda x: (x["p"], x["d"], x["c"]))],
                 rng, cluster=lambda x: x["p"])
    log("  C1 九个具名设计方法,结果=rescue factor")
    log(f"     {'方法':<26}{'原始均值':>10}{'原名次':>7}{'控靶标':>7}{'再控 placement':>15}")
    for k in sorted(C["raw"], key=lambda z: C["raw_rank"][z]):
        log(f"     {k:<26}{C['raw'][k]:>+10.4f}{C['raw_rank'][k]:>7}"
            f"{C['target_only']['rank'][k]:>7}{C['target_plus_placement']['rank'][k]:>15}")
    tp = C["target_plus_placement"]
    log(f"     Kendall tau(原始, 控 placement)= {tp['tau_vs_raw']:+.4f};"
        f" 动了 {tp['moved']}/{C['methods']} 个方法,最大移动 {tp['max_move']} 名"
        f"{'  自助 CI ' + str(tp.get('tau_boot_ci')) if tp.get('tau_boot_ci') else ''}")
    log(f"     只控靶标时 tau {C['target_only']['tau_vs_raw']:+.4f};从只控靶标再加 placement 的 tau "
        f"{tp['tau_vs_prev']:+.4f}  <- 这一步是 placement 单独的贡献")
    C["tau"] = tp["tau_vs_raw"]; C["max_move"] = tp["max_move"]
    OUT["rescue"] = C

    # -- C2: the released benchmark score, on the design-score corpus
    try:
        F = json.load(open(os.path.join(HERE, "m13_features.json")))
        it2 = []
        for fn, d in F.items():
            for r in d["rows"]:
                g = r["F"].get("gc_paired")
                if g is None or r.get("wet") is None:
                    continue
                it2.append({"m": r["method"], "y": float(r["wet"]), "p": (fn, r["puzzle"]),
                            "g": g})
        gq = np.quantile([x["g"] for x in it2], [0.2, 0.4, 0.6, 0.8])
        for x in it2:
            x["q"] = int(np.searchsorted(gq, x["g"]))
        C2 = _compare(it2, lambda x: x["m"], lambda x: x["y"], "wet",
                      [("target_only", lambda x: (x["p"],)),
                       ("target_plus_gc", lambda x: (x["p"], x["q"]))],
                      rng, cluster=lambda x: x["p"])
        log(f"\n  C2 释放的湿实验基准分,{C2['methods']} 个方法,n={C2['n']}")
        log(f"     {'方法':<10}{'原始均值':>10}{'原名次':>7}{'控靶标':>7}{'再控 GC 配对分位':>17}")
        for k in sorted(C2["raw"], key=lambda z: C2["raw_rank"][z]):
            log(f"     {k:<10}{C2['raw'][k]:>10.3f}{C2['raw_rank'][k]:>7}"
                f"{C2['target_only']['rank'][k]:>7}{C2['target_plus_gc']['rank'][k]:>17}")
        t2 = C2["target_plus_gc"]
        log(f"     Kendall tau(原始, 控 GC)= {t2['tau_vs_raw']:+.4f};动了 {t2['moved']}/{C2['methods']},"
            f"最大移动 {t2['max_move']} 名"
            f"{'  自助 CI ' + str(t2.get('tau_boot_ci')) if t2.get('tau_boot_ci') else ''}")
        OUT["benchmark_score"] = C2
    except Exception as e:  # the arm is optional; the rescue arm is the registered one
        log(f"  C2 跳过:{e}")
        OUT["benchmark_score"] = {"skipped": str(e)}
    return OUT


# --------------------------------------------------------------------------- D
def _cells(recs, cellkey, A="CG", B="AU"):
    by = collections.defaultdict(lambda: {"A": [], "B": []})
    for i, r in enumerate(recs):
        if r["pt"] == A:
            by[cellkey(r)]["A"].append(i)
        elif r["pt"] == B:
            by[cellkey(r)]["B"].append(i)
    return [(k, v) for k, v in by.items() if v["A"] and v["B"]]


def _prop4(cells, y, b, nb, m=None):
    """F1 of the M60 erratum: BOTH contrasts on ONE set of cell weights."""
    nt = nc = wt = wc = 0.0
    pa = np.zeros(nb); pb = np.zeros(nb)
    sa = np.zeros(nb); sb = np.zeros(nb); ca = np.zeros(nb); cb = np.zeros(nb)
    for _k, v in cells:
        a, bx = np.array(v["A"]), np.array(v["B"])
        w = min(len(a), len(bx))
        nt += w * (y[a].mean() - y[bx].mean()); wt += w
        acc = p = 0.0
        for k in range(nb):
            ak, bk = a[b[a] == k], bx[b[bx] == k]
            if len(ak) and len(bk):
                pk = (len(ak) + len(bk)) / (len(a) + len(bx))
                acc += pk * (y[ak].mean() - y[bk].mean()); p += pk
        if p > 0:
            nc += w * (acc / p); wc += w
        for k in range(nb):
            pa[k] += w * (b[a] == k).sum() / len(a)
            pb[k] += w * (b[bx] == k).sum() / len(bx)
        for k in range(nb):
            ak, bk = a[b[a] == k], bx[b[bx] == k]
            sa[k] += y[ak].sum(); ca[k] += len(ak); sb[k] += y[bk].sum(); cb[k] += len(bk)
    pa /= max(pa.sum(), 1e-12); pb /= max(pb.sum(), 1e-12)
    live = (ca + cb) > 0
    h = np.full(nb, np.nan)
    h[live] = (sa[live] + sb[live]) / (ca[live] + cb[live])
    dtot, dcde = nt / wt, nc / wc
    dp = np.abs(pa - pb)
    hh, pp = h[live], dp[live]
    osc = float(hh.max() - hh.min()) if live.sum() > 1 else 0.0
    dtv = 0.5 * float(dp.sum())
    tight = min(float((np.abs(hh - c) * pp).sum())
                for c in list(hh) + [float(np.average(hh, weights=pp)) if pp.sum() else 0.0])
    r = {"delta_tot": dtot, "delta_cde": dcde, "mediated": dtot - dcde,
         "osc_h": osc, "d_TV": dtv, "crude": osc * dtv, "tight": tight,
         "bins_live": int(live.sum()), "h": [round(float(x), 5) for x in h[live]],
         "pA": [round(float(x), 5) for x in pa[live]], "pB": [round(float(x), 5) for x in pb[live]]}
    if m is not None:      # the linear route, for the Prop.-3 residual identity
        mm = np.asarray(m, float)
        gA = gB = wA = 0.0
        for _k, v in cells:
            a, bx = np.array(v["A"]), np.array(v["B"])
            w = min(len(a), len(bx))
            gA += w * mm[a].mean(); gB += w * mm[bx].mean(); wA += w
        g = (gA - gB) / wA
        mc = mm - mm.mean(); yc = y - y.mean()
        beta = float((mc * yc).sum() / (mc * mc).sum()) if (mc * mc).sum() else 0.0
        r.update({"g": g, "beta": beta, "beta_g": beta * g,
                  "residual_gap": (dtot - dcde) - beta * g})
    return r


def consistency(rng):
    OUT = {}
    recs = M21.build_with_dg()
    y = np.array([r["rf"] for r in recs], float)
    dep = np.array([r["depth"] for r in recs], int)
    dg = np.array([r["dG"] for r in recs], float)
    puz = np.array([r["puzzle"] for r in recs])
    log(f"  D 用 M21 的记录 {len(recs)}")

    # ---- D1: the depth channel through Prop. 3 and Prop. 4
    cellk = lambda r: (r["design"], r["crossed"])
    cells = _cells(recs, cellk)
    nb = int(min(dep.max(), 3)) + 1
    b = np.minimum(dep, 3)
    d1 = _prop4(cells, y, b, nb, m=dep.astype(float))
    att = d1["mediated"]
    d1["crude_pct"] = 100 * d1["crude"] / abs(att) if att else float("inf")
    d1["tight_pct"] = 100 * d1["tight"] / abs(att) if att else float("inf")
    d1["linear_share_pct"] = 100 * d1["beta_g"] / att if att else float("nan")
    log("\n  D1 depth 通道:论文印的 71.2% 是什么")
    log(f"     Δ_tot(只配平 design×crossed) {d1['delta_tot']:+.5f}"
        f"   Δ_cde(再配平 depth) {d1['delta_cde']:+.5f}")
    log(f"     衰减 Δ_tot−Δ_cde = {att:+.5f}  (论文 medObs −0.0290)")
    log(f"     β_depth {d1['beta']:+.5f}  depth 差 g {d1['g']:+.4f}  βg {d1['beta_g']:+.5f}"
        f"  (论文 medPred −0.0206)")
    log(f"     Prop.3 的闭式残差 E[r|A]−E[r|B] = 衰减 − βg = {d1['residual_gap']:+.5f}")
    log(f"     线性项占衰减的 {d1['linear_share_pct']:.1f}%  <- 这就是 71.2% 的身份")
    log(f"     Prop.4:osc(h) {d1['osc_h']:.4f}  d_TV {d1['d_TV']:.4f}"
        f"  粗界 {d1['crude_pct']:.1f}%  紧界 {d1['tight_pct']:.1f}%(相对衰减本身)")
    OUT["D1_depth"] = d1

    # ---- D2: cluster bootstrap on the tight bound of the primary channel
    cellk2 = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    NBIN = 13
    def tight_pct(idx):
        sub = [recs[i] for i in idx]
        yy = y[idx]; gg = dg[idx]
        q = np.quantile(gg, np.linspace(0, 1, NBIN + 1)[1:-1])
        bb = np.searchsorted(q, gg)
        cl = _cells(sub, cellk2)
        if not cl:
            return None
        r = _prop4(cl, yy, bb, NBIN)
        return (100 * r["tight"] / abs(r["mediated"]) if r["mediated"] else None,
                100 * r["crude"] / abs(r["mediated"]) if r["mediated"] else None)
    base = tight_pct(np.arange(len(recs)))
    pz = sorted(set(puz)); bycl = {p: np.where(puz == p)[0] for p in pz}
    bs = []
    for _ in range(NBOOT_TIGHT):
        pick = [pz[i] for i in rng.integers(0, len(pz), len(pz))]
        idx = np.concatenate([bycl[p] for p in pick])
        try:
            v = tight_pct(idx)
        except Exception:
            continue
        if v and v[0] is not None and np.isfinite(v[0]):
            bs.append(v[0])
    bs = np.array(bs)
    OUT["D2_tight_ci"] = {"bins": NBIN, "point_tight_pct": round(base[0], 2),
                          "point_crude_pct": round(base[1], 2), "nboot": int(len(bs)),
                          "ci": [round(float(np.quantile(bs, 0.025)), 2),
                                 round(float(np.quantile(bs, 0.975)), 2)],
                          "frac_below_100": round(float(np.mean(bs < 100)), 4)}
    log(f"\n  D2 主通道紧界 {base[0]:.1f}%(粗界 {base[1]:.1f}%),靶标自助 "
        f"{OUT['D2_tight_ci']['ci']},落在 100% 以下的比例 "
        f"{100*OUT['D2_tight_ci']['frac_below_100']:.1f}%")

    # ---- D3: the dG null on two subsets of methods
    log("\n  D3 ΔG_loc 的零结果,按方法子集重跑(论文 eRet 101.9%,安慰剂带 93.2--107.0%)")
    LO, HI = 93.2, 107.0
    d3 = {}
    for nm, ms in (("thermodynamic", THERMO), ("neural", NEURAL), ("all", None)):
        sub = [r for r in recs if ms is None or r["method"] in ms]
        idx = np.array([i for i, r in enumerate(recs) if ms is None or r["method"] in ms])
        if len(idx) < 200:
            d3[nm] = {"skipped": len(idx)}; continue
        gg = dg[idx]
        q = np.quantile(gg, np.linspace(0, 1, 6)[1:-1])
        bb = np.searchsorted(q, gg)
        cl = _cells(sub, cellk2)
        r = _prop4(cl, y[idx], bb, 5)
        ret = 100 * r["delta_cde"] / r["delta_tot"] if r["delta_tot"] else float("nan")
        inb = LO <= ret <= HI
        d3[nm] = {"n": int(len(idx)), "methods": list(ms) if ms else "all",
                  "delta_tot": round(r["delta_tot"], 5), "delta_cde": round(r["delta_cde"], 5),
                  "retention_pct": round(ret, 1), "in_placebo_band": bool(inb),
                  "bins_live": r["bins_live"]}
        log(f"     {nm:<16}n={len(idx):5d}  Δ_tot {r['delta_tot']:+.5f}"
            f"  Δ_cde(固定 ΔG) {r['delta_cde']:+.5f}  留存 {ret:6.1f}%"
            f"  {'在安慰剂带内' if inb else '**出带**'}")
    both = bool(d3.get("thermodynamic", {}).get("in_placebo_band")
                and d3.get("neural", {}).get("in_placebo_band"))
    d3["both_hold"] = both
    d3["summary"] = (f"thermo {d3.get('thermodynamic', {}).get('retention_pct')}%, "
                     f"neural {d3.get('neural', {}).get('retention_pct')}%")
    OUT["D3"] = d3
    return OUT


if __name__ == "__main__":
    sys.exit(main())
