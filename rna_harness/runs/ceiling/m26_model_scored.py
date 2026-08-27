#!/usr/bin/env python3
"""M26: score the thermodynamic model on 4390 interventions, and ask what is left over.

Declaration m26_model_scored_declaration.json, frozen and pushed at 97daa3d4 BEFORE this file
existed. Offline, $0.

The compensatory double mutants are real synthesised molecules that differ from their own wild
type at exactly two positions -- a strict orientation flip of one target pair -- and are
byte-identical everywhere else. So for each of them ViennaRNA's partition function can state the
equilibrium probability that the flipped pair forms, which is the model's direct prediction of
what rescue measures, computed over the whole ensemble. That turns the model from a covariate
into something with a score.
"""
from __future__ import annotations

import argparse, collections, csv, json, math, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
import RNA
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m16_rescue as M16
import m21_thermo_mediator as M21

SRC = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
NBINS, NPERM, NPLACEBO = 5, 2000, 12
DELTAS = (0.00, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60)
DRAWS, PASS, SELF_MAX, P_GATE = 20, 0.8, 0.15, 0.005


def pair_prob(seq, a, b, cache):
    """equilibrium probability that positions a,b (0-based) pair, in this sequence's ensemble."""
    key = seq
    if key not in cache:
        fc = RNA.fold_compound(seq)
        fc.pf()
        bpp = fc.bpp()                       # 1-based, upper triangle
        cache[key] = bpp
    bpp = cache[key]
    i, j = min(a, b) + 1, max(a, b) + 1
    try:
        return float(bpp[i][j])
    except Exception:
        return 0.0


def build():
    """one record per compensatory double mutant whose intervention is exactly two positions."""
    csv.field_size_limit(10 ** 9)
    rows, wt = [], {}
    with open(SRC) as f:
        for d in csv.DictReader(f):
            rows.append(d)
            if d["mutA"] in ("", "NA", "nan") and d["mutB"] in ("", "NA", "nan"):
                wt[(d["puzzle"], d["method"])] = d
    cache = {}
    recs, dropped = [], []
    for d in rows:
        if d["rescue_factor"] in ("", "NA", "nan") or d["SN_filter"] not in ("1", "1.0"):
            continue
        k = (d["puzzle"], d["method"]); p = wt.get(k)
        if p is None:
            continue
        ss = int(float(d["sub_start"])); se = int(float(d["sub_end"]))
        P = M16.parse_pairs(d["target_structure"])
        try:
            ma, mb = int(float(d["mutA"])), int(float(d["mutB"]))
        except ValueError:
            continue
        a, b = ma + ss - 2, mb + ss - 2
        if P.get(a) != b:
            continue
        w, m = p["sequence"], d["sequence"]
        pt = "".join(sorted({w[a], w[b]}))
        if pt not in ("CG", "AU", "GU"):
            continue
        diff = tuple(sorted(i for i in range(ss - 1, se) if w[i] != m[i]))
        rf = float(d["rescue_factor"])
        if diff != tuple(sorted({a, b})):                      # P1: the intervention must be clean
            dropped.append({"pt": pt, "rf": rf, "ndiff": len(diff)})
            continue
        dep = 0
        while P.get(a - dep - 1) == b + dep + 1:
            dep += 1
        up = 0
        while P.get(a + up + 1) == b - up - 1:
            up += 1
        crossed = any(x < a < y < b or a < x < b < y for x, y in P.items() if x < y)
        recs.append({"design": k, "puzzle": d["puzzle"], "pt": pt, "rf": rf,
                     "depth": min(dep, up), "crossed": bool(crossed),
                     "pmut": pair_prob(m, a, b, cache),
                     "pwt": pair_prob(w, a, b, cache),
                     "wt_seq": w, "mut_seq": m, "a": a, "b": b})
    for r in recs:
        r["dp"] = r["pmut"] - r["pwt"]
    return recs, dropped


# ---------------------------------------------------------------- estimator (M21's, verbatim)
def matched(recs, A, B, sfn, key="rf"):
    by = collections.defaultdict(lambda: {A: [], B: []})
    for r in recs:
        if r["pt"] in (A, B):
            by[sfn(r)][r["pt"]].append(r[key])
    num = den = 0.0; used = 0
    for _s, d in by.items():
        if not d[A] or not d[B]:
            continue
        w = min(len(d[A]), len(d[B]))
        num += w * (float(np.mean(d[A])) - float(np.mean(d[B]))); den += w; used += 1
    return (num / den if den else None), den, used


def retained_recs(recs, A, B, sfn):
    by = collections.defaultdict(lambda: {A: [], B: []})
    for r in recs:
        if r["pt"] in (A, B):
            by[sfn(r)][r["pt"]].append(r)
    out = []
    for _s, d in by.items():
        if d[A] and d[B]:
            out += d[A] + d[B]
    return out


def perm(recs, A, B, sfn, rng, key="rf"):
    sub = [r for r in recs if r["pt"] in (A, B)]
    by = collections.defaultdict(list)
    for r in sub:
        by[sfn(r)].append(r)
    out = []
    for _s, rs in by.items():
        lab = [r["pt"] for r in rs]; rng.shuffle(lab)
        for r, L in zip(rs, lab):
            out.append({**r, "pt": L})
    return matched(out, A, B, sfn, key)[0]


def cluster_ci(recs, A, B, sfn, seed, nb=500, key="rf"):
    pz = sorted({r["puzzle"] for r in recs})
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    rng = np.random.default_rng(seed); out = []
    for _ in range(nb):
        ix = rng.integers(0, len(pz), len(pz))
        s = [r for q in ix for r in byp[pz[q]]]
        v, den, _u = matched(s, A, B, sfn, key)
        if v is not None and den:
            out.append(v)
    out = np.sort(np.array(out))
    return [round(float(out[int(.025 * len(out))]), 4), round(float(out[int(.975 * len(out))]), 4)]


def slope_in(recs, sfn, xk, yk="rf"):
    by = collections.defaultdict(list)
    for r in recs:
        by[sfn(r)].append(r)
    sxy = sxx = 0.0; n = 0
    for _s, rs in by.items():
        if len(rs) < 2:
            continue
        x = np.array([r[xk] for r in rs], float); y = np.array([r[yk] for r in rs], float)
        x -= x.mean(); y -= y.mean()
        sxy += float((x * y).sum()); sxx += float((x * x).sum()); n += len(rs)
    return (sxy / sxx if sxx else None), n


def slope_perm_p(recs, sfn, xk, obs, seed, nperm=1000):
    """permute the outcome inside the stratification -- rule 80, design held fixed."""
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for r in recs:
        by[sfn(r)].append(r)
    ge = 0
    for _ in range(nperm):
        sc = []
        for _s, rs in by.items():
            y = np.array([r["rf"] for r in rs], float)
            y = y[rng.permutation(len(y))]
            for r, v in zip(rs, y):
                sc.append({**r, "rf": float(v)})
        s, _n = slope_in(sc, sfn, xk)
        ge += (s is not None and abs(s) >= abs(obs))
    return (ge + 1) / (nperm + 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import random as _r
    print("=== M26:给模型记分 —— 4390 次它没见过的干预 ===")
    print("  声明 97daa3d4。离线 $0。\n")
    recs, dropped = build()
    OUT = {"n": len(recs), "n_dropped": len(dropped)}
    print(f"=== P1 干预纯净性 ===\n  纳入 {len(recs)},排除 {len(dropped)}(设计窗内差异位点 != {{mutA,mutB}})")
    if dropped:
        dv = np.array([x["rf"] for x in dropped]); kv = np.array([r["rf"] for r in recs])
        OUT["dropped_rf_mean"] = round(float(dv.mean()), 4)
        OUT["kept_rf_mean"] = round(float(kv.mean()), 4)
        print(f"  排除行 rescue 均值 {dv.mean():+.4f} vs 保留行 {kv.mean():+.4f}  "
              f"{'差别不大' if abs(dv.mean()-kv.mean()) < 0.05 else '**shape_4:排除是选择性的**'}")
    cnt = collections.Counter(r["pt"] for r in recs)
    print(f"  类型 {dict(cnt)}\n")

    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])

    # ---------------- D4 calibration first: it decides how D1/D2 read
    print("=== D4 模型标定:概率分箱 vs 湿读数(描述性)===")
    pv = np.array([r["pmut"] for r in recs])
    edges = [float(np.quantile(pv, q)) for q in np.linspace(0, 1, NBINS + 1)[1:-1]]
    for r in recs:
        r["bin"] = int(np.searchsorted(edges, r["pmut"], side="right"))
    OUT["bin_edges"] = [round(e, 4) for e in edges]
    by = collections.defaultdict(list)
    for r in recs:
        by[sb(r)].append(r)
    cen = {}
    for _s, rs in by.items():
        m = float(np.mean([x["rf"] for x in rs]))
        for x in rs:
            cen[id(x)] = x["rf"] - m
    cal = []
    for k in range(NBINS):
        s = [r for r in recs if r["bin"] == k]
        cal.append({"bin": k, "n": len(s),
                    "p_mean": round(float(np.mean([r["pmut"] for r in s])), 4),
                    "rf_mean": round(float(np.mean([r["rf"] for r in s])), 4),
                    "rf_centred": round(float(np.mean([cen[id(r)] for r in s])), 4)})
        print(f"  箱{k}: n={len(s):4d}  P_model {cal[-1]['p_mean']:.4f}  "
              f"rescue {cal[-1]['rf_mean']:+.4f}  层内去均值 {cal[-1]['rf_centred']:+.4f}")
    mono = all(cal[i]["rf_centred"] <= cal[i + 1]["rf_centred"] for i in range(NBINS - 1))
    OUT["D4"] = {"bins": cal, "monotone": bool(mono)}
    print(f"  **单调 {'是' if mono else '否'}**\n")

    # ---------------- D1 does the model have skill
    print("=== D1 模型有没有技能 ===")
    d1 = {}
    for xk, nm in (("pmut", "P_model(突变体)"), ("pwt", "P_model(WT)"), ("dp", "ΔP = 突变体−WT")):
        s, n = slope_in(recs, sb, xk)
        p = slope_perm_p(recs, sb, xk, s, 100 + hash(xk) % 997, nperm=500)
        d1[xk] = {"slope": round(s, 5), "n": n, "p_perm": round(p, 5)}
        print(f"  rescue 对 {nm:16s} 层内斜率 {s:+.5f}  置换 p {p:.5f}  n={n}")
    OUT["D1"] = d1
    skill = d1["pmut"]["p_perm"] <= P_GATE and d1["pmut"]["slope"] > 0
    print(f"  **{'模型有技能 ⇒ D2 可解读' if skill else 'K3:模型在此无技能 ⇒ D2 不许读成【超出模型】'}**\n")
    OUT["K3_no_skill"] = bool(not skill)

    # ---------------- D2 the caveat removal
    print("=== D2 主臂:固定模型的【完整系综预测】之后,身份还剩多少 ===")
    sd = lambda r: (r["design"], min(r["depth"], 3), r["crossed"], r["bin"])
    d0, w0, s0 = matched(recs, "CG", "AU", sb)
    d1v, w1, s1 = matched(recs, "CG", "AU", sd)
    if d1v is None or w1 < 100:
        OUT["verdict"] = "void_no_overlap"
        json.dump(OUT, open(os.path.join(HERE, "m26_model_scored.json"), "w"),
                  ensure_ascii=False, indent=1)
        print("  K4 触发,重叠不足。"); return 0
    keep = retained_recs(recs, "CG", "AU", sd)
    dsame, wsame, ssame = matched(keep, "CG", "AU", sb)
    null = np.sort(np.array([perm(recs, "CG", "AU", sd, _r.Random(a.seed + i)) or 0.0
                             for i in range(NPERM)]))
    pv1 = (float((null >= d1v).sum()) + 1) / (NPERM + 1)
    ci = cluster_ci(recs, "CG", "AU", sd, a.seed)
    ret = d1v / dsame
    print(f"  全体 · 无模型维      {d0:+.4f}  权重 {w0:.0f}")
    print(f"  留存子集 · 无模型维  {dsame:+.4f}  权重 {wsame:.0f}   <== 同群体分母")
    print(f"  **CDE(固定 P_model) {d1v:+.4f}**  权重 {w1:.0f}  置换 p {pv1:.5f}  零SD {null.std():.4f}")
    print(f"  **同群体留存 {100*ret:.1f}%**   经模型系综中介 = {100*(1-ret):+.1f}%")
    print(f"  cluster bootstrap {ci}")

    # ---------------- P2 placebo
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    pl = []
    for k in range(NPLACEBO):
        rng = np.random.default_rng(8100 + k)
        sh = []
        for _q, rs in byp.items():
            vals = [r["pmut"] for r in rs]
            ix = rng.permutation(len(vals))
            for r, j in zip(rs, ix):
                sh.append({**r, "bin": int(np.searchsorted(edges, vals[j], side="right"))})
        kp = retained_recs(sh, "CG", "AU", sd)
        v, _w, _s = matched(sh, "CG", "AU", sd)
        b, _w2, _s2 = matched(kp, "CG", "AU", sb)
        if v is not None and b:
            pl.append(v / b)
    pl = np.array(pl); pm, ps = float(pl.mean()), float(pl.std(ddof=1))
    z = (ret - pm) / (ps or 1.0)
    med = 1 - ret; mci = [med - 1.96 * ps, med + 1.96 * ps]
    print(f"  P2 伪中介留存 {100*pm:.1f}% ± {100*ps:.1f}%   真值 z = {z:+.2f}")
    print(f"  ⇒ 经模型系综中介的比例 = {100*med:+.1f}% [{100*mci[0]:+.1f}%, {100*mci[1]:+.1f}%]\n")

    # ---------------- P3 ladder
    thr = float(np.quantile(null, .995))
    lad = {}
    sub = [r for r in recs if r["pt"] in ("CG", "AU")]
    bysd = collections.defaultdict(list)
    for r in sub:
        bysd[sd(r)].append(r)
    for dl in DELTAS:
        hit = tot = 0
        for t in range(DRAWS):
            rr = _r.Random(int(dl * 1e5) + t + a.seed)
            sc = []
            for _s, rs in bysd.items():
                lab = [r["pt"] for r in rs]; rr.shuffle(lab)
                for r, L in zip(rs, lab):
                    sc.append({**r, "pt": L, "rf": r["rf"] + (dl if L == "CG" else 0.0)})
            v, den, _u = matched(sc, "CG", "AU", sd)
            if v is not None:
                hit += (v > thr); tot += 1
        lad[dl] = round(hit / tot, 3) if tot else None
    det = next((x for x in DELTAS if x > 0 and (lad.get(x) or 0) >= PASS), None)
    print("=== P3 梯级(建在零上)===")
    print("  " + "  ".join(f"{x:.3f}:{lad[x]}" for x in DELTAS))
    print(f"  δ=0 自检 {lad[0.0]}(须<=0.15)  检出阈值 {det}\n")
    OUT["D2"] = {"cde": round(d1v, 4), "base_full": round(d0, 4),
                 "base_same_population": round(dsame, 4), "retention": round(ret, 4),
                 "retention_wrong_denominator": round(d1v / d0, 4),
                 "selection_component": round(dsame / d0, 4),
                 "p_perm": round(pv1, 5), "null_sd": round(float(null.std()), 4),
                 "boot_ci": ci, "weight": w1, "n_strata": s1, "base_weight": wsame,
                 "placebo_mean": round(pm, 4), "placebo_sd": round(ps, 4),
                 "z_vs_placebo": round(z, 3), "mediated": round(med, 4),
                 "mediated_ci95": [round(mci[0], 4), round(mci[1], 4)]}
    OUT["P3_ladder"] = {str(k): v for k, v in lad.items()}
    OUT["P3_delta0"] = lad[0.0]; OUT["P3_threshold"] = det

    # ---------------- D3 alternatives
    print("=== D3 其它模型量(同一套做法)===")
    d3 = {}
    for xk in ("pwt", "dp"):
        e2 = [float(np.quantile(np.array([r[xk] for r in recs]), q))
              for q in np.linspace(0, 1, NBINS + 1)[1:-1]]
        for r in recs:
            r["bin"] = int(np.searchsorted(e2, r[xk], side="right"))
        v, w, s = matched(recs, "CG", "AU", sd)
        kp = retained_recs(recs, "CG", "AU", sd)
        b, _w, _s = matched(kp, "CG", "AU", sb)
        d3[xk] = {"cde": round(v, 4), "base_same_population": round(b, 4),
                  "retention": round(v / b, 4), "weight": w}
        print(f"  固定 {xk:5s}: CDE {v:+.4f}  同群体基线 {b:+.4f}  留存 {100*v/b:.1f}%  权重 {w:.0f}")
    for r in recs:
        r["bin"] = int(np.searchsorted(edges, r["pmut"], side="right"))

    # D3c: the LOCAL cost of the flip, which the declaration predicted would be underpowered
    # rather than null. Declared arms get run, not stubbed (ledger G7).
    CAN = {"AU", "GU", "CG"}
    fcs = {}

    def dgloc(seq, P, a, b):
        cn = lambda i, j: "".join(sorted({seq[i], seq[j]})) in CAN
        if seq not in fcs:
            fcs[seq] = RNA.fold_compound(seq)
        fc = fcs[seq]
        o = (P.get(a - 1) == b + 1) and cn(a - 1, b + 1) and cn(a, b)
        i = (P.get(a + 1) == b - 1) and cn(a + 1, b - 1) and cn(a, b)
        g = 0.0
        if o:
            g += fc.eval_int_loop(a, b + 2, a + 1, b + 1) / 100.0
        if i:
            g += fc.eval_int_loop(a + 1, b + 1, a + 2, b) / 100.0
        pt = "".join(sorted({seq[a], seq[b]}))
        g += (M21.TERM_AU if pt in ("AU", "GU") else 0.0) * ((not o) + (not i))
        return g

    csv.field_size_limit(10 ** 9)
    struct = {}
    with open(SRC) as f:
        for d in csv.DictReader(f):
            struct[(d["puzzle"], d["method"], d["sequence"])] = d["target_structure"]
    for r in recs:
        db = struct.get((r["puzzle"], r["design"][1], r["mut_seq"]))
        P = M16.parse_pairs(db)
        r["ddg"] = dgloc(r["mut_seq"], P, r["a"], r["b"]) - dgloc(r["wt_seq"], P, r["a"], r["b"])
    dv = np.array([r["ddg"] for r in recs])
    sl, nn = slope_in(recs, sb, "ddg")
    pp = slope_perm_p(recs, sb, "ddg", sl, 321, nperm=1000)
    # what slope this arm could have detected: inject a known slope on the null
    rng = np.random.default_rng(99)
    dets = {}
    ysd = float(np.std([r["rf"] for r in recs]))
    for cand in (0.01, 0.02, 0.05, 0.10, 0.20):
        hit = 0
        for t in range(20):
            sc = []
            byd = collections.defaultdict(list)
            for r in recs:
                byd[sb(r)].append(r)
            for _s2, rs in byd.items():
                y = np.array([r["rf"] for r in rs], float)
                y = y[rng.permutation(len(y))]
                for r, v in zip(rs, y):
                    sc.append({**r, "rf": float(v) - cand * r["ddg"]})
            s2, _n2 = slope_in(sc, sb, "ddg")
            hit += (s2 is not None and abs(s2) >= abs(sl) if False else
                    s2 is not None and s2 <= -abs(cand) * 0.5)
        dets[cand] = hit / 20
    print("=== D3c 取向翻转的局部代价 δΔG(声明预判:功率不足,不是无效应)===")
    print(f"  δΔG 散布 SD {dv.std(ddof=1):.4f}  |δΔG|>0.3 的比例 {(abs(dv)>0.3).mean():.3f}")
    print(f"  层内斜率 {sl:+.5f}  置换 p {pp:.5f}  n={nn}")
    print(f"  注入已知斜率的检出率 " + "  ".join(f"{k}:{v:.2f}" for k, v in dets.items()))
    det_sl = next((k for k, v in dets.items() if v >= 0.8), None)
    m21b = -0.02286
    verdict5 = ("underpowered" if (det_sl is None or abs(m21b) < det_sl) else "null")
    print(f"  可检出的最小斜率 {det_sl};M21 的层内 β_dG = {m21b:+.5f} "
          f"⇒ **{verdict5}**(声明预判 underpowered)")
    d3["ddg"] = {"slope": round(sl, 5), "p_perm": round(pp, 5), "sd": round(float(dv.std(ddof=1)), 4),
                 "frac_gt_03": round(float((abs(dv) > 0.3).mean()), 4),
                 "detectable_slope": det_sl, "verdict": verdict5}
    OUT["D3"] = d3
    print()

    ok = (lad[0.0] is not None and lad[0.0] <= SELF_MAX)
    contains0 = mci[0] <= 0.0 <= mci[1]
    if not ok:
        v = "void_ladder_selfcheck_failed"
    elif not skill:
        v = "model_has_no_skill_here__D2_not_interpretable"
    elif ret >= 0.50 and pv1 <= P_GATE and contains0:
        v = "identity_beyond_the_model_ensemble"
    elif ret >= 0.50 and pv1 <= P_GATE:
        v = "identity_beyond_the_model_ensemble__partial_mediation"
    elif ret <= 0.20:
        v = "identity_is_the_model_ensemble"
    else:
        v = "partially_beyond__report_as_a_fraction_with_its_interval"
    OUT["verdict"] = v
    OUT["staked"] = {
        "1_model_has_skill": bool(skill),
        "2_calibration_not_monotone": bool(not mono),
        "3_retention_ge_50": bool(ret >= 0.50),
        "4_mediated_ci_contains_zero": bool(contains0),
        "5_flip_dG_arm_underpowered": bool(d3["ddg"]["verdict"] == "underpowered"),
    }
    json.dump(OUT, open(os.path.join(HERE, "m26_model_scored.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"=== 裁定: {v} ===")
    print("押注逐条: " + json.dumps(OUT["staked"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
