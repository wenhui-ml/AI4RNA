#!/usr/bin/env python3
"""M21: is formation a function of the nearest-neighbour model's OWN free energy?

Declaration: m21_thermo_mediator_declaration.json, frozen and pushed at 33c83c4f BEFORE this
file existed (the twenty-second). Blind on every joint distribution of rescue with any energy.

M16 established that pair identity predicts rescue once depth and crossing are matched. Three
external reviews said the same thing about it: nearest-neighbour thermodynamics already predicts
that ordering, so the finding reads as a high-throughput reconfirmation of a textbook fact.

M21 makes the distinction decidable. dG_local is what the NN model itself says this pair
contributes -- its inner stack, its outer stack, and the terminal AU/GU penalty on whichever side
is a helix end -- taken from ViennaRNA's own parameter tables, not hardcoded. Conditioning on it
and re-estimating the identity contrast gives a controlled direct effect: what identity does to
formation through channels the model does not encode.

P3 is the probe the whole arm rests on. Adding any stratifying dimension shrinks a matched
contrast mechanically. A placebo mediator -- the same dG values scrambled within puzzle, cut at
the same fixed edges -- shrinks it by exactly the mechanical amount and no more. The difference
between the two shrinkages is the only quantity here that means anything.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
import RNA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m16_rescue as M16

NPERM, R_NOISE, DRAWS, P_GATE, PASS = 2000, 12, 20, 0.005, 0.8
DELTAS = (0.00, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60)
SELF_MAX = 0.15
NBINS = 5                      # fixed at code-writing time, before any contrast was computed
CANON = {"AU", "GU", "CG"}     # a non-canonical target "pair" does not stack (V1)
NBINS_SENS = (3, 8)            # reported as sensitivity, not pre-registered
N_PLACEBO = 12
TERM_AU = RNA.param().TerminalAU / 100.0     # from the parameter table, not hardcoded


# ---------------------------------------------------------------- dG_local
def stack_dG(fc, i, j, k, l):
    """energy of the stack closed by 0-based pair (i,j) with 0-based inner pair (k,l), kcal/mol."""
    return fc.eval_int_loop(i + 1, j + 1, k + 1, l + 1) / 100.0


def build_with_dg():
    """M16.build() plus dG_local. Cross-checked against M16.build() on every shared field."""
    rows = M16.load()
    wt, singles = {}, collections.defaultdict(dict)
    for d in rows:
        k = (d["puzzle"], d["method"])
        if d["mutA"] in ("", "NA", "nan") and d["mutB"] in ("", "NA", "nan"):
            wt[k] = d
        elif d["mutB"] in ("", "NA", "nan") and d["mutA"] not in ("", "NA", "nan"):
            singles[k][int(float(d["mutA"]))] = d
    fcs = {}
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
        if k not in fcs:
            fcs[k] = RNA.fold_compound(s)
        fc = fcs[k]
        cn = lambda i, j: "".join(sorted({s[i], s[j]})) in CANON
        # V1: ViennaRNA's eval_int_loop returns a finite but meaningless number for a
        # non-canonical "pair" (probed: +1.30 kcal/mol for A-A). dG_local is only defined when
        # both partners of the stack are canonical; otherwise that side is a helix end, which
        # is also what the NN model does with it. Touches 233/4453 = 5.2% of records.
        has_out = (P.get(a - 1) == b + 1) and cn(a - 1, b + 1) and cn(a, b)
        has_in = (P.get(a + 1) == b - 1) and cn(a + 1, b - 1) and cn(a, b)
        dG = 0.0
        if has_out:
            dG += stack_dG(fc, a - 1, b + 1, a, b)
        if has_in:
            dG += stack_dG(fc, a, b, a + 1, b - 1)
        pen = TERM_AU if pt in ("AU", "GU") else 0.0
        dG += pen * ((not has_out) + (not has_in))
        # GU adjacency, for B5
        gu_adj = False
        for x, y in ((a - 1, b + 1), (a + 1, b - 1)):
            if P.get(x) == y and "".join(sorted({s[x], s[y]})) == "GU":
                gu_adj = True
        recs.append({"puzzle": d["puzzle"], "method": d["method"], "pt": pt,
                     "rf": float(d["rf"]), "depth": min(dep, up), "crossed": bool(crossed),
                     "design": k, "dG": dG, "gu_adj": gu_adj,
                     "n_side": int(has_out) + int(has_in)})
    # 对拍: identical to M16's record set on every field M16 produces
    base = M16.build()
    key = lambda r: (r["puzzle"], r["method"], r["pt"], round(r["rf"], 6), r["depth"], r["crossed"])
    assert len(base) == len(recs), f"record count差: M16 {len(base)} vs M21 {len(recs)}"
    assert collections.Counter(map(key, base)) == collections.Counter(map(key, recs)), \
        "M21 的记录集合与 M16 不逐字相同 —— dG 之外的任何差异都是 bug"
    return recs


# ---------------------------------------------------------------- estimator
def bin_edges(recs, nb):
    """cut on the POOLED distribution once, before any contrast (rule 80: permutation-invariant)."""
    v = np.sort(np.array([r["dG"] for r in recs]))
    return [float(np.quantile(v, q)) for q in np.linspace(0, 1, nb + 1)[1:-1]]


def assign_bin(x, edges):
    return int(np.searchsorted(edges, x, side="right"))


def strat_base(r):
    return (r["design"], min(r["depth"], 3), r["crossed"])


def strat_dg(r):
    return (r["design"], min(r["depth"], 3), r["crossed"], r["bin"])


def matched_diff(recs, A, B, sfn):
    by = collections.defaultdict(lambda: {A: [], B: []})
    for r in recs:
        if r["pt"] in (A, B):
            by[sfn(r)][r["pt"]].append(r["rf"])
    num = den = 0.0; used = 0
    for st, d in by.items():
        if not d[A] or not d[B]:
            continue
        w = min(len(d[A]), len(d[B]))
        num += w * (float(np.mean(d[A])) - float(np.mean(d[B]))); den += w
        used += 1
    return (num / den if den else None), den, used


def retained(recs, A, B, sfn):
    """the records the estimator actually uses: those in a stratum holding BOTH types.

    V2: M21's first run divided the dG-stratified estimate by the baseline on the FULL
    population. Adding a stratifying dimension drops records, so that ratio mixes selection
    with mediation. The denominator has to be the same records with one fewer dimension.
    """
    by = collections.defaultdict(lambda: {A: [], B: []})
    for r in recs:
        if r["pt"] in (A, B):
            by[sfn(r)][r["pt"]].append(r)
    out = []
    for st, d in by.items():
        if d[A] and d[B]:
            out += d[A] + d[B]
    return out


def perm_diff(recs, A, B, sfn, rng):
    sub = [r for r in recs if r["pt"] in (A, B)]
    by = collections.defaultdict(list)
    for r in sub:
        by[sfn(r)].append(r)
    out = []
    for st, rs in by.items():
        lab = [r["pt"] for r in rs]
        rng.shuffle(lab)
        for r, L in zip(rs, lab):
            out.append({**r, "pt": L})
    return matched_diff(out, A, B, sfn)


def cluster_ci(recs, A, B, sfn, seed, nb=500):
    pz = sorted({r["puzzle"] for r in recs})
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    rng = np.random.default_rng(seed); out = []
    for _ in range(nb):
        idx = rng.integers(0, len(pz), len(pz))
        samp = [r for q in idx for r in byp[pz[q]]]
        d, den, _u = matched_diff(samp, A, B, sfn)
        if d is not None and den > 0:
            out.append(d)
    out = np.sort(np.array(out))
    return [round(float(out[int(.025 * len(out))]), 4), round(float(out[int(.975 * len(out))]), 4)]


def ladder(recs, A, B, sfn, seed):
    import random as _r
    null = np.sort(np.array([perm_diff(recs, A, B, sfn, _r.Random(seed + i))[0] or 0.0
                             for i in range(400)]))
    thr = float(np.quantile(null, .995))
    out = {}
    sub = [r for r in recs if r["pt"] in (A, B)]
    by = collections.defaultdict(list)
    for r in sub:
        by[sfn(r)].append(r)
    for dlt in DELTAS:
        hit = tot = 0
        for t in range(DRAWS):
            rr = _r.Random(int(dlt * 1e5) + t + seed)
            sc = []
            for st, rs in by.items():
                lab = [r["pt"] for r in rs]; rr.shuffle(lab)
                for r, L in zip(rs, lab):
                    sc.append({**r, "pt": L, "rf": r["rf"] + (dlt if L == A else 0.0)})
            v, den, _u = matched_diff(sc, A, B, sfn)
            if v is not None:
                hit += (v > thr); tot += 1
        out[dlt] = round(hit / tot, 3) if tot else None
    return out, thr


def within_stratum_slope(recs, sfn, xk="dG", yk="rf"):
    """centre x and y inside stratum, then pooled OLS through the origin."""
    by = collections.defaultdict(list)
    for r in recs:
        by[sfn(r)].append(r)
    sxy = sxx = 0.0; n = 0
    for st, rs in by.items():
        if len(rs) < 2:
            continue
        x = np.array([r[xk] for r in rs]); y = np.array([r[yk] for r in rs])
        x = x - x.mean(); y = y - y.mean()
        sxy += float((x * y).sum()); sxx += float((x * x).sum()); n += len(rs)
    return (sxy / sxx if sxx else None), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m21_thermo_mediator.json"))
    a = ap.parse_args()
    import random as _r
    print("=== M21:形成是不是最近邻模型自己那份自由能的函数 ===")
    print(f"  声明 33c83c4f(代码存在之前已推送)。离线 $0。TerminalAU(参数表)= {TERM_AU:.2f} kcal/mol\n")

    recs = build_with_dg()
    print(f"  对拍通过:记录集合与 M16 逐字相同({len(recs)} 个可用配对)\n")
    OUT = {"terminal_au_kcal": TERM_AU, "n_pairs": len(recs), "nbins": NBINS}

    # ---------------- B0: overlap, no outcome touched
    print("=== B0 重叠检查(不碰结果变量)===")
    edges = bin_edges(recs, NBINS)
    for r in recs:
        r["bin"] = assign_bin(r["dG"], edges)
    OUT["bin_edges"] = [round(e, 4) for e in edges]
    print("  dG_local 分箱边界(合并分布分位数): " + "  ".join(f"{e:+.3f}" for e in edges))
    dg_by = {}
    for t in ("CG", "AU", "GU"):
        v = np.array([r["dG"] for r in recs if r["pt"] == t])
        dg_by[t] = {"n": int(v.size), "mean": round(float(v.mean()), 4),
                    "sd": round(float(v.std(ddof=1)), 4),
                    "bin_hist": [int((np.array([r["bin"] for r in recs if r["pt"] == t]) == k).sum())
                                 for k in range(NBINS)]}
        print(f"  {t}: n={v.size:5d}  dG {v.mean():+.4f} ± {v.std(ddof=1):.4f}   "
              f"箱分布 {dg_by[t]['bin_hist']}")
    OUT["dG_by_type"] = dg_by
    d0, w0, s0 = matched_diff(recs, "CG", "AU", strat_base)
    d1, w1, s1 = matched_diff(recs, "CG", "AU", strat_dg)
    OUT["B0"] = {"base_weight": w0, "base_strata": s0, "dg_weight": w1, "dg_strata": s1,
                 "overlap_ok": bool(w1 >= 100)}
    print(f"  加 dG 维前:权重和 {w0:.0f},分层 {s0}")
    print(f"  加 dG 维后:权重和 {w1:.0f},分层 {s1}   "
          f"{'重叠充足(K4 不触发)' if w1 >= 100 else '**重叠不足 ⇒ K4**'}\n")
    if w1 < 100:
        OUT["verdict"] = "void_no_overlap"
        json.dump(OUT, open(a.out, "w"), indent=1)
        print("K4 触发,B1/B2 不报。"); return 0

    # ---------------- K5: rule-80 self-check
    print("=== K5 第 80 条自检(置换不得重建匹配集合)===")
    seen = set()
    for i in range(200):
        rr = _r.Random(9000 + i)
        sub = [r for r in recs if r["pt"] in ("CG", "AU")]
        by = collections.defaultdict(list)
        for r in sub:
            by[strat_dg(r)].append(r)
        out = []
        for st, rs in by.items():
            lab = [r["pt"] for r in rs]; rr.shuffle(lab)
            for r, L in zip(rs, lab):
                out.append({**r, "pt": L})
        _d, w, s = matched_diff(out, "CG", "AU", strat_dg)
        seen.add((round(w, 6), s))
    OUT["K5_constant"] = (len(seen) == 1)
    print(f"  200 次置换后 (权重和, 分层数) 取值集合大小 = {len(seen)}  "
          f"{'恒定 ⇒ 通过' if len(seen) == 1 else '**变动 ⇒ K0**'}\n")
    if len(seen) != 1:
        OUT["verdict"] = "void_rule80"
        json.dump(OUT, open(a.out, "w"), indent=1); return 0

    # ---------------- B1: controlled direct effect
    print("=== B1 主臂:固定住模型自己的能量之后,身份还剩多少(CDE)===")
    keep = retained(recs, "CG", "AU", strat_dg)
    d_same, w_same, s_same = matched_diff(keep, "CG", "AU", strat_base)
    null = np.sort(np.array([perm_diff(recs, "CG", "AU", strat_dg, _r.Random(a.seed + i))[0] or 0.0
                             for i in range(NPERM)]))
    p1 = (float((null >= d1).sum()) + 1) / (NPERM + 1)
    ci1 = cluster_ci(recs, "CG", "AU", strat_dg, a.seed)
    ret = d1 / d_same
    OUT["B1"] = {"cde": round(d1, 4), "base_full": round(d0, 4),
                 "base_same_population": round(d_same, 4), "retention": round(ret, 4),
                 "retention_wrong_denominator": round(d1 / d0, 4),
                 "selection_component": round(d_same / d0, 4),
                 "p_perm": round(p1, 5), "null_sd": round(float(null.std()), 4),
                 "boot_ci": ci1, "weight": w1, "n_strata": s1,
                 "base_weight": w_same, "base_strata": s_same}
    print(f"  全体 · 无能量维        {d0:+.4f}  权重 {w0:.0f}")
    print(f"  留存子集 · 无能量维    {d_same:+.4f}  权重 {w_same:.0f}   <== 分母(同群体)")
    print(f"  **CDE(固定 dG 分箱)  {d1:+.4f}**  权重 {w1:.0f}  置换 p {p1:.5f}  零SD {null.std():.4f}")
    print(f"  **同群体留存 {100*ret:.1f}%**   经能量中介 = {100*(1-ret):+.1f}%")
    print(f"  (若用全体作分母会得到 {100*d1/d0:.1f}% —— 其中 {100*d_same/d0:.1f}% 是【选择】不是中介)")
    print(f"  puzzle cluster bootstrap {ci1}\n")

    # ---------------- P3: placebo mediator  (the probe the arm rests on)
    print("=== P3 安慰剂中介(必跑:把机械衰减与真中介分开;分母同样用同群体)===")
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    pl = []
    for k in range(N_PLACEBO):
        rng = np.random.default_rng(4000 + k)
        sh = []
        for q, rs in byp.items():
            vals = [r["dG"] for r in rs]
            idx = rng.permutation(len(vals))
            for r, j in zip(rs, idx):
                sh.append({**r, "bin": assign_bin(vals[j], edges)})
        kp = retained(sh, "CG", "AU", strat_dg)
        av, wv, sv = matched_diff(sh, "CG", "AU", strat_dg)
        bv, _, _ = matched_diff(kp, "CG", "AU", strat_base)
        pl.append((av / bv, wv, sv))
    pr = np.array([x[0] for x in pl])
    z_pl = (ret - pr.mean()) / pr.std(ddof=1) if pr.std(ddof=1) > 0 else None
    OUT["P3"] = {"placebo_retention_mean": round(float(pr.mean()), 4),
                 "placebo_retention_sd": round(float(pr.std(ddof=1)), 4),
                 "placebo_retention_range": [round(float(pr.min()), 4), round(float(pr.max()), 4)],
                 "real_retention": round(ret, 4),
                 "z_vs_placebo": round(float(z_pl), 3) if z_pl is not None else None,
                 "mediated_fraction": round(1 - ret, 4),
                 "mediated_fraction_ci95": [round(float(1 - ret - 1.96 * pr.std(ddof=1)), 4),
                                            round(float(1 - ret + 1.96 * pr.std(ddof=1)), 4)]}
    print(f"  伪中介留存 {100*pr.mean():.1f}% ± {100*pr.std(ddof=1):.1f}%  "
          f"(范围 {100*pr.min():.1f}%~{100*pr.max():.1f}%)")
    print(f"  **真中介留存 {100*ret:.1f}%   z = {z_pl:+.2f}**")
    print(f"  ⇒ 经最近邻局部自由能中介的比例 = {100*(1-ret):+.1f}% "
          f"[{100*(1-ret-1.96*pr.std(ddof=1)):+.1f}%, {100*(1-ret+1.96*pr.std(ddof=1)):+.1f}%]\n")

    # ---------------- P1/P2
    print("=== P1/P2 探针 ===")
    pos = 0
    for k in range(R_NOISE):
        v = perm_diff(recs, "CG", "AU", strat_dg, _r.Random(7000 + a.seed + k))[0] or 0.0
        pos += ((float((null >= v).sum()) + 1) / (NPERM + 1)) < P_GATE
    lad, thr = ladder(recs, "CG", "AU", strat_dg, a.seed)
    OUT["P1_pos"] = pos
    OUT["P2_ladder"] = {str(k): v for k, v in lad.items()}
    OUT["P2_delta0"] = lad.get(0.00)
    det = next((x for x in DELTAS if x > 0 and (lad.get(x) or 0) >= PASS), None)
    OUT["P2_threshold"] = det
    ok = (pos <= 1) and (lad.get(0.00) is not None and lad.get(0.00) <= SELF_MAX) \
        and ((lad.get(0.60) or 0) >= PASS)
    print(f"  P1 随机标签 {pos}/{R_NOISE}(须<=1)   P2 δ=0 自检 {lad.get(0.00)}(须<=0.15)  "
          f"顶档 0.60 = {lad.get(0.60)}(须>=0.8)")
    print("     " + "  ".join(f"{x:.3f}:{lad[x]}" for x in DELTAS if x > 0))
    print(f"     加能量维后检出阈值 {det}(M16 未加时为 0.02)  {'通过' if ok else '**不通过 ⇒ K0**'}\n")
    OUT["probes_ok"] = bool(ok)

    # ---------------- B2: product of coefficients through the energy
    print("=== B2 系数乘积:经【能量】中介的部分 ===")
    beta, nb_ = within_stratum_slope(recs, strat_base)
    by = collections.defaultdict(lambda: {"CG": [], "AU": []})
    for r in recs:
        if r["pt"] in ("CG", "AU"):
            by[strat_base(r)][r["pt"]].append(r["dG"])
    num = den = 0.0
    for st, d in by.items():
        if d["CG"] and d["AU"]:
            w = min(len(d["CG"]), len(d["AU"]))
            num += w * (float(np.mean(d["CG"])) - float(np.mean(d["AU"]))); den += w
    gap = num / den
    pred = beta * gap
    OUT["B2"] = {"beta_dG": round(beta, 5), "n": nb_, "dG_gap": round(gap, 4),
                 "mediated": round(pred, 4), "total_matched": round(d0, 4),
                 "mediated_fraction": round(pred / d0, 4),
                 "b1_implied_mediated_fraction": round(1 - ret, 4)}
    print(f"  β_dG(层内) {beta:+.5f} / (kcal/mol) 于 {nb_} 个配对")
    print(f"  同层加权 dG 差(CG − AU) {gap:+.4f} kcal/mol")
    print(f"  ⇒ 经能量中介 {pred:+.4f},占总配平效应 {d0:+.4f} 的 {100*pred/d0:.1f}%")
    print(f"  B1 蕴含的中介比例 1−留存 = {100*(1-ret):.1f}%  "
          f"(**两条路线对拍**,差 {abs(100*pred/d0 - 100*(1-ret)):.1f} 个百分点)")
    # the two routes disagree; the linear route assumes monotonicity, so测 it rather than assume
    lin = []
    by3 = collections.defaultdict(list)
    for r in recs:
        by3[strat_base(r)].append(r)
    cen = {}
    for st, rs in by3.items():
        m = float(np.mean([x["rf"] for x in rs]))
        for x in rs:
            cen[id(x)] = x["rf"] - m
    for k in range(NBINS):
        sub = [r for r in recs if r["bin"] == k]
        lin.append({"bin": k, "n": len(sub),
                    "dG_mean": round(float(np.mean([r["dG"] for r in sub])), 3),
                    "rf_centred": round(float(np.mean([cen[id(r)] for r in sub])), 4)})
    OUT["B2"]["binwise"] = lin
    mono = all(lin[i]["rf_centred"] >= lin[i + 1]["rf_centred"] for i in range(NBINS - 1))
    OUT["B2"]["monotone"] = bool(mono)
    print("  层内去均值后逐箱 rescue: " + "  ".join(f"{x['dG_mean']:+.2f}:{x['rf_centred']:+.4f}"
                                                    for x in lin))
    print(f"  **单调性 {'成立' if mono else '不成立 ⇒ 系数乘积路线在此misspecified,以 B1 为准'}**\n")

    # ---------------- B3: depth ladder
    print("=== B3 深度分档(末端 A-U 罚分的作用域是 depth=0)===")
    b3 = {}
    for name, sel in (("depth0", lambda r: r["depth"] == 0),
                      ("depth1", lambda r: r["depth"] == 1),
                      ("depth2plus", lambda r: r["depth"] >= 2)):
        sub = [r for r in recs if sel(r)]
        sfn = lambda r: (r["design"], r["crossed"])
        dv, wv, sv = matched_diff(sub, "CG", "AU", sfn)
        if dv is None:
            b3[name] = None; print(f"  {name:11s} 无可用分层"); continue
        nl = np.sort(np.array([perm_diff(sub, "CG", "AU", sfn, _r.Random(a.seed + i))[0] or 0.0
                               for i in range(NPERM)]))
        pv = (float((nl >= dv).sum()) + 1) / (NPERM + 1)
        b3[name] = {"diff": round(dv, 4), "p_perm": round(pv, 5), "weight": wv, "n_strata": sv,
                    "n": len(sub)}
        print(f"  {name:11s} 配平差 {dv:+.4f}  p {pv:.5f}  权重 {wv:.0f}  分层 {sv}  n={len(sub)}")
    OUT["B3"] = b3
    print()

    # ---------------- B4: ordering vs NN
    print("=== B4 三类是否落在同一条 rescue-对-ΔG 直线上(描述性,无闸门)===")
    b4 = {}
    for A, B in (("CG", "AU"), ("CG", "GU"), ("AU", "GU")):
        dv, wv, sv = matched_diff(recs, A, B, strat_base)
        by2 = collections.defaultdict(lambda: {A: [], B: []})
        for r in recs:
            if r["pt"] in (A, B):
                by2[strat_base(r)][r["pt"]].append(r["dG"])
        n2 = d2 = 0.0
        for st, dd in by2.items():
            if dd[A] and dd[B]:
                w = min(len(dd[A]), len(dd[B]))
                n2 += w * (float(np.mean(dd[A])) - float(np.mean(dd[B]))); d2 += w
        g2 = n2 / d2 if d2 else None
        b4[f"{A}_vs_{B}"] = {"rescue_diff": round(dv, 4) if dv else None,
                             "dG_diff": round(g2, 4) if g2 else None,
                             "implied_slope": round(dv / g2, 5) if (dv and g2) else None,
                             "weight": wv}
        print(f"  {A}−{B}: rescue {dv:+.4f}   ΔdG {g2:+.4f} kcal/mol   "
              f"蕴含斜率 {dv/g2:+.5f}  权重 {wv:.0f}")
    sl = [v["implied_slope"] for v in b4.values() if v["implied_slope"] is not None]
    spread = max(abs(x) for x in sl) / min(abs(x) for x in sl)   # slopes are negative
    inside = min(sl) <= beta <= max(sl)
    b4_collinear = bool(spread <= 1.5 and inside)
    OUT["B4"] = b4
    OUT["B4_collinearity"] = {"implied_slopes": sl, "spread_ratio": round(spread, 3),
                              "beta_inside_range": bool(inside), "collinear": b4_collinear}
    print(f"  参照:B2 的层内斜率 {beta:+.5f}")
    print(f"  三条蕴含斜率之比 {spread:.2f}(共线需 <=1.5),β_dG 落在其区间内 {inside}  "
          f"⇒ **{'共线' if b4_collinear else '不共线'}**\n")

    # ---------------- B5: GU context
    print("=== B5 G-U 上下文分层(描述性,n 小)===")
    b5 = {}
    for name, sel in (("GU_adjacent", True), ("GU_isolated", False)):
        sub = [r for r in recs if r["pt"] != "GU" or r["gu_adj"] == sel]
        dv, wv, sv = matched_diff(sub, "CG", "GU", strat_base)
        ng = sum(1 for r in recs if r["pt"] == "GU" and r["gu_adj"] == sel)
        b5[name] = {"diff": round(dv, 4) if dv is not None else None, "weight": wv, "n_GU": ng}
        print(f"  {name:12s} n(GU)={ng:4d}   CG−GU 配平差 "
              f"{('%+.4f' % dv) if dv is not None else '—'}   权重 {wv:.0f}")
    OUT["B5"] = b5
    print()

    # ---------------- sensitivity: bin count
    print("=== 敏感性(非预注册):分箱数 ===")
    sens = {}
    for nb2 in NBINS_SENS:
        e2 = bin_edges(recs, nb2)
        for r in recs:
            r["bin"] = assign_bin(r["dG"], e2)
        dv, wv, sv = matched_diff(recs, "CG", "AU", strat_dg)
        kp2 = retained(recs, "CG", "AU", strat_dg)
        bv, bw, _ = matched_diff(kp2, "CG", "AU", strat_base)   # same-population denominator
        sens[str(nb2)] = {"cde": round(dv, 4), "base_same_population": round(bv, 4),
                          "retention": round(dv / bv, 4), "weight": wv, "n_strata": sv}
        print(f"  {nb2} 箱: CDE {dv:+.4f}  同群体基线 {bv:+.4f}  留存 {100*dv/bv:.1f}%  权重 {wv:.0f}")
    for r in recs:
        r["bin"] = assign_bin(r["dG"], edges)
    OUT["sensitivity_bins"] = sens

    # ---------------- verdict
    contains_zero = ci1[0] <= 0.0 <= ci1[1]
    med = 1 - ret
    med_lo, med_hi = OUT["P3"]["mediated_fraction_ci95"]
    if not ok:
        v = "void_probe_failed"
    elif contains_zero or p1 > 0.005:
        v = "no_direct_effect"
    elif med_lo <= 0.0 <= med_hi and abs(med) <= 0.20:
        v = "identity_beyond_NN_local_energy__mediated_fraction_indistinguishable_from_zero"
    elif ret >= 0.50:
        v = "identity_beyond_NN_local_energy__partial_mediation"
    elif ret <= 0.20:
        v = "identity_is_NN_local_energy"
    else:
        v = "partially_beyond__report_as_a_fraction_with_its_interval"
    OUT["verdict"] = v
    OUT["staked"] = {
        "1_retention_ge_50_and_p_le_0005": bool(ret >= 0.50 and p1 <= 0.005),
        "2_mediated_fraction_in_20_60": bool(0.20 <= med <= 0.60),
        "3_depth2plus_positive": bool((b3.get("depth2plus") or {}).get("diff", 0) > 0),
        "4_not_collinear": bool(not OUT["B4_collinearity"]["collinear"]),
        "5_overlap_ge_200": bool(w1 >= 200),
    }
    json.dump(OUT, open(a.out, "w"), indent=1)
    print(f"\n=== 裁定: {v} ===")
    print("押注逐条: " + json.dumps(OUT["staked"], ensure_ascii=False))
    print(f"写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
