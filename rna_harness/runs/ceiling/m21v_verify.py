#!/usr/bin/env python3
"""M21 verification: audit the code and the flow BEFORE letting the verdict stand.

M21 returned retention 153.3% -- the declaration's shape_1 -- and B1/B2 disagreed by 119
percentage points. Rule: review the whole logic and verify before judging.

V1  dG_local is undefined when the stacking partner is not a canonical pair. ViennaRNA's
    eval_int_loop returns a finite but meaningless number there (probed: +1.30 kcal/mol for an
    A-A "pair"). Count how many records that touches and recompute with the guard.
V2  The retention denominator in M21 compares two different populations: the dG-stratified
    estimate uses 356 pair-weights, the baseline 902. Recompute the baseline on EXACTLY the
    records that survive into the dG-stratified matched set -- same molecules, one fewer
    stratifying dimension.
V3  Same like-for-like correction applied to the placebo, so real and placebo are compared on
    the same footing.
V4  Is the rescue-on-dG relation linear? B2 assumes it. Report bin-wise means.
V5  Is there an identity x dG interaction? That is what a retention above 100% would mean.
V6  Independent recomputation of beta_dG by a route that does not call within_stratum_slope.
"""
from __future__ import annotations
import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np, RNA
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m16_rescue as M16
import m21_thermo_mediator as M21

CANON = {"AU", "GU", "CG"}


def canon(s, i, j):
    return "".join(sorted({s[i], s[j]})) in CANON


def rebuild(guard):
    """rebuild dG with (guard=True) or without (guard=False) the canonical-partner guard."""
    rows = M16.load()
    wt = {}
    for d in rows:
        if d["mutA"] in ("", "NA", "nan") and d["mutB"] in ("", "NA", "nan"):
            wt[(d["puzzle"], d["method"])] = d
    fcs = {}; recs = []; n_touch = 0
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
        out_ok = P.get(a - 1) == b + 1
        in_ok = P.get(a + 1) == b - 1
        bad = False
        if out_ok and not (canon(s, a - 1, b + 1) and canon(s, a, b)):
            bad = True
            if guard:
                out_ok = False
        if in_ok and not (canon(s, a + 1, b - 1) and canon(s, a, b)):
            bad = True
            if guard:
                in_ok = False
        n_touch += bad
        dG = 0.0
        if out_ok:
            dG += M21.stack_dG(fc, a - 1, b + 1, a, b)
        if in_ok:
            dG += M21.stack_dG(fc, a, b, a + 1, b - 1)
        pen = M21.TERM_AU if pt in ("AU", "GU") else 0.0
        dG += pen * ((not out_ok) + (not in_ok))
        recs.append({"puzzle": d["puzzle"], "method": d["method"], "pt": pt, "rf": float(d["rf"]),
                     "depth": min(dep, up), "crossed": bool(crossed), "design": k, "dG": dG})
    return recs, n_touch


def retained(recs, A, B, sfn):
    """the records that live in a stratum containing BOTH types -- the estimator's actual sample."""
    by = collections.defaultdict(lambda: {A: [], B: []})
    for r in recs:
        if r["pt"] in (A, B):
            by[sfn(r)][r["pt"]].append(r)
    out = []
    for st, d in by.items():
        if d[A] and d[B]:
            out += d[A] + d[B]
    return out


def main():
    R = {}
    print("=== V1 非规范配对污染 dG ===")
    raw, n_bad = rebuild(False)
    grd, _ = rebuild(True)
    print(f"  受影响记录(至少一侧堆叠伙伴非规范): {n_bad} / {len(raw)} = {100*n_bad/len(raw):.2f}%")
    dif = [abs(x["dG"] - y["dG"]) for x, y in zip(raw, grd)]
    print(f"  dG 最大改动 {max(dif):.3f} kcal/mol,均值 {np.mean(dif):.4f}")
    R["V1"] = {"n_contaminated": n_bad, "frac": round(n_bad / len(raw), 4),
               "max_dG_change": round(max(dif), 4)}

    recs = grd
    edges = M21.bin_edges(recs, M21.NBINS)
    for r in recs:
        r["bin"] = M21.assign_bin(r["dG"], edges)
    print(f"  守卫后分箱边界 {[round(e,3) for e in edges]}\n")

    print("=== V2 同群体基线(把 M21 的 retention 分母改对)===")
    d_full, w_full, s_full = M21.matched_diff(recs, "CG", "AU", M21.strat_base)
    keep = retained(recs, "CG", "AU", M21.strat_dg)
    d_dg, w_dg, s_dg = M21.matched_diff(recs, "CG", "AU", M21.strat_dg)
    d_same, w_same, s_same = M21.matched_diff(keep, "CG", "AU", M21.strat_base)
    print(f"  全体 · 无能量维      {d_full:+.4f}  权重 {w_full:.0f}  分层 {s_full}")
    print(f"  留存子集 · 无能量维  {d_same:+.4f}  权重 {w_same:.0f}  分层 {s_same}   <== 正确分母")
    print(f"  留存子集 · 有能量维  {d_dg:+.4f}  权重 {w_dg:.0f}  分层 {s_dg}")
    print(f"  **M21 报的 153.3% 用的是错分母。同群体留存 = {100*d_dg/d_same:.1f}%**")
    print(f"  (子集本身相对全体已经 {100*d_same/d_full:.1f}% —— 这是【选择】不是【中介】)\n")
    R["V2"] = {"full": round(d_full, 4), "same_pop_base": round(d_same, 4),
               "dg_stratified": round(d_dg, 4),
               "retention_correct": round(d_dg / d_same, 4),
               "retention_m21_reported": round(d_dg / d_full, 4),
               "selection_component": round(d_same / d_full, 4),
               "weights": [w_full, w_same, w_dg]}

    print("=== V3 安慰剂,同样用同群体基线 ===")
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    rr = []
    for k in range(M21.N_PLACEBO):
        rng = np.random.default_rng(4000 + k)
        sh = []
        for q, rs in byp.items():
            vals = [r["dG"] for r in rs]
            idx = rng.permutation(len(vals))
            for r, j in zip(rs, idx):
                sh.append({**r, "bin": M21.assign_bin(vals[j], edges)})
        kp = retained(sh, "CG", "AU", M21.strat_dg)
        a1, _, _ = M21.matched_diff(sh, "CG", "AU", M21.strat_dg)
        a0, _, _ = M21.matched_diff(kp, "CG", "AU", M21.strat_base)
        rr.append(a1 / a0)
    rr = np.array(rr)
    z = (d_dg / d_same - rr.mean()) / rr.std(ddof=1)
    print(f"  伪中介同群体留存 {100*rr.mean():.1f}% ± {100*rr.std(ddof=1):.1f}%  "
          f"(范围 {100*rr.min():.1f}~{100*rr.max():.1f})")
    print(f"  真中介同群体留存 {100*d_dg/d_same:.1f}%   z = {z:+.2f}\n")
    R["V3"] = {"placebo_mean": round(float(rr.mean()), 4), "placebo_sd": round(float(rr.std(ddof=1)), 4),
               "real": round(d_dg / d_same, 4), "z": round(float(z), 3)}

    print("=== V4 rescue 对 dG 是否线性(B2 的前提)===")
    v4 = []
    for k in range(M21.NBINS):
        sub = [r for r in recs if r["bin"] == k]
        # centre within (design,depth,crossed) so the bin means are comparable
        by = collections.defaultdict(list)
        for r in recs:
            by[M21.strat_base(r)].append(r)
        cen = {}
        for st, rs in by.items():
            m = float(np.mean([x["rf"] for x in rs]))
            for x in rs:
                cen[id(x)] = x["rf"] - m
        v = np.array([cen[id(r)] for r in sub])
        g = np.array([r["dG"] for r in sub])
        v4.append({"bin": k, "n": len(sub), "dG_mean": round(float(g.mean()), 3),
                   "rf_centred": round(float(v.mean()), 4)})
        print(f"  箱{k}: n={len(sub):4d}  dG {g.mean():+.3f}  层内去均值后 rescue {v.mean():+.4f}")
    R["V4"] = v4
    print()

    print("=== V5 身份 x dG 交互(留存>100% 的唯一自洽解释)===")
    v5 = {}
    for k in range(M21.NBINS):
        sub = [r for r in recs if r["bin"] == k]
        dv, wv, sv = M21.matched_diff(sub, "CG", "AU", M21.strat_base)
        v5[k] = {"diff": round(dv, 4) if dv is not None else None, "w": wv}
        print(f"  箱{k}(dG {'最负' if k==0 else '最弱' if k==M21.NBINS-1 else ''}): "
              f"CG−AU {('%+.4f' % dv) if dv is not None else '—':>9s}  权重 {wv:.0f}")
    R["V5"] = v5
    print()

    print("=== V6 β_dG 独立复算(不调用 within_stratum_slope)===")
    b1, n1 = M21.within_stratum_slope(recs, M21.strat_base)
    xs, ys = [], []
    by = collections.defaultdict(list)
    for r in recs:
        by[M21.strat_base(r)].append(r)
    for st, rs in by.items():
        if len(rs) < 2:
            continue
        mx = float(np.mean([r["dG"] for r in rs])); my = float(np.mean([r["rf"] for r in rs]))
        for r in rs:
            xs.append(r["dG"] - mx); ys.append(r["rf"] - my)
    b2 = float(np.polyfit(xs, ys, 1)[0])
    print(f"  within_stratum_slope {b1:+.5f}   np.polyfit 独立复算 {b2:+.5f}   "
          f"差 {abs(b1-b2):.2e}  {'一致' if abs(b1-b2)<1e-9 else '**不一致**'}")
    R["V6"] = {"slope_a": round(b1, 6), "slope_b": round(b2, 6), "agree": abs(b1 - b2) < 1e-9}

    json.dump(R, open(os.path.join(HERE, "m21v_verify.json"), "w"), indent=1)
    print("\n写出 m21v_verify.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
