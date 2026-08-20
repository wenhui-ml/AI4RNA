#!/usr/bin/env python3
"""R2: how far above the competitor floor does the target's nested skeleton sit, and should that be divided by L?

Declared in `r2_gap_declaration.json`, pushed before anything was computed.

WHAT IS BORROWED, AND WHAT IS DELIBERATELY NOT. Deng-Hani-Ma's theorem is a TRUNCATION result: recollision
diagrams are negligible, therefore the tree equation (Boltzmann) is valid. Transplanted to RNA that asserts
pseudoknots do not form -- false, and precisely why nested-only tools fail on our targets. Their smallness is
geometric and asymptotically small; RNA's crossing penalty is energetic and O(1). There is no small parameter
here, so the tree truncation is not asymptotically valid. What transfers is the OTHER side: which diagram
dominates the sum. Design means making one designated diagram win against exponentially many others, which is
a random-energy / large-deviation question, not a truncation question.

    nested structures on L bases ~ 2.6^L, so log-multiplicity ~ L log 2.6            extensive
    competitor energies: mean -cL, fluctuations ~ sigma sqrt(L)
    minimum over N = e^{aL} of them ~ -L(c + sigma sqrt(2a))                          still extensive
    the target's own energy                                                            also extensive
    => the winner is decided by PER-NUCLEOTIDE coefficients; L itself decides nothing.

That computation kills a prediction I made earlier the same day (that L contains an irreversibility
threshold): it does not -- L only sharpens the transition, since fluctuations are O(sqrt L), relatively
O(L^-1/2). It also explains a null this project measured and never explained: three equal-budget search arms
scored 62/61/60 and the axis was closed. Lowering the target's energy by design lowers the COMPETITOR FLOOR
too, because both come from the same sequence and the same energy table. The reachable gap is a difference of
two derivatives, not the derivative of the target's energy.

WHY ONLY TWO PRIMITIVES. This morning's flagship was "pin the target's crossing pairs as hard constraints and
sum the rest nested". It is NOT EXPRESSIBLE: a nested DP has no state for crossings, and ViennaRNA silently
drops the constraint and returns a number -- the returned structure leaves the requested positions unpaired,
and adding a third crossing pair moved the answer by 0.024. My own smoke test used (2,17),(3,16), which are
NESTED, so it never tested the claim. Same failure class as A15. The two primitives below evaluate only on
NESTED structures and cannot fail that way.

A STRUCTURAL FACT FORCES TWO LEVELS. L is constant within a cell (110/110 cells of the fourth sample), so
within a cell gap = L*g is the same column up to a constant and the two are indistinguishable, while L's
demeaned column is identically zero. R1's cell-fixed-effects machinery therefore CANNOT ask the
per-nucleotide question -- that question is between-target, and demeaning exists to remove between-target
variation. Level 1 tests whether the gap carries anything at all and is SILENT on normalisation; level 2 is
the actual test and its between-target confound is handled by looking at the SIGNATURE of the coefficient
across length strata rather than at its size.

COST: $0, offline, minutes. Two ViennaRNA calls per molecule. No solver, no API.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from arbitration_confirm import load_fresh                                        # noqa: E402
from r1_info_ladder import feats, r2_within, demean_by_cell, ORACLES, FINE        # noqa: E402

FLOOR_CELLS = 20
FLOOR_STRATUM = 15


def gap_one(item):
    """(E_skeleton - E_floor, L). Both evaluated on NESTED structures only -- no crossing anywhere."""
    seq, nested = item
    import RNA
    try:
        e_t = RNA.energy_of_struct(seq, nested)
        e_m = RNA.fold(seq)[1]
    except Exception:                                                             # noqa: BLE001
        return None
    return e_t - e_m, len(seq)


def boot_ci(vals, draws, seed, stat=st.median):
    rng = random.Random(seed)
    v = list(vals); n = len(v)
    if n < 5:
        return None, None, None
    o = sorted(stat([v[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return stat(v), o[int(0.025 * draws)], o[min(draws - 1, int(0.975 * draws))]


def ols1(x, y):
    """slope of y on x with an intercept; None if x is constant."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    vx = x.var()
    if vx <= 0:
        return None
    return float(((x - x.mean()) @ (y - y.mean())) / ((x - x.mean()) @ (x - x.mean())))


def run(name, path, workers, draws, nperm, seed):
    from concurrent.futures import ProcessPoolExecutor
    from rna_harness.benchmarks.rnainvbench import _round_only
    from leaderboard_design import targets
    T = targets()
    rec, _sp, _meta = load_fresh(path, "panel_recall_product")
    need = sorted({(r["puzzle"], r["seq"]) for r in rec if r["puzzle"] in T})
    nest = {p: _round_only(T[p]["target"]) for p in {p for p, _s in need}}
    print(f"\n=== {name} ===")
    print(f"  分子 {len(rec)};靶标能对上的 {len(need)};要算的唯一 (靶标,序列) {len(need)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        vals = list(ex.map(gap_one, [(s, nest[p]) for p, s in need], chunksize=64))
    G = {k: v for k, v in zip(need, vals) if v is not None}
    print(f"  算成功 {len(G)}/{len(need)};失败 {len(need)-len(G)}")

    cells = collections.defaultdict(list)
    cols = {}
    for r in rec:
        key = (r["puzzle"], r["seq"])
        f = feats(r)
        if f is None or key not in G:
            continue
        gap, L = G[key]
        r["_gap"], r["_g"], r["_L"] = gap, gap / L, L
        cols[id(r)] = f[0] + f[1] + f[2]
        cells[(r["puzzle"], r["method"])].append(r)
    rows = [(c, g) for c, g in cells.items() if len(g) >= 3]
    NF = 3 + 3 + len(ORACLES) * len(FINE)
    print(f"  格 {len(rows)}(每格 >=3 条);F2 特征数 {NF}")
    if len(rows) < FLOOR_CELLS:
        print(f"  低于下限 {FLOOR_CELLS} → 无数据"); return {"no_data": True}
    lens = {len({r["_L"] for r in g}) for _c, g in rows}
    print(f"  格内 L 唯一吗:{'是' if lens == {1} else '否 —— 声明的结构性事实不成立,停'}")
    assert lens == {1}, "格内 L 不唯一,两层设计的前提不成立"

    # ---- level 1: within cell, does the gap add R^2 beyond the whole panel? --------------------------
    with_gap = {k: v + [0.0] for k, v in cols.items()}
    for _c, g in rows:
        for r in g:
            with_gap[id(r)][-1] = r["_gap"]
    y, X2, _ci = demean_by_cell(rows, cols)
    _y, Xg, ci = demean_by_cell(rows, with_gap)
    r2_base, r2_gap = r2_within(y, X2), r2_within(y, Xg)
    rng = np.random.default_rng(seed)
    nul = []
    for _ in range(nperm):
        yp = y.copy()
        for c in range(ci.max() + 1):
            m = ci == c
            yp[m] = rng.permutation(yp[m])
        nul.append(r2_within(yp, Xg) - r2_within(yp, X2))
    print(f"  --- 第一层(格内;对『每核苷酸还是绝对』保持沉默,因为格内 gap = L·g)---")
    print(f"  F2 样本内 R² {r2_base:.4f} → F2∪{{gap}} {r2_gap:.4f}   ΔR² {r2_gap-r2_base:+.4f}"
          f"   置换零均值 {np.mean(nul):+.4f}  零 97.5% {np.quantile(nul,0.975):+.4f}")
    assert r2_gap >= r2_base - 1e-12, "嵌套单调性被破坏,估计量坏了"
    prng = random.Random(seed)
    bs = []
    for _ in range(draws):
        sel = [rows[prng.randrange(len(rows))] for _ in range(len(rows))]
        yy, Xa, _ = demean_by_cell(sel, cols)
        _yy, Xb, _ = demean_by_cell(sel, with_gap)
        a, b = r2_within(yy, Xa), r2_within(yy, Xb)
        if a is not None and b is not None:
            bs.append(b - a)
    bs.sort()
    lo, hi = bs[int(0.025 * len(bs))], bs[min(len(bs) - 1, int(0.975 * len(bs)))]
    l1_pass = lo > 0
    print(f"  ΔR² 对格聚类 CI [{lo:+.4f}, {hi:+.4f}]{'  排除 0' if l1_pass else '  跨过 0'}"
          f"   自检(嵌套 ⇒ 单调不减):✓")
    out = {"n_cells": len(rows), "level1": {"r2_F2": r2_base, "r2_F2_gap": r2_gap,
                                            "delta": r2_gap - r2_base, "ci": [lo, hi],
                                            "null_mean": float(np.mean(nul)),
                                            "excludes_zero_positive": bool(l1_pass)}}

    # ---- level 2: between cell, the coefficient's signature across length strata --------------------
    cell = [{"puzzle": c[0], "L": g[0]["_L"], "wet": st.mean([r["wet"] for r in g]),
             "gap": st.mean([r["_gap"] for r in g]), "g": st.mean([r["_g"] for r in g])}
            for c, g in rows]
    Ls = sorted(x["L"] for x in cell)
    q1, q2 = Ls[len(Ls) // 3], Ls[2 * len(Ls) // 3]
    strata = [("短 L<%d" % q1, [x for x in cell if x["L"] < q1]),
              ("中 %d..%d" % (q1, q2), [x for x in cell if q1 <= x["L"] < q2]),
              ("长 L>=%d" % q2, [x for x in cell if x["L"] >= q2])]
    print(f"  --- 第二层(格级,按 L 分三档;这一层才问归一化)---")
    print(f"  {'档':14s} {'格':>4s} {'L 中位':>7s} {'coef_g':>22s} {'coef_gap':>22s} {'coef_gap×L':>11s}")
    lvl2 = {}
    for nm, sub in strata:
        if len(sub) < FLOOR_STRATUM:
            print(f"  {nm:14s} {len(sub):4d}  —— 低于档内下限 {FLOOR_STRATUM},无数据"); lvl2[nm] = {"no_data": True}; continue
        Lm = st.median([x["L"] for x in sub])
        res = {}
        for key in ("g", "gap"):
            pt = ols1([x[key] for x in sub], [x["wet"] for x in sub])
            bb = []
            byp = collections.defaultdict(list)
            for x in sub:
                byp[x["puzzle"]].append(x)
            ks = list(byp)
            r2 = random.Random(seed + 5)
            for _ in range(draws):
                sel = [x for _ in ks for x in byp[ks[r2.randrange(len(ks))]]]
                v = ols1([x[key] for x in sel], [x["wet"] for x in sel])
                if v is not None:
                    bb.append(v)
            bb.sort()
            res[key] = {"coef": pt, "ci": [bb[int(0.025*len(bb))], bb[min(len(bb)-1, int(0.975*len(bb)))]]}
        print(f"  {nm:14s} {len(sub):4d} {Lm:7.0f} "
              f"{res['g']['coef']:+8.2f} [{res['g']['ci'][0]:+7.2f},{res['g']['ci'][1]:+7.2f}] "
              f"{res['gap']['coef']:+8.3f} [{res['gap']['ci'][0]:+6.3f},{res['gap']['ci'][1]:+6.3f}] "
              f"{res['gap']['coef']*Lm:+11.2f}")
        lvl2[nm] = {"n": len(sub), "L_median": Lm, **res, "coef_gap_times_L": res["gap"]["coef"] * Lm}
    out["level2"] = lvl2
    return out


def overlap(ivs):
    """do all intervals share a common point?"""
    lo = max(i[0] for i in ivs); hi = min(i[1] for i in ivs)
    return lo <= hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "r2_gap.json"))
    a = ap.parse_args()
    D = json.load(open(os.path.join(HERE, "r2_gap_declaration.json")))
    POPS = (("第四批(主)", "panel_f1_fourth.json"), ("样本2", "panel_f1_fresh.json"),
            ("样本3", "panel_f1_third.json"))
    res = {"declaration": D, "populations": {}}
    for nm, fn in POPS:
        res["populations"][fn] = run(nm, os.path.join(HERE, fn), a.workers, a.draws, a.nperm, a.seed)

    print(f"\n=== 判定(按声明)===")
    P = [res["populations"][fn] for _n, fn in POPS]
    if any(p.get("no_data") for p in P) or not all(p["level1"]["excludes_zero_positive"] for p in P):
        v = "gap_carries_nothing"
    else:
        def sig(p):
            ok = [s for s in p["level2"].values() if not s.get("no_data")]
            if len(ok) < 2:
                return None
            g_ov = overlap([s["g"]["ci"] for s in ok])
            gap_ov = overlap([s["gap"]["ci"] for s in ok])
            return g_ov, gap_ov
        sigs = [sig(p) for p in P]
        if any(s is None for s in sigs):
            v = "cannot_distinguish"
        elif all(g and not gp for g, gp in sigs):
            v = "per_nucleotide_supported"
        elif all(gp and not g for g, gp in sigs):
            v = "absolute_supported"
        else:
            v = "cannot_distinguish"
    for (nm, fn) in POPS:
        p = res["populations"][fn]
        if p.get("no_data"):
            print(f"  {nm}: 无数据"); continue
        l1 = p["level1"]
        print(f"  {nm}: 第一层 ΔR² {l1['delta']:+.4f} [{l1['ci'][0]:+.4f},{l1['ci'][1]:+.4f}]"
              f"{' 排除 0' if l1['excludes_zero_positive'] else ' 跨过 0'}")
    print(f"  【结论】{v} —— {D['outcomes_fixed_in_advance'][v]}")
    for lim in D["declared_limitations"]:
        print(f"  限制:{lim}")
    res["verdict"] = v
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
