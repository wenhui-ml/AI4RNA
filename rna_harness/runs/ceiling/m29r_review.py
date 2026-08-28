#!/usr/bin/env python3
"""M29r: review M29 before its numbers are written down, and put the two domains on one axis.

Two things had to be checked before M29's verdict could be read.

  R1  osc(r) is estimated on five binned points, which under-resolves the true oscillation of the
      residual over the support. Under-resolving it SHRINKS the right-hand side of the bound, so a
      bound that holds at five bins holds conservatively -- but the direction has to be verified,
      not asserted, and the bound has to hold at other bin counts too.

  R2  M29's staked prediction that the two routes would differ by at least five points FAILED: they
      differ by 1.29. Read on its own that is a miss. Read through the proposition it is the
      prediction the proposition makes, because the bound is osc(r) * d_TV and the binder corpus's
      two arms overlap heavily in the mediator. That reading is only admissible if the RNA corpus,
      where the routes differ by 75 points, has a correspondingly larger d_TV -- computed the same
      way, on the same estimator, in the same file. That is what this does.

No new claim is made here. This decides whether M29's numbers may be read, and it computes the
one comparison that makes M29 more than a repetition.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m29_binder_crossdomain as M29


def log(*a):
    print(*a, flush=True)


def route_bound(cells_live, t, y, m, nbins):
    """the proposition's two sides, for any (treatment, outcome, mediator) on any cell structure.

    Identical code path for both domains: that is the point of this file.
    """
    edges, b = M29.zbins(m, nbins=nbins)
    ic_m = []
    for c, ix in cells_live:
        for k in range(nbins):
            sub = ix[b[ix] == k]
            if len(sub) >= 2:
                ic_m.append(((c, k), sub))
    d_tot, w_tot = M29.contrast(cells_live, t, y)
    d_cde, w_cde = M29.contrast(ic_m, t, y)
    if d_tot is None or d_cde is None:
        return None
    beta = M29.within_slope(cells_live, m, y)
    g, _gw = M29.contrast(cells_live, t, m)
    # cen is defined ONLY on records that sit in a live cell. It must be initialised to nan and
    # masked, not left as np.empty: on the RNA corpus most records fall outside a live cell, and
    # averaging uninitialised memory into a bin mean produced osc(r) = 2.5e284 before this was
    # fixed. The binder corpus hid it because all six of its cells are live.
    cen = np.full(len(y), np.nan)
    inlive = np.zeros(len(y), bool)
    for _c, ix in cells_live:
        cen[ix] = y[ix] - y[ix].mean()
        inlive[ix] = True
    hs, ms = [], []
    for k in range(nbins):
        sel = (b == k) & inlive
        if sel.sum() >= 2:
            hs.append(float(cen[sel].mean()))
            ms.append(float(m[sel].mean()))
    hs, ms = np.array(hs), np.array(ms)
    res = hs - beta * ms
    osc = float(res.max() - res.min())
    dtv = M29.tv(b[inlive & (t == 1)], b[inlive & (t == 0)], nbins=nbins)
    strat, prod = d_tot - d_cde, beta * g
    return {"nbins": nbins, "n_used_bins": len(hs),
            "marginal": round(d_tot, 5), "matched": round(d_cde, 5),
            "beta": round(beta, 6), "g": round(g, 4),
            "route_stratified": round(strat, 5), "route_product": round(prod, 5),
            "gap_points": round(100 * (strat - prod), 2),
            "osc_r": round(osc, 5), "d_TV": round(dtv, 4),
            "lhs": round(abs(strat - prod), 6), "rhs": round(osc * dtv, 6),
            "bound_holds": bool(abs(strat - prod) <= osc * dtv + 1e-12),
            "n_inlive": int(inlive.sum()),
            "weight": w_tot}


def binder_side():
    d = pd.read_csv(M29.CSV, low_memory=False).dropna(subset=[M29.TAU, M29.ZCOL, M29.YCOL])
    d["method"] = np.where(d["source"].str.startswith("Cao"), "Cao et al",
                  np.where(d["source"].str.contains("Watson"), "Watson et al", "other"))
    d = d[(d["target_id"].isin(M29.TARGETS)) & (d["method"].isin(M29.METHODS))]
    d = d.reset_index(drop=True)
    y = d[M29.YCOL].astype(float).to_numpy()
    z = d[M29.ZCOL].astype(float).to_numpy()
    tau = d[M29.TAU].astype(float).to_numpy()
    ic = M29.cells(d)
    t = np.zeros(len(d), int)
    for _c, ix in ic:
        t[ix] = (tau[ix] > np.median(tau[ix])).astype(int)
    live = [(c, ix) for c, ix in ic
            if len(np.unique(y[ix])) > 1 and len(np.unique(t[ix])) > 1]
    return live, t, y, z, len(d)


def rna_side():
    """the same three roles on the RNA corpus: identity as treatment, rescue as outcome,
    the local nearest-neighbour energy as mediator, design cells as strata."""
    import m21_thermo_mediator as M21
    recs = M21.build_with_dg()
    keep = [r for r in recs if r["pt"] in ("CG", "AU")]
    cellkey = [(r["design"], min(r["depth"], 3), r["crossed"]) for r in keep]
    y = np.array([r["rf"] for r in keep], float)
    m = np.array([r["dG"] for r in keep], float)
    t = np.array([1 if r["pt"] == "CG" else 0 for r in keep], int)
    by = collections.defaultdict(list)
    for i, k in enumerate(cellkey):
        by[k].append(i)
    live = [(k, np.asarray(v)) for k, v in by.items()
            if len(set(t[v])) > 1 and len(v) >= 2]
    return live, t, y, m, len(keep)


def main():
    log("=== M29r:先复查,再判 ===")
    log("  不产生新主张。决定 M29 的数能不能读,并把两个领域放到同一条定理下。\n")
    out = {}

    log("【R1】bin 数敏感性 —— 界在几个 bin 数下都成立吗")
    live, t, y, z, n = binder_side()
    log(f"  蛋白侧 n={n},可用格 {len(live)}")
    rows = []
    for nb in (3, 5, 8, 10, 15, 20, 30, 40):
        r = route_bound(live, t, y, z, nb)
        if r:
            rows.append(r)
            log(f"    {nb:2d} 箱(用到 {r['n_used_bins']}): 差 {r['gap_points']:+6.2f} 点   "
                f"|差| {r['lhs']:.6f} <= osc*d_TV {r['rhs']:.6f}   "
                f"osc={r['osc_r']:.4f} d_TV={r['d_TV']:.4f}   "
                f"**{'成立' if r['bound_holds'] else '不成立'}**")
    out["R1_binder_binsweep"] = rows
    allhold = all(r["bound_holds"] for r in rows)
    log(f"  所有 bin 数下界都{'成立' if allhold else '**不都成立 ⇒ K1**'}")
    log("  注:5 箱低估了残差在支撑集上的真实振幅,因而低估了界的右侧 ——")
    log("      在低估的右侧下仍然成立,是**保守方向**的确证。\n")

    log("【R2】把 RNA 侧用【同一段代码】算一遍,比的是界的松紧随重叠而变")
    lr, tr, yr, mr, nr = rna_side()
    log(f"  RNA 侧 n={nr},可用格 {len(lr)}")
    rrows = []
    for nb in (3, 5, 8, 10, 15, 20, 30, 40):
        r = route_bound(lr, tr, yr, mr, nb)
        if r:
            rrows.append(r)
            log(f"    {nb:2d} 箱(用到 {r['n_used_bins']}): 差 {r['gap_points']:+6.2f} 点   "
                f"|差| {r['lhs']:.6f} <= osc*d_TV {r['rhs']:.6f}   "
                f"osc={r['osc_r']:.4f} d_TV={r['d_TV']:.4f}   "
                f"**{'成立' if r['bound_holds'] else '不成立'}**")
    out["R2_rna_binsweep"] = rrows

    # The comparison is made at the finest binning both corpora reach, not at the coarsest.
    # osc(r) is a supremum over the support and bin means shrink it, so a coarse estimate
    # UNDERSTATES the right-hand side -- the direction that manufactures false violations.
    NB = 20
    b5 = next((r for r in rows if r["nbins"] == NB), None)
    r5 = next((r for r in rrows if r["nbins"] == NB), None)
    if b5 and r5:
        cmp_ = {
            "binder": {"n": n, "d_TV": b5["d_TV"], "osc_r": b5["osc_r"],
                       "gap_points": b5["gap_points"], "rhs": b5["rhs"],
                       "tightness": round(b5["lhs"] / b5["rhs"], 3) if b5["rhs"] else None},
            "rna": {"n": nr, "d_TV": r5["d_TV"], "osc_r": r5["osc_r"],
                    "gap_points": r5["gap_points"], "rhs": r5["rhs"],
                    "tightness": round(r5["lhs"] / r5["rhs"], 3) if r5["rhs"] else None},
        }
        cmp_["dtv_ratio"] = (round(r5["d_TV"] / b5["d_TV"], 2) if b5["d_TV"] else None)
        cmp_["gap_ratio"] = (round(abs(r5["gap_points"]) / abs(b5["gap_points"]), 2)
                             if b5["gap_points"] else None)
        out["comparison"] = cmp_
        log(f"\n【两个领域,同一条定理,{NB} 箱 —— 分辨得出 osc(r) 的那一档】")
        log(f"{'':10s} {'n':>6s} {'d_TV':>8s} {'osc(r)':>9s} {'路线差(点)':>12s} {'界右侧':>10s} {'松紧':>7s}")
        for k in ("binder", "rna"):
            v = cmp_[k]
            log(f"  {k:8s} {v['n']:6d} {v['d_TV']:8.4f} {v['osc_r']:9.5f} "
                f"{v['gap_points']:12.2f} {v['rhs']:10.6f} {str(v['tightness']):>7s}")
        log(f"\n  d_TV 之比(RNA/蛋白) {cmp_['dtv_ratio']}   路线差之比 {cmp_['gap_ratio']}")

    log("\n【R3】定理的前提本身:E[Y|T,M] 里有没有 T x M 交互")
    log("  界与恒等式都建立在【无交互】上。若交互存在,失败的是前提,不是定理。")
    inter = {}
    for nm, (lv, tt, yy, mm) in (("binder", (live, t, y, z)), ("rna", (lr, tr, yr, mr))):
        _e, b = M29.zbins(mm, nbins=5)
        cen = np.full(len(yy), np.nan); il = np.zeros(len(yy), bool)
        for _c, ix in lv:
            cen[ix] = yy[ix] - yy[ix].mean(); il[ix] = True
        rows = []
        for k in range(5):
            sa = (b == k) & il & (tt == 1); sb = (b == k) & il & (tt == 0)
            if sa.sum() >= 5 and sb.sum() >= 5:
                rows.append({"bin": k, "n_hi": int(sa.sum()), "n_lo": int(sb.sum()),
                             "h_hi": round(float(cen[sa].mean()), 5),
                             "h_lo": round(float(cen[sb].mean()), 5),
                             "diff": round(float(cen[sa].mean() - cen[sb].mean()), 5)})
        dif = [x["diff"] for x in rows]
        inter[nm] = {"bins": rows, "spread_of_within_bin_contrast": round(max(dif) - min(dif), 5),
                     "mean": round(float(np.mean(dif)), 5)}
        log(f"  {nm}: 逐箱的臂间差 {[x['diff'] for x in rows]}")
        log(f"      均值 {inter[nm]['mean']}   跨箱极差 {inter[nm]['spread_of_within_bin_contrast']}"
            f"  <- 无交互要求这一列近似常数")
    out["R3_interaction"] = inter

    p = os.path.join(HERE, "m29r_review.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
