#!/usr/bin/env python3
"""M60: a real upper bound on what a mediator can carry, and the sweep that makes it a curve.

Declared in m60_carrying_bound_declaration.json, committed and pushed before this file existed.

R1 withdrew every mediated fraction and put two numbers in their place, calling them bounds.
They are product-route readings -- |effect_M * slope / effect_total| -- and R1's own
Prop. routes says a projection can carry a sign and a magnitude the true mediated part does not,
in either direction.  "Could be wrong either way" does not give "is an upper limit".

The corollary takes the proposition's own proof step and applies it to h rather than to r.
Under assumption (7) the mediated part is exactly E[h(M)|A] - E[h(M)|B], so

    |Delta_tot - Delta_cde|  <=  osc(h) * d_TV(P_{M|A}, P_{M|B}),

which needs no affinity and no mediation estimator.  Everything here is offline and reads the
committed caches; no sequence is folded and no predictor is called.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")

SEED = 20260901          # integer literal, in the released script, as every run here fixes it
BINGRID = (3, 5, 8, 10, 15, 20, 30, 40)     # written into the declaration; not chosen after
SEPGRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
FGRID = (0.0, 0.25, 0.5, 0.75, 1.0)
NMIN = 30                # a bin must hold this many records before its contrast is read
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


# --------------------------------------------------------------------------- shared primitives
def qbins(x, nbins):
    """quantile edges from the POOLED law, fixed before any treatment grouping (M29's rule)."""
    q = np.linspace(0, 1, nbins + 1)[1:-1]
    edges = np.unique([float(np.quantile(x, v)) for v in q])
    return np.searchsorted(edges, x, side="right"), len(edges) + 1


def contrast(cells, t, y):
    num = w = 0.0
    for _c, ix in cells:
        a, b = ix[t[ix] == 1], ix[t[ix] == 0]
        if len(a) == 0 or len(b) == 0:
            continue
        ww = min(len(a), len(b))
        num += ww * (y[a].mean() - y[b].mean())
        w += ww
    return (num / w if w else None), w


def within_slope(cells, x, y):
    sxy = sxx = 0.0
    for _c, ix in cells:
        if len(ix) < 2:
            continue
        xa, ya = x[ix] - x[ix].mean(), y[ix] - y[ix].mean()
        sxy += float((xa * ya).sum())
        sxx += float((xa * xa).sum())
    return sxy / sxx if sxx else None


def tv_from_bins(b_a, b_b, nb):
    pa = np.array([(b_a == k).sum() for k in range(nb)], float)
    pb = np.array([(b_b == k).sum() for k in range(nb)], float)
    pa /= max(pa.sum(), 1)
    pb /= max(pb.sum(), 1)
    return 0.5 * float(np.abs(pa - pb).sum())


def bound_row(cells, t, y, m, nbins):
    """both bounds on one (cells, treatment, outcome, mediator, binning).

    osc_r * d_TV  bounds the gap between the two ROUTES          (the proposition, eq. 9)
    osc_h * d_TV  bounds the MEDIATED PART itself                (the corollary this run adds)
    """
    b, nb = qbins(m, nbins)
    cells_m = []
    for c, ix in cells:
        for k in range(nb):
            sub = ix[b[ix] == k]
            if len(sub) >= 2:
                cells_m.append(((c, k), sub))
    d_tot, w_tot = contrast(cells, t, y)
    d_cde, _ = contrast(cells_m, t, y)
    if d_tot is None or d_cde is None or abs(d_tot) < 1e-12:
        return None
    beta = within_slope(cells, m, y)
    g, _ = contrast(cells, t, m)
    if beta is None or g is None:
        return None

    # h is the cell-centred conditional mean of y in the mediator, on records that sit in a live
    # cell only.  Initialising to nan and masking is not fussiness: M29r found that averaging
    # uninitialised memory into a bin mean returned osc = 2.5e284 on this very corpus.
    cen = np.full(len(y), np.nan)
    inlive = np.zeros(len(y), bool)
    for _c, ix in cells:
        cen[ix] = y[ix] - y[ix].mean()
        inlive[ix] = True
    hs, ms = [], []
    for k in range(nb):
        sel = (b == k) & inlive
        if sel.sum() >= 2:
            hs.append(float(cen[sel].mean()))
            ms.append(float(m[sel].mean()))
    if len(hs) < 3:
        return None
    hs, ms = np.array(hs), np.array(ms)
    res = hs - beta * ms
    osc_r = float(res.max() - res.min())
    osc_h = float(hs.max() - hs.min())
    dtv = tv_from_bins(b[inlive & (t == 1)], b[inlive & (t == 0)], nb)

    strat, prod = d_tot - d_cde, beta * g
    return {
        "nbins": nbins, "n_used_bins": len(hs), "n_inlive": int(inlive.sum()),
        "d_tot": round(d_tot, 6), "d_cde": round(d_cde, 6),
        "beta": round(beta, 8), "g": round(g, 5),
        "route_stratified": round(strat, 6), "route_product": round(prod, 6),
        "gap_points": round(100 * (strat - prod), 3),
        "osc_r": round(osc_r, 6), "osc_h": round(osc_h, 6), "d_TV": round(dtv, 5),
        # the proposition's bound, on the gap between routes
        "route_lhs": round(abs(strat - prod), 7), "route_rhs": round(osc_r * dtv, 7),
        "route_bound_holds": bool(abs(strat - prod) <= osc_r * dtv + 1e-12),
        # the corollary's bound, on the mediated part itself
        "carry_lhs": round(abs(strat), 7), "carry_rhs": round(osc_h * dtv, 7),
        "carry_bound_holds": bool(abs(strat) <= osc_h * dtv + 1e-12),
        "corollary_bound_frac": round(osc_h * dtv / abs(d_tot), 5),
        "product_reading_frac": round(abs(prod / d_tot), 5),
        "weight": w_tot,
    }


# --------------------------------------------------------------------------- corpora
def rna_records():
    import m26_model_scored as M26
    import m21_thermo_mediator as M21
    recs = M21.build_with_dg()
    pm = {}
    for r in M26.build()[0]:
        pm.setdefault((r["design"], r["pt"], round(r["rf"], 6)), r["pmut"])
    for r in recs:
        r["pmut"] = pm.get((r["design"], r["pt"], round(r["rf"], 6)))
    return recs


def rna_arm(recs, mediator):
    """identity as treatment, rescue as outcome, design cells as strata."""
    keep = [r for r in recs if r["pt"] in ("CG", "AU")]
    if mediator == "pmut":
        keep = [r for r in keep if r.get("pmut") is not None]
    y = np.array([r["rf"] for r in keep], float)
    t = np.array([1 if r["pt"] == "CG" else 0 for r in keep], int)
    m = np.array([(r["dG"] if mediator == "dG"
                   else r["pmut"] if mediator == "pmut"
                   else min(r["depth"], 3)) for r in keep], float)
    by = collections.defaultdict(list)
    for i, r in enumerate(keep):
        by[(r["design"], min(r["depth"], 3), r["crossed"])].append(i)
    if mediator == "depth":                       # depth cannot key the strata AND be the mediator
        by = collections.defaultdict(list)
        for i, r in enumerate(keep):
            by[(r["design"], r["crossed"])].append(i)
    cells = [(k, np.asarray(v)) for k, v in by.items()
             if len(set(t[v])) > 1 and len(v) >= 2]
    return cells, t, y, m, len(keep)


def binder_arm():
    import m29_binder_crossdomain as M29
    import pandas as pd
    d = pd.read_csv(M29.CSV, low_memory=False).dropna(subset=[M29.TAU, M29.ZCOL, M29.YCOL])
    d["method"] = np.where(d["source"].str.startswith("Cao"), "Cao et al",
                  np.where(d["source"].str.contains("Watson"), "Watson et al", "other"))
    d = d[(d["target_id"].isin(M29.TARGETS)) & (d["method"].isin(M29.METHODS))].reset_index(drop=True)
    y = d[M29.YCOL].astype(float).to_numpy()
    m = d[M29.ZCOL].astype(float).to_numpy()
    tau = d[M29.TAU].astype(float).to_numpy()
    ic = M29.cells(d)
    t = np.zeros(len(d), int)
    for _c, ix in ic:
        t[ix] = (tau[ix] > np.median(tau[ix])).astype(int)
    cells = [(c, ix) for c, ix in ic
             if len(np.unique(y[ix])) > 1 and len(np.unique(t[ix])) > 1]
    return cells, t, y, m, len(d)


# --------------------------------------------------------------------------- V
def v_arm(recs, out):
    log("【V】先复算已发表的数,再读任何新列")
    ok = True
    pub = json.load(open(os.path.join(HERE, "m21b_direct_contrast.json")))
    cells, t, y, m, n = rna_arm(recs, "dG")
    d_rf, _ = contrast(cells, t, y)
    d_dg, _ = contrast(cells, t, m)
    checks = []
    for nm, mine, theirs, tol in (("rescue 对比", d_rf, pub["rf"]["effect"], 5e-4),
                                  ("ΔG_loc 对比", d_dg, pub["dg"]["effect"], 5e-3)):
        d = abs(mine - theirs)
        good = d <= tol
        ok &= good
        checks.append({"name": nm, "mine": round(mine, 5), "published": theirs,
                       "delta": round(d, 6), "ok": bool(good)})
        log(f"  {nm:<16} 我 {mine:+.5f}   已发表 {theirs:+.5f}   |Δ| {d:.2e}  "
            f"-> {'通过' if good else '**不符**'}")
    m29r = json.load(open(os.path.join(HERE, "m29r_review.json")))["comparison"]
    log(f"  M29r 的 comparison 读入:binder d_TV {m29r['binder']['d_TV']}, "
        f"rna d_TV {m29r['rna']['d_TV']}")
    out["V"] = {"checks": checks, "all_ok": bool(ok), "n_rna": n}
    return ok


# --------------------------------------------------------------------------- A1
def a1_arm(recs, out):
    log("\n【A1】推论界:osc(h)·d_TV,与论文里印着的乘积读数并排")
    published = {"dG": 0.586, "pmut": 0.068}     # dDgBound / dPmBound as printed
    res = {}
    specs = [("rna", "dG"), ("rna", "pmut"), ("rna", "depth"), ("binder", "A_length")]
    for corpus, med in specs:
        if corpus == "rna":
            cells, t, y, m, n = rna_arm(recs, med)
        else:
            cells, t, y, m, n = binder_arm()
        rows = [r for r in (bound_row(cells, t, y, m, nb) for nb in BINGRID) if r]
        if not rows:
            log(f"  {corpus}/{med}: 无可用箱数,剔除")
            continue
        cb = [r["corollary_bound_frac"] for r in rows]
        key = f"{corpus}:{med}"
        res[key] = {
            "n": n, "n_cells": len(cells), "rows": rows,
            "corollary_bound_min": min(cb), "corollary_bound_max": max(cb),
            "corollary_bound_at_5bins": next((r["corollary_bound_frac"]
                                              for r in rows if r["nbins"] == 5), None),
            "product_reading_frac": rows[0]["product_reading_frac"],
            "published_bound": published.get(med),
            "route_bound_holds_all": all(r["route_bound_holds"] for r in rows),
            "carry_bound_holds_all": all(r["carry_bound_holds"] for r in rows),
            "n_bins_tested": len(rows),
        }
        log(f"\n  {key}  (n={n}, 格 {len(cells)})")
        log(f"    {'箱':>4}{'osc(h)':>10}{'d_TV':>9}{'推论界':>10}"
            f"{'|中介部分|':>12}{'界成立':>8}{'乘积读数':>10}")
        for r in rows:
            log(f"    {r['nbins']:>4}{r['osc_h']:>10.5f}{r['d_TV']:>9.4f}"
                f"{100*r['corollary_bound_frac']:>9.1f}%{abs(r['route_stratified']):>12.5f}"
                f"{'  是' if r['carry_bound_holds'] else '  **否**':>8}"
                f"{100*r['product_reading_frac']:>9.1f}%")
        if med in published:
            log(f"    -> 推论界区间 [{100*min(cb):.1f}%, {100*max(cb):.1f}%]   "
                f"论文印的乘积读数 {100*published[med]:.1f}%")
    out["A1"] = res
    return res


# --------------------------------------------------------------------------- A2
def resample_to_separation(cells, t, m, s, nbins, rng):
    """reweight each arm's mediator law toward or away from the pooled law.

    q_arm = p_pooled + s*(p_arm - p_pooled).  s=0 gives both arms the pooled law (d_TV -> 0),
    s=1 returns the corpus untouched, s>1 pushes the arms further apart than they really are.
    Records are drawn with replacement inside (cell, arm) so the cell structure is preserved.
    """
    b, nb = qbins(m, nbins)
    new_cells = []
    for c, ix in cells:
        picked = []
        for arm in (0, 1):
            sub = ix[t[ix] == arm]
            if len(sub) < 2:
                picked.append(sub)
                continue
            cnt = np.array([(b[sub] == k).sum() for k in range(nb)], float)
            p_arm = cnt / cnt.sum()
            pooled = np.array([(b[ix] == k).sum() for k in range(nb)], float)
            pooled /= pooled.sum()
            q = pooled + s * (p_arm - pooled)
            q = np.clip(q, 0.0, None)
            if q.sum() <= 0:
                picked.append(sub)
                continue
            q /= q.sum()
            w = np.where(cnt > 0, q / np.where(cnt > 0, p_arm, 1.0), 0.0)[b[sub]]
            if w.sum() <= 0:
                picked.append(sub)
                continue
            picked.append(rng.choice(sub, size=len(sub), replace=True, p=w / w.sum()))
        new_cells.append((c, np.concatenate(picked)))
    return new_cells


def a2_arm(recs, out):
    log("\n【A2】分离度扫描:把 n=2 的『tracking』变成一条曲线")
    cells, t, y, m, n = rna_arm(recs, "dG")
    rng = np.random.default_rng(SEED)

    log("\n  A2a 真实结果,只改两臂在中介物上的重叠")
    rows = []
    for s in SEPGRID:
        cs = resample_to_separation(cells, t, m, s, 5, rng)
        r = bound_row(cs, t, y, m, 5)
        if r:
            r["separation"] = s
            rows.append(r)
            log(f"    s={s:<5} d_TV {r['d_TV']:.4f}   |条件−乘积| {r['route_lhs']:.6f}   "
                f"osc(r)·d_TV {r['route_rhs']:.6f}   界 {'成立' if r['route_bound_holds'] else '**不成立**'}")
    lhs = np.array([r["route_lhs"] for r in rows])
    rhs = np.array([r["route_rhs"] for r in rows])
    from scipy.stats import spearmanr
    rho = float(spearmanr(lhs, rhs).statistic) if len(rows) > 2 else float("nan")
    log(f"    -> {len(rows)} 档上 |条件−乘积| 与 osc(r)·d_TV 的 Spearman {rho:+.4f}")

    log("\n  A2b 合成结果,注入已知中介分数 f;中介物与分层都是真的")
    # sigma is the corpus's own within-stratum residual SD, not a chosen constant.  This is the
    # answer to "how was sigma picked": it is measured here and printed.
    cen = np.concatenate([y[ix] - y[ix].mean() for _c, ix in cells])
    sigma = float(cen.std(ddof=1))
    d_tot_real, _ = contrast(cells, t, y)
    log(f"    层内残差 SD {sigma:.4f}(实测,非选定);注入的总对比取实测的 {d_tot_real:+.5f}")
    b5, nb5 = qbins(m, 5)
    hshape = np.zeros(len(m))
    for k in range(nb5):
        sel = b5 == k
        if sel.any():
            hshape[sel] = float(np.mean([y[i] - y[ix].mean()
                                         for _c, ix in cells for i in ix if b5[i] == k] or [0.0]))
    gh, _ = contrast(cells, t, hshape)
    sweep = []
    for s in SEPGRID:
        cs = resample_to_separation(cells, t, m, s, 5, rng)
        ghs, _ = contrast(cs, t, hshape)
        if ghs is None or abs(ghs) < 1e-9:
            continue
        for f in FGRID:
            ysyn = ((1.0 - f) * d_tot_real * t
                    + f * d_tot_real * hshape / ghs
                    + rng.normal(0.0, sigma, size=len(y)))
            r = bound_row(cs, t, ysyn, m, 5)
            if not r:
                continue
            sweep.append({"separation": s, "d_TV": r["d_TV"], "f_true": f,
                          "f_strat": round(r["route_stratified"] / r["d_tot"], 5)
                                     if abs(r["d_tot"]) > 1e-9 else None,
                          "f_prod": round(r["route_product"] / r["d_tot"], 5)
                                    if abs(r["d_tot"]) > 1e-9 else None,
                          "carry_bound_holds": r["carry_bound_holds"],
                          "corollary_bound_frac": r["corollary_bound_frac"]})
    spans = {}
    for s in SEPGRID:
        pts = [x for x in sweep if x["separation"] == s and x["f_strat"] is not None]
        if len(pts) >= 2:
            span = max(p["f_strat"] for p in pts) - min(p["f_strat"] for p in pts)
            truespan = max(p["f_true"] for p in pts) - min(p["f_true"] for p in pts)
            spans[s] = {"d_TV": pts[0]["d_TV"], "span_frac": round(span / truespan, 4)}
            log(f"    s={s:<5} d_TV {pts[0]['d_TV']:.4f}   读回跨度 / 真值跨度 "
                f"{100*span/truespan:6.1f}%")
    out["A2"] = {"a2a_rows": rows, "spearman_lhs_rhs": round(rho, 5),
                 "n_rungs": len(rows), "sigma_measured": round(sigma, 5),
                 "a2b_sweep": sweep, "a2b_spans": {str(k): v for k, v in spans.items()},
                 "route_bound_holds_all": all(r["route_bound_holds"] for r in rows),
                 "carry_bound_holds_all": all(x["carry_bound_holds"] for x in sweep)}
    return rho, spans


# --------------------------------------------------------------------------- A3
def a3_arm(recs, out):
    log("\n【A3】假设 (7) 的 no T×M interaction —— 从未单独报告过")
    res = {}
    for corpus, med in (("rna", "dG"), ("binder", "A_length")):
        if corpus == "rna":
            cells, t, y, m, n = rna_arm(recs, med)
        else:
            cells, t, y, m, n = binder_arm()
        b, nb = qbins(m, 5)
        bins = []
        for k in range(nb):
            cs = [(c, ix[b[ix] == k]) for c, ix in cells]
            cs = [(c, ix) for c, ix in cs if len(ix) >= 2 and len(set(t[ix])) > 1]
            nk = sum(len(ix) for _c, ix in cs)
            if nk < NMIN:
                continue
            d, w = contrast(cs, t, y)
            if d is None:
                continue
            bins.append({"bin": k, "n": int(nk), "weight": w, "contrast": round(d, 5)})
        if len(bins) < 2:
            log(f"  {corpus}/{med}: 可用箱 < 2,剔除")
            continue
        cs_ = [b_["contrast"] for b_ in bins]
        mean, spread = float(np.mean(cs_)), float(max(cs_) - min(cs_))
        same = len({np.sign(c) for c in cs_}) == 1
        res[f"{corpus}:{med}"] = {"n": n, "bins": bins, "mean": round(mean, 5),
                                  "spread": round(spread, 5),
                                  "same_sign_all_bins": bool(same),
                                  "spread_over_mean": round(spread / abs(mean), 3)
                                                      if abs(mean) > 1e-9 else None}
        log(f"  {corpus}/{med}: " + "  ".join(f"bin{b_['bin']}(n={b_['n']}) {b_['contrast']:+.5f}"
                                              for b_ in bins))
        log(f"    mean {mean:+.5f}   spread {spread:.5f}   同号 {'是' if same else '**否**'}"
            f"   spread/|mean| {spread/abs(mean):.2f}" if abs(mean) > 1e-9 else "")
    out["A3"] = res
    return res


# --------------------------------------------------------------------------- main
def main():
    log("=== M60:一个真正的上界,以及把它变成曲线的扫描 ===")
    log("  声明已冻结并推送于本文件存在之前。全离线,$0,不折叠任何序列。\n")
    out = {"seed": SEED, "bin_grid": list(BINGRID), "sep_grid": list(SEPGRID),
           "f_grid": list(FGRID), "nmin": NMIN}
    recs = rna_records()
    log(f"RNA 记录 {len(recs)},其中 P_model 连上 {sum(r.get('pmut') is not None for r in recs)}\n")

    if not v_arm(recs, out):
        log("\n**K0 触发:已发表的数复算不上。停,不读新列。**")
        out["verdict"] = "K0_v_arm_failed"
        json.dump(out, open(os.path.join(HERE, "m60_carrying_bound.json"), "w"),
                  ensure_ascii=False, indent=1)
        open(os.path.join(HERE, "m60_carrying_bound.log"), "w").write("\n".join(LOG) + "\n")
        return 1

    a1 = a1_arm(recs, out)
    rho, spans = a2_arm(recs, out)
    a3 = a3_arm(recs, out)

    # ------------------------------------------------------------------ scoring the seven stakes
    log("\n=== 预注册记分 ===")
    S = {}
    dg, pm_ = a1.get("rna:dG"), a1.get("rna:pmut")
    S["S1"] = bool(dg and dg["corollary_bound_at_5bins"] is not None
                   and dg["corollary_bound_at_5bins"] < 0.586)
    S["S2"] = bool(pm_ and pm_["corollary_bound_at_5bins"] is not None
                   and pm_["corollary_bound_at_5bins"] < 0.068)
    n_ok = sum(1 for v in a1.values() for r in v["rows"] if r["carry_bound_holds"])
    n_all = sum(len(v["rows"]) for v in a1.values())
    S["S3"] = bool(n_ok == n_all)
    S["S4"] = bool(rho >= 0.8)
    sp = [(float(k), v["span_frac"]) for k, v in out["A2"]["a2b_spans"].items()]
    sp.sort()
    S["S5"] = bool(len(sp) >= 2 and sp[0][1] <= sp[-1][1])
    rna3 = a3.get("rna:dG")
    S["S6"] = bool(rna3 and rna3["same_sign_all_bins"])
    S["S7"] = bool(rna3 and rna3["spread_over_mean"] is not None
                   and rna3["spread_over_mean"] < 2.0)
    txt = {
        "S1": f"ΔG 推论界 {100*dg['corollary_bound_at_5bins']:.1f}% vs 印着的 58.6%" if dg else "n/a",
        "S2": f"P_model 推论界 {100*pm_['corollary_bound_at_5bins']:.1f}% vs 印着的 6.8%" if pm_ else "n/a",
        "S3": f"界成立 {n_ok}/{n_all}",
        "S4": f"Spearman {rho:+.4f}",
        "S5": f"跨度比 从 {100*sp[0][1]:.1f}% 到 {100*sp[-1][1]:.1f}%" if sp else "n/a",
        "S6": f"逐箱同号 {rna3['same_sign_all_bins']}" if rna3 else "n/a",
        "S7": f"spread/|mean| {rna3['spread_over_mean']}" if rna3 else "n/a",
    }
    for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        log(f"  {k}  {'成立  ' if S[k] else '**失败**'}  {txt[k]}")
    out["stakes"], out["stake_text"] = S, txt

    if not S["S3"]:
        v = "K1_corollary_violated"
    elif not (S["S1"] and S["S2"]):
        v = "K2_product_reading_was_not_a_bound"
    elif not S["S4"]:
        v = "K3_tracking_unresolved"
    elif not S["S6"]:
        v = "K4_assumption_7_fails"
    else:
        v = "corollary_bound_replaces_the_product_reading"
    out["verdict"] = v
    log(f"\n  裁定:{v}   {sum(S.values())}/7 成立")

    json.dump(out, open(os.path.join(HERE, "m60_carrying_bound.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m60_carrying_bound.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m60_carrying_bound.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
