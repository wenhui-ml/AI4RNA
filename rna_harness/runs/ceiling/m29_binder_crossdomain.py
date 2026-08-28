#!/usr/bin/env python3
"""M29: test R1's own theorem on a corpus from a different domain.

Declaration `m29_binder_crossdomain_declaration.json`, frozen and pushed at 77e2b212 BEFORE this
file existed. Offline, $0.

This is not a replication. R1's Proposition (routes) makes two falsifiable statements about any
corpus, not about RNA:

    (i)  the stratified and product routes to a mediated part differ by exactly
         E[r(M)|A] - E[r(M)|B], where r is the residual of the conditional mean after its linear
         projection; they agree for every pair of exposure laws iff that conditional mean is affine

    (ii) |(D_tot - D_cde) - beta*g|  <=  osc(r) * d_TV(P_{M|A}, P_{M|B})

Designed protein binders give a corpus where the premise is visible without touching the outcome:
under one target, Cao et al's designs are nearly monodisperse in binder length (sd 1.48 to 4.70)
while Watson et al's are three to nine times more dispersed. The covariate distribution is the
generator's, in a domain with different molecules, different generators and a different assay.

Arm C is the primary arm and it can refute us: K1 says a violated bound goes in the main text.
K4 forbids any protein-biology claim from this file, and nothing here makes one.
"""
from __future__ import annotations

import argparse, collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = "/data/tmp/binder_final_dataset.csv"

# every one of these was written into the declaration before this file existed
TARGETS = ("IL7Ra", "InsulinR", "TrkA")
METHODS = ("Cao et al", "Watson et al")
TAU = "input_rosetta_interface_binder_hydrophobicity_percent"
ZCOL = "A_length"
YCOL = "binder"
NBINS, NPERM = 5, 2000
DELTAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)
SELF_MAX = 0.15


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------- estimators
def cells(df):
    return list(df.groupby(["target_id", "source"]).indices.items())


def contrast(idx_by_cell, t, y):
    """weighted risk difference between treatment arms inside cells, weight min(n_hi, n_lo)."""
    num = w = 0.0
    for _c, ix in idx_by_cell:
        a = ix[t[ix] == 1]
        b = ix[t[ix] == 0]
        if len(a) == 0 or len(b) == 0:
            continue
        ww = min(len(a), len(b))
        num += ww * (y[a].mean() - y[b].mean())
        w += ww
    return (num / w if w else None), w


def perm_p(idx_by_cell, t, y, obs, nperm=NPERM, seed=0, permute="outcome"):
    """rule 80: the OUTCOME is permuted inside the cell. `treatment` is reported for comparison
    only, because permuting it would rebuild the matched configuration."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(nperm):
        if permute == "outcome":
            yp, tp = y.copy(), t
            for _c, ix in idx_by_cell:
                yp[ix] = y[ix[rng.permutation(len(ix))]]
        else:
            yp, tp = y, t.copy()
            for _c, ix in idx_by_cell:
                tp[ix] = t[ix[rng.permutation(len(ix))]]
        e, _w = contrast(idx_by_cell, tp, yp)
        if e is not None:
            null.append(e)
    null = np.asarray(null)
    p = (float((np.abs(null) >= abs(obs)).sum()) + 1) / (len(null) + 1)
    return float(p), float(null.std())


def zbins(z, nbins=NBINS):
    """quantile edges from the POOLED distribution, fixed once, before any treatment grouping.

    Binning inside treatment arms would let a permutation move records between bins and rebuild
    the matched set -- the same defect rule 80 names on the treatment side."""
    q = np.linspace(0, 1, nbins + 1)[1:-1]
    edges = np.unique([float(np.quantile(z, x)) for x in q])
    return edges, np.searchsorted(edges, z, side="right")


def within_slope(idx_by_cell, x, y):
    sxy = sxx = 0.0
    for _c, ix in idx_by_cell:
        if len(ix) < 2:
            continue
        xa, ya = x[ix] - x[ix].mean(), y[ix] - y[ix].mean()
        sxy += float((xa * ya).sum())
        sxx += float((xa * xa).sum())
    return sxy / sxx if sxx else None


def h_bins(idx_by_cell, b, y, nbins=NBINS):
    """E[y|z] by bin, with each cell's mean removed so the bins are not reporting which cells
    happen to be easy. This is the h of the proposition, on the reference law."""
    cen = np.empty(len(y), float)
    for _c, ix in idx_by_cell:
        cen[ix] = y[ix] - y[ix].mean()
    out = []
    for k in range(nbins):
        m = b == k
        out.append({"bin": k, "n": int(m.sum()),
                    "h": float(cen[m].mean()) if m.any() else None,
                    "z": None} )
    return out, cen


def tv(b_a, b_b, nbins=NBINS):
    """total variation between the two arms' laws on the mediator, on the fixed bins."""
    pa = np.array([(b_a == k).sum() for k in range(nbins)], float)
    pb = np.array([(b_b == k).sum() for k in range(nbins)], float)
    pa /= max(pa.sum(), 1); pb /= max(pb.sum(), 1)
    return 0.5 * float(np.abs(pa - pb).sum())


# ----------------------------------------------------------------- arms
def run(df, label, args, out):
    log(f"\n{'='*74}\n=== {label}\n{'='*74}")
    d = df.reset_index(drop=True)
    y = d[YCOL].astype(float).to_numpy()
    z = d[ZCOL].astype(float).to_numpy()
    tau = d[TAU].astype(float).to_numpy()

    ic = cells(d)
    # treatment: split at the WITHIN-CELL median, as declared
    t = np.zeros(len(d), int)
    for _c, ix in ic:
        t[ix] = (tau[ix] > np.median(tau[ix])).astype(int)

    # K0: cells whose outcome is degenerate carry no information
    live, dead = [], 0
    for c, ix in ic:
        if len(np.unique(y[ix])) < 2 or len(np.unique(t[ix])) < 2:
            dead += 1
        else:
            live.append((c, ix))
    log(f"  记录 {len(d)}   格 {len(ic)}   结果或处理退化的格 {dead}   可用格 {len(live)}")
    r = {"label": label, "n": len(d), "n_cells": len(ic), "n_dead_cells": dead,
         "n_live_cells": len(live)}
    if len(live) < 4:
        r["void"] = "K0_too_few_live_cells"
        log("  ** K0 触发:可用格 < 4 ⇒ 作废 **")
        out.append(r)
        return

    # ---- arm A: the premise, outcome never touched
    eta = []
    for tg in sorted(set(d["target_id"])):
        m = d["target_id"] == tg
        if m.sum() < 20:
            continue
        grp = d.loc[m, "source"].to_numpy()
        zz = z[m.to_numpy()]
        gm = zz.mean()
        ssb = sum(((zz[grp == s].mean() - gm) ** 2) * (grp == s).sum() for s in set(grp))
        sst = float(((zz - gm) ** 2).sum())
        eta.append({"target": tg, "eta2": round(ssb / sst, 4) if sst else None,
                    "n": int(m.sum())})
    r["A_premise"] = {"eta2_by_target": eta,
                      "eta2_mean": round(float(np.mean([e["eta2"] for e in eta if e["eta2"]])), 4)}
    log(f"\n  【A 臂 前提】source 解释 z 方差的比例(逐靶标,不碰结果列):")
    for e in eta:
        log(f"      {e['target']:12s} eta^2 = {e['eta2']:.4f}   n={e['n']}")

    # ---- arm B: marginal vs z-matched
    edges, b = zbins(z)
    ic_z = []
    for c, ix in live:
        for k in range(NBINS):
            sub = ix[b[ix] == k]
            if len(sub) >= 2:
                ic_z.append(((c, k), sub))
    d_tot, w_tot = contrast(live, t, y)
    d_cde, w_cde = contrast(ic_z, t, y)
    p_tot, sd_tot = perm_p(live, t, y, d_tot, nperm=args.nperm)
    p_t_side, _ = perm_p(live, t, y, d_tot, nperm=args.nperm, permute="treatment")
    r["B"] = {"marginal": round(d_tot, 5), "marginal_w": w_tot, "marginal_p": round(p_tot, 5),
              "marginal_p_treatment_side": round(p_t_side, 5), "null_sd": round(sd_tot, 5),
              "matched": None if d_cde is None else round(d_cde, 5), "matched_w": w_cde,
              "z_edges": [round(e, 2) for e in edges]}
    log(f"\n  【B 臂 衰减】边际 {d_tot:+.5f}(权重 {w_tot:.0f},置换 p {p_tot:.4f})")
    log(f"                 z 配平 {d_cde:+.5f}(权重 {w_cde:.0f})")
    log(f"      规则 80 参照:打乱【处理】列得 p={p_t_side:.4f} 对 打乱【结果】列 p={p_tot:.4f}")

    # ---- arm C: the two routes, and the theorem's bound
    beta = within_slope(live, z, y)
    g, gw = contrast(live, t, z)          # E[z|hi] - E[z|lo], same weighting
    strat = d_tot - d_cde
    prod = beta * g
    hb, cen = h_bins(live, b, y)
    for k in range(NBINS):
        m = b == k
        if m.any():
            hb[k]["z"] = round(float(z[m].mean()), 3)
    seen = [x for x in hb if x["n"] and x["h"] is not None]
    zc = np.array([x["z"] for x in seen]); hc = np.array([x["h"] for x in seen])
    res = hc - beta * zc
    osc = float(res.max() - res.min())
    dtv = tv(b[t == 1], b[t == 0])
    lhs = abs(strat - prod)
    rhs = osc * dtv
    r["C"] = {"beta_z": round(beta, 6), "g": round(g, 4),
              "route_stratified": round(strat, 5), "route_product": round(prod, 5),
              "gap": round(strat - prod, 5), "gap_points": round(100 * (strat - prod), 2),
              "osc_r": round(osc, 5), "d_TV": round(dtv, 4),
              "bound_lhs": round(lhs, 6), "bound_rhs": round(rhs, 6),
              "bound_holds": bool(lhs <= rhs + 1e-12)}
    log(f"\n  【C 臂 定理检验 —— 主臂】")
    log(f"      beta_z {beta:+.6f}   g {g:+.4f}")
    log(f"      条件路线 (边际−配平) {strat:+.5f}      乘积路线 beta*g {prod:+.5f}")
    log(f"      两条路线之差 {100*(strat-prod):+.2f} 个百分点")
    log(f"      定理的界:|差| = {lhs:.6f}   <=   osc(r)*d_TV = {osc:.5f}*{dtv:.4f} = {rhs:.6f}")
    log(f"      **界{'成立' if r['C']['bound_holds'] else '不成立 ⇒ K1 触发,定理被证否'}**")

    # ---- arm D: affinity
    dev = float(np.abs(res - res.mean()).mean())
    mono = all(seen[i]["h"] <= seen[i + 1]["h"] for i in range(len(seen) - 1))
    r["D"] = {"bins": hb, "residual_after_linear": [round(float(x), 5) for x in res],
              "osc_r": round(osc, 5), "mean_abs_dev": round(dev, 5), "monotone": mono,
              "affine_to_within": round(osc, 5)}
    hstr = ", ".join(f"{x['h']:+.4f}" for x in seen)
    log(f"\n  【D 臂 仿射性】逐箱 E[y|z](格内去均值): {hstr}")
    log(f"      线性投影后的残差 {', '.join(f'{v:+.4f}' for v in res)}   osc(r) = {osc:.5f}")
    log(f"      {'单调' if mono else '非单调'}")

    # ---- arm E: the constructive calibration, transplanted
    sweep = []
    rng = np.random.default_rng(0)
    ysd = float(y.std())
    for fr in (0.0, 0.25, 0.5, 0.75, 1.0):
        gam = fr * d_tot / g if abs(g) > 1e-9 else 0.0
        dlt = (1 - fr) * d_tot
        ys = gam * z + dlt * t + rng.normal(0, ysd, len(y))
        a, _w = contrast(live, t, ys)
        c_, _w2 = contrast(ic_z, t, ys)
        sweep.append({"f": fr, "true_retention": round(1 - fr, 3),
                      "measured_retention": round(c_ / a, 4) if a and abs(a) > 1e-9 else None})
    ok = [s for s in sweep if s["measured_retention"] is not None]
    if len(ok) >= 3:
        tr = np.array([s["true_retention"] for s in ok])
        me = np.array([s["measured_retention"] for s in ok])
        corr = float(np.corrcoef(tr, me)[0, 1]) if tr.std() and me.std() else None
        mae = float(np.abs(tr - me).mean())
    else:
        corr = mae = None
    r["E"] = {"sweep": sweep, "corr": None if corr is None else round(corr, 4),
              "mae": None if mae is None else round(mae, 4)}
    log(f"\n  【E 臂 构造性标定】注入已知 f,读回留存量")
    for s in sweep:
        log(f"      f={s['f']:.2f}  真留存 {s['true_retention']:.2f}  实测 {s['measured_retention']}")
    log(f"      与真值相关 {corr}   MAE {mae}")

    # ---- ladder
    lad = {}
    for dl in DELTAS:
        hit = 0
        for k in range(args.ndraw):
            tp = t.copy()
            rr = np.random.default_rng(1000 + k)
            for _c, ix in live:
                tp[ix] = t[ix[rr.permutation(len(ix))]]
            yy = y + dl * tp
            e, _w = contrast(live, tp, yy)
            if e is None:
                continue
            pp, _s = perm_p(live, tp, yy, e, nperm=200, seed=7919 * k + 3)
            hit += (pp <= 0.05)
        lad[str(dl)] = round(hit / args.ndraw, 3)
    thr = next((float(k) for k in lad if float(k) > 0 and lad[k] >= 0.8), None)
    r["ladder"] = {"rungs": lad, "delta0": lad.get("0.0"), "threshold": thr}
    log(f"\n  【梯级】{lad}   δ=0 自检 {lad.get('0.0')} / {SELF_MAX}   检出阈值 {thr}")
    if lad.get("0.0", 0) > SELF_MAX:
        r["ladder"]["void"] = "K3_self_check_failed"
        log("      ** K3 触发:梯级自检超标 ⇒ E 臂与梯级作废 **")
    out.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=NPERM)
    ap.add_argument("--ndraw", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "m29_binder_crossdomain.json"))
    args = ap.parse_args()
    log("=== M29:用蛋白 binder 语料检验 R1 自己的定理 ===")
    log("  声明 m29_binder_crossdomain_declaration.json,冻结于 77e2b212,先于本文件。")
    log("  离线 $0。**主臂是 C(定理的界),它可以证否我们自己。**")
    log("  K4:本文件不产生任何蛋白学主张。\n")

    d = pd.read_csv(CSV, low_memory=False)
    d = d.dropna(subset=[TAU, ZCOL, YCOL])
    d["method"] = np.where(d["source"].str.startswith("Cao"), "Cao et al",
                  np.where(d["source"].str.contains("Watson"), "Watson et al", "other"))
    log(f"  语料 {len(d)} 条(τ/z/结果三列均非空)")

    main_df = d[(d["target_id"].isin(TARGETS)) & (d["method"].isin(METHODS))]
    log(f"  主臂:{len(TARGETS)} 个被两个【方法】交叉的靶标,{len(main_df)} 条设计")
    log(f"      {dict(main_df.groupby(['target_id','method']).size())}")

    out = []
    run(main_df, f"MAIN: {'+'.join(TARGETS)} (Cao x Watson), n={len(main_df)}", args, out)

    egfr = d[(d["target_id"] == "EGFR")]
    log(f"\n  敏感性(不进主裁定):EGFR n={len(egfr)},另一侧是 302 个独立参赛者而非一个方法")
    run(egfr, f"SENSITIVITY: EGFR, n={len(egfr)}", args, out)

    payload = {"arms": out, "targets": list(TARGETS), "tau": TAU, "z": ZCOL, "y": YCOL}
    json.dump(payload, open(args.out, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
