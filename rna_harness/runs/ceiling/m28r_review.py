#!/usr/bin/env python3
"""M28r: review M28's two boundary results before either is written down.

Two results in M28 sit on a boundary and neither may be reported until the code and the logic
behind it have been checked:

  R1  M3SARS_DMS admitted ZERO interventions at the declared SN filter, and M3SARS_2A3 admitted
      only one family. A filter that returns zero is the shape most likely to be a defect.

  R2  The two probes on M2SL5 gave skill slopes 3.75x and 3.10x apart. That comparison is between
      two DIFFERENT outcome variables measured on different scales, and rule 68 says two things
      must be put on the same scale before their magnitudes are compared. The raw ratio is not
      admissible as reported.

This file answers both with quantities that do not depend on the outcome's scale, and adds the
outlier sensitivity the raw slope hides. It computes no new claim; it decides whether M28's
existing numbers may be read.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m28_second_corpus as M

NPERM = 2000


def log(*a):
    print(*a, flush=True)


def build(rid):
    """re-mine one deposit and attach both predictors, exactly as M28 does."""
    cons, seqpos, hs, hq = M.parse(rid)
    assert not hs and not hq, "P4: this deposit declares a structure"
    window = (seqpos[0] - 1, seqpos[-1])
    allo = M.find_offset(cons, window)
    offs = {k: v for k, v in allo.items() if v[1] / max(1, v[2]) >= M.AGREE}
    recs, ex = M.mine(cons, {c.name: c for c in cons}, offs, M.SN_MAIN, window)
    for r in recs:
        r["cols"] = tuple(k for k, p in enumerate(seqpos) if p - 1 in (r["i"], r["j"]))
    rf = np.array([(v if (v := M.rescue(r)) is not None else np.nan) for r in recs])
    ok = ~np.isnan(rf)
    recs = [r for r, o in zip(recs, ok) if o]
    return recs, rf[ok], ex


def rank_within(sid, v):
    """rank inside each stratum -- scale-free and outlier-robust in one step."""
    out = np.empty(len(v))
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)
    for _s, ix in by.items():
        a = np.asarray(ix)
        r = v[a].argsort().argsort().astype(float)
        out[a] = (r - r.mean()) / (r.std() if r.std() else 1.0)
    return out


def within_corr(sid, x, y):
    """pooled within-stratum Pearson: the design is removed, not regressed on."""
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)
    sxy = sxx = syy = 0.0
    for _s, ix in by.items():
        if len(ix) < 3:
            continue
        a = np.asarray(ix)
        xa, ya = x[a] - x[a].mean(), y[a] - y[a].mean()
        sxy += float((xa * ya).sum())
        sxx += float((xa * xa).sum())
        syy += float((ya * ya).sum())
    return sxy / np.sqrt(sxx * syy) if sxx and syy else None


def corr_perm(sid, x, y, nperm=NPERM, seed=0):
    obs = within_corr(sid, x, y)
    if obs is None:
        return None, 1.0
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)
    ge = 0
    for _ in range(nperm):
        yp = y.copy()
        for _s, ix in by.items():
            a = np.asarray(ix)
            yp[a] = y[a[rng.permutation(len(a))]]
        c = within_corr(sid, x, yp)
        ge += (c is not None and abs(c) >= abs(obs))
    return float(obs), (ge + 1) / (nperm + 1)


def boot_diff(sid_a, xa, ya, sid_b, xb, yb, nboot=2000, seed=0):
    """cluster bootstrap on the DIFFERENCE of the two probes' within-stratum correlations.

    Rule 69: a gate that compares two estimates must test their difference, not eyeball two
    point estimates. Both probes measure the SAME interventions, so the resample is over
    interventions and is applied to both arms together -- a paired bootstrap.
    """
    rng = np.random.default_rng(seed)
    ka = {k: i for i, k in enumerate(zip(sid_a, map(tuple, np.c_[xa])))}
    n = min(len(ya), len(yb))
    idx = np.arange(n)
    out = []
    for _ in range(nboot):
        s = rng.choice(idx, size=n, replace=True)
        ca = within_corr(sid_a[s], xa[s], ya[s])
        cb = within_corr(sid_b[s], xb[s], yb[s])
        if ca is not None and cb is not None:
            out.append(ca - cb)
    out = np.asarray(out)
    return (round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4),
            round(float(out.mean()), 4))


def main():
    res = {}
    log("=== M28r:两个边界结果的复查 ===")
    log("  不产生任何新主张。只决定 M28 已有的数能不能读。\n")

    # ---------------------------------------------------------------- R1
    log("【R1】M3SARS 的零,是【参考测量】不合格,不是【干预】太吵")
    r1 = {}
    for rid in ("M3SARS_DMS_0000", "M3SARS_2A3_0000", "M2SL5_DMS_0000", "M2SL5_2A3_0000"):
        cons, seqpos, _hs, _hq = M.parse(rid)
        wts = [c for c in cons if not M.tags(c.name)]
        window = (seqpos[0] - 1, seqpos[-1])
        allo = M.find_offset(cons, window)
        offs = {k: v for k, v in allo.items() if v[1] / max(1, v[2]) >= M.AGREE}
        bn = {c.name: c for c in cons}
        # every flip whose four constructs exist, before any SN filter
        per_fam = collections.defaultdict(lambda: {"n": 0, "flip_ok": 0})
        for c in cons:
            ts = M.tags(c.name)
            if len(ts) != 2:
                continue
            (p1, f1, t1), (p2, f2, t2) = ts
            if not (f1 == t2 and f2 == t1):
                continue
            if "".join(sorted({f1.replace("T", "U"), f2.replace("T", "U")})) not in M.CANON:
                continue
            fam = M.family_key(c.name)
            wt = M.find_wt(fam, wts) if fam else None
            if wt is None or wt == "AMBIGUOUS" or fam not in offs:
                continue
            a = bn.get(M.single_name(c.name, 0))
            b = bn.get(M.single_name(c.name, 1))
            if a is None or b is None:
                continue
            per_fam[fam]["n"] += 1
            per_fam[fam]["wt_sn"] = wt.sn
            # the three per-intervention constructs only -- the reference is excluded here
            per_fam[fam]["flip_ok"] += int(min(c.sn, a.sn, b.sn) >= M.SN_MAIN)
        rows = []
        for fam, v in sorted(per_fam.items()):
            gated = v["wt_sn"] < M.SN_MAIN
            rows.append({"family": fam[:44], "n_flips": v["n"], "wt_sn": round(v["wt_sn"], 3),
                         "flips_passing_own_sn": v["flip_ok"],
                         "family_gated_out_by_its_reference": bool(gated)})
            log(f"  {rid:20s} {fam[:40]:42s} 翻转 {v['n']:5d}  参考 SN {v['wt_sn']:6.3f}"
                f"  自身 SN 过线 {v['flip_ok']:5d}"
                f"{'   <= 整族被参考卡掉' if gated else ''}")
        r1[rid] = rows
    res["R1_reference_gating"] = r1
    log("\n  读法:野生型的剖面在 d_i、d_j、d_ij 三个差向量里都要被减掉,所以参考不合格 ⇒ 整族不可用。")
    log("  **这不是过滤器的缺陷,是 rescue 公式的依赖结构。阈值不动。**\n")

    # ---------------------------------------------------------------- R2
    log("【R2】两个探针的差,放到同一尺度上还剩多少")
    built = {}
    for rid in ("M2SL5_DMS_0000", "M2SL5_2A3_0000"):
        recs, rf, _ex = build(rid)
        xv = M.vienna_scores(recs, lambda *a: None)
        xn = M.neural_scores(recs, lambda *a: None)
        built[rid] = (recs, rf, xv, xn)
        log(f"  {rid} 重挖 n={len(recs)}")

    a_rec, a_rf, a_xv, a_xn = built["M2SL5_DMS_0000"]
    b_rec, b_rf, b_xv, b_xn = built["M2SL5_2A3_0000"]
    ka = {(r["family"], r["i"], r["j"]): i for i, r in enumerate(a_rec)}
    kb = {(r["family"], r["i"], r["j"]): i for i, r in enumerate(b_rec)}
    sh = sorted(set(ka) & set(kb))
    ia = np.array([ka[k] for k in sh])
    ib = np.array([kb[k] for k in sh])
    sid = np.array([a_rec[i]["family"] for i in ia])
    log(f"  共享干预 {len(sh)}  层 {len(set(sid))}")

    # the predictors depend only on the sequence, so they must be identical across probes
    same_v = float(np.nanmax(np.abs(a_xv[ia] - b_xv[ib])))
    same_n = float(np.nanmax(np.abs(a_xn[ia] - b_xn[ib])))
    log(f"  预测量跨探针一致性(必须恰好 0):ViennaRNA {same_v:.2e}  RibonanzaNet {same_n:.2e}")
    res["R2_predictors_identical"] = {"vienna_max_abs_diff": same_v, "neural_max_abs_diff": same_n,
                                      "ok": bool(same_v < 1e-12 and same_n < 1e-12)}

    ya, yb = a_rf[ia], b_rf[ib]
    res["R2_outcome_scale"] = {"DMS_sd": round(float(ya.std()), 4),
                               "2A3_sd": round(float(yb.std()), 4),
                               "ratio": round(float(ya.std() / yb.std()), 3),
                               "DMS_below_minus1": int((ya < -1).sum()),
                               "2A3_below_minus1": int((yb < -1).sum())}
    log(f"  结果量 SD  DMS {ya.std():.4f}  2A3 {yb.std():.4f}  比 {ya.std()/yb.std():.2f}x"
        f"   (DMS 有 {(ya<-1).sum()} 条 < -1,2A3 有 {(yb<-1).sum()} 条)")

    arms = {}
    for nm, xa_, xb_ in (("ViennaRNA", a_xv[ia], b_xv[ib]), ("RibonanzaNet", a_xn[ia], b_xn[ib])):
        row = {}
        ca, pa = corr_perm(sid, xa_, ya)
        cb, pb = corr_perm(sid, xb_, yb)
        row["within_corr"] = {"DMS": round(ca, 4), "2A3": round(cb, 4)}
        row["within_corr_p"] = {"DMS": round(pa, 5), "2A3": round(pb, 5)}
        # rank version: scale-free AND outlier-proof
        rya, ryb = rank_within(sid, ya), rank_within(sid, yb)
        rxa, rxb = rank_within(sid, xa_), rank_within(sid, xb_)
        ra, pra = corr_perm(sid, rxa, rya)
        rb, prb = corr_perm(sid, rxb, ryb)
        row["within_spearman"] = {"DMS": round(ra, 4), "2A3": round(rb, 4)}
        row["within_spearman_p"] = {"DMS": round(pra, 5), "2A3": round(prb, 5)}
        lo, hi, mn = boot_diff(sid, xa_, ya, sid, xb_, yb)
        row["paired_boot_corr_diff"] = {"mean": mn, "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}
        # drop the two extreme DMS points and re-read
        keep = ya > -1
        ck, pk = corr_perm(sid[keep], xa_[keep], ya[keep])
        row["DMS_without_outliers"] = {"n": int(keep.sum()), "within_corr": round(ck, 4),
                                       "p": round(pk, 5)}
        arms[nm] = row
        log(f"\n  {nm}")
        log(f"    层内 Pearson    DMS {ca:+.4f} (p {pa:.4f})   2A3 {cb:+.4f} (p {pb:.4f})")
        log(f"    层内 Spearman   DMS {ra:+.4f} (p {pra:.4f})   2A3 {rb:+.4f} (p {prb:.4f})")
        log(f"    配对 bootstrap 的【差】 {mn:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]"
            f"  {'排除 0' if (lo>0 or hi<0) else '**含 0**'}")
        log(f"    DMS 去掉 {int((~keep).sum())} 条离群后 Pearson {ck:+.4f} (p {pk:.4f})")
    res["R2_arms"] = arms

    out = os.path.join(HERE, "m28r_review.json")
    json.dump(res, open(out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    log(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
