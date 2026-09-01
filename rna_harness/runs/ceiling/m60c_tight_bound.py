#!/usr/bin/env python3
"""M60c: the tight form of the corollary, and the under-resolution diagnostic M60b's numbers need.

M60b established two things and left one open.

  Established.  Assumption (7) holds cleanly on the primary channel (dG) and fails on P_model and
  on the binder corpus, so Prop. routes is licensed exactly where the paper leans on it and not
  where it does not.  And the plug-in estimate of osc(h)*d_TV rises monotonically with resolution
  while the left-hand side shows no trend, which is the signature of a downward-biased right-hand
  side rather than of a violated theorem.

  Open.  osc(h)*d_TV is Hoelder with the crudest possible pairing, sup-norm against L1.  The exact
  quantity is sum_k (h_k - c)(pA_k - pB_k) for any constant c, so

      | mediated part |  <=  min_c  sum_k |h_k - c| |pA_k - pB_k|,

  which is never worse and is usually strictly tighter.  Whether the tighter form is below one is
  the difference between "the corpus cannot bound this channel" and "it can, at this value".

Offline, $0.  Governed by m60_carrying_bound_declaration.json; erratum in m60_declaration_erratum.json.
"""
from __future__ import annotations

import json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m60b_carrying_bound as B

LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def tight(h, pa, pb):
    """min over c of sum_k |h_k - c| * |pA_k - pB_k|.

    The objective is piecewise linear in c with kinks at the h_k, so the minimiser is one of them:
    a |dp|-weighted median of h.  Evaluated exactly rather than optimised numerically.
    """
    dp = np.abs(pa - pb)
    best = None
    for c in np.concatenate([h, [float(np.average(h, weights=dp)) if dp.sum() else 0.0]]):
        v = float((np.abs(h - c) * dp).sum())
        best = v if best is None else min(best, v)
    return best


def row(cells, t, y, m, nbins, integer_levels=False):
    b, nb = B.levelbins(m) if integer_levels else B.qbins(m, nbins)
    d_tot, d_cde, _w = B.total_and_cde(cells, t, y, b, nb)
    if d_tot is None or abs(d_tot) < 1e-12:
        return None
    h, live, dtv = B.h_and_tv(cells, t, y, b, nb)
    if int(live.sum()) < 3:
        return None
    # the two arms' laws on the same cell weights h_and_tv used
    pa = np.zeros(nb); pb = np.zeros(nb)
    for _c, ix in cells:
        a, bx = ix[t[ix] == 1], ix[t[ix] == 0]
        if len(a) == 0 or len(bx) == 0:
            continue
        ww = min(len(a), len(bx))
        for k in range(nb):
            pa[k] += ww * (b[a] == k).sum() / len(a)
            pb[k] += ww * (b[bx] == k).sum() / len(bx)
    pa /= max(pa.sum(), 1e-12); pb /= max(pb.sum(), 1e-12)
    hs, pas, pbs = h[live], pa[live], pb[live]
    osc_h = float(hs.max() - hs.min())
    loose = osc_h * dtv
    tt = tight(hs, pas, pbs)
    lhs = abs(d_tot - d_cde)
    return {"nbins": int(nb), "used": int(live.sum()), "d_tot": round(d_tot, 6),
            "lhs": round(lhs, 7), "osc_h": round(osc_h, 6), "d_TV": round(dtv, 5),
            "loose_rhs": round(loose, 7), "tight_rhs": round(tt, 7),
            "loose_frac": round(loose / abs(d_tot), 5), "tight_frac": round(tt / abs(d_tot), 5),
            "loose_holds": bool(lhs <= loose + 1e-12), "tight_holds": bool(lhs <= tt + 1e-12),
            "h_bins": [round(float(x), 6) for x in hs],
            "dp_bins": [round(float(x), 6) for x in (pas - pbs)]}


def main():
    log("=== M60c:推论的紧形式,以及欠分辨的诊断 ===")
    log("  声明未改。全离线,$0。\n")
    out = {"governed_by": "m60_carrying_bound_declaration.json",
           "erratum": "m60_declaration_erratum.json",
           "tight_form": "min_c sum_k |h_k - c| |pA_k - pB_k|  <=  osc(h) d_TV"}
    recs = B.rna_records()
    res = {}
    for corpus, med, intlev in (("rna", "dG", False), ("rna", "pmut", False),
                                ("rna", "depth", True), ("binder", "A_length", False)):
        cells, t, y, m, n = B.rna_arm(recs, med) if corpus == "rna" else B.binder_arm()
        grid = (0,) if intlev else B.BINGRID
        rows = [r for r in (row(cells, t, y, m, nb, intlev) for nb in grid) if r]
        if not rows:
            continue
        key = f"{corpus}:{med}"
        # the under-resolution diagnostic: does the RHS climb with resolution while the LHS does not?
        from scipy.stats import spearmanr
        nb_ = [r["nbins"] for r in rows]
        rho_rhs = float(spearmanr(nb_, [r["loose_rhs"] for r in rows]).statistic) if len(rows) > 2 else float("nan")
        rho_lhs = float(spearmanr(nb_, [r["lhs"] for r in rows]).statistic) if len(rows) > 2 else float("nan")
        res[key] = {"n": n, "rows": rows, "rho_rhs_vs_nbins": round(rho_rhs, 4),
                    "rho_lhs_vs_nbins": round(rho_lhs, 4),
                    "tight_min": min(r["tight_frac"] for r in rows),
                    "tight_at_finest_holding": next(
                        (r["tight_frac"] for r in rows if r["tight_holds"]), None),
                    "loose_at_finest_holding": next(
                        (r["loose_frac"] for r in rows if r["loose_holds"]), None),
                    "any_tight_below_one": any(r["tight_frac"] < 1.0 and r["tight_holds"]
                                               for r in rows)}
        log(f"\n  {key}  n={n}")
        log(f"    {'箱':>4}{'用箱':>5}{'|中介部分|':>12}{'松界':>10}{'紧界':>10}"
            f"{'松%':>8}{'紧%':>8}{'松成立':>8}{'紧成立':>8}")
        for r in rows:
            log(f"    {r['nbins']:>4}{r['used']:>5}{r['lhs']:>12.5f}{r['loose_rhs']:>10.5f}"
                f"{r['tight_rhs']:>10.5f}{100*r['loose_frac']:>7.1f}%{100*r['tight_frac']:>7.1f}%"
                f"{'  是' if r['loose_holds'] else ' **否**':>8}"
                f"{'  是' if r['tight_holds'] else ' **否**':>8}")
        log(f"    右边随分箱数的 Spearman {rho_rhs:+.3f};左边 {rho_lhs:+.3f}"
            f"   -> {'右边被粗分箱系统性低估' if rho_rhs > 0.7 and abs(rho_lhs) < 0.7 else '无此模式'}")
    out["arms"] = res
    dg = res.get("rna:dG", {})
    out["headline"] = {
        "assumption7_holds_on_primary_channel": True,
        "loose_bound_is_vacuous_where_valid": bool((dg.get("loose_at_finest_holding") or 0) >= 1.0),
        "tight_bound_is_vacuous_where_valid": bool((dg.get("tight_at_finest_holding") or 0) >= 1.0),
        "tight_min_frac": dg.get("tight_min"),
    }
    log("\n=== 结论 ===")
    log(f"  主通道 ΔG:松界在成立处 {100*(dg.get('loose_at_finest_holding') or 0):.1f}%,"
        f"紧界在成立处 {100*(dg.get('tight_at_finest_holding') or 0):.1f}%")
    log(f"  紧界的最小值 {100*(dg.get('tight_min') or 0):.1f}%")
    json.dump(out, open(os.path.join(HERE, "m60c_tight_bound.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m60c_run.out"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m60c_tight_bound.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
