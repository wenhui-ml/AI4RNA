#!/usr/bin/env python3
"""M23 verification: is DGP2 actually a zero-effect process?

M23 reported N2 rejecting 45/240 under DGP2 at a nominal 0.005. Before that can be read as a
property of the null, the data-generating process has to be shown to have no effect. DGP2
permutes the treatment inside FIXED-GRID gamma blocks of width CAL_GC, but the estimator's
caliper is a MOVING WINDOW of the same width, so a pair straddling a block boundary keeps its
original treatment-outcome association. If that leak is real, the "type-I rates" are power.

V1  mean observed effect under each DGP, against the real data's effect.
V2  what fraction of matched pairs straddle a block boundary.
V3  a leak-free DGP: y0 = fhat(gamma, pi) + permuted residual, fhat non-parametric (kNN).
    The treatment does not enter y0 at all, so the true effect is exactly zero.
"""
from __future__ import annotations
import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
from m13b_gc_placement import assemble, match, point, POPS, DELTA, CAL_GC       # noqa: E402
import m23_exact_null as M23                                                    # noqa: E402


def real(d):
    M = match(d, [c["gcp"] for c in d["cells"]], DELTA, panel=False)
    return point(M), M["npair"]


def eff_of(d0):
    M = match(d0, [c["gcp"] for c in d0["cells"]], DELTA, panel=False)
    return (point(M), M["npair"]) if M else (None, 0)


def knn_fit(c, k=8):
    """non-parametric E[y | gamma, pi] within cell: mean of the k nearest neighbours in
    standardised (gamma, pi), the point itself excluded."""
    n = len(c["y"])
    if n <= 2:
        return c["y"].copy()
    g = (c["gcf"] - c["gcf"].mean()) / (c["gcf"].std() or 1.0)
    p = (c["pB"] - c["pB"].mean()) / (c["pB"].std() or 1.0)
    D = (g[:, None] - g[None, :]) ** 2 + (p[:, None] - p[None, :]) ** 2
    np.fill_diagonal(D, np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    return c["y"][idx].mean(1)


def dgp3(d, rng):
    """y0 = fhat(gamma, pi) + permuted residual. t never enters y0 => true effect exactly 0."""
    out = []
    for c in d["cells"]:
        f = knn_fit(c); r = c["y"] - f
        y = f + r[rng.permutation(len(r))]
        out.append({**c, "y": y - y.mean()})
    return {**d, "cells": out}


def main():
    R = {}
    fn = POPS[0][1]
    d = M23._D(fn, 0)
    e_real, np_real = real(d)
    print(f"=== V1 各 DGP 下的实测效应({POPS[0][0]})===")
    print(f"  真实数据            效应 {e_real:+.4f}  配对 {np_real}")
    rows = {}
    for name, fnc in (("DGP1", M23.dgp1), ("DGP2", M23.dgp2), ("DGP3_knn", dgp3)):
        es = []
        for k in range(40):
            rng = np.random.default_rng(1234 + k)
            e, npr = eff_of(fnc(d, rng))
            if e is not None:
                es.append(e)
        es = np.array(es)
        rows[name] = {"mean": round(float(es.mean()), 4), "sd": round(float(es.std(ddof=1)), 4),
                      "frac_positive": round(float((es > 0).mean()), 3)}
        print(f"  {name:9s}  效应均值 {es.mean():+.4f} ± {es.std(ddof=1):.4f}  "
              f"为正 {100*(es>0).mean():.0f}%   "
              f"占真实效应 {100*es.mean()/e_real:.1f}%")
    R["V1"] = {"real_effect": round(e_real, 4), **rows}
    leak = rows["DGP2"]["mean"] / e_real
    print(f"\n  **DGP2 保留了真实效应的 {100*leak:.1f}% ⇒ "
          f"{'它不是零效应过程,M23 的 DGP2 一栏是功率不是一类错误' if abs(leak) > 0.1 else '泄漏可忽略'}**")
    R["V1"]["DGP2_leak_fraction"] = round(leak, 4)

    print("\n=== V2 有多少配对跨越固定网格块边界 ===")
    tot = cross = 0
    for c in d["cells"]:
        g = c["gcf"]; t = c["gcp"]
        if len(t) < 2:
            continue
        dt = t[:, None] - t[None, :]
        ok = (dt >= DELTA) & (np.abs(g[:, None] - g[None, :]) <= CAL_GC)
        a, b = np.nonzero(ok)
        gb = np.floor(g / CAL_GC).astype(np.int64)
        tot += len(a); cross += int((gb[a] != gb[b]).sum())
    print(f"  配对 {tot},跨块 {cross} = {100*cross/max(tot,1):.1f}%  "
          f"(caliper 是宽 {CAL_GC} 的移动窗,网格块也是 {CAL_GC} ⇒ 跨块配对必然存在)")
    R["V2"] = {"pairs": tot, "cross_block": cross, "frac": round(cross / max(tot, 1), 4)}

    print("\n=== V3 泄漏无关检查:DGP3 的效应应当落在 0 附近 ===")
    print(f"  DGP3 效应均值 {rows['DGP3_knn']['mean']:+.4f}(真实 {e_real:+.4f})  "
          f"为正 {100*rows['DGP3_knn']['frac_positive']:.0f}%")
    json.dump(R, open(os.path.join(HERE, "m23v_dgp_check.json"), "w"), ensure_ascii=False, indent=1)
    print("\n写出 m23v_dgp_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
