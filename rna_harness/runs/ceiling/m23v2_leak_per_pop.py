#!/usr/bin/env python3
"""Does DGP2's leak explain M23's DGP2 rejection rates, population by population?

If it does, the DGP2 column is power against a residual real effect, not type-I error.
If it does not, N2 has a genuine type-I problem and every p-value in the paper is at risk.
"""
from __future__ import annotations
import json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
from m13b_gc_placement import match, point, POPS, DELTA                          # noqa: E402
import m23_exact_null as M23                                                     # noqa: E402


def job(spec):
    nm, fn = spec
    d = M23._D(fn, 0)
    M = match(d, [c["gcp"] for c in d["cells"]], DELTA, panel=False)
    real = point(M)
    es = []
    for k in range(40):
        rng = np.random.default_rng(1234 + k)
        d0 = M23.dgp2(d, rng)
        MM = match(d0, [c["gcp"] for c in d0["cells"]], DELTA, panel=False)
        if MM:
            es.append(point(MM))
    es = np.array(es)
    # the N2 null on ONE dgp2 draw, to get the scale the observed value is judged against
    rng = np.random.default_rng(999)
    d0 = M23.dgp2(d, rng)
    nl = M23.null_dist(d0, "N2_permute_y_cell", 77, [c["gcp"] for c in d0["cells"]], nperm=400)
    thr = float(np.quantile(nl, 0.995))
    return {"pop": nm, "real": round(real, 4), "dgp2_mean": round(float(es.mean()), 4),
            "dgp2_sd": round(float(es.std(ddof=1)), 4),
            "leak_frac": round(float(es.mean() / real), 4),
            "N2_null_sd": round(float(nl.std()), 4), "N2_thr_995": round(thr, 4),
            "frac_draws_above_thr": round(float((es > thr).mean()), 3)}


def main():
    from concurrent.futures import ProcessPoolExecutor
    print("=== DGP2 的泄漏能否解释 M23 的 DGP2 拒绝率 ===\n")
    m23 = json.load(open(os.path.join(HERE, "m23_exact_null.json")))
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(job, [(nm, fn) for nm, fn in POPS]))
    print(f"  {'人群':12s} {'真实效应':>9s} {'DGP2均值':>9s} {'泄漏%':>7s} {'N2零SD':>8s} "
          f"{'99.5%阈':>8s} {'超阈%':>7s} {'M23实测N2':>10s}")
    for r, (nm, fn) in zip(rows, POPS):
        obs = m23["DGP2"]["per_pop"][fn]["N2_permute_y_cell"]
        print(f"  {r['pop']:12s} {r['real']:+9.3f} {r['dgp2_mean']:+9.3f} "
              f"{100*r['leak_frac']:6.1f}% {r['N2_null_sd']:8.3f} {r['N2_thr_995']:8.3f} "
              f"{100*r['frac_draws_above_thr']:6.1f}% {obs['hits']:4d}/{obs['n']:<4d}")
    pred = np.array([r["frac_draws_above_thr"] for r in rows])
    act = np.array([m23["DGP2"]["per_pop"][fn]["N2_permute_y_cell"]["hits"] /
                    max(m23["DGP2"]["per_pop"][fn]["N2_permute_y_cell"]["n"], 1)
                    for _n, fn in POPS])
    print(f"\n  预测(泄漏超阈比例)与实测拒绝率的相关 {np.corrcoef(pred, act)[0,1]:+.3f}")
    print(f"  预测合计 {pred.mean():.3f}   实测合计 {act.mean():.3f}")
    ok = abs(pred.mean() - act.mean()) < 0.08
    print(f"\n  **{'泄漏解释了 DGP2 一栏 ⇒ 那一栏是【功率】不是一类错误' if ok else '泄漏解释不了 ⇒ N2 可能真有一类错误问题,必须立刻上报'}**")
    json.dump({"rows": rows, "pred_mean": float(pred.mean()), "act_mean": float(act.mean()),
               "corr": float(np.corrcoef(pred, act)[0, 1]), "leak_explains": bool(ok)},
              open(os.path.join(HERE, "m23v2_leak_per_pop.json"), "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
