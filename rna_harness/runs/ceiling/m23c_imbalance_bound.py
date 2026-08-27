#!/usr/bin/env python3
"""M23 arm C: bound the failure mode the review actually identifies.

The review is right that "y perp t within cell" does not give within-cell exchangeability, because
gamma and pi predict y. Arms A and B tried to measure the consequence by simulation and arm B
failed: NO zero-effect dataset can preserve the real outcome-covariate relation on this design,
because gamma and t are dependent and part of the y-gamma relation IS the effect. DGP2 leaked
4.7-33.7% of the real effect and DGP3 (non-parametric) leaked 47.5%.

The consequence is reachable analytically instead. Under y = h(gamma, pi) + noise, a matched pair
contributes h(gamma_i, pi_i) - h(gamma_j, pi_j), which is zero only if the caliper closes exactly.
The residual within-caliper imbalance is measured by the estimator already (it is the SMD the
design gates on), so the bias it can produce is

    bias  =  beta_gamma * mean(delta gamma)  +  beta_pi * mean(delta pi)

with both slopes estimated within cell and both mean gaps read off the matched set. Report it
against the observed effect, per population, for both the panel-free and panel-matched arms.
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


def within_cell_slopes(d):
    """OLS of y on (gamma, pi) after centring both inside cell -- the same working model the
    Freedman-Lane construction would use."""
    X, Y = [], []
    for c in d["cells"]:
        if len(c["y"]) < 3:
            continue
        g = c["gcf"] - c["gcf"].mean(); p = c["pB"] - c["pB"].mean()
        y = c["y"] - c["y"].mean()
        X.append(np.column_stack([g, p])); Y.append(y)
    X = np.vstack(X); Y = np.concatenate(Y)
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(b[0]), float(b[1]), len(Y)


def main():
    rows = []
    print("=== M23 arm C:残余 caliper 内不平衡能造成多大偏倚 ===")
    print("  (arm B 作废:在本设计上不存在既零效应又保留 y–γ 真实关系的构造)\n")
    for nm, fn in POPS:
        d = M23._D(fn, 0)
        bg, bp, n = within_cell_slopes(d)
        for panel in (False, True):
            M = match(d, [c["gcp"] for c in d["cells"]], DELTA, panel=panel)
            if M is None:
                continue
            dg = float(M["dgcf"].mean()); dp = float(M["dpB"].mean())
            eff = point(M)
            bias = bg * dg + bp * dp
            sg = float(np.std(np.concatenate([c["gcf"] for c in d["cells"]]))) or 1.0
            rows.append({"pop": nm, "panel": panel, "beta_gamma": round(bg, 3),
                         "beta_pi": round(bp, 4), "n_cellpoints": n,
                         "mean_dgamma": round(dg, 5), "mean_dpi": round(dp, 5),
                         "smd_gamma": round(abs(dg) / sg, 4),
                         "bias": round(bias, 4), "effect": round(eff, 4),
                         "bias_pct_of_effect": round(100 * bias / eff, 2),
                         "npair": M["npair"]})
    print(f"  {'人群':12s} {'面板':>4s} {'β_γ':>8s} {'β_π':>8s} {'Δγ均值':>9s} {'Δπ均值':>9s} "
          f"{'偏倚':>8s} {'效应':>8s} {'占比':>7s}")
    for r in rows:
        print(f"  {r['pop']:12s} {'是' if r['panel'] else '否':>4s} {r['beta_gamma']:8.2f} "
              f"{r['beta_pi']:8.3f} {r['mean_dgamma']:+9.5f} {r['mean_dpi']:+9.5f} "
              f"{r['bias']:+8.4f} {r['effect']:+8.4f} {r['bias_pct_of_effect']:6.2f}%")
    pan = [r for r in rows if r["panel"]]
    worst = max(abs(r["bias_pct_of_effect"]) for r in pan)
    print(f"\n  **面板配平臂:残余不平衡偏倚最大占效应的 {worst:.2f}%**")
    print(f"  逐人群占比 " + "  ".join(f"{r['bias_pct_of_effect']:+.2f}%" for r in pan))
    out = {"rows": rows, "worst_pct_panel": worst,
           "arm_B_void_reason": "DGP2 保留了真实效应的 4.7%~33.7%(逐人群),与实测拒绝率相关 +0.857;"
                                "DGP3(kNN)保留 47.5%。γ 与 t 相依 ⇒ 『y–γ 的真实关系』里有一部分【就是】"
                                "效应本身 ⇒ 本设计上不存在既零效应又保留该关系的构造。",
           "verdict": ("residual_imbalance_bias_is_bounded_and_small" if worst < 5
                       else "residual_imbalance_bias_is_material__must_be_reported_as_a_limit")}
    json.dump(out, open(os.path.join(HERE, "m23c_imbalance_bound.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n=== 裁定: {out['verdict']} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
