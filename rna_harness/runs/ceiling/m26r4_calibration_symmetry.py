#!/usr/bin/env python3
"""M26r4 review: is the summary the paper quotes the summary the calibration sweep supports?

A review, not an experiment. No new data, no folding, no estimation: this file reads
m26r3_realistic_calibration.json and recomputes summary statistics of a sweep that is already
committed. It exists because the manuscript audit flagged two macros as computed-and-never-shown,
and both belong to the SAME claimbox:

    calNCorr  = -0.7019   quoted     ("correlates with the truth at ..., the wrong sign")
    calVCorr  = +0.7026   NOT quoted
    calNFull  = +131.4%   NOT quoted
    calVFull  = -842.5%   quoted

corr is corrcoef(true_retention, measured_retention) and true_retention = 1 - f, so a correct
estimator gives a POSITIVE correlation. The neural arm's -0.7019 has the wrong sign, which is what
the paper says. The thermodynamic arm's +0.7026 has the RIGHT sign, and the paper does not say so.
Quoting the wrong-sign arm's correlation and the other arm's error, and never the reverse, is
selective reporting inside a withdrawal claim. Either the paper reports both, or the withdrawal is
resting on a summary the reader cannot check.

So the question this file settles is narrow and answerable: is +0.7026 evidence that the
thermodynamic arm tracks the truth?

  R1  report every summary for BOTH arms, on the same footing.
  R2  leave-one-out over the six sweep points. A correlation on six points that is produced by one
      point is not a correlation; the paper may not lean on it either way without saying so.
  R3  rank correlation, which the endpoint cannot dominate the way it dominates Pearson.
  R4  the honest question restated: does either arm's measured retention stay inside a range a
      reader could invert? Report the span of measured retention over f in [0, 0.75], i.e. with the
      degenerate endpoint excluded, against the true span it is supposed to cover.

COST: $0, offline, no model call, no folding. Reads one committed JSON.
"""
from __future__ import annotations

import json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "m26r3_realistic_calibration.json")
ARMS = (("RibonanzaNet", "N"), ("ViennaRNA", "V"))
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    src = json.load(open(SRC))
    out = {"source": os.path.basename(SRC), "arms": {}}

    log("=== M26r4:校准汇报的对称性复核 ===")
    for nm, tag in ARMS:
        a = src[nm]
        sw = [r for r in a["sweep"] if r["measured_retention"] is not None]
        f = np.array([r["f"] for r in sw])
        t = np.array([r["true_retention"] for r in sw])
        m = np.array([r["measured_retention"] for r in sw])

        full = float(np.corrcoef(t, m)[0, 1])
        # R2: leave-one-out over sweep points
        loo = []
        for i in range(len(sw)):
            k = [j for j in range(len(sw)) if j != i]
            loo.append({"dropped_f": float(f[i]),
                        "corr": round(float(np.corrcoef(t[k], m[k])[0, 1]), 4)})
        loo_vals = [d["corr"] for d in loo]
        # R3 rank
        rho = spearman(t, m)
        # R4 span over the non-degenerate part of the sweep
        keep = f <= 0.75
        span_meas = float(m[keep].max() - m[keep].min())
        span_true = float(t[keep].max() - t[keep].min())

        rec = {
            "corr": round(full, 4),
            "mae": a["mae"],
            "spearman": round(rho, 4),
            "sign_is_correct": bool(full > 0),
            "loo": loo,
            "loo_corr_min": round(min(loo_vals), 4),
            "loo_corr_max": round(max(loo_vals), 4),
            "loo_sign_flips": bool(min(loo_vals) < 0 < max(loo_vals)),
            "corr_without_endpoint": next(d["corr"] for d in loo if d["dropped_f"] == 1.0),
            "retention_at_f0": sw[0]["measured_retention"],
            "retention_at_f1": sw[-1]["measured_retention"],
            "span_measured_f_le_075": round(span_meas, 4),
            "span_true_f_le_075": round(span_true, 4),
            "span_ratio": round(span_meas / span_true, 4),
        }
        out["arms"][nm] = rec
        log(f"\n--- {nm}(宏前缀 cal{tag})")
        log(f"  corr(真值留存, 实测留存) = {full:+.4f}   符号正确:{rec['sign_is_correct']}")
        log(f"  秩相关 = {rho:+.4f}    平均绝对误差 = {a['mae']:.4f}")
        log(f"  f=0 实测 {100*rec['retention_at_f0']:+.1f}%    f=1 实测 {100*rec['retention_at_f1']:+.1f}%")
        log(f"  留一:{rec['loo_corr_min']:+.4f} .. {rec['loo_corr_max']:+.4f}"
            f"   去掉 f=1 那一点后 {rec['corr_without_endpoint']:+.4f}"
            f"   留一会翻号:{rec['loo_sign_flips']}")
        log(f"  f<=0.75 段:实测跨度 {span_meas:.4f} 对应真值跨度 {span_true:.4f}"
            f"(比 {rec['span_ratio']:.3f})")

    n, v = out["arms"]["RibonanzaNet"], out["arms"]["ViennaRNA"]
    # The verdict this file is allowed to reach.
    both_uninformative = (abs(n["span_ratio"]) < 0.5) and (abs(v["span_ratio"]) < 0.5)
    v_corr_is_endpoint = (v["corr"] > 0) and (v["corr_without_endpoint"] <= 0)
    out["V_correlation_is_produced_by_the_endpoint"] = bool(v_corr_is_endpoint)
    out["both_arms_span_less_than_half_the_truth"] = bool(both_uninformative)
    out["verdict"] = (
        "report_both_arms_the_positive_correlation_is_an_endpoint_artefact"
        if v_corr_is_endpoint and both_uninformative else
        "REREAD_the_withdrawal_claim_needs_restating")

    log("\n=== 复核裁定 ===")
    log(f"  ViennaRNA 那个 +{v['corr']:.4f} 是不是由 f=1 单点造出来的:"
        f"{out['V_correlation_is_produced_by_the_endpoint']}")
    log(f"  两条臂在 f<=0.75 段的实测跨度都不到真值跨度的一半:"
        f"{out['both_arms_span_less_than_half_the_truth']}")
    log(f"  {out['verdict']}")
    if out["verdict"].startswith("REREAD"):
        log("  !! 撤回声明的措辞不能照抄,必须按本文件测到的数重写")

    out["log"] = LOG
    json.dump(out, open(os.path.join(HERE, "m26r4_calibration_symmetry.json"), "w"),
              indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
