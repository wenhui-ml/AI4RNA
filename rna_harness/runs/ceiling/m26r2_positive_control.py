#!/usr/bin/env python3
"""M26r2: calibrate the retention metric against a KNOWN mediated fraction.

The first attempt at this control was malformed and it is worth saying how, because the failure
mode is the interesting part. It built a mediator M = frac*1[CG] + noise and conditioned on it.
That mediator depends on the treatment but does not predict the outcome, so it is not on any
causal path: conditioning on it destroys CG/AU overlap without shrinking the within-stratum
contrast. Retention stayed at 100% and the weight collapsed, and the script called that
"detection". It was not.

A mediator has to sit between the treatment and the outcome. So this builds a synthetic OUTCOME:

    M    = alpha * 1[CG] + noise                      (mediator, treatment-dependent)
    Ysyn = gamma * M + delta * 1[CG] + eps            (outcome, part through M, part direct)

The true mediated fraction is then known in closed form,

    mediated_true = gamma*alpha / (gamma*alpha + delta),

and retention should come back at 1 - mediated_true. Sweeping delta from all-direct to
all-mediated says what the metric can actually resolve on THIS design -- same records, same design
cells, same depth and crossing, same estimator.

If retention tracks the truth, the committed 96.3% / 100.7% are null results. If retention sits
near 100% regardless, they are uninformative and every one of them has to be re-labelled.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def retention_of(recs, mkey, ykey, nbins=5):
    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    v = np.array([r[mkey] for r in recs])
    edges = [float(np.quantile(v, q)) for q in np.linspace(0, 1, nbins + 1)[1:-1]]
    for r in recs:
        r["_b"] = int(np.searchsorted(edges, r[mkey], side="right"))
    sd = lambda r: (r["design"], min(r["depth"], 3), r["crossed"], r["_b"])
    cde, w, _ = M26.matched(recs, "CG", "AU", sd, key=ykey)
    if cde is None or w < 100:
        return None, None, (w or 0)
    keep = M26.retained_recs(recs, "CG", "AU", sd)
    base, _wb, _s = M26.matched(keep, "CG", "AU", sb, key=ykey)
    return (cde / base if base else None), cde, w


def main():
    log("=== M26r2:用【已知中介比例】标定留存这个指标 ===")
    log("  第一次的正对照是坏的:中介物只依赖处理、不预测结果,不在因果路径上。\n")
    recs, _ = M26.build()
    log(f"纳入 {len(recs)}(真实设计单元、真实 depth/crossed、真实配对类型)")
    rng = np.random.default_rng(20260827)
    ALPHA, GAMMA, SD_M, SD_E = 1.0, 1.0, 0.6, 0.25

    log(f"\n构造: M = {ALPHA}*1[CG] + N(0,{SD_M})   "
        f"Ysyn = {GAMMA}*M + delta*1[CG] + N(0,{SD_E})")
    log(f"  真值 mediated = gamma*alpha / (gamma*alpha + delta)\n")
    log(f"  {'delta':>7}{'真值中介':>10}{'真值留存':>10}   {'实测留存':>10}{'CDE':>10}{'权重':>8}")
    rows = []
    for delta in (0.00, 0.25, 0.50, 1.00, 2.00, 4.00):
        sig = np.array([1.0 if r["pt"] == "CG" else 0.0 for r in recs])
        M = ALPHA * sig + rng.normal(0, SD_M, len(recs))
        Y = GAMMA * M + delta * sig + rng.normal(0, SD_E, len(recs))
        for r, m, y in zip(recs, M, Y):
            r["_m"], r["_y"] = float(m), float(y)
        true_med = (GAMMA * ALPHA) / (GAMMA * ALPHA + delta)
        ret, cde, w = retention_of(recs, "_m", "_y")
        rows.append({"delta": delta, "true_mediated": round(true_med, 4),
                     "true_retention": round(1 - true_med, 4),
                     "measured_retention": (round(ret, 4) if ret is not None else None),
                     "weight": w})
        log(f"  {delta:7.2f}{100*true_med:9.1f}%{100*(1-true_med):9.1f}%   " +
            (f"{100*ret:9.1f}%{cde:10.4f}{w:8.0f}" if ret is not None
             else f"{'重叠丧失':>10}{'-':>10}{w:8.0f}"))

    ok = [r for r in rows if r["measured_retention"] is not None]
    if len(ok) >= 3:
        t = np.array([r["true_retention"] for r in ok])
        m = np.array([r["measured_retention"] for r in ok])
        corr = float(np.corrcoef(t, m)[0, 1])
        err = float(np.abs(t - m).mean())
        log(f"\n  真值留存 vs 实测留存:相关 {corr:+.4f}   平均绝对误差 {err:.4f}")
        tracks = corr >= 0.8 and err <= 0.20
        log(f"  **留存指标{'能' if tracks else '**不能**'}追踪真实中介比例**")
    else:
        corr, err, tracks = None, None, False
        log("\n  **可用点不足 3 个 ⇒ 无法标定,留存数只能记为不可判**")

    # what a null looks like on the same design: mediator carries no signal at all
    sig = np.array([1.0 if r["pt"] == "CG" else 0.0 for r in recs])
    Mn = rng.normal(0, SD_M, len(recs))
    Yn = 1.0 * sig + rng.normal(0, SD_E, len(recs))
    for r, m, y in zip(recs, Mn, Yn):
        r["_m"], r["_y"] = float(m), float(y)
    rn, _c, wn = retention_of(recs, "_m", "_y")
    log(f"\n  阴性对照(中介物与结果完全无关):留存 "
        f"{100*rn:.1f}%  权重 {wn:.0f}  —— 真值应为 100%")

    out = {"sweep": rows, "corr_true_vs_measured": (round(corr, 4) if corr else None),
           "mean_abs_error": (round(err, 4) if err else None), "metric_tracks_truth": bool(tracks),
           "negative_control_retention": (round(rn, 4) if rn else None)}
    json.dump(out, open(os.path.join(HERE, "m26r2_positive_control.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m26r2_positive_control.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m26r2_positive_control.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
