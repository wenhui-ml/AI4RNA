#!/usr/bin/env python3
"""M26r3: calibrate retention at the scale this corpus actually operates at.

m26r2 calibrated in a regime the data never visits. Its synthetic mediator separated the two arms
by 1.67 pooled SD; the real predictor separates them by 0.053 SD -- 31x weaker. Whatever m26r2
concluded cannot be carried over, in either direction.

This version changes one thing and keeps everything else real. The mediator IS the real predictor,
with its real near-binary distribution, its real correlation with design cells, its real binning
behaviour. Only the OUTCOME is synthetic:

    Ysyn = gamma * M_real + delta * 1[CG] + N(0, sigma)

with gamma and delta chosen so the total stratified contrast equals the corpus's own +0.0539 and a
known fraction f of it flows through the mediator:

    alpha  = the stratified CG-AU contrast IN THE MEDIATOR (what the estimator actually sees)
    gamma  = f * total / alpha
    delta  = (1 - f) * total

Sweeping f from 0 to 1 says what a measured retention means HERE. It also exposes the ceiling: with
alpha as small as it is, gamma has to become implausibly large for f to approach 1, and that is
itself a statement about what this design can support.

Reports the structural bound alongside: the largest mediated fraction the real predictor could
produce is its own arm separation times its own within-stratum slope, divided by the contrast.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

LOG = []
NBINS = 5
FRACS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)


def log(s=""):
    print(s, flush=True); LOG.append(s)


def retention_of(recs, mkey, ykey, nbins=NBINS):
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
    log("=== M26r3:在这批语料真正所处的尺度上标定留存 ===")
    log("  中介物用【真实的 P_model】,只合成结果。m26r2 的区间比这里强 31 倍,不可搬用。\n")
    recs, _ = M26.build()
    z = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {k: v for k, v in zip(z["keys"], z["mats"])}
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    log(f"纳入 {len(recs)}")

    OUT = {}
    for mkey, nm in (("nmut", "RibonanzaNet"), ("pmut", "ViennaRNA")):
        log(f"\n{'='*66}\n=== {nm} ===")
        # alpha: the contrast the estimator actually sees IN the mediator
        alpha, aw, _ = M26.matched(recs, "CG", "AU", sb, key=mkey)
        total, tw, _ = M26.matched(recs, "CG", "AU", sb, key="rf")
        # the mediator's real within-stratum slope on the real outcome
        gslope, _n = M26.slope_in(recs, sb, mkey)
        bound = abs(alpha * gslope / total) if total else float("nan")
        log(f"  分层后中介物的两臂差 alpha = {alpha:+.5f}   (权重 {aw:.0f})")
        log(f"  分层后结果的两臂差 total  = {total:+.5f}   (权重 {tw:.0f})")
        log(f"  中介物的层内斜率 gamma_real = {gslope:+.5f}")
        log(f"  **结构上限:经该中介物最多能中介 alpha*gamma/total = {100*bound:.1f}%**")
        OUT[nm] = {"alpha": round(alpha, 5), "total": round(total, 5),
                   "gamma_real": round(gslope, 5), "structural_bound": round(bound, 4)}

        rng = np.random.default_rng(31415)
        sig = np.array([1.0 if r["pt"] == "CG" else 0.0 for r in recs])
        mv = np.array([r[mkey] for r in recs])
        sd_e = float(np.array([r["rf"] for r in recs]).std())
        log(f"\n  合成结果 Ysyn = gamma*M_real + delta*1[CG] + N(0,{sd_e:.4f})")
        log(f"  {'f(真值中介)':>12}{'真值留存':>10}{'gamma':>10}{'实测留存':>10}{'CDE':>10}{'权重':>8}")
        rows = []
        for f in FRACS:
            gamma = f * total / alpha if alpha else 0.0
            delta = (1 - f) * total
            Y = gamma * mv + delta * sig + rng.normal(0, sd_e, len(recs))
            for r, y in zip(recs, Y):
                r["_y"] = float(y)
            ret, cde, w = retention_of(recs, mkey, "_y")
            rows.append({"f": f, "true_retention": round(1 - f, 4), "gamma": round(gamma, 3),
                         "measured_retention": (round(ret, 4) if ret is not None else None),
                         "weight": w})
            log(f"  {100*f:11.0f}%{100*(1-f):9.1f}%{gamma:10.2f}" +
                (f"{100*ret:9.1f}%{cde:10.4f}{w:8.0f}" if ret is not None
                 else f"{'重叠丧失':>10}{'-':>10}{w:8.0f}"))
        ok = [r for r in rows if r["measured_retention"] is not None]
        if len(ok) >= 3:
            t = np.array([r["true_retention"] for r in ok])
            m = np.array([r["measured_retention"] for r in ok])
            corr = float(np.corrcoef(t, m)[0, 1]); err = float(np.abs(t - m).mean())
            log(f"\n  真值 vs 实测:相关 {corr:+.4f}  平均绝对误差 {err:.4f}")
            OUT[nm].update({"corr": round(corr, 4), "mae": round(err, 4)})
            # invert: what true mediation is consistent with the committed measurement?
            meas = 0.963 if nm == "ViennaRNA" else 1.007
            close = min(ok, key=lambda r: abs(r["measured_retention"] - meas))
            log(f"  **已提交的实测留存 {100*meas:.1f}% 最接近 f={100*close['f']:.0f}% 那一档"
                f"(该档实测 {100*close['measured_retention']:.1f}%)**")
            OUT[nm]["committed_maps_to_f"] = close["f"]
        OUT[nm]["sweep"] = rows

    json.dump(OUT, open(os.path.join(HERE, "m26r3_realistic_calibration.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m26r3_realistic_calibration.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m26r3_realistic_calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
