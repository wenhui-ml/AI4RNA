#!/usr/bin/env python3
"""M62: the injection calibration of Section 5.5, applied to the route the paper kept.

Declaration: m62_product_route_calibration_declaration.json, frozen, committed and pushed BEFORE
this file existed. Departures go in m62_declaration_erratum.json; the declaration is not edited.

The estimator under test is m20's mediation(), called unchanged. Only the outcome is synthetic:

    Ysyn = h_true(depth) + delta * 1[CG] + N(0, sigma),   h_true(m) = a*m + c*r0(m)

with r0 the part of m^2 orthogonal to m under the pooled depth law, so the least-squares slope of
h_true on m is exactly a and the TRUE linear share of the mediated part is known before any
outcome is drawn.

COST: $0, offline, CPU only.
"""
from __future__ import annotations
import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m16_rescue as M16
import m20_designer_confounding as M20

SEED, DRAWS = 20260901, 40
FRACS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)
LAMBDAS = (1.00, 0.90, 0.75, 0.60, 0.45, 0.30)
PUBLISHED = 0.7121
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def sigma_within(recs):
    """The corpus's own within-stratum residual SD. Measured, not chosen."""
    by = collections.defaultdict(list)
    for r in recs:
        if r["pt"] in ("CG", "AU"):
            by[(r["design"], min(r["depth"], 3), r["crossed"])].append(r["rf"])
    v = []
    for _s, g in by.items():
        if len(g) >= 2:
            a = np.asarray(g, float); v.append(a - a.mean())
    return float(np.concatenate(v).std(ddof=1))


def marginal_gap(recs, key):
    """The arm gap in an arbitrary per-record quantity, on m20's marginal weighting."""
    def gap(vals):
        by = collections.defaultdict(lambda: {"CG": [], "AU": []})
        for r in recs:
            if r["pt"] in ("CG", "AU"):
                by[r["design"]][r["pt"]].append(vals[id(r)])
        num = den = 0.0
        for _s, d in by.items():
            if not d["CG"] or not d["AU"]:
                continue
            w = min(len(d["CG"]), len(d["AU"]))
            num += w * (float(np.mean(d["CG"])) - float(np.mean(d["AU"]))); den += w
        return num / den if den else float("nan")
    return gap({id(r): key(r) for r in recs})


def run(recs, m, r0, a, c, delta, sigma, rng):
    """Draw one synthetic outcome and hand it to m20's estimator unchanged."""
    y = a * m + c * r0 + delta * np.array([1.0 if r["pt"] == "CG" else 0.0 for r in recs]) \
        + rng.normal(0.0, sigma, len(recs))
    sub = [{**r, "rf": float(v)} for r, v in zip(recs, y)]
    return M20.mediation(sub)


def qs(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not len(v):
        return (float("nan"),) * 3
    return (float(np.median(v)), float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75)))


def main():
    rng = np.random.default_rng(SEED)
    OUT = {"seed": SEED, "draws": DRAWS}
    log("=== M62:把 §5.5 的注入标定,施加到论文留下的那条路线上 ===")
    log("  声明冻结在代码之前。估计量是 m20 的 mediation(),原样调用。全离线,$0。\n")

    recs = [r for r in M16.build() if r["pt"] in ("CG", "AU")]
    log(f"记录 {len(recs)}  设计格 {len({r['design'] for r in recs})}  靶标 {len({r['puzzle'] for r in recs})}")

    # ---------------------------------------------------------------- V
    V = M20.mediation(recs)
    ok_v = abs(V["mediated_fraction"] - PUBLISHED) <= 0.005
    log("\n【V】先复算已发表的分解")
    log(f"  边际 {V['marginal']:+.5f}  配平 {V['matched']:+.5f}  衰减 {V['observed_attenuation']:+.5f}")
    log(f"  g {V['gap_g']:+.4f}  β {V['beta_depth']:+.5f}  βg {V['predicted_attenuation']:+.5f}"
        f"  比值 {V['mediated_fraction']:.4f}  (已发表 {PUBLISHED})")
    log(f"  -> {'通过' if ok_v else '**不通过,下面不读**'}")
    OUT["V"] = {**V, "reproduces_published": bool(ok_v)}

    m = np.array([float(r["depth"]) for r in recs])
    mc = m - m.mean()
    q = m ** 2
    r0 = q - q.mean() - (float((mc * (q - q.mean())).sum() / (mc * mc).sum())) * mc
    r0 = r0 / r0.std()                       # scale only; orthogonality to m is preserved
    sigma = sigma_within(recs)
    total = V["matched"]
    g = V["gap_g"]
    for r, v in zip(recs, r0):
        r["_r0"] = float(v)
    dr0 = marginal_gap(recs, lambda r: r["_r0"])
    log(f"\n  层内残差 SD {sigma:.4f}(实测)  总量取语料自己的 {total:+.5f}")
    # The separation Section 5.5 identifies as the governing quantity, computed for THIS mediator.
    # Descriptive; it stakes nothing and is reported so the two calibrations can be compared.
    sep = abs(g) / float(m.std())
    log(f"  depth 的臂间差 g {g:+.4f};正交非线性分量 r0 的臂间差 {dr0:+.4f}")
    log(f"  depth 把两臂分开 {sep:.3f} 个合并 SD(§5.5 标定所用的中介物是 0.053)")
    OUT["setup"] = {"sigma": round(sigma, 5), "total": total, "g": g, "delta_r0": round(dr0, 5),
                    "depth_pooled_sd": round(float(m.std()), 4),
                    "arm_separation_pooled_sd": round(sep, 4),
                    "s55_mediator_separation": 0.053}

    # ---------------------------------------------------------------- A
    log("\n【A】线性 h,扫注入的中介分数 f。真值两重:βg/total 应读出 f,印着的比值应读出 1")
    log(f"  {'f':>6}{'βg/total 中位':>15}{'IQR':>20}{'印着的比值':>13}{'IQR':>20}{'条件路线留存':>14}")
    A = {}
    for f in FRACS:
        a = f * total / g if g else 0.0
        delta = (1.0 - f) * total
        fr, ra, ret = [], [], []
        for k in range(DRAWS):
            d = run(recs, m, r0 * 0.0, a, 0.0, delta, sigma, np.random.default_rng(SEED + 1000 * k + int(f * 97)))
            fr.append(d["predicted_attenuation"] / total)
            ra.append(d["mediated_fraction"])
            ret.append(d["matched"] / d["marginal"] if d["marginal"] else np.nan)
        mf, lf, hf = qs(fr); mr, lr, hr = qs(ra); mt, _l, _h = qs(ret)
        A[f"{f:.2f}"] = {"f": f, "a": round(a, 5), "delta": round(delta, 5),
                         "frac_median": round(mf, 4), "frac_iqr": [round(lf, 4), round(hf, 4)],
                         "ratio_median": round(mr, 4), "ratio_iqr": [round(lr, 4), round(hr, 4)],
                         "cond_route_median": round(mt, 4)}
        log(f"  {f:>6.2f}{mf:>15.4f}   [{lf:+.4f}, {hf:+.4f}]{mr:>13.4f}   [{lr:+.4f}, {hr:+.4f}]"
            f"{mt:>14.3f}")
    span_read = A["1.00"]["frac_median"] - A["0.00"]["frac_median"]
    span_true = 1.0 - 0.0
    A["span_readback"] = round(span_read, 4)
    A["span_ratio"] = round(span_read / span_true, 4)
    log(f"  -> 读回跨度 {span_read:.4f} / 真值跨度 {span_true:.2f} = {100*span_read:.1f}%"
        f"   (§5.5 的条件路线是 12.9% 与 28.7%)")
    OUT["A_fraction_sweep"] = A

    # ---------------------------------------------------------------- B
    log("\n【B】把中介分数固定在语料自己的水平,扫真实的线性份额 λ")
    f0 = PUBLISHED * V["observed_attenuation"] / total if total else 0.0
    f0 = abs(f0)
    log(f"  语料自己的中介分数取 |βg/total| = {f0:.4f}")
    log(f"  {'λ 真值':>8}{'a':>10}{'c':>10}{'印着的比值 中位':>18}{'IQR':>20}{'绝对误差':>10}")
    B = {}
    for lam in LAMBDAS:
        a = f0 * total / g if g else 0.0
        # lam = a*g / (a*g + c*dr0)  =>  c = a*g*(1-lam)/(lam*dr0)
        c = a * g * (1.0 - lam) / (lam * dr0) if (lam and dr0) else 0.0
        delta = total - (a * g + c * dr0)
        ra = []
        for k in range(DRAWS):
            d = run(recs, m, r0, a, c, delta, sigma,
                    np.random.default_rng(SEED + 7717 * k + int(lam * 313)))
            ra.append(d["mediated_fraction"])
        mr, lr, hr = qs(ra)
        B[f"{lam:.2f}"] = {"lambda_true": lam, "a": round(a, 5), "c": round(c, 5),
                           "ratio_median": round(mr, 4),
                           "ratio_iqr": [round(lr, 4), round(hr, 4)],
                           "abs_error": round(abs(mr - lam), 4)}
        log(f"  {lam:>8.2f}{a:>10.5f}{c:>10.5f}{mr:>18.4f}   [{lr:+.4f}, {hr:+.4f}]"
            f"{abs(mr-lam):>10.4f}")
    lt = np.array([B[f"{x:.2f}"]["lambda_true"] for x in LAMBDAS])
    lr_ = np.array([B[f"{x:.2f}"]["ratio_median"] for x in LAMBDAS])
    rk = lambda v: np.argsort(np.argsort(v))
    sp = float(np.corrcoef(rk(lt), rk(lr_))[0, 1])
    B["spearman"] = round(sp, 4)
    log(f"  -> 读回与真值的 Spearman {sp:+.4f}")
    # Invert the curve at the corpus's own operating point: what true linear share would produce
    # the reading the paper prints? Descriptive, and monotone by S5, so the inversion is defined.
    o = np.argsort(lr_)
    inv = float(np.interp(PUBLISHED, lr_[o], lt[o]))
    B["published_reading"] = PUBLISHED
    B["implied_true_lambda"] = round(inv, 4)
    B["bias_at_published"] = round(PUBLISHED - inv, 4)
    log(f"  -> 把印着的 {PUBLISHED:.4f} 放回这条曲线上,对应的真实线性份额约 {inv:.3f}"
        f"(高读了 {PUBLISHED-inv:+.3f})")
    OUT["B_linearity_sweep"] = B

    # ---------------------------------------------------------------- scoring
    log("\n=== 预注册记分(按冻结声明原文)===")
    S = {}
    S["S1"] = (A["0.00"]["frac_median"] < 0.10, f"f=0 读回 {A['0.00']['frac_median']:.4f}")
    S["S2"] = (A["1.00"]["frac_median"] > 0.80, f"f=1 读回 {A['1.00']['frac_median']:.4f}")
    S["S3"] = (A["span_ratio"] >= 0.60, f"跨度比 {A['span_ratio']:.4f}")
    hi = [k for k in A if k not in ("span_readback", "span_ratio") and A[k]["f"] >= 0.25]
    S["S4"] = (all(A[k]["ratio_median"] > 0.85 for k in hi),
               "f>=0.25 的各档比值 " + ", ".join(f"{A[k]['ratio_median']:.3f}" for k in sorted(hi)))
    S["S5"] = (sp >= 0.80, f"Spearman {sp:+.4f}")
    S["S6"] = (B["1.00"]["abs_error"] <= 0.20, f"λ=1 时误差 {B['1.00']['abs_error']:.4f}")
    S["S7"] = (ok_v, f"比值 {V['mediated_fraction']:.4f} vs {PUBLISHED}")
    npass = 0
    for k in sorted(S):
        ok, txt = S[k]; npass += ok
        log(f"  {k}  {'成立' if ok else '**失败**'}  {txt}")
    kills = []
    if not ok_v:
        kills.append("K0_verification_failed")
    if (not S["S3"][0]) and (not S["S1"][0] or not S["S2"][0]):
        kills.append("K1_product_route_unreadable")
    if (not S["S4"][0]) or (not S["S6"][0]):
        kills.append("K2_linear_share_not_recoverable")
    if S["S1"][0] and S["S2"][0] and S["S3"][0] and (not S["S5"][0] or not S["S6"][0]):
        kills.append("K3_only_the_fraction_survives")
    OUT["scoring"] = {k: {"pass": bool(v[0]), "note": v[1]} for k, v in S.items()}
    OUT["kill_conditions_fired"] = kills
    OUT["n_pass"] = npass
    log(f"\n  裁定:{kills if kills else '无 kill condition 触发'}   {npass}/7 成立")

    json.dump(OUT, open(os.path.join(HERE, "m62_product_route_calibration.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "m62_product_route_calibration.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m62_product_route_calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
