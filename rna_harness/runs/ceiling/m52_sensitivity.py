#!/usr/bin/env python3
"""M52: the sensitivity analysis the Discussion says is already reported.

Declaration m52_sensitivity_declaration.json, frozen and pushed BEFORE this file existed.
Offline, $0.

THE DEFECT
----------
Section 5.2 states Assumption 1, says it cannot be tested, and reports a controlled-truth
benchmark -- explicitly distinguishing that benchmark from a sensitivity analysis in the same
subsection ("no benchmark built from a confounder we chose could"). The Discussion then writes
"the honest bound is the sensitivity analysis already reported". There is none.

THE ALGEBRA
-----------
eq:est is linear in the outcome. If an omitted U enters the outcome with coefficient lambda,
    alpha_hat(y + lambda*u) = alpha_hat(y) + lambda * g_u,
where g_u is the SAME weighted arm-gap functional applied to u. With lambda estimated as the
within-stratum slope beta_u, the bias is beta_u * g_u -- the paper's own controlled-truth form.
The product is invariant to rescaling u, so covariates need no standardisation to be compared.

A consequence that falls out and matters more than the numbers: the strata are keyed on the design,
so any covariate constant within a design has g_u = 0 exactly. No design-level confounder, measured
or not, can bias this estimate. Assumption 1 only ever had to cover covariates varying BETWEEN
pairs inside one design.

Estimator, arm gap, slope, optimisation, bootstrap: written here. M26 supplies records only.
"""
from __future__ import annotations

import collections, json, math, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

DECL = os.path.join(HERE, "m52_sensitivity_declaration.json")
OUTJ = os.path.join(HERE, "m52_sensitivity.json")
CACHE = os.path.join(HERE, "m52_recs_cache.json")
D = json.load(open(DECL))
SEED, NBOOT, NRAND = 20260901, 2000, 20000
A_ARM, B_ARM = "CG", "AU"
LOG = []

# published, for V3 -- reproduced before any sensitivity number is read.
#
# DECLARATION ERRATUM (recorded, not edited away). V3 in m52_sensitivity_declaration.json names
# "cell count 144" as a number to reproduce. 144 is jdg{resCells}: the number of DESIGN CELLS in
# the per-cell sign test, a different unit from the strata (design, min(depth,3), crossed) the
# estimator sums over. The first run of this file voided on K1 because it compared the stratum
# count against 144. The manuscript publishes no macro for the stratum count, so V3 checks the two
# numbers it does publish -- the contrast and the matched weight -- and the stratum count is
# cross-checked against jdg{selNStrata}, which M30 computes independently on the same strata.
PUB_ALPHA, PUB_WEIGHT, PUB_CELLS = 0.0539, 902, 144
PUB_STRATA = 553   # jdg{selNStrata}, from M30, a different run over the same stratification
PUB_BETA, PUB_GAP, PUB_PROD = 0.06759, -0.3487, -0.0236
PUB_SD_RF = 0.219  # M50's pooled within-stratum SD of the outcome


def log(s=""):
    print(s, flush=True); LOG.append(s)


def _plain(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    raise TypeError(repr(o))


def save(OUT):
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False, default=_plain)
    log(f"\n写出 {OUTJ}")


SFN = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])


# --------------------------------------------------------------- estimator, written here
def arm_gap(rs, key):
    """eq:est applied to an arbitrary column: sum_s w_s (mean_A - mean_B) / sum_s w_s.

    For key='rf' this IS the contrast. For a covariate it is that covariate's g_u, and the two
    are the same function by construction -- which is exactly why the injection check works."""
    by = collections.defaultdict(lambda: ([], []))
    for r in rs:
        if r["pt"] == A_ARM:
            by[r["_s"]][0].append(float(r[key]))
        elif r["pt"] == B_ARM:
            by[r["_s"]][1].append(float(r[key]))
    num = den = 0.0; used = 0
    for a, b in by.values():
        if a and b:
            w = min(len(a), len(b))
            num += w * (sum(a) / len(a) - sum(b) / len(b)); den += w; used += 1
    return (num / den if den else None), den, used


def within_slope(rs, xk, yk="rf"):
    """within-stratum pooled slope of y on x."""
    by = collections.defaultdict(list)
    for r in rs:
        by[r["_s"]].append(r)
    sxy = sxx = 0.0
    for v in by.values():
        if len(v) < 2:
            continue
        x = np.array([float(r[xk]) for r in v]); y = np.array([float(r[yk]) for r in v])
        x -= x.mean(); y -= y.mean()
        sxy += float(x @ y); sxx += float(x @ x)
    return (sxy / sxx) if sxx > 0 else None


def within_moments(rs, keys, yk="rf", scale=None):
    """within-stratum cross-product matrix Sigma and cross-product vector c, on ONE pass.

    Both use the same normaliser (it cancels in every quantity below), so none is applied.
    `scale` divides each column before accumulating; the induced bias beta_u*g_u is invariant to
    column scaling, so this only fixes conditioning -- and V6 checks that invariance rather than
    assuming it."""
    by = collections.defaultdict(list)
    for r in rs:
        by[r["_s"]].append(r)
    p = len(keys)
    sc = np.ones(p) if scale is None else np.asarray(scale, float)
    S = np.zeros((p, p)); c = np.zeros(p)
    for v in by.values():
        if len(v) < 2:
            continue
        X = np.array([[float(r[k]) for k in keys] for r in v]) / sc
        y = np.array([float(r[yk]) for r in v])
        X = X - X.mean(0); y = y - y.mean()
        S += X.T @ X; c += X.T @ y
    return S, c


def within_var(rs, keys):
    """diagonal of the within-stratum cross-product matrix, per column."""
    by = collections.defaultdict(list)
    for r in rs:
        by[r["_s"]].append(r)
    out = np.zeros(len(keys))
    for v in by.values():
        if len(v) < 2:
            continue
        X = np.array([[float(r[k]) for k in keys] for r in v])
        X = X - X.mean(0)
        out += (X * X).sum(0)
    return out


def best_index(S, c, d):
    """closed form: max over unit-within-variance indices of (w'c)(w'd)/(w'Sw).

    With v = S^{1/2}w, a = S^{-1/2}c, b = S^{-1/2}d, this is max over ||v||=1 of (v'a)(v'b),
    which equals (|a||b| + a'b)/2, the minimum being (a'b - |a||b|)/2."""
    ev, EV = np.linalg.eigh(S)
    Sm = EV @ np.diag(ev ** -0.5) @ EV.T
    a_ = Sm @ c; b_ = Sm @ d
    na, nb, ab = float(np.linalg.norm(a_)), float(np.linalg.norm(b_)), float(a_ @ b_)
    return (na * nb + ab) / 2.0, (ab - na * nb) / 2.0, a_, b_, Sm, na, nb


def pct(v, lo=2.5, hi=97.5):
    v = np.sort(np.asarray(v, float))
    return [round(float(v[int(lo / 100 * len(v))]), 5), round(float(v[int(hi / 100 * len(v))]), 5)]


# --------------------------------------------------------------- covariates
COVS = [
    # key,            label,                                     in_stratification
    ("cvDepth",       "helix depth (raw, uncapped)",             "capped at 3 in the strata"),
    ("cvCrossed",     "crossing",                                "yes, exactly"),
    ("cvLen",         "design length",                           "design-level, so exactly zero"),
    ("cvGC",          "GC fraction of the design",               "design-level, so exactly zero"),
    ("cvPos",         "fractional position of the pair",         "no"),
    ("cvSpan",        "span of the pair (b - a)",                "no"),
    ("cvLocGC",       "local GC fraction, +/-10 nt window",      "no"),
    ("cvPwt",         "wild-type pair probability",              "no"),
]
CKEYS = [k for k, _l, _s in COVS]


def add_covariates(recs):
    for r in recs:
        w = r["wt_seq"]; a, b = int(r["a"]), int(r["b"]); L = len(w)
        r["cvDepth"] = float(r["depth"])
        r["cvCrossed"] = 1.0 if r["crossed"] else 0.0
        r["cvLen"] = float(L)
        r["cvGC"] = float(sum(ch in "GC" for ch in w)) / L
        r["cvPos"] = 0.5 * (a + b) / max(L - 1, 1)
        r["cvSpan"] = float(b - a)
        lo, hi = max(0, a - 10), min(L, b + 11)
        win = w[lo:hi]
        r["cvLocGC"] = float(sum(ch in "GC" for ch in win)) / max(len(win), 1)
        r["cvPwt"] = float(r["pwt"])
    return recs


# --------------------------------------------------------------- records
def load_records():
    st = os.stat(M26.SRC); stamp = {"size": st.st_size, "mtime": int(st.st_mtime)}
    if os.path.exists(CACHE):
        c = json.load(open(CACHE))
        if c.get("stamp") == stamp:
            log(f"  记录缓存命中({len(c['recs'])} 条),源文件未变")
            return [{**r, "design": tuple(r["design"])} for r in c["recs"]]
    log("  无可用缓存,重建记录(ViennaRNA 配分函数,数分钟)")
    recs, _ = M26.build()
    keep = ("design", "puzzle", "pt", "rf", "depth", "crossed", "pwt", "wt_seq", "a", "b")
    recs = [{k: r[k] for k in keep} for r in recs]
    json.dump({"stamp": stamp, "recs": [{**r, "design": list(r["design"])} for r in recs]},
              open(CACHE, "w"))
    return recs


def main():
    t0 = time.time()
    log("=== M52:给 Assumption 1 一个界,因为讨论章节声称已经有一个 ===")
    log("  声明已冻结并推送于本文件存在之前。全离线,$0。")
    recs = add_covariates(load_records())
    for r in recs:
        r["_s"] = SFN(r)
    log(f"  干预 {len(recs)}  ({time.time()-t0:.0f}s)")
    OUT = {"declaration": os.path.basename(DECL),
           "cost": "offline, $0 -- no API, no GPU, no download.",
           "n": len(recs), "covariates": [{"key": k, "label": l, "in_strata": s}
                                          for k, l, s in COVS],
           "declaration_erratum": {
               "where": "V3_reverse_control",
               "text": "V3 names 'cell count 144' among the numbers to reproduce. 144 is the "
                       "number of design cells in the per-cell sign test (jdg{resCells}); the "
                       "estimator sums over strata (design, min(depth,3), crossed), of which 553 "
                       "supply both arms. The first run voided on K1 for comparing the two. The "
                       "declaration is left as written. V3 now checks the contrast and the "
                       "matched weight, both published, and cross-checks the stratum count "
                       "against jdg{selNStrata}=553, which M30 computes independently.",
               "consequence": "none for any staked prediction; V3 is strictly harder than "
                              "declared, since 553 was not a number this run could tune to."}}

    # ---------------------------------------------------------- V, before any arm
    log("\n=== V 验证(先于任何 arm)===")
    alpha, weight, cells = arm_gap(recs, "rf")
    log(f"  V3 对比 {alpha:+.5f}(论文 {PUB_ALPHA:+.4f})  权重 {weight:.0f}(论文 {PUB_WEIGHT})")
    log(f"  V3 供两臂的层 {cells}(M30 独立算得 {PUB_STRATA};论文的 144 是设计格,不是层 —— "
        f"声明的 V3 措辞有误,已记录 declaration_erratum)")
    v3 = (abs(alpha - PUB_ALPHA) < 5e-5 and abs(weight - PUB_WEIGHT) < 0.5
          and cells == PUB_STRATA)
    prod = PUB_BETA * PUB_GAP
    log(f"  V3 基准 beta {PUB_BETA:+.5f} x gap {PUB_GAP:+.4f} = {prod:+.5f}(论文 {PUB_PROD:+.4f})")
    v3b = abs(prod - PUB_PROD) < 5e-5
    if not (v3 and v3b):
        OUT["VOID"] = "K1: V3 未复现已发表的数"; save(OUT); return 1
    OUT["n_strata_both_arms"] = cells
    OUT["matched_weight"] = round(weight, 1)
    if cells < 100:
        OUT["VOID"] = "K4: 供两臂的层不足 100"; save(OUT); return 1

    # V1 zero point -- a design-level covariate has an EXACTLY zero arm gap
    g_len, _w, _u = arm_gap(recs, "cvLen")
    g_gc, _w, _u = arm_gap(recs, "cvGC")
    log(f"  V1 设计级协变量的臂间差  长度 {g_len:.3e}  GC {g_gc:.3e}   (层键含 design,故应恒为 0)")
    v1 = abs(g_len) < 1e-12 and abs(g_gc) < 1e-12

    # V2 equivalent form -- injection. Solve lambda so that the estimate is nulled, inject, recompute.
    rng = np.random.default_rng(SEED)
    u = rng.normal(size=len(recs))
    for r, val in zip(recs, u):
        r["_u"] = float(val)
    g_u, _w, _u2 = arm_gap(recs, "_u")
    lam = -alpha / g_u
    for r in recs:
        r["_yinj"] = r["rf"] + lam * r["_u"]
    a_inj, _w, _u3 = arm_gap(recs, "_yinj")
    log(f"  V2 注入:g_u {g_u:+.6f}  解出 lambda {lam:+.6f}  注入后对比 {a_inj:+.3e}")
    v2 = abs(a_inj) < 1e-9

    # V1b injection with zero arm gap leaves the estimate untouched
    for r in recs:
        r["_u0"] = r["_u"]
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["_s"], r["pt"])].append(r)
    for v in by.values():
        m = sum(r["_u0"] for r in v) / len(v)
        for r in v:
            r["_u0"] -= m           # centred within stratum x arm -> gap identically zero
    g_u0, _w, _u4 = arm_gap(recs, "_u0")
    for r in recs:
        r["_yinj0"] = r["rf"] + 3.7 * r["_u0"]
    a_inj0, _w, _u5 = arm_gap(recs, "_yinj0")
    log(f"  V1 零臂间差注入:g {g_u0:.3e}  注入后 {a_inj0:+.9f} vs 原 {alpha:+.9f}  "
        f"|Δ| {abs(a_inj0-alpha):.3e}")
    v1b = abs(g_u0) < 1e-12 and abs(a_inj0 - alpha) < 1e-12

    # V4 linearity in lambda and oddness in U
    pts = []
    for L in (0.0, 1.0, 2.0, -3.0):
        for r in recs:
            r["_t"] = r["rf"] + L * r["_u"]
        pts.append(arm_gap(recs, "_t")[0])
    lin = max(abs(pts[i] - (pts[0] + L * g_u)) for i, L in enumerate((0.0, 1.0, 2.0, -3.0)))
    for r in recs:
        r["_un"] = -r["_u"]
    g_neg, _w, _u6 = arm_gap(recs, "_un")
    log(f"  V4 线性残差 {lin:.3e}   取负 {g_neg:+.6f} vs {-g_u:+.6f}  |Δ| {abs(g_neg+g_u):.3e}")
    v4 = lin < 1e-12 and abs(g_neg + g_u) < 1e-12

    # V5 the closed form is checked against random search AFTER A4 is computed (below)
    OUT["V"] = {"V1_zero_point": bool(v1 and v1b), "V2_injection": bool(v2),
                "V3_reverse_control": bool(v3 and v3b), "V4_linear_and_odd": bool(v4)}
    if not (v1 and v1b and v2 and v4):
        OUT["VOID"] = "K2: V1/V2/V4 未通过"; save(OUT); return 1
    log("  V1-V4 通过 —— V5 在 A4 之后,现在开始读 arm")

    # ---------------------------------------------------------- A3 observed covariates
    log("\n=== A3 每个实测协变量:若被遗漏会引入多大偏差 ===")
    A3 = {}
    for k, lab, ins in COVS:
        g, _w, _u7 = arm_gap(recs, k)
        b = within_slope(recs, k)
        # A column with no within-stratum variance has no within-stratum slope, and its arm gap
        # is exactly zero for the same reason. Its bias is 0 by construction, not unknown.
        degenerate = (b is None)
        bias = 0.0 if degenerate else b * g
        A3[k] = {"label": lab, "in_strata": ins, "g": round(g or 0.0, 8),
                 "beta": None if degenerate else round(b, 6),
                 "bias": round(bias, 8),
                 "zero_by_construction": bool(degenerate)}
        log(f"  {lab:42s} g {g or 0.0:+.5f}  "
            + ("beta   ---     bias  0 (层内无方差,恒为 0)" if degenerate
               else f"beta {b:+.5f}  bias {bias:+.6f}"
                    + ("   |bias| >= alpha  ***" if abs(bias) >= alpha else "")))
    A3_max = max(abs(v["bias"]) for v in A3.values())
    A3_arg = max(A3.values(), key=lambda v: abs(v["bias"]))["label"]
    log(f"  最大 |bias| = {A3_max:.6f}({A3_arg}),观测对比 {alpha:.6f}")
    OUT["A3_observed_covariates"] = {"per_covariate": A3, "max_abs_bias": round(A3_max, 6),
                                     "argmax": A3_arg, "alpha": round(alpha, 6)}

    # ---------------------------------------------------------- A4 best linear combination
    log("\n=== A4 所有实测协变量的最佳线性组合(闭式)===")
    var_all = within_var(recs, CKEYS)
    deg = [k for k, vv in zip(CKEYS, var_all) if vv <= 1e-12 * float(var_all.max())]
    use = [k for k in CKEYS if k not in deg]
    log(f"  层内方差恒为 0 的列(设计级或已在层键里),不构成方向,剔除:{deg}")
    sc = np.sqrt(within_var(recs, use) / max(len(recs) - 1, 1))
    S, c = within_moments(recs, use, scale=sc)
    # The scaled column is x/sc, so its arm gap is d/sc. An earlier version multiplied here, and
    # V6 -- which exists for exactly this -- caught it: the two scalings disagreed by 30x.
    d = np.array([arm_gap(recs, k)[0] or 0.0 for k in use]) / sc
    w_ev = np.linalg.eigvalsh(S)
    cond = float(w_ev.max() / w_ev.min()) if w_ev.min() > 0 else float("inf")
    log(f"  标度归一后特征值 {w_ev.min():.4e} .. {w_ev.max():.4e}  条件数 {cond:.3e}")
    if not (w_ev.min() > 0 and cond < 1e10):
        OUT["A4"] = {"refused": "K3: within-stratum cross-product matrix singular after "
                                "removing the columns with no within-stratum variance"}
        A4_max = None
        log("  K3:仍然奇异 —— 按声明拒绝报告,不做正则化")
    else:
        hi, lo, a_, b_, Sm, na, nb = best_index(S, c, d)
        A4_max = max(abs(hi), abs(lo))
        log(f"  闭式  max {hi:+.6f}   min {lo:+.6f}   max|bias| {A4_max:.6f}")
        # V6 equivalent form: the answer must not depend on the column scaling
        S2, c2 = within_moments(recs, use)
        d2 = np.array([arm_gap(recs, k)[0] or 0.0 for k in use])
        hi2, lo2, *_ = best_index(S2, c2, d2)
        v6 = abs(hi2 - hi) < 1e-6 * max(1.0, abs(hi)) and abs(lo2 - lo) < 1e-6 * max(1.0, abs(lo))
        log(f"  V6 未归一标度重算  max {hi2:+.6f}  min {lo2:+.6f}  -> {'通过' if v6 else '失败'}")
        # V5 random search may approach but never exceed
        rr = np.random.default_rng(SEED + 5)
        V = rr.normal(size=(NRAND, len(use)))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        vals = (V @ a_) * (V @ b_)
        rmax, rmin = float(vals.max()), float(vals.min())
        v5 = (rmax <= hi + 1e-9) and (rmin >= lo - 1e-9)
        log(f"  V5 随机 {NRAND} 个单位方向  max {rmax:+.6f} <= {hi:+.6f}   "
            f"min {rmin:+.6f} >= {lo:+.6f}  -> {'通过' if v5 else '失败'}")
        OUT["V"]["V5_closed_form_not_exceeded"] = bool(v5)
        OUT["V"]["V6_scale_invariant"] = bool(v6)
        if not (v5 and v6):
            OUT["VOID"] = "K2: V5/V6 未通过"; save(OUT); return 1
        v_hi = a_ / na + b_ / nb
        v_hi /= np.linalg.norm(v_hi)
        w_hi = (Sm @ v_hi) / sc
        w_hi = w_hi / np.linalg.norm(w_hi)
        OUT["A4"] = {"keys": use, "dropped_no_within_variance": deg,
                     "max": round(hi, 6), "min": round(lo, 6),
                     "max_abs_bias": round(A4_max, 6),
                     "direction_of_maximiser": {k: round(float(x), 4) for k, x in zip(use, w_hi)},
                     "condition_number": cond}

    # ---------------------------------------------------------- A1/A2 the requirement
    log("\n=== A1/A2 一个未观测混杂需要多强 ===")
    K1 = alpha / abs(prod)
    diag = math.sqrt(abs(alpha))
    log(f"  A1 K1 = {alpha:.5f} / |{prod:.5f}| = {K1:.4f}"
        f"   (置换掉的那个混杂如果不被条件化,会带来 {abs(prod):.4f} 的偏差)")
    log(f"  A2 对角要求 beta_u = g_u = sqrt({alpha:.5f}) = {diag:.5f}"
        f"   折成结局层内 SD:{diag/PUB_SD_RF:.4f} SD")

    # target-clustered bootstrap on K1: alpha recomputed in every draw
    pz = sorted({r["puzzle"] for r in recs})
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    rngb = np.random.default_rng(SEED + 1)
    ks, als, a3s, a4s = [], [], [], []
    for _ in range(NBOOT):
        ix = rngb.integers(0, len(pz), len(pz))
        s = [r for q in ix for r in byp[pz[q]]]
        av, den, _u8 = arm_gap(s, "rf")
        if av is None or not den:
            continue
        als.append(av); ks.append(av / abs(prod))
        # A3's maximum inside the same draw, so the comparison is paired
        m = 0.0
        for k in CKEYS:
            g, _w, _u9 = arm_gap(s, k); b = within_slope(s, k)
            if g is not None and b is not None:
                m = max(m, abs(b * g))
        a3s.append(m)
    log(f"  A1 K1 自助 {len(ks)} 次  {K1:.4f}  95% {pct(ks)}")
    ci_k = pct(ks)
    OUT["A1_multiplier"] = {"alpha": round(alpha, 6), "benchmark_bias": round(prod, 6),
                            "K1": round(K1, 4), "ci95": ci_k,
                            "ci_excludes_one": bool(ci_k[0] > 1.0 or ci_k[1] < 1.0),
                            "ci_above_one": bool(ci_k[0] > 1.0),
                            "alpha_ci95": pct(als)}
    OUT["A2_diagonal"] = {"required_beta_eq_g": round(diag, 5),
                          "in_outcome_within_sd": round(diag / PUB_SD_RF, 4),
                          "outcome_within_sd": PUB_SD_RF}
    # paired margin: alpha minus the largest observed-covariate bias, same draw
    marg = [a - b for a, b in zip(als, a3s)]
    ci_m = pct(marg)
    log(f"  A3 配对余量 alpha - max|bias_c| = {alpha - A3_max:+.6f}  95% {ci_m}  "
        f"{'不含 0' if (ci_m[0] > 0 or ci_m[1] < 0) else '含 0'}")
    OUT["A3_observed_covariates"]["paired_margin"] = {
        "point": round(alpha - A3_max, 6), "ci95": ci_m,
        "ci_excludes_zero": bool(ci_m[0] > 0 or ci_m[1] < 0)}

    # ---------------------------------------------------------- prereg scoring
    S1 = bool(K1 > 1.0)
    S2 = bool(ci_k[0] > 1.0)
    S3 = bool(A3_max < alpha)
    S4 = bool(A4_max is not None and A4_max < alpha)
    log("\n=== 预注册记分 ===")
    for sid, ok_, det in (
            ("S1", S1, f"K1 = {K1:.4f}"),
            ("S2", S2, f"K1 95% {ci_k}"),
            ("S3", S3, f"max|bias_c| {A3_max:.6f} vs alpha {alpha:.6f}({A3_arg})"),
            ("S4", S4, f"最佳线性组合 {('%.6f' % A4_max) if A4_max is not None else '拒绝'} "
                       f"vs alpha {alpha:.6f}")):
        log(f"  {sid}  {'成立' if ok_ else '**失败**'}   {det}")
    OUT["staked"] = {
        "S1": {"text": D["staked_predictions"][0]["text"], "pass": S1, "detail": f"K1 = {K1:.4f}"},
        "S2": {"text": D["staked_predictions"][1]["text"], "pass": S2, "detail": f"95% {ci_k}"},
        "S3": {"text": D["staked_predictions"][2]["text"], "pass": S3,
               "detail": f"max |beta_c g_c| = {A3_max:.6f} ({A3_arg}) vs alpha {alpha:.6f}"},
        "S4": {"text": D["staked_predictions"][3]["text"], "pass": S4,
               "detail": (f"{A4_max:.6f} vs alpha {alpha:.6f}" if A4_max is not None else "refused")},
    }
    n_ok = sum(x["pass"] for x in OUT["staked"].values())
    OUT["verdict"] = ("unmeasured_confounder_must_exceed_the_one_conditioned_on"
                      if (S1 and S2) else "bound_does_not_resolve")
    log(f"\n  裁定:{OUT['verdict']}   {n_ok}/4 成立   用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
