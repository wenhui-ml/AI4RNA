#!/usr/bin/env python3
"""M50: put section 5.4's four columns on a common scale before subtracting them.

Declaration m50_units_declaration.json, frozen and pushed BEFORE this file existed.
Offline, $0.

THE DEFECT
----------
Section 5.1 states the rule and obeys it: two readouts on different columns are not on one scale,
so it divides each contrast by that column's own pooled within-stratum SD before differencing.
Section 5.4 holds four columns -- the measured rescue factor and three predictors' pair
probabilities -- and subtracts across them, reporting the measurement as "indistinguishable from
the transformer" at -0.00662. Forty lines later the same section multiplies a probability contrast
by that predictor's within-stratum slope on the outcome before dividing by the measured contrast,
which is how it obtains 6.8% and 23.2%. The conversion factor is in the section; the subtraction
did not use it.

THE BRIDGE
----------
Because g = rho * SD_R / SD_p, the printed bound equals rho * (Delta_p/SD_p) / (alpha_R/SD_R).
The slope route IS the SD route attenuated by rho. They answer different questions and neither
corrects the other; what is neither question is the raw subtraction. This run computes both, and
checks the bridge by computing rho INDEPENDENTLY as the within-stratum correlation rather than
by rearranging the slope, so the check can fail on a coding error.

Estimator, SD, correlation, bootstrap: written here. M26 supplies records only.
"""
from __future__ import annotations

import collections, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

DECL = os.path.join(HERE, "m50_units_declaration.json")
OUTJ = os.path.join(HERE, "m50_units.json")
CACHE = os.path.join(HERE, "m50_recs_cache.json")
D = json.load(open(DECL))
SEED = 20260831
NBOOT = 2000
LOG = []

PUB = {"rf": 0.0539, "nmut": 0.0605, "pmut": -0.0304, "emut": -0.00569}
PUB_PDMN = -0.00662
PUB_BOUND = {"pmut": 0.068, "nmut": 0.232}
PUB_SLOPE = {"pmut": 0.12059, "nmut": 0.20692}
NAME = {"rf": "measurement", "nmut": "RibonanzaNet", "emut": "EternaFold", "pmut": "ViennaRNA"}


def log(s=""):
    print(s, flush=True); LOG.append(s)


def save(OUT):
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False)
    log(f"\n写出 {OUTJ}")


# --------------------------------------------------------------- estimator, written here
def matched(rs, key, A="CG", B="AU"):
    """eq:est -- weight min(n_A,n_B) inside each stratum, strata with one arm contribute nothing."""
    by = collections.defaultdict(lambda: ([], []))
    for r in rs:
        if r["pt"] == A:
            by[r["_s"]][0].append(r[key])
        elif r["pt"] == B:
            by[r["_s"]][1].append(r[key])
    num = den = 0.0
    for a, b in by.values():
        if a and b:
            w = min(len(a), len(b))
            num += w * (sum(a) / len(a) - sum(b) / len(b)); den += w
    return (num / den if den else None), den


def within_sd(rs, key):
    """pooled within-stratum SD, M36's definition: sqrt(sum SS / sum (n_s - 1))."""
    by = collections.defaultdict(list)
    for r in rs:
        by[r["_s"]].append(float(r[key]))
    num = den = 0.0
    for v in by.values():
        if len(v) < 2:
            continue
        a = np.asarray(v); num += float(((a - a.mean()) ** 2).sum()); den += len(a) - 1
    return float(np.sqrt(num / den)) if den else None


def within_slope_and_rho(rs, xk, yk="rf"):
    """within-stratum pooled slope AND the within-stratum correlation, computed separately so the
    bridge identity g = rho * SD_y/SD_x can fail on a coding error rather than hold by algebra."""
    by = collections.defaultdict(list)
    for r in rs:
        by[r["_s"]].append(r)
    sxy = sxx = syy = 0.0
    for v in by.values():
        if len(v) < 2:
            continue
        x = np.array([r[xk] for r in v], float); y = np.array([r[yk] for r in v], float)
        x -= x.mean(); y -= y.mean()
        sxy += float(x @ y); sxx += float(x @ x); syy += float(y @ y)
    if sxx <= 0 or syy <= 0:
        return None, None
    return sxy / sxx, sxy / np.sqrt(sxx * syy)


def pct(v, lo=2.5, hi=97.5):
    v = np.sort(np.asarray(v, float))
    return [round(float(v[int(lo / 100 * len(v))]), 5), round(float(v[int(hi / 100 * len(v))]), 5)]


# --------------------------------------------------------------- records
def load_records():
    src_stat = os.stat(M26.SRC)
    stamp = {"size": src_stat.st_size, "mtime": int(src_stat.st_mtime)}
    if os.path.exists(CACHE):
        c = json.load(open(CACHE))
        if c.get("stamp") == stamp:
            log(f"  记录缓存命中({len(c['recs'])} 条),源文件未变")
            return [{**r, "design": tuple(r["design"])} for r in c["recs"]]
    log("  无可用缓存,重建记录(ViennaRNA 配分函数,数分钟)")
    recs, _ = M26.build()
    zn = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {k: v for k, v in zip(zn["keys"], zn["mats"])}
    ze = np.load(os.path.join(HERE, "m32_eterna_cache.npz"), allow_pickle=True)
    # Index conventions differ and it is not cosmetic: the neural cache is a dense 0-based matrix,
    # EternaFold's posteriors are a SPARSE 1-based dict in which an absent pair means 0.0. M35
    # states this; reading it as a 0-based matrix raises KeyError on the first record, which is
    # how this was caught rather than silently returning the wrong pair.
    pe = {str(k): (v.item() if hasattr(v, "item") else v) for k, v in zip(ze["keys"], ze["maps"])}
    keep = []
    miss_n = miss_e = 0
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        if r["mut_seq"] not in pn:
            miss_n += 1; continue
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
        r["emut"] = (float(pe[r["mut_seq"]].get((i + 1, j + 1), 0.0))
                     if r["mut_seq"] in pe else None)
        if r["emut"] is None:
            miss_e += 1
        keep.append({k: r[k] for k in ("design", "puzzle", "pt", "rf", "depth", "crossed",
                                       "pmut", "nmut", "emut")})
    log(f"  记录 {len(keep)};神经缺 {miss_n},EternaFold 缺 {miss_e}")
    json.dump({"stamp": stamp, "recs": keep}, open(CACHE, "w"))
    return [{**r, "design": tuple(r["design"])} for r in keep]


def main():
    t0 = time.time()
    log("=== M50:把 §5.4 的四列放到同一把尺子上再相减 ===")
    log("  声明已冻结并推送于本文件存在之前。全离线,$0。")
    OUT = {"declaration": os.path.basename(DECL), "cost": D["cost"],
           "bridge": D["algebraic_bridge_established_before_the_run"]}

    recs = load_records()
    for r in recs:
        r["_s"] = (r["design"], min(r["depth"], 3), bool(r["crossed"]))
    COLS = ["rf", "nmut", "emut", "pmut"]
    have_e = [r for r in recs if r["emut"] is not None]
    log(f"  干预 {len(recs)},其中有 EternaFold 值的 {len(have_e)}  ({time.time()-t0:.0f}s)")
    OUT["n"] = {"all": len(recs), "with_eternafold": len(have_e)}

    def rs_for(col):
        return recs if col != "emut" else have_e

    # ================================================================ V verification
    log("\n=== V 验证(先于任何 arm)===")
    ok = {}

    # V1 zero point
    for r in recs:
        r["_const"] = 1.0
    c_sd = within_sd(recs, "_const")
    v_self = matched(recs, "rf")[0]
    ok["V1_constant_sd_is_zero"] = (c_sd is not None and c_sd == 0.0)
    log(f"  V1 常数列层内 SD = {c_sd}  -> {'通过' if ok['V1_constant_sd_is_zero'] else '**失败**'}")

    # V3 reverse control on the four published contrasts
    raw, sd, drift = {}, {}, {}
    for c in COLS:
        rs = rs_for(c)
        raw[c] = matched(rs, c)[0]
        sd[c] = within_sd(rs, c)
        drift[c] = abs(raw[c] - PUB[c])
        log(f"  V3 {NAME[c]:14s} 对比 {raw[c]:+.5f}  论文 {PUB[c]:+.5f}  |Δ| {drift[c]:.2e}   "
            f"层内 SD {sd[c]:.4f}")
    pdmn = raw["rf"] - raw["nmut"]
    drift["pdMN"] = abs(pdmn - PUB_PDMN)
    log(f"  V3 原始相减 measurement-RNet {pdmn:+.5f}  论文 {PUB_PDMN:+.5f}  |Δ| {drift['pdMN']:.2e}")
    ok["V3_reverse_control"] = max(drift.values()) < 1e-4
    OUT["A6_reverse_control"] = {"contrasts": {c: round(raw[c], 5) for c in COLS},
                                 "published": PUB, "raw_pdMN": round(pdmn, 5),
                                 "published_pdMN": PUB_PDMN,
                                 "max_abs_drift": round(float(max(drift.values())), 8)}
    OUT["A1_within_sd"] = {c: round(sd[c], 4) for c in COLS}

    # K3
    k3 = [c for c in COLS if sd[c] is not None and sd[c] < 0.01]
    OUT["K3_columns_refused"] = k3
    if k3:
        log(f"  **K3:{k3} 的层内 SD < 0.01,不做标准化**")

    # V2 equivalent form: divide after vs standardise the column first
    v2 = []
    for c in COLS:
        rs = rs_for(c)
        for r in rs:
            r["_z"] = r[c] / sd[c]
        v2.append(abs(raw[c] / sd[c] - matched(rs, "_z")[0]))
    ok["V2_equivalent_form"] = max(v2) < 1e-9
    OUT["V2_max_gap"] = float(max(v2))
    log(f"  V2 先除 vs 后除 max|Δ| = {max(v2):.2e}  -> "
        f"{'通过' if ok['V2_equivalent_form'] else '**失败**'}")

    # V4 bridge: slope and correlation computed separately must satisfy g = rho * SD_rf/SD_p,
    # and the published bound must equal rho * (std_p / std_rf)
    bridge = {}
    for c in ("pmut", "nmut"):
        g, rho = within_slope_and_rho(recs, c)
        lhs = g
        rhs = rho * sd["rf"] / sd[c]
        b_slope = raw[c] * g / raw["rf"]
        b_std = rho * (raw[c] / sd[c]) / (raw["rf"] / sd["rf"])
        bridge[c] = {"slope": round(g, 5), "published_slope": PUB_SLOPE[c],
                     "rho_within": round(rho, 5),
                     "g_vs_rho_SDratio": round(abs(lhs - rhs), 12),
                     "bound_slope_route": round(b_slope, 5),
                     "bound_via_rho_and_std": round(b_std, 5),
                     "published_bound": PUB_BOUND[c],
                     "bridge_gap": round(abs(b_slope - b_std), 12)}
        log(f"  V4 {NAME[c]:14s} g {g:+.5f}(论文 {PUB_SLOPE[c]:+.5f})  rho {rho:+.5f}  "
            f"|g - rho*SDr/SDp| {abs(lhs-rhs):.2e}")
        log(f"      上界 slope 路 {b_slope:+.5f}  rho x SD 路 {b_std:+.5f}  "
            f"论文 {PUB_BOUND[c]:+.4f}  桥 |Δ| {abs(b_slope-b_std):.2e}")
    OUT["A5_bridge"] = bridge
    ok["V4_bridge"] = max(v["bridge_gap"] for v in bridge.values()) < 1e-9 and \
        max(v["g_vs_rho_SDratio"] for v in bridge.values()) < 1e-9

    OUT["V"] = ok
    if not ok["V3_reverse_control"]:
        log("\n  **K1 触发:复现不出论文已印的对比 ⇒ 作废**")
        OUT["verdict"] = "VOID_K1_cannot_reproduce_published_contrasts"; save(OUT); return 1
    if not (ok["V1_constant_sd_is_zero"] and ok["V2_equivalent_form"] and ok["V4_bridge"]):
        log("\n  **K2 触发 ⇒ 作废**")
        OUT["verdict"] = "VOID_K2_verification_failed"; save(OUT); return 1
    log("  V 全部通过 —— 现在才读 arm")

    # ================================================================ A2/A3/A4 one bootstrap
    log("\n=== A2/A3/A4 目标聚类自助(四列同一次抽样)===")
    pz = sorted({r["puzzle"] for r in recs})
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    g_ = np.random.default_rng(SEED)
    std_draw = {c: [] for c in COLS}
    dif_std = {c: [] for c in COLS if c != "rf"}
    dif_slope = {c: [] for c in ("pmut", "nmut", "emut")}
    self_diff = []
    for _ in range(NBOOT):
        ix = g_.integers(0, len(pz), len(pz))
        S = [r for i in ix for r in byp[pz[i]]]
        Se = [r for r in S if r["emut"] is not None]
        vals, sds = {}, {}
        bad = False
        for c in COLS:
            rs = S if c != "emut" else Se
            v, _d = matched(rs, c); s = within_sd(rs, c)
            if v is None or not s or s <= 0:
                bad = True; break
            vals[c] = v; sds[c] = s
        if bad:
            continue
        for c in COLS:
            std_draw[c].append(vals[c] / sds[c])
        for c in COLS:
            if c != "rf":
                dif_std[c].append(vals["rf"] / sds["rf"] - vals[c] / sds[c])
        self_diff.append(vals["rf"] / sds["rf"] - vals["rf"] / sds["rf"])
        for c in ("pmut", "nmut", "emut"):
            rs = S if c != "emut" else Se
            gg, _rr = within_slope_and_rho(rs, c)
            if gg is not None:
                dif_slope[c].append(vals["rf"] - vals[c] * gg)

    ok["V1_self_difference_exactly_zero"] = bool(self_diff and max(map(abs, self_diff)) == 0.0)
    log(f"  V1 一列对自己的标准化差 max|.| = {max(map(abs, self_diff)) if self_diff else 'n/a'}")
    OUT["V"] = ok
    if not ok["V1_self_difference_exactly_zero"]:
        log("\n  **K2 触发 ⇒ 作废**"); OUT["verdict"] = "VOID_K2_self_difference"; save(OUT); return 1

    A2 = {c: {"standardised_contrast": round(raw[c] / sd[c], 5), "ci95": pct(std_draw[c]),
              "within_sd": round(sd[c], 4)} for c in COLS}
    OUT["A2_standardised"] = A2
    log("\n  A2 每列除以自己的层内 SD:")
    for c in COLS:
        log(f"    {NAME[c]:14s} {A2[c]['standardised_contrast']:+.5f} SD  {A2[c]['ci95']}")

    A3 = {}
    log("\n  A3 配对差(measurement − 预测器,标准化后,同一次抽样):")
    for c in ("nmut", "emut", "pmut"):
        pt = raw["rf"] / sd["rf"] - raw[c] / sd[c]
        ci = pct(dif_std[c])
        A3[c] = {"point": round(pt, 5), "ci95": ci, "ci_contains_zero": bool(ci[0] <= 0 <= ci[1])}
        log(f"    measurement − {NAME[c]:14s} {pt:+.5f} SD  {ci}"
            f"{'  含 0' if A3[c]['ci_contains_zero'] else '  不含 0'}")
    OUT["A3_paired_standardised_difference"] = A3

    A4 = {}
    log("\n  A4 slope 路(预测器对比 × 自身层内斜率,折进 rescue 单位;斜率同抽样内重算):")
    for c in ("nmut", "emut", "pmut"):
        gg, _ = within_slope_and_rho(recs, c)
        pt = raw["rf"] - raw[c] * gg
        ci = pct(dif_slope[c])
        A4[c] = {"converted_contrast": round(raw[c] * gg, 5), "point": round(pt, 5), "ci95": ci,
                 "ci_contains_zero": bool(ci[0] <= 0 <= ci[1]), "slope": round(gg, 5)}
        log(f"    {NAME[c]:14s} 折算后 {raw[c]*gg:+.5f}   measurement − 它 = {pt:+.5f}  {ci}"
            f"{'  含 0' if A4[c]['ci_contains_zero'] else '  不含 0'}")
    OUT["A4_slope_route"] = A4

    # K4: do the two routes disagree in sign on the same comparison?
    k4 = {c: (np.sign(A3[c]["point"]) != np.sign(A4[c]["point"])) for c in ("nmut", "emut", "pmut")}
    OUT["K4_routes_disagree_in_sign"] = {c: bool(v) for c, v in k4.items()}
    if any(k4.values()):
        log(f"  **K4:{[c for c,v in k4.items() if v]} 两条路符号相反 —— 两条都报,不选一条当答案**")

    # ================================================================ scorecard
    log("\n=== 预注册记分 ===")
    st = {}
    st["S1"] = {"text": D["staked_predictions"][0]["text"],
                "pass": not A3["nmut"]["ci_contains_zero"],
                "detail": f"标准化后 measurement − RNet = {A3['nmut']['point']:+.5f} SD "
                          f"{A3['nmut']['ci95']}"}
    st["S2"] = {"text": D["staked_predictions"][1]["text"],
                "pass": not A4["nmut"]["ci_contains_zero"],
                "detail": f"slope 路 {A4['nmut']['point']:+.5f} {A4['nmut']['ci95']}"}
    st["S3"] = {"text": D["staked_predictions"][2]["text"],
                "pass": bool(A3["nmut"]["point"] > 0),
                "detail": f"符号 {'+' if A3['nmut']['point']>0 else '-'}"}
    st["S4"] = {"text": D["staked_predictions"][3]["text"], "pass": bool(ok["V4_bridge"]),
                "detail": f"桥 max|Δ| {max(v['bridge_gap'] for v in bridge.values()):.2e}"}
    for k, v in st.items():
        log(f"  {k}  {'成立' if v['pass'] else '**失败**'}   {v['detail']}")
    OUT["staked"] = st
    OUT["verdict"] = ("indistinguishable_does_not_survive_standardisation"
                      if st["S1"]["pass"] else "indistinguishable_survives_standardisation")
    log(f"\n  裁定:{OUT['verdict']}   用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
