#!/usr/bin/env python3
"""M56: a second interval on the three numbers that carry claims.

Declaration m56_bca_declaration.json, frozen and pushed BEFORE this file existed. Offline, $0.

Section 3.6 asserts that every interval here should be read as at least as wide as printed and
does not compute it. This computes it, for the three intervals that carry claims, by two methods
that fail differently from the percentile one and from each other.

The estimators are the PUBLISHED ones, imported rather than rewritten, because what is under test
here is the interval method and not the estimator; V2 and V3 are what make that legitimate --
every point estimate and every percentile interval must reproduce before a second interval is read.
"""
from __future__ import annotations

import collections, json, math, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DECL = os.path.join(HERE, "m56_bca_declaration.json")
OUTJ = os.path.join(HERE, "m56_bca.json")
D = json.load(open(DECL))
LOG = []

def _J(n):
    return json.load(open(os.path.join(HERE, n)))


# Published values are READ from the artifacts that produced them, not typed. The manuscript
# prints these to four decimals, and a four-decimal number carries +/-5e-5 of rounding, so a
# tolerance of 5e-5 against the printed value is arithmetically unreachable. V3 refused Q3 twice
# for that reason before the target was corrected; the tolerance below is half the last digit the
# artifact itself stores.
PUB = {
    "Q1": {"point": 0.05393, "ci": _J("m26v_alpha_verify.json")["rf"]["ci"], "tol": 5e-5,
           "cluster": "puzzle", "nperm": 1000, "nboot": 300, "seed": 0, "quant": "floor"},
    "Q2": {"point": 0.1552, "ci": _J("m36_readout_scale.json")["standardised"]["ci"], "tol": 5e-6,
           "cluster": "design", "nboot": 2000, "seed": 3601, "quant": "floor"},
    "Q3": {"point": _J("m44v_verify.json")["staked2_missing_interval"]["0.1"]["point"],
           "ci": _J("m44v_verify.json")["staked2_missing_interval"]["0.1"]["ci"], "tol": 5e-6,
           "cluster": "puzzle", "nboot": 500, "seed": 20260902, "quant": "interp"},
}
NAME = {"Q1": "matched C-G minus A-U contrast on the measured outcome",
        "Q2": "paired standardised interventional-minus-agreement difference",
        "Q3": "budgeted recovery gap at the tightest budget"}


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


# --------------------------------------------------------------- interval methods, written here
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phinv(p):
    """Acklam's inverse normal CDF; only ever called on 0.025, 0.975 and bootstrap fractions."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5; r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    e = _phi(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _q(a, p, kind):
    """The two runs that produced these intervals do not use the same quantile convention: two
    take the floor-indexed order statistic, one calls np.quantile, which interpolates. Reproducing
    a published endpoint means matching the convention of the code that wrote it, and V3 refuses
    the quantity otherwise -- it did, on Q3, at 1.9e-4."""
    return float(a[int(p * len(a))]) if kind == "floor" else float(np.quantile(a, p))


def percentile(draws, lo=0.025, hi=0.975, kind="floor"):
    a = np.sort(np.asarray(draws, float))
    return [_q(a, lo, kind), _q(a, hi, kind)]


def basic(draws, point, lo=0.025, hi=0.975, kind="floor"):
    p = percentile(draws, lo, hi, kind)
    return [2 * point - p[1], 2 * point - p[0]]


def bca(draws, point, jack, lo=0.025, hi=0.975, force_zero=False, kind="floor"):
    """BCa. force_zero sets z0 = a = 0, which must return the percentile interval exactly (V1)."""
    a = np.sort(np.asarray(draws, float))
    if force_zero:
        z0 = acc = 0.0
    else:
        frac = float((a < point).sum()) / len(a)
        z0 = _phinv(min(max(frac, 1.0 / (2 * len(a))), 1 - 1.0 / (2 * len(a))))
        j = np.asarray(jack, float); dv = j.mean() - j
        s2 = float((dv ** 2).sum()); s3 = float((dv ** 3).sum())
        acc = s3 / (6.0 * s2 ** 1.5) if s2 > 0 else 0.0
    out = []
    for q in (lo, hi):
        z = _phinv(q)
        adj = z0 + (z0 + z) / (1.0 - acc * (z0 + z))
        out.append(_q(a, min(max(_phi(adj), 0.0), 0.999999), kind))
    return out, z0, acc


# --------------------------------------------------------------- the three quantities
def q1():
    """the matched contrast, exactly designaudit's own call in m26v.

    The published interval is NOT a fresh bootstrap: core.contrast draws its permutations and its
    bootstrap from ONE generator, so the bootstrap stream starts 1000 permutations in. Reproducing
    the published endpoints therefore means consuming that stream identically, which is why the
    package's own _weighted and _perm_within are called here rather than reimplemented. A first
    version of this function drew a fresh bootstrap and missed the published interval by 3.5e-3;
    V3 is what caught it."""
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "tools")))
    from designaudit import core
    recs = [{**r, "design": tuple(r["design"])} for r in
            json.load(open(os.path.join(HERE, "m52_recs_cache.json")))["recs"]]
    dr = [{"design": r["design"], "pt": r["pt"], "depth3": min(r["depth"], 3),
           "crossed": r["crossed"], "puzzle": r["puzzle"], "rf": r["rf"]} for r in recs]
    c = core.Corpus.from_records(dr, design="design", treatment="pt", outcome="rf",
                                 covariates=("depth3", "crossed"), cluster="puzzle")
    lv = ("CG", "AU")
    m = np.isin(c.treatment, lv)
    sid = c.strata(("depth3", "crossed"), None)[m]
    t = (c.treatment[m] == lv[0]); y = c.outcome[m].astype(float)
    point = core._weighted(sid, t, y)[0]
    rng = np.random.default_rng(PUB["Q1"]["seed"])
    for _ in range(PUB["Q1"]["nperm"]):                       # the stream must be advanced first
        core._weighted(sid, core._perm_within(sid, t, rng), y)
    cl = c.cluster[m]; units = np.unique(cl)
    draws = []
    for _ in range(PUB["Q1"]["nboot"]):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([np.nonzero(cl == units[j])[0] for j in pick])
        v = core._weighted(sid[idx], t[idx], y[idx])[0]
        if v is not None:
            draws.append(v)
    jack = []
    for u in units:
        idx = np.nonzero(cl != u)[0]
        jack.append(core._weighted(sid[idx], t[idx], y[idx])[0])
    return point, draws, jack, list(units)


def q2():
    """the paired standardised difference, clustered on designs, exactly m36's loop."""
    import m25_estimator_benchmark as M25
    from m53_probing_reference import wild_types, rnet_structures, m25_keyset, within_sd, SF
    wt = wild_types(); rnet, _f = rnet_structures(wt)
    resc, called, keys, _n = m25_keyset(wt, rnet)
    R1 = [{**resc[k]} for k in keys]
    R2 = [{**resc[k], "y": called[k]} for k in keys]
    sd1 = within_sd(R1, "y", SF); sd2 = within_sd(R2, "y", SF)
    point = M25.E4_matched(R1, "y") / sd1 - M25.E4_matched(R2, "y") / sd2
    des = sorted({r["design"] for r in R1})
    by = collections.defaultdict(lambda: ([], []))
    for a_, b_ in zip(R1, R2):
        by[a_["design"]][0].append(a_); by[a_["design"]][1].append(b_)
    rng = np.random.default_rng(PUB["Q2"]["seed"]); draws = []
    for _ in range(PUB["Q2"]["nboot"]):
        ix = rng.integers(0, len(des), len(des))
        s1 = [x for j in ix for x in by[des[j]][0]]
        s2 = [x for j in ix for x in by[des[j]][1]]
        v1, v2 = M25.E4_matched(s1, "y"), M25.E4_matched(s2, "y")
        if v1 is None or v2 is None:
            continue
        a1, a2 = within_sd(s1, "y", SF), within_sd(s2, "y", SF)
        if not (a1 > 0 and a2 > 0):
            continue
        draws.append(v1 / a1 - v2 / a2)
    jack = []
    for d in des:
        s1 = [x for k in des if k != d for x in by[k][0]]
        s2 = [x for k in des if k != d for x in by[k][1]]
        jack.append(M25.E4_matched(s1, "y") / within_sd(s1, "y", SF)
                    - M25.E4_matched(s2, "y") / within_sd(s2, "y", SF))
    return point, draws, jack, des


def q3():
    """the budgeted gap at the tightest budget, clustered on puzzles, exactly m44v's loop."""
    import m44v_verify as M44V
    cells = M44V.cells_from_csv()
    sgn = M44V.depth_sign(cells)
    frac = min(M44V.BUDGETS)
    gap = lambda cs: (M44V.recover(cs, "RibonanzaNet", sgn, frac)
                      - M44V.recover(cs, "ViennaRNA", sgn, frac))
    point = gap(cells)
    pz = sorted({c["puzzle"] for c in cells.values()})
    bypz = collections.defaultdict(dict)
    for k, c in cells.items():
        bypz[c["puzzle"]][k] = c
    rng = np.random.default_rng(PUB["Q3"]["seed"]); draws = []
    for _ in range(PUB["Q3"]["nboot"]):
        ix = rng.integers(0, len(pz), len(pz))
        sub = {}
        for j, q in enumerate(ix):
            for k, c in bypz[pz[q]].items():
                sub[(j, k)] = c
        a = M44V.recover(sub, "RibonanzaNet", sgn, frac)
        b = M44V.recover(sub, "ViennaRNA", sgn, frac)
        if a is not None and b is not None:
            draws.append(a - b)
    jack = []
    for u in pz:
        sub = {k: c for k, c in cells.items() if c["puzzle"] != u}
        jack.append(gap(sub))
    return point, draws, jack, pz


def main():
    t0 = time.time()
    log("=== M56:给三个承载主张的区间加第二个算法 ===")
    log("  声明已冻结并推送于本文件存在之前。全离线,$0。")
    OUT = {"declaration": os.path.basename(DECL), "cost": "offline, $0", "quantities": {}}

    # V1 first: the identity that makes BCa a generalisation of the percentile interval
    rr = np.random.default_rng(7); dd = rr.normal(size=2000).tolist()
    p0 = percentile(dd); b0, _z, _a = bca(dd, 0.0, [0.0] * 20, force_zero=True)
    v1 = max(abs(p0[0] - b0[0]), abs(p0[1] - b0[1])) < 1e-12
    log(f"  V1 z0=a=0 时 BCa 与百分位 max|Δ| {max(abs(p0[0]-b0[0]), abs(p0[1]-b0[1])):.2e}"
        f"  -> {'通过' if v1 else '失败'}")
    OUT["V"] = {"V1_degenerate_equals_percentile": bool(v1)}
    if not v1:
        OUT["VOID"] = "K2: V1 未通过"; save(OUT); return 1

    rows = {}
    for tag, fn in (("Q1", q1), ("Q2", q2), ("Q3", q3)):
        log(f"\n=== {tag}: {NAME[tag]} ===")
        point, draws, jack, units = fn()
        pub = PUB[tag]
        kind = pub.get("quant", "floor")
        pc = percentile(draws, kind=kind)
        dpt = abs(point - pub["point"])
        dci = max(abs(pc[0] - pub["ci"][0]), abs(pc[1] - pub["ci"][1]))
        log(f"  点估计 {point:+.5f}(论文 {pub['point']:+.4f})  |Δ| {dpt:.2e}")
        log(f"  百分位 [{pc[0]:+.5f}, {pc[1]:+.5f}](论文 {pub['ci']})  max|Δ| {dci:.2e}")
        log(f"  聚类单元 {pub['cluster']},共 {len(units)} 个;自助 {len(draws)}/{pub['nboot']} 次;"
            f"jackknife {len(jack)} 个")
        v2 = dpt < 5e-5; v3 = dci < pub["tol"]
        v4 = len(jack) == len(units) and all(x is not None and np.isfinite(x) for x in jack)
        if not (v2 and v3 and v4):
            log(f"  **K1/K2:V2 {v2} V3 {v3} V4 {v4} —— {tag} 不报告第二个区间**")
            rows[tag] = {"VOID": f"V2={v2} V3={v3} V4={v4}", "cluster": pub["cluster"],
                         "n_clusters": len(units)}
            continue
        bc, z0, acc = bca(draws, point, jack, kind=kind)
        bs = basic(draws, point, kind=kind)
        wp = pc[1] - pc[0]
        log(f"  BCa    [{bc[0]:+.5f}, {bc[1]:+.5f}]  z0 {z0:+.4f}  a {acc:+.4f}  "
            f"宽度比 {(bc[1]-bc[0])/wp:.3f}")
        log(f"  basic  [{bs[0]:+.5f}, {bs[1]:+.5f}]  宽度比 {(bs[1]-bs[0])/wp:.3f}")
        for nm, iv in (("BCa", bc), ("basic", bs)):
            log(f"    {nm} {'含 0' if iv[0] <= 0 <= iv[1] else '不含 0'}")
        rows[tag] = {"name": NAME[tag], "point": round(point, 5),
                     "cluster": pub["cluster"], "n_clusters": len(units),
                     "n_draws": len(draws),
                     "percentile": [round(x, 5) for x in pc],
                     "bca": [round(x, 5) for x in bc],
                     "basic": [round(x, 5) for x in bs],
                     "z0": round(z0, 4), "acceleration": round(acc, 4),
                     "acceleration_large": bool(abs(acc) > 0.25),
                     "width_ratio_bca": round((bc[1] - bc[0]) / wp, 4),
                     "width_ratio_basic": round((bs[1] - bs[0]) / wp, 4),
                     "bca_excludes_zero": bool(bc[0] > 0 or bc[1] < 0),
                     "basic_excludes_zero": bool(bs[0] > 0 or bs[1] < 0),
                     "percentile_excludes_zero": bool(pc[0] > 0 or pc[1] < 0)}
    OUT["quantities"] = rows

    ok = [t for t in ("Q1", "Q2", "Q3") if "VOID" not in rows[t]]
    S1 = bool("Q1" in ok and rows["Q1"]["bca_excludes_zero"])
    S2 = bool("Q2" in ok and rows["Q2"]["bca_excludes_zero"])
    S3 = bool("Q3" in ok and rows["Q3"]["bca_excludes_zero"])
    S4 = bool(sum(1 for t in ok if rows[t]["width_ratio_bca"] > 1.0) >= 2)
    log("\n=== 预注册记分 ===")
    det = [f"Q1 BCa {rows.get('Q1',{}).get('bca')}", f"Q2 BCa {rows.get('Q2',{}).get('bca')}",
           f"Q3 BCa {rows.get('Q3',{}).get('bca')}",
           "宽度比 " + ", ".join(f"{t} {rows[t]['width_ratio_bca']:.3f}" for t in ok)]
    for i, (sid, okk) in enumerate((("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4))):
        log(f"  {sid}  {'成立' if okk else '**失败**'}   {det[i]}")
    OUT["staked"] = {f"S{i+1}": {"text": D["staked_predictions"][i]["text"], "pass": p,
                                 "detail": det[i]} for i, p in enumerate([S1, S2, S3, S4])}
    OUT["verdict"] = ("all_three_survive_a_second_interval" if (S1 and S2 and S3)
                      else "a_claim_interval_does_not_survive")
    log(f"\n  裁定:{OUT['verdict']}   {sum([S1,S2,S3,S4])}/4 成立   用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
