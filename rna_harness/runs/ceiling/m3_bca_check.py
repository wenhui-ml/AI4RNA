#!/usr/bin/env python3
"""Measure the two things I used as hedges instead of measuring.

HEDGE 1: "percentile bootstrap coverage under skew at n=56 is unverified." Then verify it. BCa (bias-
corrected and accelerated) adjusts for exactly the skew and bias that make percentile intervals
mis-cover. If BCa still excludes zero, the hedge is retired.

HEDGE 2: "the sign test is weak." The sign test asks about the MEDIAN; the frozen gate asks about the
MEAN. Both are reported here as what they are, alongside a trimmed mean and the skewness, so the shape of
the g_t distribution is on the record rather than being used as an unfrozen discount.

Nothing here can change M3's frozen verdict. It answers questions I raised after the fact, which is the
advance move rather than the retreat move.
"""
from __future__ import annotations
import collections, json, math, os, random, statistics as st, sys
sys.path.insert(0, "/data/wenhui/openclaude_rna"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_ensemble_marginals import base_block, load_pop, marginals, E_FREE
from h1_heldout_ensemble_defect import fit_predict
import h2_heldout_cluster_ci as H
from m3_gc_cancelled import acc_matched

SPLITS, B = 200, 5000


def _phi(z):  return 0.5 * (1 + math.erf(z / math.sqrt(2)))
def _phinv(p):
    # Acklam-style rational approximation, plenty for a 95% interval
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    pl=0.02425
    if p<pl:
        q=math.sqrt(-2*math.log(p));  return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p>1-pl:
        q=math.sqrt(-2*math.log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q=p-0.5; r=q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def bca(v, seed=17):
    """BCa interval for the mean. Corrects for bias (z0) and skew-driven acceleration (a)."""
    n = len(v); th = st.fmean(v)
    rng = random.Random(seed)
    boots = sorted(st.fmean([v[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    prop = sum(1 for x in boots if x < th) / B
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = _phinv(prop)
    jack = [st.fmean(v[:i] + v[i+1:]) for i in range(n)]
    jm = st.fmean(jack)
    num = sum((jm - x) ** 3 for x in jack); den = 6 * (sum((jm - x) ** 2 for x in jack) ** 1.5)
    a = num / den if den else 0.0
    out = []
    for al in (0.025, 0.975):
        z = z0 + (z0 + _phinv(al)) / (1 - a * (z0 + _phinv(al)))
        out.append(boots[min(B - 1, max(0, int(_phi(z) * B)))])
    return [round(out[0], 4), round(out[1], 4)], z0, a


def gvec(fn, formable, seed=0):
    from leaderboard_design import targets
    T = targets()
    rec, _s, _m = load_pop(fn)
    keep = [r for r in rec if base_block(r) is not None]
    from concurrent.futures import ProcessPoolExecutor
    M = []
    with ProcessPoolExecutor(max_workers=14) as ex:
        for m in ex.map(marginals, [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep], chunksize=8):
            M.append(m)
    cells, cols, key = collections.defaultdict(list), {}, {}
    for r, m in zip(keep, M):
        if "error" in m or (formable and "ens_defect_skel" not in m):
            continue
        rr = dict(r)
        ext = [m["ens_defect_skel"]] if formable else [m[k] for k in E_FREE]
        cols[id(rr)] = base_block(r) + [m["gc_cross"], m["gc_frac"]] + ext
        key[id(rr)] = (m["gc_cross"], round(m["gc_frac"], 3))
        cells[(r["puzzle"], r["method"])].append(rr)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 2]
    nB = len(base_block(keep[0])); nE = 1 if formable else len(E_FREE)
    iA = list(range(nB)); iX = iA + list(range(nB + 2, nB + 2 + nE))
    by_t = collections.defaultdict(list)
    for c, gp in rws:
        by_t[c[0]].append((c, gp))
    order = list(sorted(by_t)); rng = random.Random(seed)
    per = {"A": {t: [] for t in order}, "X": {t: [] for t in order}}
    for _s in range(SPLITS):
        rng.shuffle(order); cut = max(1, int(0.7 * len(order)))
        tr, te = set(order[:cut]), set(order[cut:])
        TR = [x for t in tr for x in by_t[t]]; TE = [x for t in te for x in by_t[t]]
        if not TR or not TE: continue
        P = {"A": fit_predict(TR, TE, cols, iA), "X": fit_predict(TR, TE, cols, iX)}
        for t in te:
            va, na, _tt, _tp, _sk = acc_matched(by_t[t], P["A"], key)
            vx, _n, _t2, _p2, _s2 = acc_matched(by_t[t], P["X"], key)
            if na < 30 or va is None or vx is None: continue
            per["A"][t].append(va); per["X"][t].append(vx)
    used = [t for t in order if len(per["A"][t]) >= H.MIN_SPLITS_PER_TARGET]
    return [st.fmean(per["X"][t]) - st.fmean(per["A"][t]) for t in used], used


out = {}
for nm, fn, arm, formable in (("样本2(主) 臂1", "panel_f1_fresh.json", "arm1", True),
                              ("样本2(主) 臂2", "panel_f1_fresh.json", "arm2", False),
                              ("样本3 臂1", "panel_f1_third.json", "arm1", True),
                              ("第四批 臂1", "panel_f1_fourth.json", "arm1", True),
                              ("基准4560 臂1", "panel_f1_designs.json", "arm1", True)):
    g, used = gvec(fn, formable)
    pct, _n = H.boot_ci(g, 17)
    bc, z0, a = bca(list(g))
    m = st.fmean(g); srt = sorted(g); k = max(1, len(g) // 10)
    trim = st.fmean(srt[k:-k]); med = st.median(g)
    sk = (sum((x - m) ** 3 for x in g) / len(g)) / (st.pstdev(g) ** 3) if st.pstdev(g) else 0
    print(f"\n=== {nm} (cluster {len(g)}) ===")
    print(f"  均值 {m:+.4f}  中位 {med:+.4f}  10% 截尾均值 {trim:+.4f}  偏度 {sk:+.2f}")
    print(f"  百分位 bootstrap {pct}  {'排除 0' if pct[0]>0 or pct[1]<0 else '含 0'}")
    print(f"  **BCa(校正偏倚 z0={z0:+.3f} 与偏度 a={a:+.4f})** {bc}"
          f"  {'排除 0' if bc[0]>0 or bc[1]<0 else '含 0'}"
          f"  {'← 与百分位一致' if (pct[0]>0)==(bc[0]>0) else '← **与百分位不一致**'}")
    out[nm] = {"mean": round(m, 4), "median": round(med, 4), "trim10": round(trim, 4),
               "skew": round(sk, 2), "pct": pct, "bca": bc, "z0": round(z0, 3), "a": round(a, 4),
               "n": len(g)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m3_bca_check.json"), "w"),
          indent=1, ensure_ascii=False)
print("\n写出 m3_bca_check.json")
