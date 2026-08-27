#!/usr/bin/env python3
"""M26b: four parameter sets, two of them machine-learned, on the same 4317 interventions.

Declaration m26b_model_panel_declaration.json, frozen and pushed at b99164c3 BEFORE this file
existed. Offline, $0.

Parameters are global state in ViennaRNA, so each set runs in its own process.
"""
from __future__ import annotations
import argparse, collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

MODELS = ("Turner2004", "Turner1999", "Andronescu2007", "Langdon2018")
NBINS, NPERM, NPLACEBO, P_GATE = 5, 2000, 12, 0.005


def job(name):
    import RNA
    import m26_model_scored as M26
    getattr(RNA, f"params_load_RNA_{name}")()
    recs, dropped = M26.build()
    for r in recs:
        r["dp"] = r["pmut"] - r["pwt"]
    keys = [(r["design"], r["a"], r["b"]) for r in recs]
    pm = np.array([r["pmut"] for r in recs])
    sb = lambda x: (x["design"], min(x["depth"], 3), x["crossed"])
    edges = [float(np.quantile(pm, q)) for q in np.linspace(0, 1, NBINS + 1)[1:-1]]
    for r in recs:
        r["bin"] = int(np.searchsorted(edges, r["pmut"], side="right"))
    sd = lambda x: (x["design"], min(x["depth"], 3), x["crossed"], x["bin"])

    sl, n = M26.slope_in(recs, sb, "pmut")
    p = M26.slope_perm_p(recs, sb, "pmut", sl, 7, nperm=500)
    by = collections.defaultdict(list)
    for r in recs:
        by[sb(r)].append(r)
    cen = {}
    for _s, rs in by.items():
        m = float(np.mean([x["rf"] for x in rs]))
        for x in rs:
            cen[id(x)] = x["rf"] - m
    cal = [{"bin": k, "n": int((np.array([r["bin"] for r in recs]) == k).sum()),
            "p_mean": round(float(np.mean([r["pmut"] for r in recs if r["bin"] == k])), 4),
            "rf_mean": round(float(np.mean([r["rf"] for r in recs if r["bin"] == k])), 4),
            "rf_centred": round(float(np.mean([cen[id(r)] for r in recs if r["bin"] == k])), 4)}
           for k in range(NBINS)]
    mono = all(cal[i]["rf_centred"] <= cal[i + 1]["rf_centred"] for i in range(NBINS - 1))

    import random as _r
    d0, w0, _s0 = M26.matched(recs, "CG", "AU", sb)
    dc, wc, sc = M26.matched(recs, "CG", "AU", sd)
    keep = M26.retained_recs(recs, "CG", "AU", sd)
    dsame, wsame, _ = M26.matched(keep, "CG", "AU", sb)
    null = np.sort(np.array([M26.perm(recs, "CG", "AU", sd, _r.Random(i)) or 0.0
                             for i in range(NPERM)]))
    pv = (float((null >= dc).sum()) + 1) / (NPERM + 1)
    ret = dc / dsame
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    pl = []
    for k in range(NPLACEBO):
        rng = np.random.default_rng(900 + k)
        sh = []
        for _q, rs in byp.items():
            vals = [x["pmut"] for x in rs]
            ix = rng.permutation(len(vals))
            for x, j in zip(rs, ix):
                sh.append({**x, "bin": int(np.searchsorted(edges, vals[j], side="right"))})
        kp = M26.retained_recs(sh, "CG", "AU", sd)
        v, _w, _s = M26.matched(sh, "CG", "AU", sd)
        b, _w2, _s2 = M26.matched(kp, "CG", "AU", sb)
        if v is not None and b:
            pl.append(v / b)
    pl = np.array(pl); pmn, psd = float(pl.mean()), float(pl.std(ddof=1))
    med = 1 - ret
    return {"model": name, "n": len(recs), "keys": keys, "pmut": pm.tolist(),
            "skill_slope": round(sl, 5), "skill_p": round(p, 5), "calib": cal,
            "monotone": bool(mono), "cde": round(dc, 4),
            "base_same_population": round(dsame, 4), "base_full": round(d0, 4),
            "retention": round(ret, 4), "p_perm": round(pv, 5), "weight": wc, "strata": sc,
            "placebo_mean": round(pmn, 4), "placebo_sd": round(psd, 4),
            "z_vs_placebo": round((ret - pmn) / (psd or 1), 3),
            "mediated": round(med, 4),
            "mediated_ci95": [round(med - 1.96 * psd, 4), round(med + 1.96 * psd, 4)]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out",
        default=os.path.join(HERE, "m26b_model_panel.json"))
    a = ap.parse_args()
    print("=== M26b:四套参数(其中两套是机器学习拟合)在同一批 4317 次干预上 ===")
    print("  声明 b99164c3。离线 $0。\n")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(job, MODELS))
    ref = res[0]
    print("=== P2 同一批配对 ===")
    same = all(r["keys"] == ref["keys"] for r in res)
    print(f"  四套的纳入集合逐字相同: {same}  (n={ref['n']})")
    assert same, "P2 违反"
    print("\n=== P1 参数确实换了 ===")
    P = {r["model"]: np.array(r["pmut"]) for r in res}
    for r in res[1:]:
        v = P[r["model"]]
        c = float(np.corrcoef(P["Turner2004"], v)[0, 1])
        print(f"  {r['model']:16s} 与 Turner2004 相关 {c:+.4f}  最大差 "
              f"{float(np.abs(P['Turner2004'] - v).max()):.4f}  均值 {v.mean():.4f}")
    cors = {}
    for i, a1 in enumerate(MODELS):
        for b1 in MODELS[i + 1:]:
            cors[f"{a1}|{b1}"] = round(float(np.corrcoef(P[a1], P[b1])[0, 1]), 4)
    print(f"  两两相关 {cors}")

    print("\n=== E1/E2 逐模型 ===")
    print(f"  {'model':16s} {'技能斜率':>9s} {'p':>8s} {'单调':>5s} {'CDE':>9s} "
          f"{'同群体基线':>10s} {'留存':>8s} {'安慰剂':>12s} {'中介':>9s} {'区间含0':>7s}")
    for r in res:
        c0 = r["mediated_ci95"][0] <= 0 <= r["mediated_ci95"][1]
        print(f"  {r['model']:16s} {r['skill_slope']:+9.4f} {r['skill_p']:8.4f} "
              f"{'是' if r['monotone'] else '否':>5s} {r['cde']:+9.4f} "
              f"{r['base_same_population']:+10.4f} {100*r['retention']:7.1f}% "
              f"{100*r['placebo_mean']:6.1f}±{100*r['placebo_sd']:<4.1f} "
              f"{100*r['mediated']:+8.1f}% {'是' if c0 else '否':>7s}")

    skill = all(r["skill_p"] <= P_GATE and r["skill_slope"] > 0 for r in res)
    ret80 = all(r["retention"] >= 0.80 for r in res)
    ci0 = all(r["mediated_ci95"][0] <= 0 <= r["mediated_ci95"][1] for r in res)
    ml = all(r["skill_slope"] >= res[0]["skill_slope"] for r in res
             if r["model"] in ("Andronescu2007", "Langdon2018"))
    cor7 = all(v >= 0.7 for v in cors.values())
    out = {"models": [{k: v for k, v in r.items() if k not in ("keys", "pmut")} for r in res],
           "pairwise_corr": cors, "n": ref["n"],
           "staked": {"1_all_have_skill": bool(skill), "2_all_retention_ge_80": bool(ret80),
                      "3_all_mediated_ci_contains_zero": bool(ci0),
                      "4_ml_params_skill_not_below_turner2004": bool(ml),
                      "5_pairwise_corr_ge_07": bool(cor7)},
           "verdict": ("identity_survives_every_parameter_set" if (skill and ret80 and ci0)
                       else "at_least_one_parameter_set_absorbs_it")}
    lo = min(r["retention"] for r in res)
    if lo <= 0.50:
        out["K1_absorbing_model"] = [r["model"] for r in res if r["retention"] <= 0.50]
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n=== 裁定: {out['verdict']} ===")
    print("押注逐条: " + json.dumps(out["staked"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
