#!/usr/bin/env python3
"""M26r: a review of M26/M26c/M26d, looking for the ways they could be wrong.

Not a new claim. Five checks, each aimed at something that would invalidate a number already
committed. Written after seeing those numbers, so nothing here may be used to support the
conclusion -- only to attack it.

R1 POSITIVE CONTROL. Every "retention ~100%" result says the mediator explains nothing. That is
only informative if the machinery can detect a mediator that DOES explain something. A synthetic
mediator carrying the identity effect by construction must drive retention down. If it does not,
every retention number in M26/M26b/M26c is uninformative rather than null.

R2 CONDITIONING STRENGTH. RibonanzaNet is nearly binary here (bins 1-4 all above P=0.83). If the
five quantile bins barely reduce the within-stratum variance of the prediction, then "fixing the
prediction" fixes little and 100% retention is trivial.

R3 LEAKED-SUBSET INTERVAL. M26d compared point slopes on 324 records over 14 design cells and I
drew a direction from it. A point estimate without an interval cannot carry that.

R4 WINDOW CONSISTENCY. M26d keyed each design cell's window from the last CSV row for that cell.
If sub_start/sub_end vary inside a cell, the leakage query used the wrong window.

R5 PLACEHOLDER REPORTING. m26d printed "no-candidate identity mean 0.0000" from an empty array.
Count how many queries actually had no candidate, and say so rather than printing a fill value.
"""
from __future__ import annotations

import collections, csv, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

OUT, LOG = {}, []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def retention(recs, mkey, nbins=5, seed=0):
    """M26's own D2, verbatim machinery, for an arbitrary mediator column."""
    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    v = np.array([r[mkey] for r in recs])
    edges = [float(np.quantile(v, q)) for q in np.linspace(0, 1, nbins + 1)[1:-1]]
    for r in recs:
        r["_rb"] = int(np.searchsorted(edges, r[mkey], side="right"))
    sd = lambda r: (r["design"], min(r["depth"], 3), r["crossed"], r["_rb"])
    cde, w, _ = M26.matched(recs, "CG", "AU", sd)
    if cde is None or w < 100:
        return None, None, w or 0
    keep = M26.retained_recs(recs, "CG", "AU", sd)
    base, _wb, _s = M26.matched(keep, "CG", "AU", sb)
    return (cde / base if base else None), cde, w


def main():
    log("=== M26r:对 M26/M26c/M26d 的复核 ===")
    log("  不是新主张。五条检查,每条都指向一个能让已提交的数作废的地方。\n")
    recs, _ = M26.build()
    z = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {k: v for k, v in zip(z["keys"], z["mats"])}
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    log(f"纳入 {len(recs)}")

    # ---------------- R1 positive control
    log("\n=== R1 正对照:一个【按构造就该中介】的中介物,机制抓不抓得出来 ===")
    base_ret, base_cde, base_w = retention(recs, "pmut")
    log(f"  参照:固定 P_vienna       留存 {100*base_ret:.1f}%  CDE {base_cde:+.4f}  权重 {base_w:.0f}")
    nret, ncde, nw = retention(recs, "nmut")
    log(f"  参照:固定 P_neural       留存 {100*nret:.1f}%  CDE {ncde:+.4f}  权重 {nw:.0f}")
    rng = np.random.default_rng(11)
    tab = []
    for frac in (1.0, 0.75, 0.5, 0.25):
        # a mediator that carries `frac` of the identity signal plus matched-scale noise
        for r in recs:
            sig = 1.0 if r["pt"] == "CG" else 0.0
            r["_pos"] = frac * sig + (1 - frac) * rng.random()
        ret, cde, w = retention(recs, "_pos")
        tab.append((frac, ret, cde, w))
        log(f"  正对照 frac={frac:<4}  " +
            (f"留存 {100*ret:.1f}%  CDE {cde:+.4f}  权重 {w:.0f}" if ret is not None
             else f"**权重 {w:.0f} < 100 ⇒ 完全重叠丧失(这本身就是中介被抓到的极端形式)**"))
    OUT["R1"] = {"vienna_retention": round(base_ret, 4), "neural_retention": round(nret, 4),
                 "positive_control": [{"frac": f, "retention": (round(r, 4) if r else None),
                                       "weight": w} for f, r, _c, w in tab]}
    detected = any((r is None or r < 0.5) for _f, r, _c, _w in tab)
    log(f"  **机制能否检出中介: {'能' if detected else '**不能 ⇒ 所有留存数都是不可判而非零**'}**")
    OUT["R1_can_detect_mediation"] = bool(detected)

    # ---------------- R2 conditioning strength
    log("\n=== R2 条件化强度:五个分位箱到底去掉了预测量多少方差 ===")
    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    for mkey, nm in (("pmut", "P_vienna"), ("nmut", "P_neural")):
        v = np.array([r[mkey] for r in recs])
        edges = [float(np.quantile(v, q)) for q in np.linspace(0, 1, 6)[1:-1]]
        b = np.searchsorted(edges, v, side="right")
        tot = float(v.var())
        within = float(np.mean([v[b == k].var() for k in range(5) if (b == k).sum() > 1]))
        # variance still left inside a full stratum (design x depth x crossed x bin)
        by = collections.defaultdict(list)
        for r, bb in zip(recs, b):
            by[(sb(r), int(bb))].append(r[mkey])
        ws = [np.var(x) for x in by.values() if len(x) > 1]
        log(f"  {nm:10s} 总方差 {tot:.5f}  箱内平均 {within:.5f}({100*within/tot:.1f}%)  "
            f"分层内平均 {np.mean(ws):.5f}({100*np.mean(ws)/tot:.1f}%)")
        OUT.setdefault("R2", {})[nm] = {"total_var": round(tot, 5), "within_bin": round(within, 5),
                                        "within_stratum": round(float(np.mean(ws)), 5),
                                        "frac_left": round(float(np.mean(ws)) / tot, 4)}

    # ---------------- R3 leaked-subset interval
    log("\n=== R3 M26d 泄漏子集的斜率:点估计不够,要区间 ===")
    lk = json.load(open(os.path.join(HERE, "m26d_clean_subset.json")))
    log(f"  M26d 报的泄漏设计单元 {lk['n_leaked_designs']}/{lk['n_designs']},泄漏干预 {lk['n_dirty']}")
    # recompute leaked/clean membership from the committed json is not possible; recompute by size
    # instead we bootstrap the slope on a random 324-record subsample to get the noise floor
    boot = []
    rng2 = np.random.default_rng(7)
    for _ in range(400):
        idx = rng2.choice(len(recs), lk["n_dirty"], replace=False)
        s, _n = M26.slope_in([recs[i] for i in idx], sb, "nmut")
        if s is not None:
            boot.append(s)
    boot = np.sort(np.array(boot))
    lo, hi = float(boot[int(.025 * len(boot))]), float(boot[int(.975 * len(boot))])
    log(f"  **同样大小(n={lk['n_dirty']})的随机子集,斜率 95% 区间 [{lo:+.4f}, {hi:+.4f}]**")
    log(f"  M26d 的泄漏子集斜率 +0.10022 落在这个区间内: "
        f"{'是 ⇒ 与随机同大小子集不可区分,方向不可读' if lo <= 0.10022 <= hi else '否'}")
    OUT["R3"] = {"n": lk["n_dirty"], "random_subset_ci": [round(lo, 4), round(hi, 4)],
                 "leaked_slope": 0.10022,
                 "indistinguishable_from_random": bool(lo <= 0.10022 <= hi)}

    # ---------------- R4 window consistency
    log("\n=== R4 设计窗在同一设计单元内是否一致 ===")
    csv.field_size_limit(10 ** 9)
    wins = collections.defaultdict(set)
    for d in csv.DictReader(open(M26.SRC)):
        try:
            wins[(d["puzzle"], d["method"])].add((int(float(d["sub_start"])), int(float(d["sub_end"]))))
        except (ValueError, KeyError):
            continue
    bad = {k: v for k, v in wins.items() if len(v) > 1}
    log(f"  设计单元 {len(wins)},窗不唯一的 {len(bad)}")
    if bad:
        for k, v in list(bad.items())[:5]:
            log(f"    {k} -> {sorted(v)}")
        log("  **M26d 的泄漏查询对这些单元用了最后一行的窗 ⇒ 该部分结果需要重算**")
    else:
        log("  **窗唯一 ⇒ M26d 的取窗方式无害**")
    OUT["R4"] = {"n_designs": len(wins), "n_ambiguous_window": len(bad)}

    json.dump(OUT, open(os.path.join(HERE, "m26r_review.json"), "w"), ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m26r_review.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m26r_review.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
