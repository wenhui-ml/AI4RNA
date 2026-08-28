#!/usr/bin/env python3
"""M32r: review M32 before its verdict is read.

M32 returned five of five staked predictions held. That is the shape M31 returned before its
verdict turned out to be wrong, so it is reviewed rather than believed. Two checks that M32's
declaration should have contained and did not:

  R1  DYNAMIC RANGE. The manuscript already reports, for both scored models, the fraction of
      interventions above 0.99 and below 0.08 and the median, because a predictor with no room
      to move cannot express a contrast and its null is an artefact of its range rather than a
      statement about the molecule. M32 never ran that check on EternaFold. If EternaFold is
      more saturated than either incumbent, "it does not separate" is not evidence about model
      classes at all.

  R2  CENSORING. EternaFold emits nothing below 1e-4, so 176 mutants were set to 0.0 by hand.
      C-G pairs are more stable than A-U, so censoring is plausibly correlated with pair type,
      and a pair-type-correlated floor pushes the C-G minus A-U contrast in a definite
      direction. The contrast is recomputed on the uncensored subset alone.

  R3  the claim M32's verdict string invites -- "EternaFold behaves like ViennaRNA" -- is checked
      against the intervals rather than the point estimates. -0.00569 and -0.03039 share a sign,
      which is not the same as behaving alike.

COST: $0, offline, reuses M32's cached posteriors.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26
import m32_third_model_class as M32

LOG = []
HI, LO = 0.99, 0.08                      # the manuscript's own dynamic-range thresholds


def log(s=""):
    print(s, flush=True); LOG.append(s)


def main() -> int:
    log("=== M32r:在读裁定之前复核 M32 ===\n")
    prev = json.load(open(os.path.join(HERE, "m32_third_model_class.json")))
    recs, _ = M26.build()
    seqs = sorted({r["wt_seq"] for r in recs} | {r["mut_seq"] for r in recs})
    post = M32.score_sequences(seqs)

    for r in recs:
        i, j = min(r["a"], r["b"]) + 1, max(r["a"], r["b"]) + 1
        em = post[r["mut_seq"]].get((i, j))
        r["censored"] = em is None
        r["emut"] = 0.0 if em is None else em

    OUT = {}

    # ---------------- R1 dynamic range, the manuscript's own three numbers
    log("--- R1 动态范围(用论文对另外两个模型用的同一套阈值)---")
    e = np.array([r["emut"] for r in recs], float)
    hi, lo = float((e > HI).mean()), float((e < LO).mean())
    log(f"  EternaFold  >{HI} 占 {hi:.1%}   <{LO} 占 {lo:.1%}   中位 {np.median(e):.4f}   "
        f"p10 {np.quantile(e,.1):.4f}  p90 {np.quantile(e,.9):.4f}")
    log(f"  对照:ViennaRNA >0.99 28.1% / <0.08 33.1% 中位 0.7451;"
        f"RibonanzaNet 44.1% / 13.4% 中位 0.9839")
    mid = 1.0 - hi - lo
    OUT["R1_dynamic_range"] = {"frac_hi": round(hi, 4), "frac_lo": round(lo, 4),
                               "frac_mid": round(mid, 4), "median": round(float(np.median(e)), 4),
                               "p10": round(float(np.quantile(e, .1)), 4),
                               "p90": round(float(np.quantile(e, .9)), 4)}
    # the range is only disqualifying if it is WORSE than the models that did express a contrast
    worse_than_both = hi > 0.441 and mid < 0.425
    OUT["R1_range_could_explain_null"] = bool(worse_than_both)
    log(f"  比两个现有模型都更饱和(会让零结果成为量程假象){worse_than_both}")

    # ---------------- R2 censoring, and the contrast without it
    log("\n--- R2 截断是否与配对类型相关 ---")
    byp = collections.Counter((r["pt"], r["censored"]) for r in recs)
    rates = {}
    for pt in ("CG", "AU", "GU"):
        n = byp[(pt, True)] + byp[(pt, False)]
        rates[pt] = round(byp[(pt, True)] / n, 4) if n else None
        log(f"  {pt}  被截断 {byp[(pt, True)]}/{n} = {rates[pt]}")
    OUT["R2_censor_rate_by_pairtype"] = rates

    clean = [r for r in recs if not r["censored"]]
    log(f"  未被截断的记录 {len(clean)}/{len(recs)}")
    a_all = M32.contrast_arm(recs, "emut", seed=3240, nperm=2000, nboot=2000)
    a_cln = M32.contrast_arm(clean, "emut", seed=3241, nperm=2000, nboot=2000)
    log(f"  全部     {a_all['effect']:+.5f}  CI {a_all['ci']}  p {a_all['p_perm']:.5f}")
    log(f"  去掉截断 {a_cln['effect']:+.5f}  CI {a_cln['ci']}  p {a_cln['p_perm']:.5f}")
    OUT["R2_contrast_all"] = a_all
    OUT["R2_contrast_uncensored"] = a_cln
    flip = (a_all["effect"] < 0) != (a_cln["effect"] < 0)
    sep = a_cln["ci"][0] is not None and a_cln["ci"][0] > 0
    OUT["R2_sign_flips_when_censoring_removed"] = bool(flip)
    OUT["R2_separates_without_censoring"] = bool(sep)
    log(f"  去掉截断后换号 {flip};去掉截断后按实测方向分离 {sep}")

    # ---------------- R3 does "behaves like ViennaRNA" survive the intervals?
    log("\n--- R3 『它表现得像 ViennaRNA』这句话经得起区间吗 ---")
    ee, ve = a_all["effect"], -0.03039
    eci = a_all["ci"]
    log(f"  EternaFold {ee:+.5f} {eci};ViennaRNA {ve:+.5f} [-0.06491, 0.00994]")
    log(f"  EternaFold 的区间含 0:{eci[0] <= 0 <= eci[1]}")
    log(f"  ViennaRNA 的区间也含 0:True —— 两者都只是『没有分离』,不是『同向』")
    OUT["R3"] = {
        "eterna_ci_contains_zero": bool(eci[0] <= 0 <= eci[1]),
        "licensed_sentence": ("EternaFold does not separate the two pair types in the measured "
                              "direction; its interval contains zero"),
        "refused_sentence": ("EternaFold behaves like ViennaRNA -- both point estimates are "
                             "negative but EternaFold's is five times smaller and both intervals "
                             "cover zero, so a shared sign is not a shared behaviour")}

    log("\n=== 复核裁定 ===")
    void = OUT["R1_range_could_explain_null"] or OUT["R2_separates_without_censoring"]
    if void:
        log("  **M32 的 S2 不可按原样解读**")
    else:
        log("  S2 站得住,但只能按 R3 许可的那句话写")
    OUT["verdict"] = ("S2_not_readable_as_stated" if void
                      else "S2_stands_but_only_as_a_non_separation")
    log(f"  {OUT['verdict']}")

    json.dump(OUT, open(os.path.join(HERE, "m32r_review.json"), "w"), indent=2,
              ensure_ascii=False)
    open(os.path.join(HERE, "m32r_run.out"), "w").write("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
