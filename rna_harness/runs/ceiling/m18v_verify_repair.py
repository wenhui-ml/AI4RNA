#!/usr/bin/env python3
"""M18v: verify the repair rather than trust the diagnostic that motivated it.

Four independent checks on the repaired null. This is not a new experiment and adjudicates
nothing; if any check fails, M18 and M19 are wrong and both come back out.

  V1  joint-covariate permutation == permute-y.  Shuffling molecule identity within cell and
      carrying gcp, gcf, pB and dg together is the treatment-side permutation that keeps the
      design intact. Averaged over all permutations it is the SAME null as permuting y, so the
      two must agree. If they do, "permute y" is not a workaround, it is the correct null.
  V2  the null is centred. A design-fixed permutation null must have mean ~0.
  V3  brute force. On the smallest population, enumerate the statistic directly with an
      independent implementation that does not import match(), and compare.
  V4 and V5 are NOT in this file -- they are in m18v2_patch_check.py, written after this one
  because declaring a gate here and not running it is exactly failure mode G7.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
from m13b_gc_placement import assemble, match, point, POPS, DELTA, CAL_GC, CAL_P     # noqa: E402
from m18_repaired_null import permute_y, null_y                                      # noqa: E402

NPERM = 4000


def permute_covariates(d, rng):
    """the other formulation: shuffle molecule identity within cell, carrying EVERY covariate.
    y stays put. Must give the same null as permute_y."""
    cells = []
    for c in d["cells"]:
        p = rng.permutation(len(c["y"]))
        cells.append({**c, "gcp": c["gcp"][p], "gcf": c["gcf"][p], "pB": c["pB"][p],
                      "dg": c["dg"][p], "ISGC": c["ISGC"][p]})
    return {**d, "cells": cells}


def null_cov(d, seed, nperm, panel):
    rng = np.random.default_rng(seed); out = []
    for _ in range(nperm):
        d2 = permute_covariates(d, rng)
        M = match(d2, [c["gcp"] for c in d2["cells"]], DELTA, panel=panel)
        out.append(point(M) if M else 0.0)
    return np.sort(np.array(out))


def brute(d, panel):
    """V3: recompute the observed statistic from scratch, no match() involved."""
    sp = d["sp"]; num = den = 0.0
    for c in d["cells"]:
        y, gcp, gcf, pB = c["y"], c["gcp"], c["gcf"], c["pB"]
        for i in range(len(y)):
            for j in range(len(y)):
                if gcp[i] - gcp[j] < DELTA:
                    continue
                if abs(gcf[i] - gcf[j]) > CAL_GC:
                    continue
                if panel and abs(pB[i] - pB[j]) > CAL_P * sp:
                    continue
                num += y[i] - y[j]; den += 1
    return (num / den if den else float("nan")), int(den)


def ks(a, b):
    """two-sample KS statistic, no scipy."""
    z = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(a, z, "right") / len(a)
                               - np.searchsorted(b, z, "right") / len(b))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--nperm", type=int, default=NPERM)
    a = ap.parse_args()
    print("=== M18v:验证修法本身,而不是相信促成它的那个诊断 ===")
    print("  四项检查。任一不过,M18 与 M19 都要退回。离线 $0。\n")
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    M18 = json.load(open(os.path.join(HERE, "m18_repaired_null.json")))["pops"]
    OK = {"V1": True, "V2": True, "V3": True}
    OUT = {}

    print("--- V1/V2:联合重排协变量 vs 打乱 y,以及零分布是否居中 ---")
    print(f"  {'人群':12s} {'臂':>4s} {'打乱y 均值±SD':>20s} {'联合重排 均值±SD':>20s} "
          f"{'SD比':>6s} {'KS':>6s} {'居中?':>6s}")
    for nm, fn in POPS:
        d = assemble(C[fn]["rows"], E[fn], a.seed)
        for panel, lab in ((False, "free"), (True, "panel")):
            ny = null_y(d, 1000 + a.seed, a.nperm, panel=panel)
            nc = null_cov(d, 2000 + a.seed, a.nperm, panel=panel)
            r = float(nc.std()) / (float(ny.std()) or 1e-12)
            k = ks(ny, nc)
            centred = abs(float(ny.mean())) < 0.15 * float(ny.std())
            # KS 5% critical value for two samples of size n
            crit = 1.358 * np.sqrt(2.0 / a.nperm)
            v1 = bool(0.90 <= r <= 1.10 and k <= crit)
            OK["V1"] &= v1; OK["V2"] &= centred
            OUT[f"{fn}|{lab}"] = {"sd_ratio": round(r, 4), "ks": round(k, 4),
                                  "ks_crit_5pct": round(float(crit), 4),
                                  "mean_y": round(float(ny.mean()), 4),
                                  "sd_y": round(float(ny.std()), 4),
                                  "mean_cov": round(float(nc.mean()), 4),
                                  "sd_cov": round(float(nc.std()), 4), "pass": v1}
            print(f"  {nm:12s} {lab:>5s} {ny.mean():+10.4f}±{ny.std():<8.4f} "
                  f"{nc.mean():+10.4f}±{nc.std():<8.4f} {r:6.3f} {k:6.4f} "
                  f"{'是' if centred else '**否**':>6s}  {'✓' if v1 else '**✗**'}")
    print(f"  KS 5% 临界值 {1.358*np.sqrt(2.0/a.nperm):.4f}(n={a.nperm} 两样本)")

    print("\n--- V3:独立蛮力复算观测统计量(不调用 match()) ---")
    for nm, fn in POPS[-2:]:
        d = assemble(C[fn]["rows"], E[fn], a.seed)
        for panel, key in ((False, "A_oracle_free"), (True, "B_panel_matched")):
            b, n = brute(d, panel)
            ref = M18[fn]["arms"][key]
            same = abs(b - ref["effect"]) < 5e-4 and n == ref["npair"]
            OK["V3"] &= same
            print(f"  {nm:12s} {key:16s} 蛮力 {b:+8.4f}/{n:6d}  M18 {ref['effect']:+8.4f}/"
                  f"{ref['npair']:6d}  {'一致' if same else '**不一致**'}")

    print("\n=== 验证结论 ===")
    for k, v in OK.items():
        print(f"  {k} {'通过' if v else '**不通过**'}")
    allok = all(OK.values())
    print(f"  **{'修法经得起检验 —— M18/M19 的裁定维持' if allok else '修法有问题 —— M18/M19 必须退回'}**")
    json.dump({"checks": OK, "detail": OUT, "all_pass": allok},
              open(os.path.join(HERE, "m18v_verify_repair.json"), "w"), ensure_ascii=False, indent=1)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
