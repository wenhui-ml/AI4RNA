#!/usr/bin/env python3
"""M61c: does the outcome's MEASUREMENT ERROR differ by pair type?

A review question M61 left open. M61 measured how noisy the rescue factor is; it did not ask
whether that noise is larger for one pair type than the other. If it were, the identity contrast
could in principle be an artifact of unequal precision rather than of formation, so the question
has to be answered rather than assumed away.

This is a REVIEW artifact: written after the run it checks, and it stakes nothing. It reports the
per-pair measurement SD of R by pair type, the same contrast re-run with inverse-variance weights,
and the same contrast on the half of the records with the smallest measurement error.

COST: $0, offline, CPU only.
"""
from __future__ import annotations
import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m61_readout_and_consequence as M61

SEED, NDRAW = 20260901, 200
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def main():
    rng = np.random.default_rng(SEED)
    log("=== M61c:结果变量的测量误差按对类型分 ===")
    log("  review 类产出:写在它检查的那次运行之后,不下任何赌注。全离线,$0。\n")
    recs, _et = M61.build()
    rows = M61.flat(recs)
    ev = M61.ctr(rows, "rf")
    log(f"  基线 rescue 对比 {ev.effect:+.5f}  [{ev.ci[0]:+.4f}, {ev.ci[1]:+.4f}]")

    sds = []
    for r in recs:
        P = np.array(r["prof"]); E = np.array(r["errs"])
        m = np.isfinite(P).all(0) & np.isfinite(E).all(0)
        if m.sum() < 10:
            sds.append(np.nan); continue
        p, e = P[:, m], np.abs(E[:, m])
        z = rng.standard_normal((NDRAW, 4, int(m.sum())))
        dd = p[None, :, :] + z * e[None, :, :]
        w = dd[:, 0, :]
        ri = np.sqrt(((w - dd[:, 1, :]) ** 2).mean(1))
        rj = np.sqrt(((w - dd[:, 2, :]) ** 2).mean(1))
        rij = np.sqrt(((w - dd[:, 3, :]) ** 2).mean(1))
        with np.errstate(divide="ignore", invalid="ignore"):
            sds.append(float(np.nanstd(1 - rij / np.sqrt(ri ** 2 + rj ** 2))))
    sds = np.array(sds, float)
    OUT = {"seed": SEED, "draws": NDRAW, "baseline": round(ev.effect, 5)}

    log("\n  每对 R 的测量 SD,按对类型")
    by = {}
    for t in ("CG", "AU", "GU"):
        v = np.array([s for s, r in zip(sds, recs) if r["pt"] == t and np.isfinite(s)])
        by[t] = {"n": len(v), "median": round(float(np.median(v)), 5),
                 "mean": round(float(v.mean()), 5)}
        log(f"    {t}: n={len(v):5d}  中位 {np.median(v):.5f}  均值 {v.mean():.5f}")
    OUT["sd_by_pair_type"] = by
    OUT["median_ratio_CG_over_AU"] = round(by["CG"]["median"] / by["AU"]["median"], 3)
    log(f"    G-C 与 A-U 的中位数之比 {OUT['median_ratio_CG_over_AU']:.3f}")

    # the same contrast, matched, with the measurement SD as the OUTCOME
    good = [(r, f, s) for r, f, s in zip(recs, rows, sds) if np.isfinite(s)]
    r2 = [{**f, "sdR": s} for _r, f, s in good]
    e_sd = M61.ctr(r2, "sdR")
    OUT["contrast_in_measurement_sd"] = {
        "effect": round(e_sd.effect, 6), "ci": [round(x, 5) for x in e_sd.ci],
        "p": round(e_sd.p, 5)}
    log(f"\n  把测量 SD 本身当结果跑同一个配平对比:{e_sd.effect:+.6f}"
        f"  [{e_sd.ci[0]:+.5f}, {e_sd.ci[1]:+.5f}]  p={e_sd.p:.4f}")

    # the contrast on the low-error half only
    med = float(np.nanmedian(sds))
    low = [f for _r, f, s in good if s <= med]
    e_low = M61.ctr(low, "rf")
    OUT["low_error_half"] = {"n": len(low), "effect": round(e_low.effect, 5),
                             "ci": [round(x, 4) for x in e_low.ci], "p": round(e_low.p, 5),
                             "retention": round(float(e_low.effect / ev.effect), 4)}
    log(f"  只用测量误差较小的那一半(n={len(low)}):{e_low.effect:+.5f}"
        f"  [{e_low.ci[0]:+.4f}, {e_low.ci[1]:+.4f}]  p={e_low.p:.4f}"
        f"  留存 {100*e_low.effect/ev.effect:.1f}%")

    json.dump(OUT, open(os.path.join(HERE, "m61c_error_by_type.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "m61c_error_by_type.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m61c_error_by_type.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
