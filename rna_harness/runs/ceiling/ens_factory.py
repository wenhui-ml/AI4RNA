#!/usr/bin/env python3
"""The factory process for the ensemble co-transcriptional criterion. Every gate can fail.

WHY A FACTORY PROCESS AND NOT A SWAP. ens_max/share is the only measured improvement of the period -- it
predicts an independent simulator markedly better (+0.147 in |rho| against the previous best idea's +0.018)
and is STRONGER on the null where composition is fixed by construction (+20.2% against +10.7%). None of that
entitles it to be the shipping rule. A criterion that ships carries a published error rate, and swapping one in
because it looked better on the comparison that motivated it is how a project talks itself into a false
positive.

THE CHECKLIST WAS WRONG AND WAS CORRECTED IN THE PRE-RUN REVIEW. An earlier version required the register-shift
(0/386) and nested (0/183) decoys to pass again. Those are nulls for the PSEUDOKNOT criterion knot_confirmed
and a change here cannot touch them. This criterion has exactly two nulls of its own.

GATES, all declared before the run:

  G1 SCALE      the threshold grid must straddle the statistic's measured scale. instrument.Declared enforces
                it -- and note that .against() only measures while .value enforces, which the first version of
                calibrate_theta got wrong by reading .scale and never .value.
  G2 SELECTIVITY the operating point must reproduce the shipping rule's ACCEPTANCE RATE on the same pool. A
                looser rule accepts more and therefore delivers more; that is loosening. Any comparison at
                unequal selectivity is refused rather than reported.
  G3 JUDGE      at equal selectivity, the new rule's accepted set must score higher on the visit-based Kinfold
                reach than the shipping rule's. Equal or worse ends the process.
  G4 ISOMER NULL within a family the base multiset is identical BY CONSTRUCTION, so a composition proxy cannot
                separate at all. The shipping rule scores +10.7 [+7.1, +14.3]; the replacement may not be
                weaker, or it bought its advantage by loosening in a way G2 did not catch.
  G5 GROW INVARIANCE the gap must keep its sign across a 16x span of transcription speed, as the shipping rule
                does (+16.8 / +14.4 / +17.7).

FAILING ANY GATE LEAVES IT AS A MEASURE. No threshold retuning afterwards -- that is the move this whole file
exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rna_harness.benchmarks.ensemble_kinetic import (calibrate_theta,               # noqa: E402
                                                     compare_at_equal_selectivity, held,
                                                     off_share_profile, summarise)
from rna_harness.instrument import InstrumentError, has_dynamic_range               # noqa: E402

SHIPPING_MAX = 0.10        # the rule in production, unchanged
ISOMER_FLOOR = 0.107       # the shipping rule's own isomer-null gap; the replacement may not be weaker
STATISTIC = "share_mean"   # declared; share_max is reported alongside but the gate is on this one


def work(job):
    tag, tid, seq, target, extra = job
    try:
        s = summarise(off_share_profile(seq, target))
    except Exception as e:                                                     # noqa: BLE001
        return {"tag": tag, "task_id": tid, "error": repr(e)[:120]}
    s.pop("shares", None)
    return {"tag": tag, "task_id": tid, "n": len(target), **s, **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "ens_factory.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    big = json.load(open(os.path.join(HERE, "kinetic_calibrate_big.json")))
    rv = json.load(open(os.path.join(HERE, "revisit_calibration.json")))
    tmap = {str(r["id"]): str(r["target"]) for r in load_bprna_nested(big.get("pool", 400))}
    idx = {}
    for r in big["rows"]:
        if r.get("error"):
            continue
        for c in r["candidates"]:
            k = (r["task_id"], round(c["gc"], 4), round(c["trap_fraction"], 4))
            idx[k] = None if k in idx else c["seq"]
    jobs = []
    for r in rv["rows"]:
        k = (r["task_id"], round(r["gc"], 4), round(r["trap_fraction"], 4))
        seq, tg = idx.get(k), tmap.get(r["task_id"])
        if seq and tg and len(tg) == len(seq):
            jobs.append(("pool", r["task_id"], seq, tg, {"visit_reach": r["visit_reach"]}))
    # G4's population: the isomer families, where composition is identical by construction
    iso = json.load(open(os.path.join(HERE, "kinetic_null_isomer2.json")))
    tm2 = {str(r["id"]): str(r["target"]) for r in load_bprna_nested(1200)}
    for r in iso["rows"]:
        if r.get("error") or r.get("skipped"):
            continue
        tg = tm2.get(r["task_id"])
        if not tg:
            continue
        for c in r["candidates"]:
            if c.get("kinfold"):
                jobs.append(("isomer", r["task_id"], c["seq"], tg, {"reach": c["kinfold"]["reach"]}))
    print(f"出厂流程:{sum(1 for j in jobs if j[0]=='pool')} 条设计池 + "
          f"{sum(1 for j in jobs if j[0]=='isomer')} 条异构体\n"
          f"  五道闸全部可以不过;不过就留在度量层,【不调阈值再过】\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 200 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    import numpy as np
    from scipy.stats import spearmanr, wilcoxon
    pool = [r for r in R if r["tag"] == "pool" and not r.get("error")]
    isor = [r for r in R if r["tag"] == "isomer" and not r.get("error")]
    verdict, out = {}, {}

    print(f"\n=== G1 量纲 ===")
    try:
        cal = calibrate_theta(pool, STATISTIC, SHIPPING_MAX)
        print(f"  {STATISTIC} 的实测量纲 10%/中位/90% = {cal['scale']['p10']:.3f} / "
              f"{cal['scale']['median']:.3f} / {cal['scale']['p90']:.3f}")
        print(f"  声明网格跨过量纲:{cal['scale']['straddles_scale']}")
        verdict["G1"] = True
    except InstrumentError as e:
        print(f"  !! {e}")
        verdict["G1"] = False
        cal = None

    if cal:
        print(f"\n=== G2 等选择性 ===")
        print(f"  出厂规则在这批池上的接受率 {cal['shipping_acceptance']:.1%}")
        print(f"  标定后的 theta={cal['theta']:.4f},新规则接受率 {cal['matched_acceptance']:.1%}")
        print(f"  匹配成功:{cal['selectivity_matched']}")
        verdict["G2"] = bool(cal["selectivity_matched"])

        print(f"\n=== G3 裁判(访问式 Kinfold 到达率,等选择性下比)===")
        cmp_ = compare_at_equal_selectivity(pool, [r["visit_reach"] for r in pool],
                                            STATISTIC, SHIPPING_MAX)
        out["compare"] = cmp_
        if "refused" in cmp_:
            print(f"  拒绝比较:{cmp_['refused']}")
            verdict["G3"] = False
        else:
            print(f"  出厂规则接受 {cmp_['n_shipping']} 条,平均到达率 {cmp_['judge_shipping']:.4f}")
            print(f"  新规则接受 {cmp_['n_new']} 条,平均到达率 {cmp_['judge_new']:.4f}")
            print(f"  差 {cmp_['delta']:+.4f}")
            verdict["G3"] = bool(cmp_["delta"] > 0)
        rho_s = spearmanr([r["trap_fraction"] for r in pool], [r["visit_reach"] for r in pool])
        rho_n = spearmanr([r[STATISTIC] for r in pool], [r["visit_reach"] for r in pool])
        print(f"  顺带(不是闸):Spearman 出厂 {rho_s[0]:+.3f}   新 {rho_n[0]:+.3f}")
        out["spearman"] = {"shipping": float(rho_s[0]), "new": float(rho_n[0])}

        print(f"\n=== G4 同分异构体零假设(下限 +{ISOMER_FLOOR:.1%},不许比出厂规则弱)===")
        fam = {}
        for r in isor:
            fam.setdefault(r["task_id"], []).append(r)
        rows = {}
        for name, keyf in (("trap_fraction", lambda r: r["trap_fraction"] <= SHIPPING_MAX),
                           (STATISTIC, lambda r: held(r, cal["theta"], STATISTIC))):
            A, B = [], []
            for v in fam.values():
                lo = [x["reach"] for x in v if keyf(x)]
                hi = [x["reach"] for x in v if not keyf(x)]
                if lo and hi:
                    A.append(st.mean(lo))
                    B.append(st.mean(hi))
            if len(A) >= 6:
                d = np.array(A) - np.array(B)
                p = float(wilcoxon(np.array(A), np.array(B), alternative="greater")[1])
                rows[name] = {"n_fam": len(A), "gap": float(d.mean()), "p": p,
                              "n_better": int((d > 0).sum())}
                print(f"  {name:14s} {d.mean():+.1%}  更好 {int((d>0).sum())}/{len(d)}  p={p:.3g}"
                      f"  (家族 {len(A)})")
            else:
                rows[name] = {"n_fam": len(A), "gap": None, "p": None}
                print(f"  {name:14s} 家族不足 6({len(A)}),不下结论")
        out["isomer"] = rows
        g = rows.get(STATISTIC, {}).get("gap")
        verdict["G4"] = bool(g is not None and g >= ISOMER_FLOOR)

    print(f"\n=== 出厂判定 ===")
    for k in ("G1", "G2", "G3", "G4"):
        print(f"  {k}  {'通过' if verdict.get(k) else '不过' if k in verdict else '未评估'}")
    print(f"  G5  转录速度不变性 —— 需要重跑 Kinfold,单独一步,本轮未评估")
    passed = all(verdict.get(k) for k in ("G1", "G2", "G3", "G4"))
    print(f"\n  -> {'前四道闸通过,可进入 G5' if passed else '未通过 —— 留在度量层,不调阈值再过'}")
    json.dump({"statistic": STATISTIC, "shipping_max": SHIPPING_MAX, "isomer_floor": ISOMER_FLOOR,
               "verdict": verdict, "calibration": cal, **out,
               "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
