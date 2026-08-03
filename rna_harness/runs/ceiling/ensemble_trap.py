#!/usr/bin/env python3
"""The other half of the crossing: move the CRITERION from a structure to a distribution.

The judge was moved first (neighbour_judge.py) and that turned up a bias -- --stop undercounts by 6.5 points
-- but it did not widen anything. The criterion itself is still pointwise and binary:

    trap_fraction = #{k : MFE(seq[:k]) has ANY pair the target lacks} / #{k}

MFE is one structure out of the ensemble, and "any pair" is a 0/1 verdict on a continuous quantity. Measured
consequence: on natural riboswitches this fires on 91.7% of transcription intermediates, and two pre-registered
attempts to fix that by re-cutting the SET of traps (sticky/non-sticky, and scale) both failed -- one of them
backwards. Neither touched the fact that the verdict is binary.

The ensemble version asks how much probability mass sits on pairs the target does not have:

    off_mass(k) = SUM over (i,j) NOT in target of p_ij(seq[:k])        (one partition function per prefix)
    ens_mean    = mean over k of off_mass(k)
    ens_frac_t  = #{k : off_mass(k) > t} / #{k}                        for a declared grid of t

Why this is expected to help where the other two did not: a prefix whose MFE carries one marginal off-target
pair at p=0.05 and one where p=0.9 are the SAME EVENT today. Directly observed on a real design -- at k=20 the
MFE reports no off-target pair while 1.34 pairs' worth of probability mass is still off-target.

THREE PRE-REGISTERED CONDITIONS, and passing fewer than all three means this is not an improvement:

  1. PREDICTION. Some ensemble variant tracks visit-based Kinfold reach more strongly than trap_fraction does,
     with a bootstrap resampled over TARGETS (candidates of one target are not independent).
  2. THE FIRING RATE. On natural constructs it must fire on materially less than 91.7%. A variant that
     predicts better and still fires on everything has not fixed what is broken -- that is exactly what the
     scale variant did (+0.018 in rho, 88.6% firing).
  3. THE NULL MUST SURVIVE. Inside an isomer family the base multiset is identical by construction, so a
     criterion that is secretly a composition proxy cannot separate there. trap_fraction scores +10.7 points
     [+7.1, +14.3] on that null. The ensemble variant must too. Buying dynamic range by loosening is the
     failure mode, and this is the test that catches it.

Reported whatever it says, including the case where the ensemble is simply worse.
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

MIN_PREFIX = 12
THETAS = (0.1, 0.25, 0.5, 1.0, 2.0)


def ensemble_profile(seq: str, target: str):
    """Off-target probability mass at every prefix, plus today's binary flag from the same folding."""
    import RNA
    from rna_harness.benchmarks.kinetic_suite import pair_set
    tgt = pair_set(target)
    prof = []
    for k in range(MIN_PREFIX, len(seq) + 1):
        sub = seq[:k]
        fc = RNA.fold_compound(sub)
        fc.pf()
        b = fc.bpp()
        m = 0.0
        for i in range(1, k + 1):
            row = b[i]
            for j in range(i + 1, k + 1):
                if (i - 1, j - 1) not in tgt:
                    m += row[j]
        prof.append({"k": k, "off_mass": m,
                     "mfe_trap": bool(pair_set(RNA.fold(sub)[0]) - tgt)})
    return prof


def variants(prof):
    n = max(1, len(prof))
    out = {"trap_fraction": sum(1 for p in prof if p["mfe_trap"]) / n,
           "ens_mean": sum(p["off_mass"] for p in prof) / n,
           "ens_max": max((p["off_mass"] for p in prof), default=0.0)}
    for t in THETAS:
        out[f"ens_frac_{t}"] = sum(1 for p in prof if p["off_mass"] > t) / n
    return out


def work(job):
    tag, tid, seq, target, extra = job
    try:
        v = variants(ensemble_profile(seq, target))
    except Exception as e:                                                     # noqa: BLE001
        return {"tag": tag, "task_id": tid, "error": repr(e)[:120]}
    v.update({"tag": tag, "task_id": tid, "n": len(target), **extra})
    return v


def boot(rows, a, b, key, n_boot=4000, seed=0):
    import numpy as np
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["task_id"], []).append(r)
    keys = list(by)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        sub = [r for i in pick for r in by[keys[i]]]
        if len(sub) < 10:
            continue
        ra = spearmanr([r[a] for r in sub], [r[key] for r in sub])[0]
        rb = spearmanr([r[b] for r in sub], [r[key] for r in sub])[0]
        if ra == ra and rb == rb:
            diffs.append(abs(ra) - abs(rb))
    if not diffs:
        return None
    diffs.sort()
    return {"mean": float(sum(diffs) / len(diffs)),
            "lo95": float(diffs[int(0.025 * len(diffs))]),
            "hi95": float(diffs[int(0.975 * len(diffs))]),
            "p_one_sided": float(sum(1 for d in diffs if d <= 0) / len(diffs))}


def run(mode, workers, out):
    import numpy as np
    from scipy.stats import spearmanr, wilcoxon
    jobs = []

    if mode == "predict":
        # visit-based reach, because the marker undercounts -- see revisit_calibration.py
        from rna_harness.benchmarks.bprna_nested import load_bprna_nested
        big = json.load(open(os.path.join(HERE, "kinetic_calibrate_big.json")))
        rv = json.load(open(os.path.join(HERE, "revisit_calibration.json")))
        tmap = {str(r["id"]): str(r["target"]) for r in load_bprna_nested(big.get("pool", 400))}
        # the re-judged rows carry no sequence, so they are matched back on (target, gc, trap_fraction);
        # any key that is not unique is DROPPED rather than guessed at
        idx = {}
        for r in big["rows"]:
            if r.get("error"):
                continue
            for c in r["candidates"]:
                k = (r["task_id"], round(c["gc"], 4), round(c["trap_fraction"], 4))
                idx[k] = None if k in idx else c["seq"]
        dropped = 0
        for r in rv["rows"]:
            k = (r["task_id"], round(r["gc"], 4), round(r["trap_fraction"], 4))
            seq, tg = idx.get(k), tmap.get(r["task_id"])
            if not seq or not tg or len(tg) != len(seq):
                dropped += 1
                continue
            jobs.append(("predict", r["task_id"], seq, tg,
                         {"visit_reach": r["visit_reach"], "gc": r["gc"], "accepted": r["accepted"]}))
        print(f"预测检验:{len(jobs)} 条(回填序列时丢弃了 {dropped} 条键不唯一或对不上的)\n", flush=True)

    elif mode == "natives":
        import glob
        from experiment_check import parse_rdat
        ec = json.load(open(os.path.join(HERE, "experiment_check.json")))
        tgt = {r["rmdb_id"]: r["target"] for r in ec["rows"] if not r.get("skipped")}
        for p in sorted(glob.glob(os.path.join(HERE, "rmdb_cache", "*.rdat"))):
            rid = os.path.basename(p)[:-5]
            if rid not in tgt:
                continue
            rec = parse_rdat(open(p, encoding="utf8", errors="ignore").read())
            jobs.append(("natives", rid, rec["sequence"], tgt[rid], {}))
        print(f"天然触发率:{len(jobs)} 个参照合格的构建体\n", flush=True)

    else:  # isomer
        from rna_harness.benchmarks.bprna_nested import load_bprna_nested
        iso = json.load(open(os.path.join(HERE, "kinetic_null_isomer2.json")))
        tmap = {str(r["id"]): str(r["target"]) for r in load_bprna_nested(1200)}
        for r in iso["rows"]:
            if r.get("error") or r.get("skipped"):
                continue
            tg = tmap.get(r["task_id"])
            if not tg:
                continue
            for c in r["candidates"]:
                if c.get("kinfold"):
                    jobs.append(("isomer", r["task_id"], c["seq"], tg,
                                 {"reach": c["kinfold"]["reach"], "accepted": c["accepted"]}))
        print(f"同分异构体零假设:{len(jobs)} 条异构体,来自 {len({j[1] for j in jobs})} 个家族\n"
              f"  注意:这批的到达率是【标记式】的旧口径(低估),但两个判据用的是同一批数值,比较仍成立\n",
              flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 100 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    rows = [r for r in R if not r.get("error")]
    names = ["trap_fraction", "ens_mean", "ens_max"] + [f"ens_frac_{t}" for t in THETAS]

    if mode == "natives":
        print(f"\n=== 各定义在天然构建体上的触发率(构建体平均)===")
        agg = {}
        for nm in names:
            v = [r[nm] for r in rows]
            agg[nm] = float(np.mean(v))
            mark = "  <- 今天的规则" if nm == "trap_fraction" else ""
            unit = "" if nm.startswith("ens_frac") or nm == "trap_fraction" else " (质量,非比例)"
            print(f"  {nm:16s} {np.mean(v):.3f}{unit}{mark}")
        json.dump({"mode": mode, "mean": agg, "rows": rows}, open(out, "w"),
                  ensure_ascii=False, indent=1)
        print(f"\n  wrote {out}")
        return

    key = "visit_reach" if mode == "predict" else "reach"
    print(f"\n=== 与 Kinfold 到达率的 Spearman(越负越好)===")
    sp = {}
    for nm in names:
        xs = [r[nm] for r in rows]
        if len(set(xs)) < 3:
            continue
        rho, p = spearmanr(xs, [r[key] for r in rows])
        sp[nm] = {"rho": float(rho), "p": float(p)}
        mark = "  <- 今天的规则" if nm == "trap_fraction" else ""
        print(f"  {nm:16s} rho={rho:+.3f}  p={p:.2g}{mark}")
    best = max((n for n in sp if n != "trap_fraction"),
               key=lambda n: abs(sp[n]["rho"]), default=None)
    b = None
    if best:
        b = boot(rows, best, "trap_fraction", key)
        print(f"\n  最强的系综变体:{best}(rho={sp[best]['rho']:+.3f})")
        if b:
            print(f"  |rho({best})| - |rho(trap_fraction)| = {b['mean']:+.3f}  "
                  f"95% [{b['lo95']:+.3f}, {b['hi95']:+.3f}]  单侧 p={b['p_one_sided']:.3g}")
            print(f"  -> {'系综版更强' if b['lo95'] > 0 else '分不开或更弱 —— 照实报'}")

    extra = {}
    if mode == "isomer":
        # inside a family composition is identical BY CONSTRUCTION, so a composition proxy cannot separate
        print(f"\n  [零假设] 家族内配对:低陷阱减高陷阱的到达率")
        fam = {}
        for r in rows:
            fam.setdefault(r["task_id"], []).append(r)
        for nm in ("trap_fraction", best):
            if not nm:
                continue
            A, B = [], []
            for v in fam.values():
                lo = [x[key] for x in v if x[nm] <= (0.10 if nm == "trap_fraction"
                                                     else st.median([y[nm] for y in v]))]
                hi = [x[key] for x in v if x[nm] > (0.10 if nm == "trap_fraction"
                                                    else st.median([y[nm] for y in v]))]
                if lo and hi:
                    A.append(st.mean(lo))
                    B.append(st.mean(hi))
            if len(A) >= 6:
                dif = np.array(A) - np.array(B)
                p = float(wilcoxon(np.array(A), np.array(B), alternative="greater")[1])
                extra[nm] = {"n_fam": len(A), "gap": float(dif.mean()), "p": p}
                print(f"    {nm:16s} {dif.mean():+.1%}  更好 {int((dif>0).sum())}/{len(dif)}  p={p:.3g}")
            else:
                print(f"    {nm:16s} 家族不足 6,不下结论")
    json.dump({"mode": mode, "spearman": sp, "best": best, "boot_vs_trap": b,
               "isomer_null": extra, "rows": rows}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("predict", "natives", "isomer"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.mode, a.workers, a.out or os.path.join(HERE, f"ensemble_trap_{a.mode}.json"))


if __name__ == "__main__":
    main()
