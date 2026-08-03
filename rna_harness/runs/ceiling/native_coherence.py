#!/usr/bin/env python3
"""Does the criterion behave the same way on NATURAL RNAs as it does on our designs?

THE TENSION THIS CLOSES. trap_fraction predicts Kinfold well on DesiRNA candidates (rho -0.47) and fires on
91.7% of natural transcription intermediates. Two attempts to fix that firing rate were pre-registered and
both died: the sticky/non-sticky split is backwards (sticky rho +0.014, non-sticky -0.481), and scale buys
+0.018 while barely moving the rate. What survived is a hypothesis, not a conclusion:

    natural riboswitches really ARE full of transient off-target pairing during transcription, and that is
    the same class that predicts folding failure in designs -- so 91.7% is biology, not miscounting.

That hypothesis makes one sharp prediction, and it is cheap to check because both halves are already on disk.
If the criterion is one coherent quantity rather than something tuned to our own generator, then on NATURAL
sequences it must ALSO track how often Kinfold reaches the target.

    it tracks on natives too        -> the criterion is coherent across populations; 91.7% says natural
                                       riboswitches would not pass a design filter, which is a fact about
                                       riboswitches and not a defect
    it tracks only on our designs   -> the criterion is a property of DesiRNA output, not of RNA, and its
                                       scope has to be written down as such. That is a hard limit on the
                                       second ruler and would be the more important finding.

DESIGN, and the reason it is not a copy of the design-side test. There is no "target" handed to a natural
RNA by us: the reference comes from RNAstructure's SHAPE-directed fold on the full-length experimental
reactivity, the same reference experiment_check.py uses, and only constructs whose reference explains their
own data (AUROC >= 0.65) are admitted -- the gate that turned that file's headline from p=0.049 to p=0.109.

The unit of variation is the PREFIX, not the molecule: with ~22 constructs there is no power in a
between-molecule correlation. So each construct contributes many (prefix, reached?) observations, and the
question is asked WITHIN a construct -- does Kinfold, started on a longer template, reach the reference
structure less often when our criterion says the path is trappy? Standard errors are clustered on the
molecule because deposits of one riboswitch are not independent RNAs.

DECLARED BEFORE THE RUN: the criterion is coherent if the within-construct correlation between
trap_fraction-of-the-prefix-path and Kinfold reach is negative, with a molecule-level sign test over
constructs. Fewer than 6 usable molecules -> report 样本不足, not a null. Same gate as everywhere else here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from experiment_check import parse_rdat                                       # noqa: E402
from kinetic_calibrate import kinfold_reach                                   # noqa: E402

MIN_MOLECULES = 6
MIN_PREFIX = 12


def path_trap_fraction(seq: str, target: str, upto: int) -> float:
    """trap_fraction of the path that produced a transcript of length `upto`.

    This is the criterion applied to exactly what Kinfold will be asked to grow: prefixes MIN_PREFIX..upto,
    judged against the reference restricted to those positions. Judging against the FULL reference would
    count pairs that cannot exist yet as if the molecule had failed to form them.
    """
    import RNA
    from rna_harness.benchmarks.kinetic_suite import pair_set
    tgt = pair_set(target)
    ks = list(range(MIN_PREFIX, upto + 1))
    if not ks:
        return 0.0
    traps = 0
    for k in ks:
        if pair_set(RNA.fold(seq[:k])[0]) - tgt:
            traps += 1
    return traps / len(ks)


def work(job):
    rid, seq, target, lengths = job
    t0 = time.time()
    rows = []
    for k in lengths:
        pre = seq[:k]
        sub = target[:k]
        # the reference truncated to this prefix must still be a legal structure to hand Kinfold
        depth, keep = 0, []
        for ch in sub:
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    keep.append(".")
                    continue
                depth -= 1
            keep.append(ch)
        sub = "".join(keep)
        if depth:                      # unmatched openers: drop them rather than hand Kinfold garbage
            out, d = [], depth
            for ch in reversed(sub):
                if ch == "(" and d > 0:
                    out.append(".")
                    d -= 1
                else:
                    out.append(ch)
            sub = "".join(reversed(out))
        if sub.count("(") < 2:
            continue
        kf = kinfold_reach(pre, sub)
        if not kf:
            continue
        rows.append({"k": k, "reach": kf["reach"], "n_traj": kf["n_traj"],
                     "trap_fraction": path_trap_fraction(seq, target, k)})
    return {"rmdb_id": rid, "n": len(seq), "n_points": len(rows), "rows": rows,
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=6, help="sample every Nth transcript length")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "native_coherence.json"))
    a = ap.parse_args()

    ec = json.load(open(os.path.join(HERE, "experiment_check.json")))
    tgt = {r["rmdb_id"]: r["target"] for r in ec["rows"] if not r.get("skipped")}
    jobs = []
    for p in sorted(glob.glob(os.path.join(HERE, "rmdb_cache", "*.rdat"))):
        rid = os.path.basename(p)[:-5]
        if rid not in tgt:
            continue
        rec = parse_rdat(open(p, encoding="utf8", errors="ignore").read())
        ks = sorted(k for k in rec["lengths"] if 25 <= k <= 160)[::a.step]
        if len(ks) >= 5:
            jobs.append((rid, rec["sequence"], tgt[rid], ks))
    print(f"天然序列上的一致性检验:{len(jobs)} 个参照合格的构建体\n"
          f"  裁判 Kinfold(与设计那一侧同一个),判据 = 该前缀路径上的 trap_fraction\n"
          f"  只用参照能解释自己数据的构建体(AUROC>=0.65),门槛与 experiment_check 相同\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            print(f"    {n}/{len(jobs)}  {R[-1]['rmdb_id']:22s} {R[-1]['n_points']} 点  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    import numpy as np
    from scipy.stats import spearmanr, wilcoxon
    mol = lambda r: re.match(r"[A-Z0-9]+", r).group(0)                         # noqa: E731
    per = {}
    for r in R:
        if len(r["rows"]) < 5:
            continue
        rho = spearmanr([x["trap_fraction"] for x in r["rows"]],
                        [x["reach"] for x in r["rows"]])[0]
        if rho == rho:
            per.setdefault(mol(r["rmdb_id"]), []).append(rho)
    print(f"\n=== 构建体内部 Spearman(trap_fraction vs Kinfold 到达率)===")
    ms = [(m, float(np.mean(v)), len(v)) for m, v in sorted(per.items())]
    for m, v, n in ms:
        print(f"  {m:10s} {v:+.3f}  ({n} 个构建体)")
    if len(ms) < MIN_MOLECULES:
        print(f"\n  !! 只有 {len(ms)} 个分子,未达跑前声明的 {MIN_MOLECULES} —— 报样本不足,不报 null")
        json.dump({"gate": "insufficient", "per_molecule": ms, "rows": R},
                  open(a.out, "w"), ensure_ascii=False, indent=1)
        return
    A = np.array([v for _, v, _ in ms])
    neg = int((A < 0).sum())
    p = float(wilcoxon(A, alternative="less")[1])
    print(f"\n  {neg}/{len(A)} 个分子为负   均值 {A.mean():+.3f}   Wilcoxon vs 0  p={p:.3g}")
    print(f"  设计那一侧的同一个量:rho=-0.469(1596 条候选)")
    ok = p < 0.05 and A.mean() < 0
    print(f"\n  预注册:天然序列上也为负 -> "
          f"{'通过 —— 判据在两个总体上一致,91.7% 是生物学不是数错' if ok else '未通过 —— 判据可能只是 DesiRNA 输出的性质,适用范围要写死'}")
    json.dump({"per_molecule": ms, "mean_rho": float(A.mean()), "n_negative": neg,
               "wilcoxon_p": p, "coherent": bool(ok), "design_side_rho": -0.469,
               "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
