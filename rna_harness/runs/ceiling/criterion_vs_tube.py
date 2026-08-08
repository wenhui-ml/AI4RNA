#!/usr/bin/env python3
"""Does OUR criterion predict the tube? Head-to-head against RibonanzaNet, on the same molecules.

WHY THIS IS THE EXPERIMENT THAT MATTERS. The audit of 2026-08-08 closed the selection story: picking better
out of our own pool is worth at most +0.55 wet-lab points, and the 11th-of-14 placement is a GENERATION gap
against methods at our own novelty (28.5% identity to wild type; they score 0.93-0.98 crossed-pair F1, we
score 0.816). A generator, though, is only ever as good as the thing it optimises. Ours optimises a
thermodynamic panel -- pKiss, ProbKnot, ShapeKnots, agreeing on the crossing stems. So the question under
the gap is whether that panel tracks reality at all.

    I have twice quoted "our panel correlates with the tube at +0.155, RibonanzaNet at +0.364" as though it
    were measured. It is not. There is no artifact on disk for either number, exactly as there was none for
    the "+1.74 wet-lab points" retracted this morning. Two unsourced numbers in two days is a pattern, and
    the fix is to measure it once, properly, and write it down.

WHAT MAKES THIS ANSWERABLE. OpenKnot publishes 34,750 real designs that were actually synthesised, each with
a measured `target_openknot_score`. Our panel can be run on those same sequences. So both criteria can be
scored against the tube, on identical molecules, paired.

    This also tests something the project has never tested. Our certificate's entire meaning is that the
    panel confirmed the crossing stems. Whether a panel-confirmed molecule behaves better IN A TUBE than an
    unconfirmed one has never once been measured, on any data. G3 measures it.

PRE-REGISTERED, before any adjudication runs. The independent unit is the TARGET, never the design: designs
within a puzzle share a structure and a difficulty, and pooling them is the pseudo-replication that produced
the retracted p = 2.05e-28. Correlations are computed within target and combined across targets.

    G2  CRITERION HEAD-TO-HEAD (primary). Within target, Spearman against the wet-lab score for
        (a) the published RNet crossed-pair F1 and (b) our panel's mean crossing recall. Paired difference
        per target, Wilcoxon over targets, one-sided.
        PASS: RNet's rho exceeds ours, p < 0.025 -> our generator optimises a criterion that tracks reality
        less well than a model we already run, which is a sufficient explanation for the generation gap.
        KILL: no difference, or ours is better -> the criterion is NOT the bottleneck and the gap is in the
        search or in the metric, so this whole line closes and the next move is elsewhere.

    G3  DOES OUR CERTIFICATE MEAN ANYTHING IN A TUBE (equally important, reported whatever G2 says).
        Within target, wet-lab score of designs our panel confirms (k >= 1 oracle at recall 1.0, the shipped
        operating point) versus designs it does not. Wilcoxon over targets on the within-target difference.
        This is the first external validation our shipped criterion has ever had. A null here is a far bigger
        result than anything in G2, and it is reported at the same prominence either way.

    G4  DO THE TWO CRITERIA EVEN AGREE. Within-target Spearman between our panel's recall and RNet's F1.
        Descriptive; it tells us whether they are rivals measuring one thing or two different things.

COST: $0 API. Local CPU, local oracle binaries.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
ALPHA = 0.025


def work(item):
    """Adjudicate one real design with the shipped panel. Returns the criterion summaries only."""
    puz, seq, target, f1c, wet, method = item
    from rna_harness.cognitive.certify import _knot_recall
    try:
        kr = _knot_recall(seq, target)
    except Exception as e:                                                     # noqa: BLE001
        return {"puzzle": puz, "error": repr(e)[:120]}
    vals = [v for v in kr["per_oracle"].values() if v is not None]
    if not vals:
        return {"puzzle": puz, "error": "all oracles failed to run"}
    return {"puzzle": puz, "method": method, "f1c": f1c, "wet": wet,
            "panel_mean": sum(vals) / len(vals), "panel_max": max(vals),
            "panel_k": sum(1 for v in vals if v >= 1.0),
            "confirmed": bool(kr["knot_confirmed"]), "n_cross": kr["n_cross"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-target", type=int, default=60)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "criterion_vs_tube.json"))
    a = ap.parse_args()

    from leaderboard_design import targets
    from m2r_pairwise import spearman, wilcoxon

    T = targets()
    csv.field_size_limit(10 ** 9)
    pool = collections.defaultdict(list)
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {k: i for i, k in enumerate(next(r))}
        for x in r:
            if x[idx["SN_filter"]] != "1":
                continue
            p, s = x[idx["puzzle"]], x[idx["design_sequence"]]
            # only designs scored against the SAME target string we would design for; six puzzles carry
            # two variants and four of those differ in length
            if p not in T or not s or x[idx["target_structure"]] != T[p]["target"]:
                continue
            try:
                f1c = float(x[idx["RNet_F1_crossed_pair"]])
                wet = float(x[idx["target_openknot_score"]])
            except (TypeError, ValueError):
                continue
            pool[p].append((s, f1c, wet, x[idx["method"]]))

    rng = random.Random(a.seed)
    jobs = []
    for p, rows in sorted(pool.items()):
        if len(rows) < 20:
            continue
        for s, f1c, wet, m in rng.sample(rows, min(a.per_target, len(rows))):
            jobs.append((p, s, T[p]["target"], f1c, wet, m))
    print(f"{len({j[0] for j in jobs})} 个靶标,随机抽 {len(jobs)} 条真实设计(每靶标最多 {a.per_target} 条)")
    print(f"用出厂面板逐条裁决,{a.workers} 进程\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    import time
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, res in enumerate(ex.map(work, jobs, chunksize=4), 1):
            R.append(res)
            if n % 400 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    bad = [r for r in R if "error" in r]
    R = [r for r in R if "error" not in r]
    print(f"\n  裁决成功 {len(R)},失败 {len(bad)}"
          + (f"({collections.Counter(r['error'] for r in bad).most_common(1)[0][0]})" if bad else ""))

    by = collections.defaultdict(list)
    for r in R:
        by[r["puzzle"]].append(r)

    # ---- G2 ------------------------------------------------------------------------------------------
    rows = []
    for p, rs in sorted(by.items()):
        if len(rs) < 15:
            continue
        wet = [r["wet"] for r in rs]
        if len(set(wet)) < 3:
            continue
        a1 = spearman([r["f1c"] for r in rs], wet)
        a2 = spearman([r["panel_mean"] for r in rs], wet)
        if a1 is None or a2 is None:
            continue
        rows.append((p, a1, a2, len(rs)))
    d = [x[1] - x[2] for x in rows]
    w, p2, n2 = wilcoxon(d, "greater")
    print(f"\n=== G2 哪个判据更能预测试管(靶标内 Spearman,靶标为独立单元)===")
    print(f"  {len(rows)} 个靶标")
    print(f"  RibonanzaNet 交叉配对 F1   rho 中位 {st.median([x[1] for x in rows]):+.3f}")
    print(f"  我们的面板 平均交叉召回    rho 中位 {st.median([x[2] for x in rows]):+.3f}")
    print(f"  配对差 中位 {st.median(d):+.3f},RNet 更好的靶标 {sum(1 for x in d if x>0)}/{len(d)},"
          f"p = {'n/a' if p2 is None else f'{p2:.3g}'}")
    g2 = bool(p2 is not None and p2 < ALPHA and st.median(d) > 0)
    print(f"  {'通过 —— 我们优化的判据比 RNet 更不贴近现实,足以解释生成缺口' if g2 else 'KILL —— 判据不是瓶颈,缺口在别处'}")

    # ---- G3 ------------------------------------------------------------------------------------------
    g3rows = []
    for p, rs in sorted(by.items()):
        yes = [r["wet"] for r in rs if r["confirmed"]]
        no = [r["wet"] for r in rs if not r["confirmed"]]
        if len(yes) >= 3 and len(no) >= 3:
            g3rows.append((p, st.median(yes) - st.median(no), len(yes), len(no)))
    dd = [x[1] for x in g3rows]
    w3, p3, n3 = wilcoxon(dd, "greater")
    conf = sum(1 for r in R if r["confirmed"])
    print(f"\n=== G3 面板确认过的分子,在试管里真的更好吗(出厂判据的首次外部检验)===")
    print(f"  {len(R)} 条真实设计里,面板确认 {conf} 条({conf/max(1,len(R)):.1%})")
    print(f"  {len(g3rows)} 个靶标两组都够大:确认组 − 未确认组 的湿实验中位差 "
          f"{st.median(dd) if dd else float('nan'):+.2f} 分,"
          f"确认组更好 {sum(1 for x in dd if x>0)}/{len(dd)},p = {'n/a' if p3 is None else f'{p3:.3g}'}")
    g3 = bool(p3 is not None and p3 < ALPHA and dd and st.median(dd) > 0)
    print(f"  {'通过 —— 我们的证书在试管里有意义' if g3 else 'KILL —— 面板确认与试管表现没有可检出的关系,这是本项目最重要的一个空值'}")

    # ---- G4 ------------------------------------------------------------------------------------------
    agree = []
    for p, rs in sorted(by.items()):
        if len(rs) >= 15:
            s = spearman([r["panel_mean"] for r in rs], [r["f1c"] for r in rs])
            if s is not None:
                agree.append(s)
    print(f"\n=== G4 两个判据彼此有多一致 ===")
    print(f"  靶标内 Spearman(面板召回, RNet F1)中位 {st.median(agree):+.3f}({len(agree)} 个靶标)")

    json.dump({"n_designs": len(R), "n_failed": len(bad), "per_target_sampled": a.per_target,
               "G2": {"n_targets": len(rows), "rho_rnet_median": st.median([x[1] for x in rows]),
                      "rho_panel_median": st.median([x[2] for x in rows]),
                      "paired_diff_median": st.median(d), "p": p2, "pass": g2,
                      "wins": sum(1 for x in d if x > 0)},
               "G3": {"n_targets": len(g3rows), "confirmed_frac": conf / max(1, len(R)),
                      "wetlab_diff_median": st.median(dd) if dd else None, "p": p3, "pass": g3,
                      "wins": sum(1 for x in dd if x > 0)},
               "G4": {"n_targets": len(agree), "rho_median": st.median(agree)},
               "per_target": {x[0]: {"rho_rnet": x[1], "rho_panel": x[2], "n": x[3]} for x in rows}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
