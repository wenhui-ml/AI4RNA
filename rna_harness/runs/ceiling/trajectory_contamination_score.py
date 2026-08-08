#!/usr/bin/env python3
"""Stage B: what does the trajectory contamination cost us, in the units we are graded in?

RUNS IN THE RibonanzaNet venv. Stage A ran the solver once per target and kept every candidate labelled by
the file it came from, so both arms come from the SAME run and no comparison depends on rerunning anything.

    T1  SEPARATION. Per target, median RNet crossed-pair F1 of DesiRNA's reported answers versus the
        trajectory states our scrape mixes in with them. Paired within target, Wilcoxon, one-sided greater.
        PASS: p < 0.025.
        KILL: no difference -> the trace states are as good as the answers and the scrape is harmless.

    T3  COST. The pool median as the product builds it today (first 200 in walk order, whatever they are)
        against the pool median over reported answers alone. This is the size of the win available from
        excluding one directory.

    T4  DOES IT CLOSE THE GAP. The reported-only pool median against the novelty-matched peer median that
        `generation_gap.py` measured us 0.209 below on 57 of 57 targets. Whatever remains after the fix is
        the honest size of the generation gap.

COST: $0. Local CPU.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pool_rerank_score import _score_chunk                                      # noqa: E402

DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
PEERS = ["Struct2SeQ", "MPNN-RFdiff", "gRNAde-no3d", "codesign-RFdiff"]
ALPHA = 0.025


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=os.path.join(HERE, "trajectory_contamination_pools.json"))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--cap", type=int, default=200, help="the shipped adapter's pool cap")
    ap.add_argument("--out", default=os.path.join(HERE, "trajectory_contamination_score.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    P = json.load(open(a.pools))
    T = P["targets"]
    rows = [r for r in P["rows"] if "error" not in r]
    jobs = []
    for r in rows:
        tgt = T[r["puzzle"]]["target"]
        for i, s in enumerate(r["reported"]):
            jobs.append((("rep", r["puzzle"], i), s, tgt))
        for i, s in enumerate(r["trace"]):
            jobs.append((("tr", r["puzzle"], i), s, tgt))
    print(f"{len(rows)} 个靶标,{len(jobs)} 条候选,{a.workers} 进程\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 8))
    chunks = [jobs[i:i + CH] for i in range(0, len(jobs), CH)]
    sc, t0, done = {}, time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, chunks):
            for k, v in res:
                sc[k] = v
            done += 1
            if done % 40 == 0 or done == len(chunks):
                print(f"    {done}/{len(chunks)}  {(time.time()-t0)/60:.1f} min", flush=True)

    per = {}
    for r in rows:
        p = r["puzzle"]
        rep = [sc[("rep", p, i)] for i in range(len(r["reported"])) if ("rep", p, i) in sc]
        tr = [sc[("tr", p, i)] for i in range(len(r["trace"])) if ("tr", p, i) in sc]
        # Rebuild the pool the shipped adapter would have kept: walk order, capped, dedup already applied.
        kinds = r["shipped_slice_kinds"][:a.cap]
        ri, ti, shipped = 0, 0, []
        for k in kinds:
            if k == "reported" and ri < len(rep):
                shipped.append(rep[ri]); ri += 1
            elif k == "trace" and ti < len(tr):
                shipped.append(tr[ti]); ti += 1
        per[p] = {"rep": rep, "tr": tr, "shipped": shipped}

    def med(v):
        return st.median(v) if v else None

    print(f"\n=== T1 上报答案 vs 轨迹状态(同一次运行,同一个靶标)===")
    ps = [p for p in per if per[p]["rep"] and per[p]["tr"]]
    d = [med(per[p]["rep"]) - med(per[p]["tr"]) for p in ps]
    w, p1, _ = wilcoxon(d, "greater")
    print(f"  {len(ps)} 个靶标:上报中位 {st.median([med(per[p]['rep']) for p in ps]):.3f},"
          f"轨迹中位 {st.median([med(per[p]['tr']) for p in ps]):.3f}")
    print(f"  配对差 中位 {st.median(d):+.3f},上报更好 {sum(1 for x in d if x>0)}/{len(ps)},"
          f"p = {'n/a' if p1 is None else f'{p1:.3g}'}")
    t1 = bool(p1 is not None and p1 < ALPHA and st.median(d) > 0)
    print(f"  {'通过 —— 轨迹状态确实更差,混进池子是纯粹的污染' if t1 else 'KILL —— 轨迹状态并不更差,这个抓取方式无害'}")

    print(f"\n=== T3 修掉之后,池子会好多少 ===")
    ps2 = [p for p in per if per[p]["rep"] and per[p]["shipped"]]
    d2 = [med(per[p]["rep"]) - med(per[p]["shipped"]) for p in ps2]
    w, p2, _ = wilcoxon(d2, "greater")
    print(f"  现行池中位 {st.median([med(per[p]['shipped']) for p in ps2]):.3f}  ->  "
          f"只留上报 {st.median([med(per[p]['rep']) for p in ps2]):.3f}")
    print(f"  配对差 中位 {st.median(d2):+.3f},变好 {sum(1 for x in d2 if x>0)}/{len(ps2)},"
          f"p = {'n/a' if p2 is None else f'{p2:.3g}'}")

    # ---- T4 ------------------------------------------------------------------------------------------
    csv.field_size_limit(10 ** 9)
    theirs = collections.defaultdict(list)
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {k: i for i, k in enumerate(next(r))}
        for x in r:
            if x[idx["SN_filter"]] != "1" or x[idx["method"]] not in PEERS:
                continue
            p = x[idx["puzzle"]]
            if p not in T or x[idx["target_structure"]] != T[p]["target"]:
                continue
            try:
                theirs[p].append(float(x[idx["RNet_F1_crossed_pair"]]))
            except (TypeError, ValueError):
                pass
    print(f"\n=== T4 修掉之后,与同新颖度同行的缺口还剩多少 ===")
    out4 = {}
    for label, key in (("现行(含轨迹)", "shipped"), ("只留上报答案", "rep")):
        ts = [p for p in per if theirs.get(p) and per[p][key]]
        dd = [med(per[p][key]) - st.median(theirs[p]) for p in ts]
        w, pv, _ = wilcoxon(dd, "two-sided")
        print(f"  {label:14s} {len(ts)} 个靶标,差中位 {st.median(dd):+.3f},"
              f"我们更好 {sum(1 for x in dd if x>0):2d}/{len(ts)},"
              f"p = {'n/a' if pv is None else f'{pv:.3g}'}")
        out4[key] = {"n_targets": len(ts), "median_diff": st.median(dd), "p": pv,
                     "wins": sum(1 for x in dd if x > 0)}
    print(f"  (generation_gap.py 当时报的是 -0.209,0/57 —— 那是拿他们的成品比我们的草稿)")

    json.dump({"T1": {"n_targets": len(ps), "median_diff": st.median(d), "p": p1, "pass": t1,
                      "rep_median": st.median([med(per[p]["rep"]) for p in ps]),
                      "trace_median": st.median([med(per[p]["tr"]) for p in ps])},
               "T3": {"n_targets": len(ps2), "median_diff": st.median(d2), "p": p2,
                      "shipped_median": st.median([med(per[p]["shipped"]) for p in ps2]),
                      "reported_median": st.median([med(per[p]["rep"]) for p in ps2])},
               "T4": out4,
               "per_target": {p: {"rep_median": med(per[p]["rep"]), "trace_median": med(per[p]["tr"]),
                                  "shipped_median": med(per[p]["shipped"]),
                                  "n_rep": len(per[p]["rep"]), "n_trace": len(per[p]["tr"])}
                              for p in per}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
