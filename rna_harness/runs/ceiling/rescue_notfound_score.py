#!/usr/bin/env python3
"""Stage B: does the field-calibrated screen transfer to our own designs, and what would it hand over?

RUNS IN THE RibonanzaNet venv. Stage A collected pools for the 14 undelivered targets on the shipped ladder
and panel-screened every candidate. The panel is the SCREEN and RibonanzaNet is the JUDGE; RNet never
touched the design loop and the screen was calibrated on other people's molecules, so it cannot have been
fitted to this judge on these sequences.

    N2  TRANSFER (this is the test that can fail). Within our own pools, do candidates passing the screen
        score higher on RNet crossed-pair F1 than those failing it? Paired within target, one-sided.
        PASS: p < 0.025 and median difference >= 0.05.
        KILL: the screen does not separate inside our pools -> withdraw the coverage gain.
    N3  What a rescued delivery would actually be, per target, against the current answer of nothing.

COST: $0. Local CPU.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pool_rerank_score import _score_chunk                                      # noqa: E402

ALPHA = 0.025


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "rescue_notfound.json"))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(HERE, "rescue_notfound_score.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    S = json.load(open(a.src))
    T, thr = S["targets"], S["threshold"]
    rows = S["rows"]
    jobs = [((i,), r["seq"], T[r["puzzle"]]["target"]) for i, r in enumerate(rows)]
    print(f"{len(jobs)} 条候选打分,{a.workers} 进程\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 8))
    chunks = [jobs[i:i + CH] for i in range(0, len(jobs), CH)]
    sc = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, chunks):
            for k, v in res:
                sc[k[0]] = v
    for i, r in enumerate(rows):
        r["f1c"] = sc.get(i)
    rows = [r for r in rows if r.get("f1c") is not None]

    by = collections.defaultdict(list)
    for r in rows:
        by[r["puzzle"]].append(r)

    # ---- N2 -----------------------------------------------------------------------------------------
    d, detail = [], []
    for p in sorted(by):
        yes = [r["f1c"] for r in by[p] if r["panel_mean"] >= thr]
        no = [r["f1c"] for r in by[p] if r["panel_mean"] < thr]
        if len(yes) >= 3 and len(no) >= 3:
            d.append(st.median(yes) - st.median(no))
            detail.append((p, st.median(yes), st.median(no), len(yes), len(no)))
    w, p2, _ = wilcoxon(d, "greater")
    print(f"=== N2 这道筛在【我们自己的池子】里分得开吗(RNet 当裁判)===")
    print(f"  {'靶标':6s} {'过筛中位':>9s} {'未过筛中位':>10s} {'过筛数':>7s} {'未过数':>7s}")
    for p, y, n, ny, nn in detail:
        print(f"  {p:6s} {y:9.3f} {n:10.3f} {ny:7d} {nn:7d}")
    if d:
        print(f"\n  {len(d)} 个靶标:配对差中位 {st.median(d):+.3f},过筛更好 {sum(1 for x in d if x>0)}/{len(d)},"
              f"p = {'n/a' if p2 is None else f'{p2:.3g}'}")
        ok = bool(p2 is not None and p2 < ALPHA and st.median(d) >= 0.05)
        print(f"  {'通过 —— 在同行分子上标定的筛,迁移到我们自己的设计上依然有效' if ok else 'KILL —— 迁移失败,覆盖增益撤回,这 14 个维持未交付'}")
    else:
        ok = False
        print(f"  两组都够大的靶标不足,无法检验迁移")

    # ---- N3 -----------------------------------------------------------------------------------------
    print(f"\n=== N3 真按新筛交付,每个靶标会拿到什么(现状是什么都拿不到)===")
    print(f"  {'靶标':6s} {'nt':>4s} {'过筛数':>7s} {'交付的 F1':>10s} {'该池最好':>9s} {'平均召回':>9s}")
    deliver = {}
    for p in sorted(by):
        cand = [r for r in by[p] if r["panel_mean"] >= thr]
        best_pool = max((r["f1c"] for r in by[p]), default=0.0)
        if not cand:
            print(f"  {p:6s} {T[p]['n']:4d} {0:7d} {'—':>10s} {best_pool:9.3f} {'—':>9s}")
            continue
        pick = max(cand, key=lambda r: (r["panel_k"], r["panel_mean"]))
        deliver[p] = {"f1c": pick["f1c"], "panel_mean": pick["panel_mean"],
                      "panel_k": pick["panel_k"], "n_pass": len(cand), "pool_best": best_pool}
        print(f"  {p:6s} {T[p]['n']:4d} {len(cand):7d} {pick['f1c']:10.3f} {best_pool:9.3f} "
              f"{pick['panel_mean']:9.3f}")
    if deliver:
        v = [x["f1c"] for x in deliver.values()]
        print(f"\n  救回 {len(deliver)}/{len(T)} 个靶标,交付 F1 中位 {st.median(v):.3f}"
              f"(其中 >=0.5 的 {sum(1 for x in v if x>=0.5)} 个,=0.0 的 {sum(1 for x in v if x==0)} 个)")
        print(f"  覆盖:43/57 -> {43+len(deliver)}/57")

    json.dump({"threshold": thr, "N2": {"n_targets": len(d),
                                        "median_diff": (st.median(d) if d else None), "p": p2,
                                        "pass": ok,
                                        "per_target": [{"puzzle": x[0], "pass_median": x[1],
                                                        "fail_median": x[2], "n_pass": x[3],
                                                        "n_fail": x[4]} for x in detail]},
               "N3": deliver, "coverage_after": 43 + len(deliver)},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
