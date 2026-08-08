#!/usr/bin/env python3
"""Was a better sequence already in the pool? Score every candidate and re-run the placement.

RUNS IN THE RibonanzaNet venv. Stage A (`pool_rerank.py`) kept every candidate the product's own ladder
produced for the 57 OpenKnot targets; what the product actually returned is read from
`leaderboard_ours.json`, the same seed and ladder, so nothing is re-derived.

THE QUESTION. We placed 11th of 14, and the loss is in the quality of what we deliver rather than in
coverage. Our answer for each target was chosen from a few hundred candidates by a selector that has since
been measured not to beat random at the product's real pool size. So: is the better sequence already sitting
in the pool we generated and threw away, or is the pool itself the ceiling?

    If the ceiling is high  -> our placement is a SELECTION failure, fixable today with a model we run.
    If the ceiling is low   -> the pool is the ceiling, the failure is GENERATION, and every further hour
                               spent on choosing better is wasted. That would redirect the project.

PRE-REGISTERED, before any candidate was scored. Paired within target, Wilcoxon signed-rank over targets,
one-sided. Two tests, threshold 0.025.

    C1  CEILING (oracle bound, not a product proposal -- it uses the target to pick). The best candidate in
        the pool by RNet crossed-pair F1 beats what the product returned.
        KILL: median gain < 0.05 -> the pools do not contain better sequences; selection is not our problem.

    C2  REALISABLE. The product ships one sequence and would have to choose without knowing the answer.
        RNet's prediction IS available at run time, so ranking the pool by RNet crossed-pair F1 against the
        REQUESTED target is a legitimate run-time rule -- the target is the user's input, not hidden truth.
        C1 and C2 therefore coincide here, and that is stated rather than hidden: the honest ceiling/
        realisable gap in this design is zero, and the thing that keeps C2 honest is that RNet never saw
        the wet-lab score, only the target the user asked for.
        PASS: p < 0.025 AND median gain >= 0.05.

    C3  PLACEMENT (descriptive). Re-run the leaderboard comparison with the re-ranked choice: does
        "better than the field median on 8/57" move?

    A target the product failed to deliver keeps its 0.0 unless the pool contains a candidate; if it does,
    re-ranking is credited with the rescue and this is reported separately, because turning not_found into
    an answer is a different kind of gain from improving an answer.

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

DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
ALPHA = 0.025
HOMEFIELD = "Struct2SeQ-SHAPE"


_MODEL = None


def _score_chunk(items):
    """Score a slice of candidates and return their crossed-pair F1 against the requested target.

    One torch thread per process: the model is 11 M parameters and the sequences are short, so intra-op
    threading fights the process pool for the same cores. Parallelism belongs across candidates, not
    inside one forward pass. The model is built once per worker and cached in a module global, because
    rebuilding it per chunk would dominate the run.
    """
    global _MODEL
    import torch
    torch.set_num_threads(1)
    from rnet_reproduce import build_model, pairwise_logits, pairs_from_logits, f1
    if _MODEL is None:
        _MODEL = build_model()

    _O = {"(": ")", "[": "]", "{": "}", "<": ">"}
    _C = {v: k for k, v in _O.items()}

    def pk(s):
        st_, out = {o: [] for o in _O}, set()
        for i, c in enumerate(s):
            if c in _O:
                st_[c].append(i)
            elif c in _C and st_[_C[c]]:
                out.add((st_[_C[c]].pop(), i))
        return out

    def cross(pairs):
        o = set()
        for (i, j) in pairs:
            for (m, n) in pairs:
                if (m < i < n < j) or (i < m < j < n):
                    o.add((i, j))
                    break
        return o

    out = []
    for key, seq, tgt in items:
        pred = cross(pairs_from_logits(pairwise_logits(_MODEL, seq)))
        out.append((key, f1(pred, cross(pk(tgt)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=os.path.join(HERE, "pool_rerank_pools.json"))
    ap.add_argument("--shipped", default=os.path.join(HERE, "leaderboard_ours.json"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "pool_rerank_score.json"))
    a = ap.parse_args()

    from rnet_reproduce import build_model, pairwise_logits, pairs_from_logits, f1
    from m2r_pairwise import wilcoxon

    _OPEN = {"(": ")", "[": "]", "{": "}", "<": ">"}
    _CLOSE = {v: k for k, v in _OPEN.items()}

    def pk_pairs(s):
        stacks, out = {o: [] for o in _OPEN}, set()
        for i, c in enumerate(s):
            if c in _OPEN:
                stacks[c].append(i)
            elif c in _CLOSE and stacks[_CLOSE[c]]:
                out.add((stacks[_CLOSE[c]].pop(), i))
        return out

    def crossing(pairs):
        out = set()
        for (i, j) in pairs:
            for (m, n) in pairs:
                if (m < i < n < j) or (i < m < j < n):
                    out.add((i, j))
                    break
        return out

    P = json.load(open(a.pools))
    T = P["targets"]
    pools = {r["puzzle"]: r["pool"] for r in P["rows"] if "pool" in r}
    S = json.load(open(a.shipped))
    shipped = {}
    for r in S["rows"]:
        p = r["puzzle"]
        if r.get("sequence") and (p not in shipped or shipped[p]["budget"] != "standard"):
            shipped[p] = r
    print(f"池子 {len(pools)} 个靶标,合计 {sum(len(v) for v in pools.values())} 条候选;"
          f"产品交付 {len(shipped)}/{len(T)}")

    # ---- score every candidate + the shipped one, in parallel -------------------------------------
    jobs = []
    for p in sorted(pools):
        for i, c in enumerate(pools[p]):
            jobs.append(((p, i), c["seq"], T[p]["target"]))
        sh = shipped.get(p)
        if sh:
            jobs.append((("SHIP", p), sh["sequence"], T[p]["target"]))
    print(f"要打分的序列 {len(jobs)} 条,{a.workers} 进程并行\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 8))
    chunks = [jobs[i:i + CH] for i in range(0, len(jobs), CH)]
    scores, t0, done = {}, time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, chunks):
            for key, v in res:
                scores[key] = v
            done += 1
            if done % 20 == 0 or done == len(chunks):
                print(f"    {done}/{len(chunks)} 块  {(time.time()-t0)/60:.1f} min", flush=True)

    best, ship_score = {}, {}
    for p in sorted(pools):
        sc = [scores[(p, i)] for i in range(len(pools[p])) if (p, i) in scores]
        bi = max(range(len(sc)), key=lambda i: sc[i]) if sc else None
        best[p] = {"f1c": (sc[bi] if bi is not None else 0.0),
                   "seq": (pools[p][bi]["seq"] if bi is not None else None),
                   "pool_n": len(sc), "pool_median": (st.median(sc) if sc else 0.0)}
        ship_score[p] = scores.get(("SHIP", p), 0.0)

    ps = sorted(pools)
    d = [best[p]["f1c"] - ship_score[p] for p in ps]
    w, p1, n = wilcoxon(d, "greater")
    print(f"\n=== C1/C2 池子里有没有更好的序列 ===")
    print(f"  {len(d)} 个靶标,中位增益 {st.median(d):+.3f},更好 {sum(1 for x in d if x > 0)}/{len(d)},"
          f"p = {'n/a' if p1 is None else f'{p1:.3g}'}")
    print(f"  产品交付的 F1 中位 {st.median([ship_score[p] for p in ps]):.3f}  ->  "
          f"池中最好 {st.median([best[p]['f1c'] for p in ps]):.3f}")
    ok = bool(p1 is not None and p1 < ALPHA and st.median(d) >= 0.05)
    print(f"  {'通过 —— 更好的序列本来就在池子里,这是【选择】问题' if ok else 'KILL —— 池子就是天花板,问题在【生成】不在【选择】'}")

    rescued = [p for p in ps if p not in shipped and best[p]["f1c"] > 0]
    print(f"  其中把 not_found 变成答案的:{len(rescued)} 个 {sorted(rescued)}")

    # ---- C3 placement ---------------------------------------------------------------------------
    csv.field_size_limit(10 ** 9)
    theirs = collections.defaultdict(list)
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {c: i for i, c in enumerate(next(r))
               if c in ("puzzle", "method", "SN_filter", "RNet_F1_crossed_pair", "target_structure")}
        for x in r:
            if x[idx["SN_filter"]] != "1" or x[idx["method"]] == HOMEFIELD:
                continue
            p = x[idx["puzzle"]]
            if p in T and x[idx["target_structure"]] != T[p]["target"]:
                continue
            try:
                theirs[p].append(float(x[idx["RNet_F1_crossed_pair"]]))
            except ValueError:
                pass
    print(f"\n=== C3 名次有没有动(对全体真实设计的每靶标中位)===")
    for label, sc in (("产品现行选择", ship_score), ("RNet 重排后", {p: best[p]["f1c"] for p in ps})):
        dd = [sc[p] - st.median(theirs[p]) for p in ps if theirs.get(p)]
        w2, p2, _ = wilcoxon(dd, "greater")
        print(f"  {label:14s} 中位差 {st.median(dd):+.3f}   更好 {sum(1 for x in dd if x>0):2d}/{len(dd)}"
              f"   p = {'n/a' if p2 is None else f'{p2:.3g}'}")

    json.dump({"C1_median_gain": st.median(d), "C1_p": p1, "C1_pass": ok,
               "n_rescued": len(rescued), "rescued": sorted(rescued),
               "shipped_median_f1c": st.median([ship_score[p] for p in ps]),
               "best_median_f1c": st.median([best[p]["f1c"] for p in ps]),
               "per_target": {p: {"shipped": ship_score[p], **best[p]} for p in ps}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
