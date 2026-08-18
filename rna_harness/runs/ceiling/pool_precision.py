#!/usr/bin/env python3
"""Re-adjudicate OUR OWN 8,730 dumped candidates with the precision term the shipped criterion throws away.

RUNS IN THE `rna` env. Long job -- launch under tmux, not nohup.

WHY OUR POOLS AND NOT THE 4,560 AGAIN. `ruler_gradient_f1.json` showed that the panel's apparent loss to
RibonanzaNet at matched coverage was entirely an artefact of comparing OUR RECALL against THEIR F1: within
(target, method) cells the loss went from +2.9 points (p = 0.014) on the recall column to +0.0 points
(p = 0.40) on the F1 column, and at 10% coverage the panel came out 6 points ahead. But that was measured on
OTHER PEOPLE'S molecules -- designs that were selected before being synthesised. On the 40 raw candidates
sampled from our own pools the precision is WORSE (median 0.091 against 0.154), so the effect on our own
selection problem could be larger or could be noisier, and nothing about the field's molecules settles it.

WHAT THIS PRODUCES. For every unique candidate in `pool_dump/` (57 pools, 8,730 sequences), the patched
`_knot_recall` output: the shipped per-oracle recall (unchanged, and asserted here to match what the dump
already recorded), plus per-oracle precision, F1 and the predicted-crossing count. The dump itself was written
BEFORE the precision patch, which is why it has to be recomputed rather than read.

    C0  IDENTITY CHECK, independently failable. The recomputed `per_oracle` must equal the recall the dump
        recorded for that sequence, on every candidate. A mismatch means the oracles are not deterministic
        across runs or the dump is not the pool we think it is; either way nothing downstream is usable.

Downstream consumers (a precision-aware selection arm, and the operating-point recalibration) read this file.

COST: $0. Local folds. About 8,730 sequences x 3 oracles; the comparable run over 4,560 field designs took
64 minutes at 16 workers on a loaded box.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)


def work(item):
    puz, seq, target = item
    from rna_harness.cognitive.certify import _knot_recall
    try:
        kr = _knot_recall(seq, target)
    except Exception as e:                                                     # noqa: BLE001
        return {"puzzle": puz, "seq": seq, "error": repr(e)[:120]}
    def mean(d):
        v = [x for x in d.values() if x is not None]
        return (sum(v) / len(v)) if v else None
    return {"puzzle": puz, "seq": seq,
            "recall": mean(kr["per_oracle"]),                 # the shipped quantity
            "precision": mean(kr["per_oracle_precision"]),
            "f1": mean(kr["per_oracle_f1"]),
            "f1_max": max([x for x in kr["per_oracle_f1"].values() if x is not None], default=None),
            "n_pred_cross": mean(kr["n_pred_cross"]),
            "n_cross": kr["n_cross"],
            "confirmed_by": kr["confirmed_by"],
            "knot_confirmed": kr["knot_confirmed"],
            "per_oracle": kr["per_oracle"],
            "per_oracle_precision": kr["per_oracle_precision"],
            "per_oracle_f1": kr["per_oracle_f1"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "pool_dump"))
    ap.add_argument("--run", default=os.path.join(HERE, "leaderboard_replay.json"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "pool_precision.json"))
    a = ap.parse_args()

    from selector_replay import load_dump
    R = json.load(open(a.run))
    T = {p: v["target"] for p, v in R["targets"].items()}
    P = load_dump(a.dump, T)
    jobs = [(p, r["seq"], T[p]) for p in sorted(P) for r in P[p]]
    was = {(p, r["seq"]): r["per_oracle"] for p in P for r in P[p]}
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(P)} 个池,{len(jobs)} 条候选,{a.workers} 进程", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    R2, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, res in enumerate(ex.map(work, jobs, chunksize=4), 1):
            R2.append(res)
            if n % 400 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    bad = [r for r in R2 if "error" in r]
    ok = [r for r in R2 if "error" not in r]
    print(f"\n  成功 {len(ok)},失败 {len(bad)}"
          + (f" ({collections.Counter(r['error'] for r in bad).most_common(1)[0][0]})" if bad else ""))

    # ---- C0 identity check ----------------------------------------------------------------------------
    mism = 0
    for r in ok:
        old = was.get((r["puzzle"], r["seq"]))
        if old is None:
            continue
        if any((old.get(k) is None) != (v is None) or
               (v is not None and abs(old.get(k) - v) > 1e-9) for k, v in r["per_oracle"].items()):
            mism += 1
    print(f"  C0 与落盘时记录的 per_oracle 召回比对:不一致 {mism}/{len(ok)}"
          + ("  ✓" if mism == 0 else "  ✗ 折叠不可复现或池对不上,下游全部不可用"))

    for k in ("recall", "precision", "f1"):
        v = [r[k] for r in ok if r[k] is not None]
        print(f"  {k:10s} 中位 {st.median(v):.3f}  均值 {st.mean(v):.3f}")
    npc = [r["n_pred_cross"] for r in ok if r["n_pred_cross"] is not None]
    ncr = [r["n_cross"] for r in ok]
    print(f"  预测交叉配对 中位 {st.median(npc):.1f}  对 目标交叉配对 中位 {st.median(ncr):.0f}"
          f"  → 过预测 {st.median(npc)/max(1, st.median(ncr)):.1f} 倍")

    json.dump({"n": len(ok), "n_failed": len(bad), "identity_mismatches": mism,
               "minutes": (time.time() - t0) / 60, "candidates": ok},
              open(a.out, "w"))
    print(f"写出 {a.out}")
    return 0 if mism == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
