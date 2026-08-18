#!/usr/bin/env python3
"""If the certificate starts reading precision, where does the operating point go? Recalibrate on ground truth.

RUNS IN THE `rna` env. Launch under tmux.

WHY THE VERDICT CANNOT SIMPLY BE SWITCHED TO F1. The shipped rung is "at least one oracle recovers ALL of the
target's crossing pairs", and the number that licenses it is a CALIBRATION: 84.0% sensitivity on known natural
structures at 0.4% false accept on random sequences. Both halves of that pair were measured on the RECALL
quantity. A precision-aware rule is a different rule, so its sensitivity and its false-accept rate are
unmeasured, and adopting it on the strength of the matched-coverage result would be choosing an operating point
by taste -- which is what `calibrate.py` exists to stop.

THE TWO POPULATIONS, unchanged from `calibrate.py` so the numbers are comparable:

    natives   the wild-type sequences of the Pseudobase targets, recovered from the antaRNA constraint files
              and matched to targets by PAIR SET (bracket-convention invariant). Ground truth: the wild type
              really does form the published structure, so a rule's pass rate here is SENSITIVITY.
    randoms   one random ACGU sequence per native, same length, seed 0. A rule that passes these is a rubber
              stamp, so its pass rate here is the FALSE-ACCEPT rate.

RULE FAMILY SCORED (each is a predicate on one design's three per-oracle readings):

    k>=1 recall==1              what ships today; reproduced here as the reference point, not as a baseline
                                being argued against
    max f1 >= t                 t on a grid -- the release's own statistic
    k>=1 recall==1 AND prec>=p  keep the shipped rung and merely refuse over-predictors
    mean recall >= 0.1          the loose screen `knot_screened` ships, for the same two populations

WHAT WOULD JUSTIFY A CHANGE, stated before the grid is scored: a rule that holds sensitivity at or above the
shipped point's while cutting false accepts, or holds false accepts at or below it while raising sensitivity.
Anything that trades one for the other is a product decision, not a measurement, and is reported as a
trade-off curve rather than a recommendation.

COST: $0. 250 natives + 250 randoms x 3 oracles, folded once each.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)


def work(item):
    kind, tid, seq, target = item
    from rna_harness.cognitive.certify import _knot_recall
    try:
        kr = _knot_recall(seq, target)
    except Exception as e:                                                     # noqa: BLE001
        return {"kind": kind, "id": tid, "error": repr(e)[:120]}
    return {"kind": kind, "id": tid, "n": len(seq), "n_cross": kr["n_cross"],
            "recall": kr["per_oracle"], "precision": kr["per_oracle_precision"],
            "f1": kr["per_oracle_f1"]}


def rules():
    """(name, predicate) over one row's three readings. None from an oracle is never an accept."""
    def vals(d):
        return [x for x in d.values() if x is not None]

    out = [("SHIPPED k>=1 recall==1", lambda r: any(v >= 1.0 for v in vals(r["recall"])))]
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        out.append((f"max f1 >= {t:.1f}", lambda r, t=t: any(v >= t for v in vals(r["f1"]))))
    for p in (0.1, 0.2, 0.3, 0.4, 0.5):
        out.append((f"k>=1 recall==1 AND prec >= {p:.1f}",
                    lambda r, p=p: any((r["recall"][o] is not None and r["recall"][o] >= 1.0
                                        and r["precision"][o] is not None and r["precision"][o] >= p)
                                       for o in r["recall"])))
    out.append(("SCREEN mean recall >= 0.1",
                lambda r: bool(vals(r["recall"])) and sum(vals(r["recall"])) / len(vals(r["recall"])) >= 0.1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "calibrate_precision.json"))
    a = ap.parse_args()

    # imported as a package module, not by path: axis1_native_control uses relative imports
    from rna_harness.experiments.axis1_native_control import parse_natives
    from rna_harness.benchmarks.rnainvbench import load_pseudobase, pk_pairs

    # parse_natives() is keyed (length, frozenset(pairs)) -> (name, seq, dot-bracket), so the match is
    # bracket-convention invariant: a stem the native file writes {{{ and the target writes [[[ has the same
    # pair set and matches. Same keying as axis1_native_control, so the populations are the archived ones.
    nat = parse_natives()
    tasks = {t.task_id: t.target for t in load_pseudobase()}
    items = []
    for tid, tgt in sorted(tasks.items()):
        hit = nat.get((len(tgt), frozenset(pk_pairs(tgt))))
        if hit:
            items.append(("native", tid, hit[1], tgt))
    print(f"野生型条目 {len(nat)} 个,Pseudobase 靶标 {len(tasks)} 个,按(长度,配对集)匹配上 {len(items)} 条")
    rng = random.Random(a.seed)
    for kind, tid, seq, tgt in list(items):
        items.append(("random", tid, "".join(rng.choice("ACGU") for _ in seq), tgt))
    if a.limit:
        items = items[:a.limit] + items[len(items) // 2:len(items) // 2 + a.limit]
    print(f"共 {len(items)} 条(野生型 + 同长随机),{a.workers} 进程", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, res in enumerate(ex.map(work, items, chunksize=4), 1):
            R.append(res)
            if n % 100 == 0 or n == len(items):
                print(f"    {n}/{len(items)}  {(time.time()-t0)/60:.1f} min", flush=True)
    R = [r for r in R if "error" not in r]
    nat_rows = [r for r in R if r["kind"] == "native"]
    rnd_rows = [r for r in R if r["kind"] == "random"]
    print(f"\n  可用:野生型 {len(nat_rows)},随机 {len(rnd_rows)}")

    print(f"\n{'规则':34s} {'灵敏度(野生型)':>14s} {'假收率(随机)':>12s}   判读")
    ship = None
    out = []
    for name, fn in rules():
        se = sum(1 for r in nat_rows if fn(r)) / max(1, len(nat_rows))
        fp = sum(1 for r in rnd_rows if fn(r)) / max(1, len(rnd_rows))
        if name.startswith("SHIPPED"):
            ship = (se, fp)
        tag = ""
        if ship and not name.startswith("SHIPPED"):
            if se >= ship[0] and fp < ship[1]:
                tag = "  <- 灵敏度不降、假收更低"
            elif fp <= ship[1] and se > ship[0]:
                tag = "  <- 假收不升、灵敏度更高"
        print(f"{name:34s} {se:13.1%} {fp:12.1%}{tag}")
        out.append({"rule": name, "sensitivity": se, "false_accept": fp, "tag": tag.strip()})

    print(f"\n  出厂点:灵敏度 {ship[0]:.1%},假收率 {ship[1]:.1%}"
          f"(档案里记的是 84.0% / 0.4%;差异来自匹配上的野生型条数与随机种子,以本次为准)")
    print(f"  没有带箭头的行 = 没有一条规则在不牺牲另一半的前提下更好,"
          f"那么换判据就是一次取舍,而不是一次改进。")
    json.dump({"n_native": len(nat_rows), "n_random": len(rnd_rows), "shipped": ship,
               "rows": out, "minutes": (time.time() - t0) / 60,
               "raw": R}, open(a.out, "w"))
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
