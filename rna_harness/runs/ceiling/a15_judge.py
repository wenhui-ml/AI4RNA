#!/usr/bin/env python3
"""A15 stage 1: judge the fourth sample with the never-in-loop thermodynamic judges.

Declared in `a15_declaration.json`. Two things about the judge set are deliberate:

FIXED, NOT "WHATEVER SUCCEEDED". The primary quality is (ipknot + pkplex)/2 and BOTH must succeed or the
molecule is dropped. `mean3`-style averaging over whatever happened to work is not a reproducible quantity:
Knotty is wall-clock bounded, and on 91 sequences judged twice its availability moved (37 -> 36 targets), so
"average of the available" was two different quantities in two runs of the same code on the same input.

KNOTTY STILL RUNS, SEPARATELY. `--judges knotty` writes its own file, and A15's analysis reads it only as a
coverage-reported secondary. It is slow (~12 s/sequence at these lengths), which is exactly why it must not
be allowed to decide which molecules are in the primary population.
"""
from __future__ import annotations

import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

JUDGES = {"ipknot": ("ipknot_judge", "ipknot_pairs"),
          "pkplex": ("external_judge", "pkplex_pairs"),
          "knotty": ("knotty_judge", "knotty_pairs")}


def one(item):
    puz, seq, target, names = item
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from rnet_reproduce import f1
    from selector_heldout import crossing_subset
    import importlib
    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    out = {}
    for nm in names:
        mod, fn = JUDGES[nm]
        try:
            ps = getattr(importlib.import_module(mod), fn)(seq)
        except Exception:                                                        # noqa: BLE001
            ps = None
        out[nm] = None if ps is None else f1(crossing_subset(ps), cross)
    return puz, seq, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "panel_f1_fourth.json"))
    ap.add_argument("--judges", default="ipknot,pkplex")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = tuple(a.judges.split(","))
    assert all(n in JUDGES for n in names), f"未知判官 {names}"
    out = a.out or os.path.join(HERE, f"a15_judged_{'_'.join(names)}.json")

    from leaderboard_design import targets
    T = targets()
    D = json.load(open(a.src))
    assert D.get("fourth_per_cell"), "不是第四批,拒绝"
    need = sorted({(r["puzzle"], r["seq"]) for r in D["designs"]})
    print(f"第四批 {len(D['designs'])} 条,唯一 (靶标,序列) {len(need)} 条;判官 {names};{a.workers} 进程",
          flush=True)
    jobs = [(p, s, str(T[p]), names) for p, s in need if p in T]
    print(f"  靶标能对上的 {len(jobs)}/{len(need)}", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    J, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, (p, s, o) in enumerate(ex.map(one, jobs, chunksize=4), 1):
            J[f"{p}\t{s}"] = o
            if n % 200 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    for nm in names:
        bad = sum(1 for v in J.values() if v[nm] is None)
        print(f"  {nm:8s} 失败 {bad}/{len(J)} = {bad/max(1,len(J)):.1%}")
    json.dump(J, open(out, "w"), indent=1)
    print(f"写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
