#!/usr/bin/env python3
"""Extra per-molecule columns M13b needs that M13's cache did not store.

Aligned element-wise to m13_features.json by replaying build()'s exact keep filter and asserting
puzzle/method/wet agreement row by row. Never assumed.

  seq         -- needed to compute GC on a CIRCULARLY SHIFTED paired-position set (P4)
  paired      -- the target's paired positions, as a sorted list
  n           -- length
  dG_target   -- eval_structure(seq, nested(target)), for the mechanism arm M1
  helix_mean  -- mean helix length of the target, for the moderation arm M3
COST: $0, offline.
"""
from __future__ import annotations
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from m12_sequence_intrinsic import parse_pairs, helices, load_pop, base_block                # noqa: E402
from m13_cache import POPS                                                                  # noqa: E402


def one(job):
    puz, seq, target = job
    import RNA
    P = sorted(parse_pairs(target))
    pos = sorted({i for pr in P for i in pr})
    nest = "".join(c if c in "()." else "." for c in target)
    H = helices(set(P))
    return {"seq": seq, "paired": pos, "n": len(seq),
            "dG_target": float(RNA.eval_structure_simple(seq, nest)),
            "helix_mean": float(sum(H) / len(H))}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--cache", default=os.path.join(HERE, "m13_features.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "m13b_extra.json"))
    a = ap.parse_args()
    from leaderboard_design import targets as load_targets
    from concurrent.futures import ProcessPoolExecutor
    T = load_targets(); C = json.load(open(a.cache)); OUT = {}
    for nm, fn in POPS:
        rec, _s, _m = load_pop(fn)
        keep = [r for r in rec if base_block(r) is not None and r.get("seq")
                and r.get("wet") is not None and r["puzzle"] in T]
        rows = C[fn]["rows"]
        if len(keep) != len(rows):
            raise RuntimeError(f"{fn}: keep {len(keep)} != cached {len(rows)} -- alignment broken")
        bad = [k for k, (a_, b_) in enumerate(zip(keep, rows))
               if a_["puzzle"] != b_["puzzle"] or a_["method"] != b_["method"]
               or abs(float(a_["wet"]) - b_["wet"]) > 1e-9]
        if bad:
            raise RuntimeError(f"{fn}: row mismatch at {bad[:5]}")
        jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            OUT[fn] = list(ex.map(one, jobs, chunksize=64))
        print(f"{nm:12s} {len(OUT[fn])} 行,逐行断言 (puzzle, method, wet) 一致 0 不符", flush=True)
    json.dump(OUT, open(a.out, "w"))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
