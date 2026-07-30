#!/usr/bin/env python3
"""Why does each residual target resist? A structural diagnosis, computed exactly and for free.

WHY THIS AND NOT ANOTHER SEARCH. Seven mechanisms have now been tried on the Eterna100 residual and none
recovered a target. But "16 unsolved" has been treated as one bucket, and it is not one thing. The published
DesiRNA result is 97/100 within 24 hours, so MOST of our 16 are solvable and simply cost more than the tiers
we run -- a cost boundary, not a scientific one. What matters is telling those apart from the ones that
resist for a structural reason, because an honest "no sequence does this" is a deliverable and "we ran out of
budget" is a price.

EVERY QUANTITY HERE IS EXACT AND COSTS NOTHING. They are properties of the TARGET, not of any search:

  lonely pairs        a base pair with no stacked neighbour. ViennaRNA's model penalises these heavily
                      (noLonelyPairs is a standard folding option), so a target that requires many is
                      fighting the energy model itself.
  min helix           the shortest run of stacked pairs. Measured earlier: on a 2-branch junction, 2 bp and
                      3 bp helices were both undesignable and 4 bp was not.
  branch load         the largest number of helices meeting at one multiloop. Multiloop initiation is
                      entropically expensive and grows with branch count.
  max loop            the largest unpaired stretch closed by a pair. Big loops are entropically costly and
                      hard to keep from pairing with something.
  repeat load         how much of the target is made of REPEATED sub-structures. Identical units invite
                      cross-pairing between copies -- the failure the decomposition experiment measured
                      directly (only 29% of individually-correct units survive being embedded).
  designable probe    for targets short enough for it to answer (<=110 nt, measured), ViennaRNA's own
                      inverse fold: does a sequence for this shape exist at all?

The point is not to predict solvability from features -- this project measured that null twice (0/2 authored
operators, 0 of 16 caught by learned abstention). The point is to SUBDIVIDE the residual so that the next
experiment is aimed at one mechanism instead of at "hard targets".
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

HOPELESS = [60, 68, 71, 73, 76, 79, 81, 86, 87, 90, 91, 94, 96, 97, 99, 100]


def diagnose(target: str) -> dict:
    from rna_harness.structure_utils import pair_table
    from rna_harness.cognitive.agent import _min_helix
    pt = pair_table(target)
    pairs = [(i, j) for i, j in enumerate(pt) if j > i]
    lonely = sum(1 for i, j in pairs
                 if not ((i > 0 and pt[i - 1] == j + 1) or (pt[i + 1] == j - 1 if i + 1 < len(pt) else False)))
    # branch load: helices meeting at each multiloop closing pair
    branch = 0
    for i, j in pairs:
        k, n_in = i + 1, 0
        while k < j:
            if pt[k] > k and pt[k] < j:
                n_in += 1
                k = pt[k] + 1
            else:
                k += 1
        branch = max(branch, n_in)
    # largest unpaired stretch closed by a pair
    maxloop = 0
    for i, j in pairs:
        run = 0
        for k in range(i + 1, j):
            run = run + 1 if pt[k] < 0 else 0
            maxloop = max(maxloop, run)
    # repeat load: fraction of the target's POSITIONS covered by some substring that occurs more than once.
    # Counting occurrences x width double-counted overlaps and produced coverages above 1.0, which is not a
    # coverage. Marking the covered positions in a set is exact.
    best = 0
    for w in range(len(target) // 2, 7, -1):
        seen = {}
        for i in range(len(target) - w + 1):
            seen.setdefault(target[i:i + w], []).append(i)
        rep = [v for v in seen.values() if len(v) > 1]
        if rep:
            covered = set()
            for starts in rep:
                for i0 in starts:
                    covered.update(range(i0, i0 + w))
            best = len(covered)
            break
    return {"n": len(target), "n_pairs": len(pairs), "lonely_pairs": lonely,
            "min_helix": _min_helix(target), "max_branch": branch, "max_loop": maxloop,
            "repeat_cover": round(best / len(target), 2)}


def main():
    from rna_harness.benchmarks.eterna100 import load_eterna100
    from rna_harness.cognitive.agent import PROBE_MAX_N, designable_probe
    rows = {int(r["id"]): r for r in load_eterna100()}
    allrows = [diagnose(r["target"]) for r in rows.values()]

    def med(key):
        v = sorted(x[key] for x in allrows)
        return v[len(v) // 2]

    print(f"Eterna100 全体的中位数(作为对照): "
          + "  ".join(f"{k}={med(k)}" for k in ("n", "lonely_pairs", "min_helix", "max_branch",
                                                "max_loop", "repeat_cover")))
    print(f"\n  {'id':>4} {'n':>4} {'孤立对':>6} {'最短螺旋':>8} {'最大分支':>8} {'最大环':>7} "
          f"{'重复覆盖':>8} {'可设计?':>8}  name")
    out = []
    for tid in HOPELESS:
        t = rows[tid]["target"]
        d = diagnose(t)
        probe = designable_probe(t)
        d.update({"id": tid, "name": rows[tid].get("name"),
                  "designable": probe.get("designable"),
                  "probe_note": "answered" if probe.get("designable") is not None
                                else f">{PROBE_MAX_N}nt, 探针不适用"})
        out.append(d)
        print(f"  {tid:>4} {d['n']:>4} {d['lonely_pairs']:>6} {d['min_helix']:>8} {d['max_branch']:>8} "
              f"{d['max_loop']:>7} {d['repeat_cover']:>8.2f} {str(d['designable']):>8}  "
              f"{str(d['name'])[:30]}")

    print("\n  按机制分组(每条只归入它最突出的那一项):")
    groups = {"探针说不可设计": [], "重复结构 >=0.5": [], "孤立对 >= 5": [],
              "分支 >= 5": [], "大环 >= 20": [], "以上都不突出": []}
    for d in out:
        if d["designable"] is False:
            groups["探针说不可设计"].append(d["id"])
        elif d["repeat_cover"] >= 0.5:
            groups["重复结构 >=0.5"].append(d["id"])
        elif d["lonely_pairs"] >= 5:
            groups["孤立对 >= 5"].append(d["id"])
        elif d["max_branch"] >= 5:
            groups["分支 >= 5"].append(d["id"])
        elif d["max_loop"] >= 20:
            groups["大环 >= 20"].append(d["id"])
        else:
            groups["以上都不突出"].append(d["id"])
    for k, v in groups.items():
        print(f"    {k:<18} {len(v):>2} 条  {v}")
    json.dump({"hopeless": out, "groups": {k: v for k, v in groups.items()},
               "eterna_medians": {k: med(k) for k in ("n", "lonely_pairs", "min_helix", "max_branch",
                                                      "max_loop", "repeat_cover")}},
              open(os.path.join(HERE, "residual_taxonomy.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/residual_taxonomy.json")


if __name__ == "__main__":
    main()
