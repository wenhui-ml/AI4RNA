#!/usr/bin/env python3
"""M33 census: how many independent clusters is every interval in this paper resampling?

A census, not an experiment. No effect is estimated here and none may be -- the same category as
M27, and it is filed that way in the manuscript's provenance.json.

Every confidence interval in the paper is a cluster bootstrap that resamples the PUZZLE, because
designs competing on one target are not independent of each other. That makes the number of
puzzles, not the number of interventions, the quantity a reader needs in order to know how much
the intervals can carry -- and the manuscript never printed it. A networked review asked for it,
which is how the omission was found.

The filtering here is not reimplemented. M26.pair_prob is replaced with a constant so that
M26.build() runs its own admission rules unchanged and only the folding is skipped; anything else
would risk censusing a different corpus from the one the paper analyses.

COST: $0, offline, no folding.
"""
from __future__ import annotations

import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26


def main() -> int:
    M26.pair_prob = lambda seq, a, b, cache: 0.0     # identical admission, no partition function
    recs, _ = M26.build()

    byp = collections.Counter(r["puzzle"] for r in recs)
    byd = collections.Counter(r["design"] for r in recs)
    counts = np.array(sorted(byp.values(), reverse=True))
    dcounts = np.array(sorted(byd.values(), reverse=True))

    out = {
        "n_interventions": len(recs),
        "n_puzzles": len(byp),
        "n_design_cells": len(byd),
        "per_puzzle": {
            "min": int(counts.min()), "median": int(np.median(counts)), "max": int(counts.max()),
            "largest_share": round(float(counts.max()) / len(recs), 4),
            "counts": [int(x) for x in counts],
        },
        "per_design_cell": {
            "min": int(dcounts.min()), "median": int(np.median(dcounts)),
            "max": int(dcounts.max()),
        },
        "what_this_is": ("a census of the clustering, so that a reader can see what the bootstrap "
                         "resamples. No effect is estimated here."),
    }
    print(f"干预 {out['n_interventions']}  独立簇(puzzle){out['n_puzzles']}  "
          f"设计格 {out['n_design_cells']}")
    print(f"每簇干预数 min {out['per_puzzle']['min']} 中位 {out['per_puzzle']['median']} "
          f"max {out['per_puzzle']['max']};最大簇占 {out['per_puzzle']['largest_share']:.1%}")
    json.dump(out, open(os.path.join(HERE, "m33_cluster_census.json"), "w"),
              indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
