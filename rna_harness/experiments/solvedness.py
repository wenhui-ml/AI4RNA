"""Per-puzzle SOLVEDNESS profile: a continuous "degree of resolution" so we can track last-mile
(A/A2) progress even while solve-rate stays 0.

Biology-expert opinion instantiated: bp-distance to target is the PRIMARY signal, solve-rate is
SECONDARY, and a design can be *meaningfully approaching* the target while solve-rate = 0. The
binary solve flag and even the integer bp are too coarse for the last mile; we add SMOOTH
(sub-integer) components that move within a single bp level:

  solved            (bool)  SECONDARY  -- exact MFE == target
  bp                (int)   PRIMARY    -- base-pair distance to target (expert's metric)
  pct_pairs_correct (float)            -- % of target base pairs realized (bp normalized by size,
                                          so a bp=1 miss is comparable across puzzle lengths)
  ensemble_defect   (float) PRIMARY-smooth -- expected # wrongly-paired nt over the Boltzmann
                                          ensemble; MOVES even when integer bp is frozen ->
                                          the "meaningfully approaching" signal the expert means
  min_pair_prob     (float)            -- Boltzmann prob of the WEAKEST target pair forming; at
                                          bp=1 tells us if the missing pair is "almost there" (0.4)
                                          or "hopeless" (0.01) -- sub-integer progress within bp=1
  n_defect_regions  (int)              -- # separate broken spots (1 localized => easy endgame)

Aggregate (across a puzzle set) is reported the expert's way: PRIMARY = bp / ensemble-defect
distribution + medians; SECONDARY = solve-rate.
"""
from __future__ import annotations

import collections
import statistics
from typing import Dict, List

import RNA

from ..structure_utils import pair_table
from ..mcp.finisher import _fold


def solvedness(seq: str, target: str) -> Dict:
    """Continuous degree-of-resolution profile for one design against one target (all exact/ViennaRNA)."""
    n = len(target)
    pt = pair_table(target)
    st = _fold(seq)
    ps = pair_table(st)
    bp = RNA.bp_distance(st, target)
    total_pairs = sum(1 for i in range(n) if pt[i] > i)
    correct = sum(1 for i in range(n) if pt[i] > i and ps[i] == pt[i])
    fc = RNA.fold_compound(seq)
    fc.pf()
    bpp = fc.bpp()  # 1-indexed
    probs = [float(bpp[i + 1][pt[i] + 1]) for i in range(n) if pt[i] > i]
    ed = fc.ensemble_defect(target) if hasattr(fc, "ensemble_defect") else None
    mis = [i for i in range(n) if ps[i] != pt[i]]
    regions = 0
    if mis:
        regions = 1
        for a, b in zip(mis, mis[1:]):
            if b != a + 1:
                regions += 1
    return {
        "solved": bool(bp == 0 and st == target),
        "bp": bp,
        "pct_pairs_correct": round(100.0 * correct / total_pairs, 1) if total_pairs else 100.0,
        "ensemble_defect": round(float(ed), 3) if ed is not None else None,
        "min_pair_prob": round(min(probs), 3) if probs else 1.0,
        "n_defect_regions": regions,
    }


def aggregate(profiles: List[Dict]) -> Dict:
    """Expert-aligned aggregate: PRIMARY = bp / ensemble-defect distribution + medians; SECONDARY =
    solve-rate. Report the DISTRIBUTION, not just the median (the residual is multi-modal)."""
    if not profiles:
        return {"n": 0}
    bps = [p["bp"] for p in profiles]
    eds = [p["ensemble_defect"] for p in profiles if p["ensemble_defect"] is not None]
    buckets = collections.Counter(
        "0" if b == 0 else "1-3" if b <= 3 else "4-10" if b <= 10 else ">10" for b in bps)
    return {
        "n": len(profiles),
        "solve_rate": round(sum(1 for p in profiles if p["solved"]) / len(profiles), 3),   # SECONDARY
        "median_bp": statistics.median(bps),                                                # PRIMARY
        "mean_bp": round(statistics.mean(bps), 2),
        "median_ensemble_defect": round(statistics.median(eds), 3) if eds else None,        # PRIMARY smooth
        "bp_buckets": {k: buckets.get(k, 0) for k in ("0", "1-3", "4-10", ">10")},
    }
