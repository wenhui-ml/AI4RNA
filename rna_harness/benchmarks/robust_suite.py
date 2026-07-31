"""Mutational robustness -- a SECOND test of the point-versus-neighbourhood hypothesis.

THE HYPOTHESIS THIS EXISTS TO TEST, stated before the run so it can lose. Six requirements have now been
measured against the standard tool and the pattern in them is sharper than the explanation I first gave:

    umfe / pfe          no gap          the objective already implies it
    banned 4-mer        no gap          free inside the objective's level sets
    GC / Tm / ortho     ONE gap         all three are the same gap; the GC control collapses them
    near-reference      no gap          M1's ceiling -- nothing in the pool is close enough
    co-transcriptional  a gap           and it does NOT collapse under the GC control

I explained the last one as "a permutation property rather than a compositional one". That is too narrow and
it hides the regularity. DesiRNA's objective is a function of ONE POINT: one sequence, one structure,
equilibrium, 37 C, no partner, no time. Everything it computes -- MFE, ensemble defect, suboptimal gap, dimer
energy -- is a state function at that point. So:

    POINT properties      compete for the same scalar, therefore they collapse into at most ONE gap,
                          and that one gap is found and measured out (GC).
    NEIGHBOURHOOD         properties the objective is not a function of at all. Co-transcriptional folding
    properties            is a neighbourhood in TIME (the prefix family). It did not collapse.

That reframing predicts the number of remaining gaps is not "some unknown number of requirements" but the
number of independent directions you can unpin the point in -- time, SEQUENCE, temperature, partner,
structure. This module tests the SEQUENCE direction, which is the sharpest of the four untested ones for a
reason that is not about convenience:

    AN OPTIMISER PRODUCES FRAGILE ANSWERS BY CONSTRUCTION. Finding an extremum is finding a peak, and a peak
    is where one step in any direction goes down. So "every single-point mutant still folds to the target"
    is directly opposed to what the search is doing, while being no part of what it measures.

AND IT SHOULD BE ORTHOGONAL TO COMPOSITION, for the same reason kinetics was: two sequences with identical GC
can differ widely in robustness, because robustness depends on WHERE the GC sits. A G-C pair closing a 2 bp
stem is fragile; the same pair buried in a 6 bp stem is buffered by its neighbours. That is a placement
property, not a composition one -- but it is a PREDICTION, and the paired GC-band control is what decides it.

THE MEASUREMENT. For a sequence of length n there are exactly 3n single-point mutants. neutral_fraction is
the share of them whose MFE is still exactly the target. Deterministic, exhaustive, no sampling.

TWO STATISTICS, because a threshold should not be the whole result:

  * THRESHOLD-FREE and primary: the paired difference in neutral_fraction between the tool's own answer and
    what screening the same pool returns. A sign test on the pairs needs no cutoff at all.
  * REQUIREMENT-SHAPED: neutral_fraction >= 0.50, declared here before the run, plus the full sweep of both
    arms across 0.30-0.80. As with the kinetic benchmark, the sweep is what carries the conclusion; a gap
    that exists only at one cutoff is a cutoff, not a gap.

Nature would have been the better anchor -- "as robust as the sequence evolution actually uses for this
shape" -- and it is not available: the bpRNA-derived source stores its sequences masked to all N. Said here
rather than quietly substituting a number and calling it principled.

THREE WAYS THIS DIES, written down first:
  1. the tool's own answers are already robust     -> the point/neighbourhood split is wrong
  2. almost nothing in the pool is robust          -> M1's ceiling; needs a better search (M6), which we
                                                      do not have, and screening cannot conjure it
  3. the gap vanishes under the GC-band control    -> robustness is a fourth costume on the composition gap;
                                                      "point properties collapse to one" survives, the
                                                      neighbourhood claim weakens
Only a gap that survives the band counts.
"""
from __future__ import annotations

from typing import Dict, List, Optional

NEUTRAL_MIN = 0.50            # declared before the run; the sweep below is what the conclusion rests on
SWEEP = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
MAX_SCREEN = 60               # candidates evaluated per target; declared, and the count screened is reported
ALPHABET = "ACGU"


def neutral_fraction(seq: str, target: str) -> Dict:
    """Exhaustive single-point mutational scan: 3n mutants, each folded exactly once.

    No sampling and no early exit, so the number is the number -- two runs on the same sequence agree to the
    last digit, which matters because every other quantity in this project had to be re-run to find out it
    was a draw.
    """
    import RNA
    n = len(seq)
    kept, total = 0, 0
    per_pos = []
    for i, ch in enumerate(seq):
        ok = 0
        for b in ALPHABET:
            if b == ch:
                continue
            total += 1
            m = seq[:i] + b + seq[i + 1:]
            if RNA.fold(m)[0] == target:
                ok += 1
                kept += 1
        per_pos.append(ok)
    return {"neutral_fraction": round(kept / max(1, total), 4), "n_mutants": total,
            "n_neutral": kept,
            # A position where all three substitutions survive is a genuinely free position; a position where
            # none does is a single point of failure. Reported because the mean can hide both.
            "n_free_positions": sum(1 for x in per_pos if x == 3),
            "n_critical_positions": sum(1 for x in per_pos if x == 0)}


def gc_percent(seq: str) -> float:
    return 100.0 * sum(1 for c in seq if c in "GC") / max(1, len(seq))


def screen(pool: List[str], target: str, gc_band=None, cap: int = MAX_SCREEN) -> Dict:
    """Evaluate the pool and report BOTH what a shipped screener would return and the pool's ceiling.

    `first_passing` is the honest arm -- it is what the product does. `best` is a diagnostic: it separates
    "screening picked badly" from "the pool has nothing better", which is the difference between a fixable
    shortfall and M1's ceiling.
    """
    from ..cognitive.certify import certify_sequence
    first, best, rows, n_folded = None, None, [], 0
    for s in pool:
        if len(rows) >= cap:
            break
        if gc_band and not (gc_band[0] <= gc_percent(s) <= gc_band[1]):
            continue
        if certify_sequence(s, target).verdict not in ("certified", "knot_confirmed"):
            continue
        n_folded += 1
        nf = neutral_fraction(s, target)
        row = {"gc": round(gc_percent(s), 1), "neutral_fraction": nf["neutral_fraction"],
               "n_critical_positions": nf["n_critical_positions"]}
        rows.append(row)
        if best is None or nf["neutral_fraction"] > best["neutral_fraction"]:
            best = {"sequence": s, **row}
        if first is None and nf["neutral_fraction"] >= NEUTRAL_MIN:
            first = {"sequence": s, **row}
    return {"first_passing": first, "best": best, "n_screened": len(rows), "n_folded": n_folded,
            "capped": len(rows) >= cap, "rows": rows}
