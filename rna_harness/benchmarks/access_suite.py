"""Accessibility -- the FIFTH and last coordinate, and the prediction here is that we have NO capability.

THE MAP THIS COMPLETES. The solver's objective is a function of one POINT: one sequence, one structure,
equilibrium, 37 C, no partner, no time. Unpin the point in a direction and you get a candidate requirement;
whether it pays is settled by one rule, arrived at through one refutation and confirmed once prospectively:

    A NEIGHBOURHOOD PROPERTY OPENS A GAP ONLY WHEN THE NEIGHBOURHOOD SPANS DIFFERENT ENERGY LANDSCAPES.

      time / prefixes    each prefix is a different molecule            GAP    8/24  -> 19/24
      sequence / mutants a small perturbation of the SAME landscape     null   pool already 93% compliant
      temperature        the same landscape, scaled                     null   Tm collapsed into GC
      partner            monomer against complex, two molecules         GAP    35/58 -> 54/58 at equal GC
      structure          THIS FILE

WHY THE PREDICTION IS NULL, and it is worth stating plainly that this is the UNFAVOURABLE prediction -- it
says the project does not have a fourth capability. Accessibility is not a neighbourhood at all. It is a
different READOUT of the very partition function the solver already computes at the same point: the same
molecule, the same temperature, no partner, no time. The solver even exposes an objective built from it
(`-sf Edef`, ensemble defect). By the rule above that is Level 0 -- implied, or at worst trading against the
same scalar everything else traded against -- so the expected outcome is no gap, or a gap that dies under the
composition control like the first three did.

A GAP HERE WOULD REFUTE THE RULE. That is the point of running it rather than asserting the cell. If
screening its own pool beats its own answer at IDENTICAL composition, the "different landscapes" formulation
is wrong and the map is not finished.

THE REQUIREMENT, declared before the run. The window is the target's LONGEST unpaired run -- the loop or
linker a ribosome, an antisense oligo or a probe would have to reach. Accessibility is the mean probability
that those positions are unpaired IN THE EQUILIBRIUM ENSEMBLE, not merely in the MFE structure: a sequence
can fold to the target as its single best structure while that loop is paired in much of the ensemble, and
that is exactly the failure a bench experiment would see.

    accessibility(seq) = mean over window of (1 - sum of that position's pairing probabilities)
    requirement:         accessibility >= 0.80

Swept 0.70-0.95 with both arms recomputed, because a single cutoff should never carry a conclusion.

INSTRUMENT CALIBRATED FIRST, and declared because it looked at data. Fifteen inverse-fold sequences per
target -- ViennaRNA's own designs, neither benchmark arm -- gave median accessibilities of 0.822, 0.831,
0.778 and 0.799 with 0.68-0.83 spread, and the 0.80 line split them 3/11, 4/15, 6/8 and 3/5. The measurement
discriminates; it is neither free nor impossible. Targets whose longest unpaired run is under 5 nt are
excluded and counted, because a 3 nt window makes the mean meaningless.
"""
from __future__ import annotations

from typing import Dict, List, Optional

ACCESS_MIN = 0.80              # declared before the run; the sweep is what the conclusion rests on
SWEEP = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
MIN_WINDOW = 5                 # below this the mean is noise, and such targets are dropped and reported


def longest_unpaired(target: str):
    """(start, length) of the longest unpaired run -- the part something else would have to bind."""
    best_len, best_at, run = 0, 0, 0
    for i, c in enumerate(target):
        if c == ".":
            run += 1
            if run > best_len:
                best_len, best_at = run, i - run + 1
        else:
            run = 0
    return (best_at, best_len) if best_len else (0, 0)


def accessibility(seq: str, at: int, length: int) -> float:
    """Mean equilibrium unpaired probability over the window.

    The ENSEMBLE, not the MFE. A design whose loop is open in its single best structure and paired across
    much of the ensemble is exactly the design that disappoints at the bench, and the MFE cannot see it.
    """
    import RNA
    fc = RNA.fold_compound(seq)
    fc.pf()
    bpp = fc.bpp()
    n = len(seq)
    up = [1.0] * n
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = bpp[i][j]
            if p:
                up[i - 1] -= p
                up[j - 1] -= p
    return sum(up[at:at + length]) / max(1, length)


def gc_percent(seq: str) -> float:
    return 100.0 * sum(1 for c in seq if c in "GC") / max(1, len(seq))


def screen(pool: List[str], target: str, at: int, length: int, gc_band=None,
           cap: int = 60) -> Dict:
    """First candidate that folds AND is accessible, plus the per-candidate rows the controls need."""
    from ..cognitive.certify import certify_sequence
    chosen, rows = None, []
    for s in pool:
        if len(rows) >= cap:
            break
        if gc_band and not (gc_band[0] <= gc_percent(s) <= gc_band[1]):
            continue
        if certify_sequence(s, target).verdict not in ("certified", "knot_confirmed"):
            continue
        acc = accessibility(s, at, length)
        rows.append({"gc": round(gc_percent(s), 1), "accessibility": round(acc, 4),
                     "held": acc >= ACCESS_MIN})
        if acc >= ACCESS_MIN and chosen is None:
            chosen = {"sequence": s, "gc": round(gc_percent(s), 1), "accessibility": round(acc, 4)}
    n_pass = sum(1 for r in rows if r["held"])
    return {"chosen": chosen, "n_folding": len(rows), "n_pass": n_pass,
            "pool_pass_rate": round(n_pass / max(1, len(rows)), 4), "capped": len(rows) >= cap,
            "rows": rows}
