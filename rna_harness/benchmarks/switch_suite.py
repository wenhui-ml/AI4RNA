"""A two-state switch -- the LAST coordinate the sharpened criterion still predicts a gap on.

WHERE THIS SITS IN THE SEQUENCE OF REFUTATIONS. Requirements that turned out to be free: umfe, pfe, banned
4-mers, and now mutational robustness. Requirements that opened a gap: composition (with melting temperature
and orthogonality collapsing into it under the GC control), and co-transcriptional folding. The rule that
survives all of them is not "the objective does not contain it" and not "it is a neighbourhood property" --
robustness is both of those and opened nothing:

    A NEIGHBOURHOOD PROPERTY OPENS A GAP ONLY WHEN THE NEIGHBOURHOOD SPANS DIFFERENT ENERGY LANDSCAPES.

      prefixes      each prefix is a different molecule; full-length well depth says nothing   GAP
      mutants       a small perturbation of the SAME landscape, and depth transfers            null
      temperature   the same landscape scaling smoothly                                        null (Tm)
      partner       monomer against complex -- two molecules, two landscapes                   THIS FILE

Robustness is the informative null: MFE optimisation produces a deep, well-separated well, a point mutation
perturbs the energy by a few kcal/mol, and the well is deeper than that. Stability IMPLIES robustness, so in
substance it belongs with umfe/pfe. The partner direction cannot be free that way, because nothing about the
monomer's well depth determines what happens once a second strand is in the box.

WHAT THE SOLVER CAN AND CANNOT SAY, read from its source rather than assumed, because the whole claim would
be worthless if the requirement were merely awkward to reach:

    monomer folds to S1                             yes, the ordinary target
    monomer can ALSO adopt S2                       yes -- but `alt_sec_struct` is scored as
                                                    fold_comp.eval_structure(alt_dbn) on the MONOMER's own
                                                    fold compound. It is a BISTABLE SINGLE MOLECULE, and it
                                                    has nothing to do with a partner being present.
    a complex A&B folds to X                        yes
    monomer folds to S1 AND complex A&B folds to S2  NO. There is no term that conditions one structure on
                                                    the presence of another strand.

THE REQUIREMENT, declared before the run, and it is a TOEHOLD switch because the first version was not.
The first design pointed the trigger at the stem alone. Smoke-tested before it ever ran, it did the opposite
of what it was for: whenever the free state was correct the trigger opened ZERO of 8 stem pairs, and the only
cases with full binding were the ones whose free fold was already wrong. An 8 nt duplex loses to a helix
braced by the whole molecule -- so that benchmark would have scored 0 in both arms and been reported as "no
gap" when it was really "impossible by construction". Real toehold switches work because a SINGLE-STRANDED
toehold nucleates the trigger, which then invades the adjacent helix. So:

    window     an unpaired run of TOEHOLD nt immediately 5' of a stacked stem, plus INVADE pairs of that stem
    trigger    the exact reverse complement of the candidate's own window, derived per candidate because in a
               real design the trigger is designed alongside the switch; both arms are held to the same rule
    free       RNA.fold(s) == target                      the stem is formed, the toehold is open
    bound      in cofold(s & trigger), how many of the INVADE stem positions pair with a TRIGGER base

The two states pull against each other: the free state wants that helix formed, the bound state wants the
trigger to win against it. The solver optimises the free state alone.

NO SINGLE CUTOFF IS LOAD-BEARING. Invasion depth is reported as a number from 0 to INVADE and the requirement
is swept over all of it, both arms recomputed at each point. Full invasion is the headline.

THE INSTRUMENT WAS CALIBRATED BEFORE THE REQUIREMENT WAS SCORED, and that pilot is declared rather than
hidden, because it did look at data. Twenty inverse-fold sequences per target -- ViennaRNA's own designs, not
DesiRNA's, and not either benchmark arm -- were measured to check that the measurement DISCRIMINATES at all.
Full-invasion rates came out at 73%, 92% and 24% on three targets. Neither free nor impossible, which is what
a usable instrument looks like, and the alternative was shipping the broken version above.

THREE WAYS THIS DIES, written first:
  1. the tool's own answers already switch    -> free inside the objective's level sets, like robustness
  2. almost nothing in the pool switches      -> M1's ceiling; screening cannot conjure what is not there
  3. the gap vanishes in the GC band          -> a fourth costume on the composition gap
Only a gap that survives the band counts, and if this one dies too, the honest account is that the project
has exactly two capabilities and the search for a third is over.
"""
from __future__ import annotations

from typing import Dict, List, Optional

TOEHOLD = 4                    # declared: unpaired nucleation run 5' of the stem
INVADE = 5                     # declared: stem pairs the trigger must displace
COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


def pair_table(struct: str) -> List[int]:
    pt, st = [-1] * len(struct), []
    for i, ch in enumerate(struct):
        if ch == "(":
            st.append(i)
        elif ch == ")" and st:
            j = st.pop()
            pt[i], pt[j] = j, i
    return pt


def switch_window(target: str, toe: int = TOEHOLD, inv: int = INVADE):
    """(start, toe, inv) for the first stacked stem of `inv` pairs with an unpaired run of `toe` just 5' of it.

    Both halves are required. Without the toehold the trigger has nowhere to nucleate and the requirement is
    unsatisfiable rather than hard -- which is exactly how the first version of this file failed.
    """
    pt = pair_table(target)
    for i in range(len(target)):
        j = pt[i]
        if j <= i:
            continue
        if not all(0 <= i + k < len(pt) and pt[i + k] == j - k for k in range(inv)):
            continue
        if j - inv + 1 <= i + inv - 1:                      # the stem must not close on itself
            continue
        if i - toe < 0 or any(pt[i - 1 - k] >= 0 for k in range(toe)):
            continue
        return i - toe, toe, inv
    return None


def trigger_for(seq: str, at: int, toe: int = TOEHOLD, inv: int = INVADE) -> str:
    """The exact reverse complement of this candidate's own toehold + invaded stem."""
    return "".join(COMPLEMENT[c] for c in reversed(seq[at:at + toe + inv]))


def switches(seq: str, target: str, at: int, toe: int = TOEHOLD, inv: int = INVADE) -> Dict:
    """Both states, exactly. Deterministic: ViennaRNA MFE on the monomer and on the two-chain complex."""
    import RNA
    free = RNA.fold(seq)[0]
    trig = trigger_for(seq, at, toe, inv)
    n = len(seq)
    st, dg = RNA.cofold(seq + "&" + trig)
    pt = pair_table(st)
    # Paired to a TRIGGER base, not merely paired: an unchanged internal helix also shows '(' here, and
    # counting brackets instead of partners would score "the switch did nothing" as a success.
    depth = sum(1 for i in range(at + toe, at + toe + inv) if pt[i] >= n)
    return {"free_ok": free == target, "free_structure": free, "trigger": trig,
            "invasion": depth, "held": bool(free == target and depth == inv),
            "complex_dg": round(dg, 2), "complex_structure": st}


def gc_percent(seq: str) -> float:
    return 100.0 * sum(1 for c in seq if c in "GC") / max(1, len(seq))


def screen(pool: List[str], target: str, at: int, gc_band=None, cap: int = 60) -> Dict:
    """First candidate meeting BOTH states, plus the pass rate that separates a win from M1's ceiling."""
    chosen, rows = None, []
    for s in pool:
        if len(rows) >= cap:
            break
        if gc_band and not (gc_band[0] <= gc_percent(s) <= gc_band[1]):
            continue
        r = switches(s, target, at)
        if not r["free_ok"]:
            continue                      # not a candidate for the requirement at all
        rows.append({"gc": round(gc_percent(s), 1), "invasion": r["invasion"], "held": r["held"]})
        if r["held"] and chosen is None:
            chosen = {"sequence": s, "gc": round(gc_percent(s), 1), **r}
    n_pass = sum(1 for r in rows if r["held"])
    return {"chosen": chosen, "n_folding": len(rows), "n_pass": n_pass,
            "pool_pass_rate": round(n_pass / max(1, len(rows)), 4), "capped": len(rows) >= cap,
            "rows": rows}
