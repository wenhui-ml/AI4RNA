"""A requirement that is NOT a composition requirement -- the first candidate for a SECOND gap.

WHY THIS SPECIFIC REQUIREMENT. Three gaps have been confirmed against the standard tool (composition, low
melting temperature, orthogonal sets) and then a controlled test collapsed all three into ONE: fix the GC band
at 45-58% and the tool contributes 0-1 sequences, so its failure on every one of them reduces to composition.
The window criterion says a wrapper pays when the requirement forces a loss in the solver's own objective, and
GC is the dominant lever on folding stability, so almost any "must pay stability" requirement re-expresses
itself as a GC requirement. To claim a second capability at all, the requirement has to be ORTHOGONAL TO
COMPOSITION -- and that is a property to be MEASURED here, not asserted.

Co-transcriptional folding is the sharpest candidate available to us:

  * IT IS A PERMUTATION PROPERTY, NOT A COMPOSITION ONE. Two sequences with identical GC content -- identical
    base counts, even -- can differ completely on whether the growing 5' prefix folds into a trap. That is the
    exact respect in which the previous three requirements failed.
  * THE SOLVER HAS NO TERM FOR IT. DesiRNA's entire option surface is equilibrium: MFE, ensemble defect
    (-sf Edef), suboptimal gap (-nd), oligomerisation (-o), dimer (-d), composition (-acgu_content), targeted
    mutations. Grepping the source for cotranscriptional/kinetic/barrier/findpath returns nothing. It is not
    that the option is awkward to reach -- the objective does not contain the quantity.
  * WE ALREADY HAVE AN EXACT ORACLE. Prefix folding is deterministic ViennaRNA, and the saddle energy between
    two structures comes from fold_compound.path_findpath_saddle -- no new dependency, no stochastic Kinfold.

THE REQUIREMENT, declared before the run. Transcription runs 5'->3', so at every prefix length the partially
synthesised molecule folds into whatever is best FOR THAT PREFIX. A pair formed early that is absent from the
final target has to be broken later, and breaking a helix is exactly what a kinetic trap is. So:

    a prefix is a TRAP if MFE(seq[:k]) contains any base pair that the target does not contain
    trap_fraction = (# trap prefixes) / (# prefixes examined)
    the requirement is  trap_fraction <= 0.10

THE THRESHOLD IS DECLARED, NOT FITTED. 0.10 is written here before the benchmark runs, and the full
distribution is reported alongside every verdict so a reader can see whether the number is load-bearing. A
threshold chosen after looking at the two arms would be the whole result.

WHAT WOULD REFUTE THE WHOLE IDEA, stated in advance, because two earlier predictions in this project were
refuted by their own pre-registration and that is the only reason the surviving ones are worth anything:

  1. NO GAP. If the tool's own answers already pass, the requirement is free inside the objective's level set
     -- the same way banned 4-mers turned out to be. Then there is no second capability here.
  2. M1's CEILING. If essentially NOTHING in the pool passes, we cannot select what is not there. That is not
     a gap we can fill by screening; it needs a better search (M6), which this project does not have. The
     measurement that decides it is pool_pass_rate, reported for every target.
  3. IT COLLAPSES TO GC AGAIN. If the gap disappears once both arms are held inside 45-58% GC, then this is a
     fourth costume on the same one gap, and the honest account is that the project has ONE extra capability.
     That outcome is a result and gets written down as one.

Only outcome 4 -- a gap that survives the GC control -- would make this a second capability.
"""
from __future__ import annotations

from typing import Dict, List, Optional

TRAP_FRACTION_MAX = 0.10     # declared before the run; see the docstring
MIN_PREFIX = 12              # below this a prefix has no room to fold into anything meaningful
PREFIX_STEP = 1              # every length; these molecules are short enough that this is affordable


def pair_set(struct: str):
    """Every base pair, including crossing ones. Brackets only -- unpaired characters are ignored."""
    out, stacks = set(), {}
    close = {")": "(", "]": "[", "}": "{", ">": "<"}
    for i, ch in enumerate(struct):
        if ch in "([{<":
            stacks.setdefault(ch, []).append(i)
        elif ch in close:
            st = stacks.get(close[ch]) or []
            if st:
                out.add((st.pop(), i))
    return out


def cotranscriptional(seq: str, target: str) -> Dict:
    """Fold every 5' prefix and count the ones that form a pair the target does not have.

    Deterministic and exact under the stated model: this is ViennaRNA's MFE on each prefix, not a stochastic
    kinetic simulation. It is a LOWER bound on trouble -- a real polymerase also has speed and pausing, which
    we do not model and do not claim to.
    """
    import RNA
    tgt = pair_set(target)
    n = len(seq)
    traps, worst, worst_off = [], None, -1
    ks = list(range(MIN_PREFIX, n + 1, PREFIX_STEP))
    for k in ks:
        st = RNA.fold(seq[:k])[0]
        off = [p for p in pair_set(st) if p not in tgt]
        if off:
            traps.append(k)
            if len(off) > worst_off:
                worst_off, worst = len(off), {"k": k, "structure": st, "n_off_pairs": len(off)}
    frac = len(traps) / max(1, len(ks))
    return {"n_prefixes": len(ks), "n_traps": len(traps), "trap_fraction": round(frac, 4),
            "held": frac <= TRAP_FRACTION_MAX, "worst": worst,
            "first_trap_k": traps[0] if traps else None}


def refold_barrier(seq: str, frm: str, to: str, width: int = 20) -> Optional[float]:
    """kcal/mol saddle on the direct path between two structures -- how deep the trap is, if there is one.

    findpath is a heuristic UPPER bound on the true barrier, which is the safe direction here: it can only
    make a trap look no worse than it is, never worse than it is.
    """
    import RNA
    try:
        fc = RNA.fold_compound(seq)
        return fc.path_findpath_saddle(frm, to, width) / 100.0     # dekacal/mol -> kcal/mol
    except Exception:                                              # noqa: BLE001
        return None


def gc_percent(seq: str) -> float:
    return 100.0 * sum(1 for c in seq if c in "GC") / max(1, len(seq))


def screen(pool: List[str], target: str, gc_band=None) -> Dict:
    """Ask of a pool the two questions that separate the three refutation modes.

    Returns the first sequence that both folds to the target AND passes the kinetic requirement, plus the
    pass rate -- which is the number that tells M1's ceiling apart from a real selection win.
    """
    from ..cognitive.certify import certify_sequence
    chosen, n_folded, n_pass, rows = None, 0, 0, []
    for s in pool:
        if gc_band and not (gc_band[0] <= gc_percent(s) <= gc_band[1]):
            continue
        cert = certify_sequence(s, target)
        folds = cert.verdict in ("certified", "knot_confirmed")
        n_folded += bool(folds)
        if not folds:
            continue
        k = cotranscriptional(s, target)
        n_pass += bool(k["held"])
        rows.append({"gc": round(gc_percent(s), 1), "trap_fraction": k["trap_fraction"],
                     "held": k["held"]})
        if k["held"] and chosen is None:
            chosen = {"sequence": s, "kinetics": k, "gc": round(gc_percent(s), 1)}
    return {"chosen": chosen, "n_in_band": len(rows), "n_folded": n_folded, "n_pass": n_pass,
            "pool_pass_rate": round(n_pass / max(1, n_folded), 4), "rows": rows}
