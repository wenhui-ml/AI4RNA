"""Distill checkable RNA design strategies from solved cases, and measure their adoption.

Strategies are deliberately *checkable* (deterministic predicates over a (structure,
sequence)) so "was this remembered strategy adopted?" is exact -- that is what makes
A_memory a real actuation measurement rather than a vibe.

Two strategy kinds (both real RNA design heuristics that transfer across cases):
  * hairpin_motif : for a hairpin loop of length L, use a specific stable loop sequence.
  * gc_closing    : close helices with strong G-C pairs (fraction of loop-closing pairs).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from ..structure_utils import pair_table
from .memstore import Strategy


def _hairpin_loops(target: str) -> List[Tuple[int, int, int]]:
    """Return (i, j, loop_len) for each hairpin (a pair enclosing only unpaired bases)."""
    pt = pair_table(target)
    out = []
    for i in range(len(target)):
        j = pt[i]
        if j > i and all(pt[k] == -1 for k in range(i + 1, j)):
            out.append((i, j, j - i - 1))
    return out


def _closing_pairs(target: str) -> List[Tuple[int, int]]:
    """Loop-closing pairs: a pair (i,j) not immediately stacked inside by another pair."""
    pt = pair_table(target)
    out = []
    for i in range(len(target)):
        j = pt[i]
        if j > i and not (i + 1 < j - 1 and pt[i + 1] == j - 1):
            out.append((i, j))
    return out


def _is_gc(a: str, b: str) -> bool:
    return {a, b} == {"G", "C"}


# --------------------------------------------------------------------------- distill
def distill_strategies(target: str, sequence: str) -> List[Strategy]:
    """Extract strategies exhibited by a *solved* (target, sequence)."""
    seq = sequence.upper().replace("T", "U")
    out: List[Strategy] = []
    for i, j, L in _hairpin_loops(target):
        motif = seq[i + 1 : j]
        if set(motif) <= set("ACGU") and 3 <= L <= 6:
            out.append(Strategy(
                name=f"rna-hairpin-{L}-{motif.lower()}",
                kind="hairpin_motif",
                trigger={"loop_len": L},
                rule={"motif": motif},
                description=f"Use '{motif}' for {L}-nt hairpin loops",
            ))
    cps = _closing_pairs(target)
    if cps:
        gc = sum(1 for i, j in cps if _is_gc(seq[i], seq[j])) / len(cps)
        if gc >= 0.5:
            out.append(Strategy(
                name="rna-gc-closing",
                kind="gc_closing",
                trigger={},
                rule={"min_gc_closing": 0.6},
                description="Close helices with G-C pairs",
            ))
    return out


def merge_strategies(existing: List[Strategy], new: List[Strategy]) -> List[Strategy]:
    """Consolidate strategies. For hairpin_motif of a given loop_len keep the most-supported
    motif; sum supports. For gc_closing sum supports."""
    # collect motif votes per loop_len, and gc_closing support
    motif_votes: Dict[int, Counter] = defaultdict(Counter)
    gc_support = 0
    passthrough: Dict[tuple, Strategy] = {}
    for s in list(existing) + list(new):
        if s.kind == "hairpin_motif":
            motif_votes[s.trigger["loop_len"]][s.rule["motif"]] += s.support
        elif s.kind == "gc_closing":
            gc_support += s.support
        else:
            passthrough[s.key()] = s
    out: List[Strategy] = []
    for L, votes in sorted(motif_votes.items()):
        motif, support = votes.most_common(1)[0]
        out.append(Strategy(
            name=f"rna-hairpin-{L}-{motif.lower()}", kind="hairpin_motif",
            trigger={"loop_len": L}, rule={"motif": motif},
            description=f"Use '{motif}' for {L}-nt hairpin loops (winning motif of {sum(votes.values())} obs)",
            support=int(sum(votes.values())),
        ))
    if gc_support > 0:
        out.append(Strategy(
            name="rna-gc-closing", kind="gc_closing", trigger={}, rule={"min_gc_closing": 0.6},
            description="Close helices with G-C pairs", support=gc_support,
        ))
    out.extend(passthrough.values())
    return out


# --------------------------------------------------------------------------- adoption
def applicable_strategies(strategies: List[Strategy], target: str) -> List[Strategy]:
    loops = _hairpin_loops(target)
    loop_lens = {L for _, _, L in loops}
    has_pairs = bool(_closing_pairs(target))
    out = []
    for s in strategies:
        if s.kind == "hairpin_motif" and s.trigger.get("loop_len") in loop_lens:
            out.append(s)
        elif s.kind == "gc_closing" and has_pairs:
            out.append(s)
    return out


def _adopted(strategy: Strategy, target: str, sequence: str) -> bool:
    seq = sequence.upper().replace("T", "U")
    if strategy.kind == "hairpin_motif":
        L = strategy.trigger["loop_len"]
        motif = strategy.rule["motif"]
        for i, j, ll in _hairpin_loops(target):
            if ll == L and seq[i + 1 : j] == motif:
                return True
        return False
    if strategy.kind == "gc_closing":
        cps = _closing_pairs(target)
        if not cps:
            return False
        gc = sum(1 for i, j in cps if _is_gc(seq[i], seq[j])) / len(cps)
        return gc >= float(strategy.rule.get("min_gc_closing", 0.6))
    return False


def memory_adoption(strategies: List[Strategy], target: str, sequence: str) -> Dict[str, object]:
    """A_memory: of the recalled strategies applicable to this case, how many did the
    produced solution actually exhibit?"""
    appl = applicable_strategies(strategies, target)
    adopted = [s for s in appl if _adopted(s, target, sequence)]
    return {
        "n_applicable": len(appl),
        "n_adopted": len(adopted),
        "applicable": [s.name for s in appl],
        "adopted": [s.name for s in adopted],
        "A_memory": (len(adopted) / len(appl)) if appl else None,
    }
