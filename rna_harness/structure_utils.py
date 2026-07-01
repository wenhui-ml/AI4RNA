"""Secondary-structure utilities: dot-bracket parsing, pair tables, motif scanning.

All structures are dot-bracket strings over the alphabet ``().`` (unpaired ``.``);
we additionally tolerate extended pseudoknot brackets ``[]{}<>`` for parsing/pair
tables (ViennaRNA's nested folding cannot *design* to these, but tasks may carry
them and we want to fail loudly rather than silently).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Matching bracket pairs we understand. The nested (non-pseudoknot) folding oracle
# only round-trips the first pair; the rest are here so parsing never crashes.
_BRACKETS: Dict[str, str] = {")": "(", "]": "[", "}": "{", ">": "<"}
_OPEN = set(_BRACKETS.values())
_CLOSE = set(_BRACKETS.keys())

RNA_BASES = ("A", "C", "G", "U")


def is_dotbracket(struct: str) -> bool:
    return all(c in ".()[]{}<>" for c in struct)


def has_pseudoknot(struct: str) -> bool:
    return any(c in "[]{}<>" for c in struct)


def pair_table(struct: str) -> List[int]:
    """Return a 0-indexed pair table.

    ``pt[i]`` is the index of the base paired with position ``i``, or ``-1`` if
    ``i`` is unpaired. Supports the four bracket families independently so
    pseudoknotted targets parse correctly.
    """
    pt = [-1] * len(struct)
    stacks: Dict[str, List[int]] = {b: [] for b in _OPEN}
    for i, c in enumerate(struct):
        if c in _OPEN:
            stacks[c].append(i)
        elif c in _CLOSE:
            opener = _BRACKETS[c]
            if not stacks[opener]:
                raise ValueError(f"Unbalanced structure at position {i}: {struct!r}")
            j = stacks[opener].pop()
            pt[i] = j
            pt[j] = i
        elif c != ".":
            raise ValueError(f"Illegal structure character {c!r} in {struct!r}")
    for b, s in stacks.items():
        if s:
            raise ValueError(f"Unbalanced '{b}' in structure: {struct!r}")
    return pt


def base_pairs(struct: str) -> List[Tuple[int, int]]:
    """Sorted list of (i, j) pairs with i < j (0-indexed)."""
    pt = pair_table(struct)
    return sorted((i, j) for i, j in enumerate(pt) if j > i)


def paired_positions(struct: str) -> List[bool]:
    return [c != "." for c in struct]


def dot_bracket_matches(a: str, b: str) -> bool:
    """Exact character-level equality (the strict uMFE criterion)."""
    return a == b


def valid_pair(b1: str, b2: str, allow_gu: bool = True) -> bool:
    """Whether two bases can form a canonical (Watson-Crick / wobble) pair."""
    p = {b1, b2}
    if p == {"A", "U"} or p == {"G", "C"}:
        return True
    if allow_gu and p == {"G", "U"}:
        return True
    return False


def complement(base: str, partner_prefer_gc: bool = True) -> str:
    """A canonical partner for a base. Prefers G-C strength when asked."""
    table = {"A": "U", "U": "A", "G": "C", "C": "G"}
    return table[base]


def loops_and_stems(struct: str) -> Dict[str, List[int]]:
    """Classify positions as 'paired' or 'unpaired' index lists."""
    paired, unpaired = [], []
    for i, c in enumerate(struct):
        (paired if c != "." else unpaired).append(i)
    return {"paired": paired, "unpaired": unpaired}


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    return sum(1 for x, y in zip(a, b) if x != y)


def positional_mismatches(target: str, folded: str) -> List[int]:
    """Positions where the pairing *state or partner* differs between target and
    folded structures. This is the raw material for a faithful repair directive:
    it localizes exactly which nucleotides the fold got wrong.
    """
    pt_t = pair_table(target)
    pt_f = pair_table(folded)
    bad = []
    for i in range(len(target)):
        if pt_t[i] != pt_f[i]:
            bad.append(i)
    return bad


def find_motif(sequence: str, motif: str, start: int = 0) -> int:
    """First index >= start where ``motif`` occurs, or -1. IUPAC-aware if motif
    contains ambiguity codes."""
    if _is_plain(motif):
        return sequence.find(motif, start)
    n, m = len(sequence), len(motif)
    for i in range(start, n - m + 1):
        if all(_iupac_match(sequence[i + k], motif[k]) for k in range(m)):
            return i
    return -1


_IUPAC = {
    "A": set("A"), "C": set("C"), "G": set("G"), "U": set("U"), "T": set("U"),
    "R": set("AG"), "Y": set("CU"), "S": set("GC"), "W": set("AU"),
    "K": set("GU"), "M": set("AC"), "B": set("CGU"), "D": set("AGU"),
    "H": set("ACU"), "V": set("ACG"), "N": set("ACGU"),
}


def _is_plain(motif: str) -> bool:
    return all(c in "ACGU" for c in motif)


def _iupac_match(base: str, code: str) -> bool:
    return base in _IUPAC.get(code, {code})
