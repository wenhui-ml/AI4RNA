"""Structured folding-mismatch diagnosis == the T_trace (trace-determinism) signal.

Because the oracle is exact, a failure localizes precisely: we can name exactly which base
pairs the target wants that the fold missed, and which spurious pairs the fold introduced.
This is the "directive" whose adoption the actuation instrumentation later checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from ..structure_utils import base_pairs, pair_table


@dataclass
class Diagnosis:
    target: str
    folded: str
    bp_distance: int
    missing_pairs: List[Tuple[int, int]]     # in target, absent in fold  (0-indexed)
    spurious_pairs: List[Tuple[int, int]]    # in fold, absent in target
    mismatched_positions: List[int]          # union of positions whose pairing state differs
    directive: str                           # agent-facing natural-language localization

    def diagnosed_position_set(self) -> Set[int]:
        return set(self.mismatched_positions)

    def to_dict(self) -> Dict[str, object]:
        return {
            "bp_distance": self.bp_distance,
            "missing_pairs": [list(p) for p in self.missing_pairs],
            "spurious_pairs": [list(p) for p in self.spurious_pairs],
            "mismatched_positions": self.mismatched_positions,
            "n_mismatched": len(self.mismatched_positions),
            "directive": self.directive,
        }


def diagnose(target: str, folded: str) -> Diagnosis:
    """Localize the difference between what the target wants and how the sequence folds."""
    if len(folded) != len(target):
        raise ValueError(f"folded length {len(folded)} != target length {len(target)}")
    tgt_pairs: Set[Tuple[int, int]] = set(base_pairs(target))
    fold_pairs: Set[Tuple[int, int]] = set(base_pairs(folded))
    missing = sorted(tgt_pairs - fold_pairs)
    spurious = sorted(fold_pairs - tgt_pairs)

    pt_t = pair_table(target)
    pt_f = pair_table(folded)
    mism = [i for i in range(len(target)) if pt_t[i] != pt_f[i]]

    directive = _render_directive(target, folded, missing, spurious, mism)
    # bp-distance = size of the symmetric difference of the pair sets; consistent with our
    # pair_table (which handles pseudoknot brackets that RNA.bp_distance silently ignores).
    return Diagnosis(
        target=target,
        folded=folded,
        bp_distance=len(missing) + len(spurious),
        missing_pairs=missing,
        spurious_pairs=spurious,
        mismatched_positions=mism,
        directive=directive,
    )


def _render_directive(
    target: str,
    folded: str,
    missing: List[Tuple[int, int]],
    spurious: List[Tuple[int, int]],
    mism: List[int],
) -> str:
    if not missing and not spurious:
        return "The sequence already folds to the target under this oracle. No repair needed."
    lines: List[str] = []
    lines.append(
        f"The MFE fold differs from the target at {len(mism)} position(s) "
        f"({len(missing)} target pair(s) missing, {len(spurious)} spurious pair(s))."
    )
    if missing:
        ex = ", ".join(f"({i}-{j})" for i, j in missing[:8])
        lines.append(
            f"MISSING pairs (target wants these but they did not form): {ex}"
            + (" ..." if len(missing) > 8 else "")
            + ". Strengthen these helices (e.g. use G-C at the closing/opening positions, "
            "or fix the complementarity of the two paired bases)."
        )
    if spurious:
        ex = ", ".join(f"({i}-{j})" for i, j in spurious[:8])
        lines.append(
            f"SPURIOUS pairs (formed but not wanted): {ex}"
            + (" ..." if len(spurious) > 8 else "")
            + ". Break these (e.g. introduce a mismatch or weaken to A-U/G-U) so the "
            "intended structure becomes the ground state."
        )
    lines.append(
        "Only change positions implicated above; leave already-correct regions untouched."
    )
    return "\n".join(lines)
