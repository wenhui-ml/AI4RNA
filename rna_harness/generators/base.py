"""Generator abstraction + the constraint-guided initialization / structured mutation
primitives shared by classical baselines.
"""
from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..constraints import ConstraintSpec
from ..structure_utils import pair_table
from ..task import DesignTask

# The six canonical base-pairs (Watson-Crick + wobble).
_PAIRS = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]
_GC_PAIRS = [("G", "C"), ("C", "G")]
_AU_GU_PAIRS = [("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]
_BASES = ["A", "C", "G", "U"]


@dataclass
class GenerationResult:
    task_id: str
    best_sequence: str
    solved: bool
    best_bp_distance: int
    n_evals: int
    candidates: List[str] = field(default_factory=list)  # search path (proposed/accepted seqs)
    method: str = "generator"
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "best_sequence": self.best_sequence,
            "solved": self.solved,
            "best_bp_distance": self.best_bp_distance,
            "n_evals": self.n_evals,
            "n_candidates": len(self.candidates),
            "method": self.method,
            "meta": self.meta,
        }


class Generator(abc.ABC):
    name: str = "generator"

    @abc.abstractmethod
    def design(self, task: DesignTask, budget: int = 2000, seed: Optional[int] = None) -> GenerationResult:
        ...


# ---------------------------------------------------------------------------
# Shared constraint-guided primitives
# ---------------------------------------------------------------------------
def gc_target_prob(spec: ConstraintSpec, default: float = 0.5) -> float:
    """A GC-pairing propensity derived from the GC window midpoint (best-effort)."""
    if spec.gc_min is None and spec.gc_max is None:
        return default
    lo = spec.gc_min if spec.gc_min is not None else 0.3
    hi = spec.gc_max if spec.gc_max is not None else 0.7
    return max(0.0, min(1.0, (lo + hi) / 2.0))


def _apply_motifs(seq: List[str], spec: ConstraintSpec) -> None:
    """Overwrite positions with fixed literal motifs (IUPAC codes left as-is if ambiguous)."""
    for pos, motif in spec.motifs:
        m = motif.upper().replace("T", "U")
        for k, ch in enumerate(m):
            if pos + k < len(seq) and ch in "ACGU":
                seq[pos + k] = ch


def init_sequence(
    target: str, spec: ConstraintSpec, rng: random.Random, p_gc: Optional[float] = None
) -> str:
    """Constraint-guided initialization: pair positions get complementary pairs, unpaired
    positions get a base biased by the GC target; fixed motifs are stamped in."""
    if p_gc is None:
        p_gc = gc_target_prob(spec)
    n = len(target)
    pt = pair_table(target)
    seq: List[str] = ["A"] * n
    for i in range(n):
        j = pt[i]
        if j > i:  # opening of a pair
            pair = rng.choice(_GC_PAIRS) if rng.random() < p_gc else rng.choice(_AU_GU_PAIRS)
            seq[i], seq[j] = pair
        elif j == -1:  # unpaired
            seq[i] = "G" if rng.random() < p_gc / 2 else ("C" if rng.random() < p_gc else rng.choice(["A", "U"]))
    _apply_motifs(seq, spec)
    return "".join(seq)


def structured_mutate(
    seq: str,
    pos: int,
    target: str,
    spec: ConstraintSpec,
    rng: random.Random,
    p_gc: Optional[float] = None,
    frozen: Optional[set] = None,
) -> str:
    """Mutate correlated positions together: a target pair is resampled as a whole; an
    unpaired position is resampled to a random base. Never mutates half a pair. Positions in
    ``frozen`` (e.g. fixed-motif positions) are left untouched."""
    if p_gc is None:
        p_gc = gc_target_prob(spec)
    frozen = frozen or set()
    pt = pair_table(target)
    s = list(seq)
    j = pt[pos]
    if j == -1:
        if pos not in frozen:
            s[pos] = "G" if rng.random() < p_gc / 2 else ("C" if rng.random() < p_gc else rng.choice(["A", "U"]))
    else:
        a, b = pos, j
        if a in frozen or b in frozen:
            return seq
        pair = rng.choice(_GC_PAIRS) if rng.random() < p_gc else rng.choice(_AU_GU_PAIRS)
        s[a], s[b] = pair
        # ~20% co-mutate a stacked neighbour pair to fix two-pair helix defects
        if rng.random() < 0.2:
            for da in (-1, 1):
                na = a + da
                if 0 <= na < len(seq) and pt[na] not in (-1,) and na not in frozen and pt[na] not in frozen:
                    nb = pt[na]
                    npair = rng.choice(_GC_PAIRS) if rng.random() < p_gc else rng.choice(_AU_GU_PAIRS)
                    s[na], s[nb] = npair
                    break
    return "".join(s)


def frozen_motif_positions(spec: ConstraintSpec) -> set:
    out = set()
    for pos, motif in spec.motifs:
        for k in range(len(motif)):
            out.add(pos + k)
    return out
