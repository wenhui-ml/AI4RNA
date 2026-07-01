"""Structural + ensemble metrics over folding results.

Pure, deterministic functions. Structural distances are exact; ensemble quantities come
from the partition function of the given oracle. Everything here is defined *relative to a
fixed oracle configuration* -- the caller must record (param_set, temperature, dangles).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import RNA

from .oracles.base import Oracle
from .oracles.vienna import ViennaRNAOracle
from .structure_utils import base_pairs, hamming, pair_table, positional_mismatches


def bp_distance(a: str, b: str) -> int:
    """Base-pair distance: symmetric difference of the two pair sets. 0 iff pair-identical."""
    return int(RNA.bp_distance(a, b))


def dotbracket_hamming(a: str, b: str) -> int:
    return hamming(a, b)


def normalized_structural_distance(a: str, b: str) -> float:
    """bp_distance normalized by length, in [0, ~1]. Used for reward shaping."""
    n = max(len(a), 1)
    return bp_distance(a, b) / n


@dataclass
class EnsembleMetrics:
    mfe_structure: str
    mfe_energy: float
    ensemble_free_energy: float
    prob_of_target: Optional[float]
    ensemble_defect: Optional[float]
    mfe_frequency: Optional[float]
    bp_distance_to_target: int
    mfe_matches_target: bool
    is_umfe: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "mfe_structure": self.mfe_structure,
            "mfe_energy": round(self.mfe_energy, 4),
            "ensemble_free_energy": round(self.ensemble_free_energy, 4)
            if self.ensemble_free_energy is not None
            else None,
            "prob_of_target": self.prob_of_target,
            "ensemble_defect": self.ensemble_defect,
            "mfe_frequency": self.mfe_frequency,
            "bp_distance_to_target": self.bp_distance_to_target,
            "mfe_matches_target": self.mfe_matches_target,
            "is_umfe": self.is_umfe,
        }


def ensemble_metrics(oracle: Oracle, sequence: str, target: str) -> EnsembleMetrics:
    """Full ensemble analysis of ``sequence`` against ``target`` under ``oracle``."""
    res = oracle.analyze(sequence, target)
    d = bp_distance(res.mfe_structure, target)
    umfe = False
    if isinstance(oracle, ViennaRNAOracle):
        try:
            umfe = oracle.is_umfe(sequence, target)
        except Exception:
            umfe = res.mfe_structure == target
    else:
        umfe = res.mfe_structure == target
    return EnsembleMetrics(
        mfe_structure=res.mfe_structure,
        mfe_energy=res.mfe_energy,
        ensemble_free_energy=res.ensemble_free_energy,
        prob_of_target=res.prob_of_target,
        ensemble_defect=res.ensemble_defect,
        mfe_frequency=res.mfe_frequency,
        bp_distance_to_target=d,
        mfe_matches_target=(res.mfe_structure == target),
        is_umfe=umfe,
    )


def gc_content(sequence: str) -> float:
    s = sequence.upper()
    return sum(c in "GC" for c in s) / max(len(s), 1)


def sequence_diversity(sequences: List[str]) -> float:
    """Mean pairwise normalized Hamming distance over a set of equal-length sequences.

    A cheap diversity signal for a batch of designs (higher = more diverse)."""
    seqs = [s for s in sequences if s]
    if len(seqs) < 2:
        return 0.0
    n = len(seqs[0])
    if any(len(s) != n for s in seqs):
        return float("nan")
    total, pairs = 0, 0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            total += sum(a != b for a, b in zip(seqs[i], seqs[j]))
            pairs += 1
    return (total / pairs) / max(n, 1) if pairs else 0.0
