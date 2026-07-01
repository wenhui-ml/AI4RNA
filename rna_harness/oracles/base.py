"""Oracle abstraction.

The *oracle* is the load-bearing component of this whole project. Per the Oracle
Residue Thesis, we deliberately choose a domain whose reward oracle is **cheap,
exact, and self-checkable** (a dynamic-programming folding computation) rather than
an expensive, approximate, non-self-checkable physical oracle (DFT / wet-lab). Every
oracle here computes a *definite* answer at negligible cost, so the amortizable
fraction alpha_f is high and there is no error-bearing surrogate standing in for
ground truth inside the loop.

Red line (from the report): the verification object is *design success under a
computational folding oracle*. We never claim in-vivo function.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class FoldResult:
    """Result of folding one sequence under one oracle configuration."""

    sequence: str
    mfe_structure: str
    mfe_energy: float
    # Ensemble quantities (filled by analyze(); None if only MFE was requested).
    ensemble_free_energy: Optional[float] = None
    prob_of_target: Optional[float] = None          # p(target | ensemble)
    ensemble_defect: Optional[float] = None          # expected wrong positions / n
    mfe_frequency: Optional[float] = None            # p(mfe | ensemble)
    oracle_name: str = "unknown"

    @property
    def n(self) -> int:
        return len(self.sequence)


class Oracle(abc.ABC):
    """A folding engine at a fixed model configuration.

    Implementations must be *picklable* (they hold only lightweight config) so the
    harness can fan out across processes; the heavy engine state is created
    per-call.
    """

    name: str = "oracle"

    @abc.abstractmethod
    def fold_mfe(self, sequence: str) -> Tuple[str, float]:
        """Cheap path: MFE structure and its free energy. Called in inner loops."""

    @abc.abstractmethod
    def analyze(self, sequence: str, target: Optional[str] = None) -> FoldResult:
        """Full path: MFE + partition function + (if target given) p(target),
        ensemble defect. More expensive; used for final scoring / ED-guided design."""

    def fold_many_mfe(self, sequences: Sequence[str]) -> List[Tuple[str, float]]:
        return [self.fold_mfe(s) for s in sequences]

    def solves(self, sequence: str, target: str) -> bool:
        """uMFE criterion: the MFE structure equals the target exactly."""
        struct, _ = self.fold_mfe(sequence)
        return struct == target
