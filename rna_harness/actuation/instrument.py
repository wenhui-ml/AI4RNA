"""Actuation-fidelity instrumentation == the A coordinate.

Paper 1 (Actuation Fidelity): a self-improving loop only learns if the correct diagnosis
actually controls the next action. A = P(a correctly-diagnosed correction is adopted by the
next execution). It decomposes A = A_harness * A_model:

  * A_model   -- did the actor's chosen edit target the diagnosed positions? (selected)
  * A_harness -- did the emitted directive reach the executor and get executed, with no
                 silent fallback dropping it? (read & executed)

We log the full chain emitted -> read -> selected -> executed -> affected per repair step,
so the "Diagnosis-Actuation Gap" (A -> 0 while a playbook grows) is directly measurable.
This is what the catalysis harness never instrumented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set


def edited_positions(before: str, after: str) -> List[int]:
    """Positions changed between two equal-length sequences."""
    m = min(len(before), len(after))
    pos = [i for i in range(m) if before[i] != after[i]]
    if len(after) != len(before):
        pos += list(range(m, max(len(before), len(after))))
    return pos


@dataclass
class ActuationStep:
    step: int
    directive_positions: List[int]   # emitted (the diagnosed mismatch set)
    read: bool                       # directive was available to the actor
    edited_positions: List[int]      # what the actor actually changed
    selected: bool                   # edit overlapped the directive
    adoption: float                  # |edited ∩ directive| / |edited| (graded on-target rate)
    executed: bool                   # the edit was applied + scored
    affected: bool                   # the outcome improved (bp_distance decreased)
    d_before: int
    d_after: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "n_directive": len(self.directive_positions),
            "read": self.read,
            "n_edited": len(self.edited_positions),
            "selected": self.selected,
            "adoption": round(self.adoption, 4),
            "executed": self.executed,
            "affected": self.affected,
            "d_before": self.d_before,
            "d_after": self.d_after,
        }


class ActuationLogger:
    """Accumulates actuation steps for one or more repair episodes and aggregates A."""

    def __init__(self) -> None:
        self.steps: List[ActuationStep] = []

    def log(
        self,
        directive_positions: Sequence[int],
        before: str,
        after: str,
        d_before: int,
        d_after: int,
        read: bool = True,
        executed: bool = True,
    ) -> ActuationStep:
        edited = edited_positions(before, after)
        dset: Set[int] = set(directive_positions)
        on_target = sum(1 for i in edited if i in dset)
        adoption = (on_target / len(edited)) if edited else 0.0
        selected = on_target > 0
        step = ActuationStep(
            step=len(self.steps),
            directive_positions=list(directive_positions),
            read=read,
            edited_positions=edited,
            selected=selected,
            adoption=adoption,
            executed=executed,
            affected=(d_after < d_before),
            d_before=d_before,
            d_after=d_after,
        )
        self.steps.append(step)
        return step

    def actuation_fidelity(self) -> Dict[str, object]:
        """Aggregate A over all logged steps. A = mean(selected & executed & read)."""
        if not self.steps:
            return {"A": None, "A_model": None, "A_harness": None, "adoption": None, "n_steps": 0}
        n = len(self.steps)
        a_model = sum(1 for s in self.steps if s.selected) / n            # chose the right positions
        a_harness = sum(1 for s in self.steps if s.read and s.executed) / n  # directive survived to execution
        a = sum(1 for s in self.steps if s.selected and s.executed and s.read) / n
        adoption = sum(s.adoption for s in self.steps) / n
        # among steps where the correct directive WAS adopted, how often did it help?
        adopted = [s for s in self.steps if s.selected and s.executed]
        efficacy = (sum(1 for s in adopted if s.affected) / len(adopted)) if adopted else None
        return {
            "A": round(a, 4),
            "A_model": round(a_model, 4),
            "A_harness": round(a_harness, 4),
            "adoption": round(adoption, 4),
            "efficacy_when_adopted": round(efficacy, 4) if efficacy is not None else None,
            "n_steps": n,
        }
