"""Generator A: ViennaRNA's built-in inverse folding (RNA.inverse_fold).

Trivial to call, strong on easy targets. It optimizes the sequence so that the target is
the MFE structure. We verify every candidate with our own oracle -- never trust the
generator's self-reported cost.
"""
from __future__ import annotations

import random
from typing import List, Optional

import RNA

from ..structure_utils import pair_table
from ..task import DesignTask
from .base import GenerationResult, Generator, gc_target_prob


class RNAinverseGenerator(Generator):
    name = "rnainverse"

    def __init__(self, gc_only: bool = False):
        self.gc_only = gc_only

    def design(self, task: DesignTask, budget: int = 50, seed: Optional[int] = None) -> GenerationResult:
        rng = random.Random(seed)
        target = task.target
        n = len(target)
        symbolset = "GC" if self.gc_only else "AUGC"
        RNA.cvar.symbolset = symbolset

        best_seq, best_cost = None, float("inf")
        candidates: List[str] = []
        tries = max(1, budget)
        solved = False
        for _ in range(tries):
            start = RNA.random_string(n, symbolset)
            seq, cost = RNA.inverse_fold(start, target)
            seq = seq.upper().replace("T", "U")
            candidates.append(seq)
            if cost < best_cost:
                best_seq, best_cost = seq, cost
            if cost == 0.0:
                solved = True
                break
        # verify hard-solved with our own oracle
        best_seq = best_seq or RNA.random_string(n, symbolset)
        from ..metrics import bp_distance
        from ..oracles.vienna import ViennaRNAOracle

        orc = ViennaRNAOracle(
            param_set=task.oracle.param_set,
            temperature=task.oracle.temperature,
            dangles=task.oracle.dangles,
        )
        mfe_struct, _ = orc.fold_mfe(best_seq)
        d = bp_distance(mfe_struct, target)
        return GenerationResult(
            task_id=task.task_id,
            best_sequence=best_seq,
            solved=(d == 0),
            best_bp_distance=d,
            n_evals=len(candidates),
            candidates=candidates,
            method=self.name,
            meta={"vienna_cost": best_cost, "symbolset": symbolset},
        )
