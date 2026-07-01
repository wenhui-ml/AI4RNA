"""Generator B: mismatch-directed structured local search (SAMFEO-style, lighter variant).

Fast, MFE-only inner loop. Initializes constraint-guided, then repeatedly:
  * folds the current sequence (MFE),
  * finds positions whose pairing state differs from the target (the mismatch set),
  * applies a *structured* mutation at a mismatched position (mutating a target pair as a
    whole, or an unpaired base), with GC bias and fixed-motif freezing,
  * accepts by simulated-annealing (always if bp-distance improves; sideways/uphill with
    exp(-Δd/T)); random-restarts on stall; K multi-starts.

The per-mismatch targeting is exactly the T_trace signal in action -- the exact folding
mismatch tells us which positions to fix.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

from ..metrics import bp_distance
from ..oracles.vienna import ViennaRNAOracle
from ..structure_utils import pair_table, positional_mismatches
from ..task import DesignTask
from .base import (
    GenerationResult,
    Generator,
    frozen_motif_positions,
    gc_target_prob,
    init_sequence,
    structured_mutate,
)


class LocalSearchGenerator(Generator):
    name = "local_search"

    def __init__(self, restarts: int = 8, T0: float = 1.0, cooling: float = 0.997, stall_limit: int = 60):
        self.restarts = restarts
        self.T0 = T0
        self.cooling = cooling
        self.stall_limit = stall_limit

    def design(
        self,
        task: DesignTask,
        budget: int = 3000,
        seed: Optional[int] = None,
        warm_starts: Optional[List[str]] = None,
    ) -> GenerationResult:
        rng = random.Random(seed)
        target = task.target
        n = len(target)
        spec = task.constraints
        p_gc = gc_target_prob(spec)
        frozen = frozen_motif_positions(spec)
        warm_starts = [w for w in (warm_starts or []) if len(w) == n]
        orc = ViennaRNAOracle(
            param_set=task.oracle.param_set,
            temperature=task.oracle.temperature,
            dangles=task.oracle.dangles,
        )

        best_seq, best_d = None, n + 1
        candidates: List[str] = []
        evals = 0
        solved = False

        per_restart_budget = max(1, budget // max(1, self.restarts))
        for restart_i in range(self.restarts):
            if evals >= budget or solved:
                break
            # Warm-start the first restart(s) from memory-biased inits, if provided.
            seq = warm_starts[restart_i] if restart_i < len(warm_starts) else init_sequence(target, spec, rng, p_gc)
            struct, _ = orc.fold_mfe(seq)
            cur_d = bp_distance(struct, target)
            evals += 1
            candidates.append(seq)
            if cur_d < best_d:
                best_seq, best_d = seq, cur_d
            if cur_d == 0:
                solved = True
                break
            T = self.T0
            stall = 0
            steps = 0
            while evals < budget and steps < per_restart_budget:
                steps += 1
                mism = positional_mismatches(target, struct)
                pool = [i for i in mism if i not in frozen] or [i for i in range(n) if i not in frozen]
                if not pool:
                    break
                i = rng.choice(pool)
                seq2 = structured_mutate(seq, i, target, spec, rng, p_gc, frozen)
                struct2, _ = orc.fold_mfe(seq2)
                d2 = bp_distance(struct2, target)
                evals += 1
                dd = d2 - cur_d
                accept = dd < 0 or (dd == 0 and rng.random() < 0.5) or rng.random() < math.exp(-dd / max(T, 1e-6))
                if accept:
                    seq, struct, cur_d = seq2, struct2, d2
                    candidates.append(seq2)
                    stall = 0
                else:
                    stall += 1
                if cur_d < best_d:
                    best_seq, best_d = seq, cur_d
                if cur_d == 0:
                    solved = True
                    break
                T *= self.cooling
                if stall >= self.stall_limit:
                    break

        best_seq = best_seq or init_sequence(target, spec, rng, p_gc)
        return GenerationResult(
            task_id=task.task_id,
            best_sequence=best_seq,
            solved=(best_d == 0),
            best_bp_distance=best_d,
            n_evals=evals,
            candidates=candidates,
            method=self.name,
            meta={"restarts": self.restarts, "p_gc": p_gc},
        )
