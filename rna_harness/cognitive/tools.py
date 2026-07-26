"""Uniform tool interface. Every solver is a Tool; `tool.attempt(target, seed)` returns a ToolResult
carrying a COST. Verification uses the CORRECT oracle (pKiss for pseudoknots, ViennaRNA MFE otherwise)
so 'solved' is never a false positive. ALL domain knowledge lives here (in the wrapped tools); the
router/core above is domain-agnostic and only ever sees features + (solved, cost)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

from ..structure_utils import has_pseudoknot
from ..task import DesignTask, OracleConfig
from .cost import solver_cost_usd

_TURNER2004 = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
# 'pfe' strictness: the target's Boltzmann probability must clear this to count as solved (robust fold,
# not a knife-edge single-structure match). Declarable via env; only used when strictness == "pfe".
_PFE_MIN = float(os.environ.get("COGCORE_PFE_MIN", "0.90"))


@dataclass
class ToolResult:
    tool: str
    solved: bool
    bp: int
    wall_s: float
    cores: int
    cost_usd: float             # solver CPU cost + oracle/verify cost (the FULL price of the attempt)
    oracle: str                 # 'mfe' | 'mfe:umfe' | 'mfe:pfe' | 'pkiss' | 'none'
    seq: Optional[str]          # only set when solved (verified)
    available: bool = True
    oracle_s: float = 0.0       # wall spent in verification (negligible for MFE; can dominate for pKiss)
    stats: Optional[dict] = None  # optional per-attempt budget accounting (oracle calls, iterations, ...)


def verify(seq: Optional[str], target: str, strictness: str = "match"):
    """Authoritative verification: the DECLARED decidable predicate evaluated on `seq`. The truth value
    of "does seq solve target?" is obtained by EVALUATION (fold once, compare) -- SAT-analogous: fold is
    the total computable function, `==` the decidable check. Returns (solved, bp, oracle_name).

    Three axes of the verifier live here:
      * WHICH oracle (axis 1): pseudoknot target -> pKiss; nested -> ViennaRNA MFE. Picking wrong (MFE on
        a pseudoknot) is a category error -- MFE literally cannot emit '[', so it is INCAPABLE of the truth.
      * HOW STRICT (axis 2), MFE targets only -- `strictness`:
          'match' fold(seq)==target                        (exact match; default, knife-edge)
          'umfe'  match AND target is the UNIQUE MFE        (closes the degenerate-co-optimal loophole)
          'pfe'   P(target) in the Boltzmann ensemble >= _PFE_MIN  (robust fold, not a fragile 1-of-many)
      * NO oracle (axis 3): unjudgeable input (e.g. length mismatch, or -- outside this fn -- a functional
        spec with no computable predicate) returns oracle 'none' -> the core must ABSTAIN, never claim solved.
    Pseudoknot targets are always judged by pKiss; `strictness` applies only to the ViennaRNA MFE path."""
    if not seq or len(seq) != len(target):
        return False, len(target), "none"
    if has_pseudoknot(target):
        from ..benchmarks.rnainvbench import verify_pk    # pKiss; pseudoknot-aware
        r = verify_pk(seq, target)
        return bool(r.get("solved")), int(r.get("bp", len(target))), "pkiss"
    import RNA
    st, _ = RNA.fold(seq)
    bp = RNA.bp_distance(st, target)
    match = (st == target)
    if strictness == "match":
        return match, bp, "mfe"
    if strictness == "umfe":
        fc = RNA.fold_compound(seq)
        _, mfe_e = fc.mfe()
        # uMFE = target is the sole ground state. We count ONLY structures at the MFE energy, so a
        # narrow window (0.01 kcal/mol = MFE +/- rounding) is both correct and BOUNDED. A wide band
        # (e.g. subopt(50) = 0.5 kcal/mol) enumerates an exponential set on large/degenerate seqs --
        # all discarded by the <1e-6 filter -- and hangs the scaled multi-seed run. subopt(1) yields
        # a bit-identical n_coopt (same filtered set) without the blow-up.
        n_coopt = sum(1 for s in fc.subopt(1) if abs(s.energy - mfe_e) < 1e-6)  # co-optimal @ MFE
        return (match and n_coopt == 1), bp, "mfe:umfe"
    if strictness == "pfe":
        fc = RNA.fold_compound(seq)
        _, mfe_e = fc.mfe()
        fc.exp_params_rescale(mfe_e)
        fc.pf()
        return (fc.pr_structure(target) >= _PFE_MIN), bp, "mfe:pfe"
    raise ValueError(f"unknown strictness {strictness!r}; use 'match' | 'umfe' | 'pfe'")


class Tool:
    name = "tool"
    cores = 1
    budget_s = 0.0

    def nominal_cost_usd(self) -> float:
        """A-priori worst-case cost of firing this tool = cores x its budget seconds. The router
        orders actions cheapest-first by this before it has outcome data."""
        return solver_cost_usd(self.budget_s, self.cores)

    def attempt(self, target: str, seed: int = 0) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


class RemcTool(Tool):
    """Our replica-exchange MC (anneal.py), bounded by a wall-second budget."""

    def __init__(self, wall_s: float, n_replicas: int = 8):
        self.wall_s_budget = float(wall_s)
        self.budget_s = float(wall_s)
        self.cores = int(n_replicas)
        self.name = f"remc@{int(wall_s)}s"

    def attempt(self, target: str, seed: int = 0) -> ToolResult:
        from ..experiments.anneal import replica_exchange
        # REMC optimises the ViennaRNA MFE structure, which CANNOT represent pseudoknots. On a pk target
        # it designs the nested SKELETON (its best achievable) -- anneal/warm-start assume nested and would
        # crash on '[' otherwise -- and pKiss then judges the FULL target, so REMC fails GRACEFULLY on a
        # pseudoknot (the crossing stem is out of its model) instead of raising.
        solve_target = "".join(c if c in "()." else "." for c in target) if has_pseudoknot(target) else target
        t0 = time.time()
        r = replica_exchange(solve_target, n_replicas=self.cores, wall_s=self.wall_s_budget, seed=seed)
        wall = time.time() - t0
        seq = r.get("best_seq")
        v0 = time.time()
        solved, bp, oracle = verify(seq, target)      # re-verify (anneal optimises MFE; pk => pKiss)
        ov = time.time() - v0
        cost = solver_cost_usd(wall, self.cores) + solver_cost_usd(ov, cores=1)   # solver + oracle
        return ToolResult(self.name, solved, bp, round(wall, 3), self.cores,
                          cost, oracle, seq if solved else None, oracle_s=round(ov, 3))


class DesirnaTool(Tool):
    """External SOTA DesiRNA via the oracle-verified subprocess adapter, bounded by its -t timelimit."""

    def __init__(self, timelimit_s: float, replicas: int = 10):
        self.tl = int(timelimit_s)
        self.budget_s = float(timelimit_s)
        self.cores = int(replicas)
        self.name = f"desirna@{int(timelimit_s)}s"

    def attempt(self, target: str, seed: int = 0) -> ToolResult:
        from ..generators.desirna import DesiRNAGenerator, desirna_available
        if not desirna_available():
            return ToolResult(self.name, False, len(target), 0.0, self.cores, 0.0, "none", None,
                              available=False)
        t0 = time.time()
        res = DesiRNAGenerator(replicas=self.cores, timelimit_s=self.tl, results_number=20).design(
            DesignTask(task_id="cog", target=target, oracle=_TURNER2004), seed=seed)
        wall = time.time() - t0
        best_seq = res.best_sequence or None
        ov = 0.0
        v0 = time.time()
        solved, bp, oracle = verify(best_seq, target)
        ov += time.time() - v0
        # pseudoknot target: DesiRNA is MFE (out-of-model), but re-check every candidate under pKiss
        if has_pseudoknot(target) and not solved:
            for c in (getattr(res, "candidates", None) or []):
                v0 = time.time()
                s, b, o = verify(c, target)
                ov += time.time() - v0
                if b < bp:
                    solved, bp, oracle, best_seq = s, b, o, c
                if solved:
                    break
        cost = solver_cost_usd(wall, self.cores) + solver_cost_usd(ov, cores=1)   # solver + oracle (pKiss dominates)
        return ToolResult(self.name, solved, bp, round(wall, 3), self.cores,
                          cost, oracle, best_seq if solved else None, oracle_s=round(ov, 3))


def default_toolbox() -> List[Tool]:
    """The action set the router chooses from: cheap->expensive, our-solver + SOTA. Each (tool,budget)
    is one action; the router learns which to fire (and in what order) per problem, to minimise cost."""
    return [
        RemcTool(wall_s=5),
        RemcTool(wall_s=30),
        RemcTool(wall_s=120),
        DesirnaTool(timelimit_s=60),
        DesirnaTool(timelimit_s=180),
    ]
