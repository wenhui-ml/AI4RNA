"""Uniform tool interface. Every solver is a Tool; `tool.attempt(target, seed)` returns a ToolResult
carrying a COST. Verification uses the CORRECT oracle (pKiss for pseudoknots, ViennaRNA MFE otherwise)
so 'solved' is never a false positive. ALL domain knowledge lives here (in the wrapped tools); the
router/core above is domain-agnostic and only ever sees features + (solved, cost)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from ..structure_utils import has_pseudoknot
from ..task import DesignTask, OracleConfig
from .cost import solver_cost_usd

_TURNER2004 = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)


@dataclass
class ToolResult:
    tool: str
    solved: bool
    bp: int
    wall_s: float
    cores: int
    cost_usd: float
    oracle: str                 # 'mfe' | 'pkiss' | 'none'
    seq: Optional[str]          # only set when solved (verified)
    available: bool = True


def verify(seq: Optional[str], target: str):
    """Authoritative verification with the CORRECT oracle. Returns (solved, bp, oracle_name)."""
    if not seq or len(seq) != len(target):
        return False, len(target), "none"
    if has_pseudoknot(target):
        from ..benchmarks.rnainvbench import verify_pk    # pKiss; pseudoknot-aware
        r = verify_pk(seq, target)
        return bool(r.get("solved")), int(r.get("bp", len(target))), "pkiss"
    import RNA
    st, _ = RNA.fold(seq)
    return (st == target), RNA.bp_distance(st, target), "mfe"


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
        t0 = time.time()
        r = replica_exchange(target, n_replicas=self.cores, wall_s=self.wall_s_budget, seed=seed)
        wall = time.time() - t0
        seq = r.get("best_seq")
        solved, bp, oracle = verify(seq, target)      # re-verify (anneal optimises MFE; pk => pKiss)
        return ToolResult(self.name, solved, bp, round(wall, 3), self.cores,
                          solver_cost_usd(wall, self.cores), oracle, seq if solved else None)


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
        solved, bp, oracle = verify(best_seq, target)
        # pseudoknot target: DesiRNA is MFE (out-of-model), but re-check every candidate under pKiss
        if has_pseudoknot(target) and not solved:
            for c in (getattr(res, "candidates", None) or []):
                s, b, o = verify(c, target)
                if b < bp:
                    solved, bp, oracle, best_seq = s, b, o, c
                if solved:
                    break
        return ToolResult(self.name, solved, bp, round(wall, 3), self.cores,
                          solver_cost_usd(wall, self.cores), oracle, best_seq if solved else None)


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
