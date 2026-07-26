"""The cognitive core loop and the fixed-policy baselines it must beat.

CognitiveCore.solve = analyse (features) -> route (ordered tools) -> invoke -> verify -> escalate on
failure, accumulating COST and recording every outcome so the router learns. It returns
(solved, total cost_usd, trace). The falsifiable claim is checked by comparing its cost-to-solve on a
puzzle mix against FixedPolicy baselines (always-cheap, always-SOTA). A cheap-first cascade already
beats always-SOTA on an easy-heavy mix; the LEARNED router further trims the cost by skipping doomed
cheap attempts on predicted-hard problems -- that trim, measured in $, is the value of learning.
"""
from __future__ import annotations

from typing import List, Optional

from .features import features as extract_features
from .router import Router
from .tools import Tool


class CognitiveCore:
    def __init__(self, tools: List[Tool], router: Optional[Router] = None, learn: bool = True):
        self.tools = list(tools)
        self.router = router or Router(self.tools)
        self.learn = learn

    def solve(self, target: str, seed: int = 0, max_actions: Optional[int] = None) -> dict:
        feat = extract_features(target)
        plan = self.router.plan(feat)
        if max_actions:
            plan = plan[:max_actions]
        total_cost, trace = 0.0, []
        for tool in plan:
            r = tool.attempt(target, seed=seed)
            total_cost += r.cost_usd
            trace.append({"tool": tool.name, "solved": r.solved, "bp": r.bp,
                          "cost_usd": round(r.cost_usd, 8), "wall_s": r.wall_s, "oracle": r.oracle,
                          "oracle_s": r.oracle_s})
            if self.learn:
                self.router.record(feat, tool.name, r.solved, r.cost_usd)
            if r.solved:
                return {"solved": True, "cost_usd": total_cost, "seq": r.seq,
                        "n_actions": len(trace), "trace": trace}
        return {"solved": False, "cost_usd": total_cost, "seq": None,
                "n_actions": len(trace), "trace": trace}


class FixedPolicy:
    """Baseline: fire a FIXED tool sequence regardless of the problem (e.g. [DesiRNA@180] = always-SOTA,
    or [remc@5] = always-cheap). The cognitive core must beat these on cost-at-quality to earn its name."""

    def __init__(self, tools: List[Tool], name: str = "fixed"):
        self.tools = list(tools)
        self.name = name

    def solve(self, target: str, seed: int = 0) -> dict:
        total_cost, trace = 0.0, []
        for tool in self.tools:
            r = tool.attempt(target, seed=seed)
            total_cost += r.cost_usd
            trace.append({"tool": tool.name, "solved": r.solved, "bp": r.bp,
                          "cost_usd": round(r.cost_usd, 8), "oracle": r.oracle, "oracle_s": r.oracle_s,
                          "stats": r.stats})
            if r.solved:
                return {"solved": True, "cost_usd": total_cost, "seq": r.seq,
                        "n_actions": len(trace), "trace": trace, "stats": r.stats}
        return {"solved": False, "cost_usd": total_cost, "seq": None,
                "n_actions": len(trace), "trace": trace, "stats": r.stats}
