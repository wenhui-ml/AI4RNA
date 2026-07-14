"""Routing policy: features -> ordered tools to try (cheapest expected cost-to-solve first).

Cold-start = mechanical cascade ordered by each tool's a-priori nominal cost (cheap first). With
outcome data it LEARNS per-tool solvability via k-NN over standardized features and re-orders by
expected cost-to-solve (cheap AND high-success first, so predicted-hard problems skip the doomed cheap
attempts). This learned routing IS the transferable procedure (A_theta) -- a policy, NOT memorized
answers -- and it persists across runs. Domain-agnostic: it only ever sees the feature vector +
(solved, cost) outcomes; it never touches a nucleotide.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List

from .features import FEATURE_NAMES


class Router:
    def __init__(self, tools):
        self.tools = list(tools)                     # list[Tool]
        self.records: List[dict] = []                # {"x":[...], "tool":name, "solved":bool, "cost":float}

    # -- outcome memory (the learned routing procedure) ----------------------
    def record(self, feat: Dict[str, float], tool: str, solved: bool, cost: float) -> None:
        self.records.append({"x": [feat[k] for k in FEATURE_NAMES], "tool": tool,
                             "solved": bool(solved), "cost": float(cost)})

    def _scales(self) -> List[float]:
        if not self.records:
            return [1.0] * len(FEATURE_NAMES)
        cols = list(zip(*[r["x"] for r in self.records]))
        out = []
        for c in cols:
            m = sum(c) / len(c)
            var = sum((v - m) ** 2 for v in c) / len(c)
            out.append(math.sqrt(var) or 1.0)
        return out

    def _knn(self, x: List[float], tool: str, k: int = 5):
        """(p_solve, mean_cost) among the k nearest same-tool records, or None if too few data."""
        recs = [r for r in self.records if r["tool"] == tool]
        if len(recs) < max(3, k):
            return None
        sc = self._scales()
        recs = sorted(recs, key=lambda r: sum(((a - b) / s) ** 2 for a, b, s in zip(x, r["x"], sc)))
        near = recs[:k]
        p = sum(1 for r in near if r["solved"]) / len(near)
        c = sum(r["cost"] for r in near) / len(near)
        return p, c

    # -- planning ------------------------------------------------------------
    def _cold_order(self) -> List:
        return sorted(self.tools, key=lambda t: t.nominal_cost_usd())

    def plan(self, feat: Dict[str, float]) -> List:
        """Ordered tools to try. Learned (order by cost/P(solve)) when data allows; else cold-start."""
        x = [feat[k] for k in FEATURE_NAMES]
        scored, have_learned = [], False
        for t in self.tools:
            est = self._knn(x, t.name)
            if est is not None:
                have_learned = True
                p, c = est
                eff = c / max(p, 1e-3)               # $ per unit success prob; lower => try earlier
                scored.append((eff, t))
            else:
                scored.append((t.nominal_cost_usd() / 0.5, t))   # neutral 0.5 prior when unlearned
        if not have_learned:
            return self._cold_order()
        scored.sort(key=lambda z: z[0])
        return [t for _e, t in scored]

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        json.dump({"features": FEATURE_NAMES, "records": self.records}, open(path, "w"))

    def load(self, path: str) -> None:
        self.records = json.load(open(path)).get("records", [])
