"""Gaming-proof "was it actually learned?" ledger (S2 + S4).

Borrowed from OpenScience's usage-based skill lifecycle (lifecycle.ts: usage counters, archive-unused,
promote-high-performers) and sharpened with our MIT discovery-criterion gate. A distilled skill /
memory item counts as LEARNED only if it is RE-SELECTED across >= min_reuse DISTINCT (held-out)
episodes AND its presence correlates with a solve-lift — NOT because a self-score said so. Deterministic
and cheap: no LLM judge (S4 — when the outcome signal is exact/cheap, an LLM critic adds nothing;
OpenScience itself left its LLM critic as dead code and gated on a heuristic).

This turns "self-improvement" into a falsifiable metric: accumulate items, then ask whether ANY of
them are reused-and-helpful on held-out puzzles. In our green zone the expected answer is 'none'
(the null); the point is to measure it in a way a skeptic cannot dismiss as self-congratulation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Set


class TransferLedger:
    def __init__(self):
        self.loads: Dict[str, Set[str]] = defaultdict(set)   # item -> set of episode_ids that used it
        self.episodes: Dict[str, dict] = {}                  # episode_id -> {"items": set, "solved": bool}

    def record_episode(self, episode_id: str, items_used: List[str], solved: bool) -> None:
        items = set(items_used)
        self.episodes[episode_id] = {"items": items, "solved": bool(solved)}
        for it in items:
            self.loads[it].add(episode_id)

    def reuse(self, item: str) -> int:
        return len(self.loads.get(item, ()))

    def lift(self, item: str) -> float:
        """solve-rate(episodes that used item) - solve-rate(episodes that did not). 0.0 if either side empty."""
        with_ = [e["solved"] for e in self.episodes.values() if item in e["items"]]
        without = [e["solved"] for e in self.episodes.values() if item not in e["items"]]
        if not with_ or not without:
            return 0.0
        return (sum(with_) / len(with_)) - (sum(without) / len(without))

    def is_learned(self, item: str, min_reuse: int = 3, min_lift: float = 0.0) -> bool:
        return self.reuse(item) >= min_reuse and self.lift(item) > min_lift

    def report(self, min_reuse: int = 3, min_lift: float = 0.0) -> dict:
        items = {it: {"reuse": self.reuse(it), "lift": round(self.lift(it), 3),
                      "learned": self.is_learned(it, min_reuse, min_lift)}
                 for it in self.loads}
        n_learned = sum(1 for v in items.values() if v["learned"])
        return {"n_items": len(items), "n_learned": n_learned,
                "n_episodes": len(self.episodes), "items": items}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"loads": {k: sorted(v) for k, v in self.loads.items()},
                       "episodes": {k: {"items": sorted(v["items"]), "solved": v["solved"]}
                                    for k, v in self.episodes.items()}}, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "TransferLedger":
        led = cls()
        with open(path) as f:
            d = json.load(f)
        led.loads = defaultdict(set, {k: set(v) for k, v in d.get("loads", {}).items()})
        led.episodes = {k: {"items": set(v["items"]), "solved": v["solved"]}
                        for k, v in d.get("episodes", {}).items()}
        return led
