"""The RLVR trajectory record. Model-agnostic (prompt/response/reward)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RLSample:
    task_id: str
    task_family: str
    target: str
    prompt: str                       # the instruction the policy saw
    response: str                     # the action (a sequence, or a repaired sequence)
    reward: float                     # canonical deterministic reward
    solved: bool
    generator: str                    # who produced it (local_search / rnainverse / opus-4.8 / ...)
    # optional structured signal
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    mfe_match: bool = False
    is_umfe: bool = False
    ensemble_defect: Optional[float] = None
    cross_oracle_consistency: Optional[float] = None
    constraints_satisfied: Optional[bool] = None
    # sequential / group structure
    step: int = 0                     # step index within a repair episode
    group_id: Optional[str] = None    # GRPO group (same prompt, K samples)
    advantage: Optional[float] = None # group-relative advantage (filled at group close)
    # actuation (repair steps)
    actuation: Optional[Dict[str, object]] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "task_family": self.task_family,
            "target": self.target,
            "prompt": self.prompt,
            "response": self.response,
            "reward": round(self.reward, 6),
            "solved": self.solved,
            "generator": self.generator,
            "reward_breakdown": {k: round(v, 6) for k, v in self.reward_breakdown.items()},
            "mfe_match": self.mfe_match,
            "is_umfe": self.is_umfe,
            "ensemble_defect": self.ensemble_defect,
            "cross_oracle_consistency": self.cross_oracle_consistency,
            "constraints_satisfied": self.constraints_satisfied,
            "step": self.step,
            "group_id": self.group_id,
            "advantage": self.advantage,
            "actuation": self.actuation,
            "meta": self.meta,
        }
