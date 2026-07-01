"""Task + reward + oracle configuration.

A DesignTask is an *open problem* handed to the agent: "design an RNA sequence that folds
to this target under this oracle, satisfying these constraints." It carries no control
flow -- how the agent solves it is entirely up to openclaude.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional

from .constraints import ConstraintSpec


@dataclass
class OracleConfig:
    """Fixes the reward oracle. uMFE / designability are all defined relative to these."""

    param_set: str = "turner2004"
    temperature: float = 37.0
    dangles: int = 2

    def to_dict(self) -> Dict[str, object]:
        return {"param_set": self.param_set, "temperature": self.temperature, "dangles": self.dangles}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, object]]) -> "OracleConfig":
        if not d:
            return cls()
        return cls(
            param_set=str(d.get("param_set", "turner2004")),
            temperature=float(d.get("temperature", 37.0)),
            dangles=int(d.get("dangles", 2)),
        )


@dataclass
class RewardConfig:
    """Weights for the shaped, deterministic reward.

    The design is intentionally verifier-first: a *solved* design gets the dominant bonus,
    and everything else is dense shaping so a weak policy still gets gradient toward the
    solution. All terms are exact ViennaRNA quantities -- this operationalizes R_det=high.
    """

    w_solved: float = 1.0            # structure solved AND all constraints satisfied
    w_mfe_match: float = 0.25        # MFE string-match without full constraints (partial credit)
    w_umfe: float = 0.1              # extra bonus if the target is the UNIQUE MFE
    w_shape: float = 0.5             # (1 - bp_distance/n) structural closeness
    w_ensemble_defect: float = 0.25  # -ED (ensemble quality)
    w_constraints: float = 0.5       # fraction of hybrid constraints satisfied
    w_cross_oracle: float = 0.25     # cross-oracle robustness fraction
    w_negative: float = 0.25         # -p(best competitor) (negative design)
    # "mfe_match" reproduces the Eterna100 score (MFE string equality); "umfe" is stricter.
    solved_criterion: str = "mfe_match"
    require_constraints_for_solved: bool = True  # constraints are a hard gate on "solved"

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass
class DesignTask:
    task_id: str
    target: str
    constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
    family: str = "generic"
    version: str = "V2"
    oracle: OracleConfig = field(default_factory=OracleConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.target)

    def content_hash(self) -> str:
        h = hashlib.sha1()
        h.update(self.target.encode())
        h.update(repr(self.constraints.to_dict()).encode())
        h.update(repr(self.oracle.to_dict()).encode())
        return h.hexdigest()[:12]

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "target": self.target,
            "n": self.n,
            "constraints": self.constraints.to_dict(),
            "family": self.family,
            "version": self.version,
            "oracle": self.oracle.to_dict(),
            "reward": self.reward.to_dict(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "DesignTask":
        return cls(
            task_id=str(d.get("task_id", "task")),
            target=str(d["target"]),
            constraints=ConstraintSpec.from_dict(d.get("constraints")),
            family=str(d.get("family", "generic")),
            version=str(d.get("version", "V2")),
            oracle=OracleConfig.from_dict(d.get("oracle")),
            reward=RewardConfig(**d["reward"]) if isinstance(d.get("reward"), dict) else RewardConfig(),
            meta=dict(d.get("meta", {}) or {}),
        )
