"""Pre-registration + noise-floor verdict for memory / self-improvement experiments.

Borrowed from OpenScience (research.txt: pre-specify success criteria BEFORE analysis; "improvement
must exceed the noise floor"). The point for us: a NULL claim is only credible if the bar was COMMITTED
before seeing the result — otherwise a reviewer suspects post-hoc metric fishing. You write a Prereg
(baseline, noise floor, the min effect that would count as "memory helped") and git-commit it BEFORE
running; afterwards `verdict()` mechanically classifies the observed result as supported / null /
refuted against that committed bar. Pairs with git-per-experiment (commit prereg, then result).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class Prereg:
    name: str
    hypothesis: str
    metric: str                 # e.g. "solve_rate" or "mean_bp"
    direction: str              # "higher" or "lower": which way the metric must move to SUPPORT the hypothesis
    baseline: float             # no-treatment (e.g. no-memory) baseline value of the metric
    noise_floor: float          # min |delta| considered distinguishable from baseline (below => indistinguishable)
    min_effect: float           # pre-committed delta magnitude (in `direction`) required to DECLARE support
    n: int                      # committed sample size
    seeds: int = 1
    created: Optional[str] = None

    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.direction not in ("higher", "lower"):
            raise ValueError("direction must be 'higher' or 'lower'")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Prereg":
        with open(path) as f:
            return cls(**json.load(f))


def verdict(pre: Prereg, observed: float) -> dict:
    """Classify an observed metric against the PRE-REGISTERED bar. `supported` only if it beats the
    baseline by >= min_effect in the committed direction AND the delta exceeds the noise floor;
    within the noise floor => `null` (indistinguishable — the honest outcome our project keeps hitting)."""
    delta = observed - pre.baseline
    signed = delta if pre.direction == "higher" else -delta   # >0 means "moved the hypothesized way"
    if abs(delta) < pre.noise_floor:
        return {"verdict": "null", "delta": round(delta, 4),
                "reason": f"|delta|={abs(delta):.4g} < noise_floor={pre.noise_floor} "
                          f"(indistinguishable from baseline={pre.baseline})"}
    if signed >= pre.min_effect:
        return {"verdict": "supported", "delta": round(delta, 4),
                "reason": f"signed_delta={signed:.4g} >= min_effect={pre.min_effect} ({pre.direction})"}
    if signed > 0:
        return {"verdict": "insufficient", "delta": round(delta, 4),
                "reason": f"right direction but signed_delta={signed:.4g} < min_effect={pre.min_effect} "
                          f"(real effect above noise, but below the pre-committed bar)"}
    return {"verdict": "refuted", "delta": round(delta, 4),
            "reason": f"moved the WRONG way beyond noise (signed_delta={signed:.4g} <= 0)"}


def git_commit(paths: List[str], message: str, repo: str = ".") -> bool:
    """Best-effort git-per-experiment: stage + commit the prereg/result files. Returns False (no raise)
    if git is unavailable or nothing changed, so it never breaks an experiment run."""
    try:
        subprocess.run(["git", "-C", repo, "add", *paths], check=True, capture_output=True)
        subprocess.run(["git", "-C", repo, "commit", "-m", message], check=True, capture_output=True)
        return True
    except Exception:
        return False
