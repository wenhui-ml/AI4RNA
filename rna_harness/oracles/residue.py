"""Oracle residue == a controlled, dial-able degradation of the REWARD-PATH oracle.

This is the R_det knob for the "when does self-improvement compound?" study. The GROUND
TRUTH stays the exact ViennaRNA verifier (turner2004); a `RewardOracle` is what the AGENT's
loop actually SEES when it decides "solved?" (search-termination) and "distill this strategy?"
(the memory/distillation gate). Injecting residue here -- and ONLY here -- lets us dial
R_det = 1 - residue while holding everything else (diagnosis/T_trace, task, actor) fixed.

Design invariants:
  * Residue corrupts the `solved` verdict (and, consistently, the solved bonus in the dense
    reward). It does NOT touch the diagnosis (that is the T_trace axis) or the bp_distance
    used for progress -- those stay exact, so R_det is cleanly separable from the other
    coordinates.
  * NoisyRewardOracle flips deterministically per (task, sequence) => the residue is a BIAS,
    not variance: re-scoring the same sequence yields the same corrupted verdict (so it also
    poisons a reused/distilled strategy consistently, which is the amplification mechanism).
  * SurrogateRewardOracle judges "solved" under a DIFFERENT energy model (structured residue),
    folding in an isolated subprocess to dodge ViennaRNA's sticky global param set.

Every verdict carries BOTH the (possibly corrupted) `solved` the agent sees and the
`exact_solved` ground truth, so `corrupted` is auditable and `estimate_r_det` can measure the
realized residue (= fraction of disagreements) rather than the nominal rate.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from ..task import DesignTask
from ..verifier import verify


@dataclass
class RewardVerdict:
    solved: bool           # what the AGENT sees (possibly corrupted)
    reward: float          # dense reward consistent with `solved`
    bp_distance: Optional[int]  # exact ground-truth structural distance (never corrupted)
    exact_solved: bool     # ground-truth solved (for audit / R_det measurement)
    corrupted: bool        # solved != exact_solved
    source: str

    def to_dict(self):
        return {
            "solved": self.solved, "reward": round(self.reward, 6), "bp_distance": self.bp_distance,
            "exact_solved": self.exact_solved, "corrupted": self.corrupted, "source": self.source,
        }


class RewardOracle:
    """The reward-path oracle the AGENT sees. Base = exact (R_det = 1, no residue)."""

    name = "exact"

    def evaluate(self, task: DesignTask, sequence: str) -> RewardVerdict:  # pragma: no cover
        raise NotImplementedError


class ExactRewardOracle(RewardOracle):
    """R_det = 1: the reward path IS the ground-truth ViennaRNA verifier."""

    name = "exact"

    def evaluate(self, task: DesignTask, sequence: str) -> RewardVerdict:
        v = verify(task, sequence, compute_cross_oracle=False)
        bp = v.ensemble.get("bp_distance_to_target") if v.ensemble else None
        return RewardVerdict(bool(v.solved), float(v.reward), bp, bool(v.solved), False, self.name)


class NoisyRewardOracle(RewardOracle):
    """Deterministic false-positive / false-negative flip of the exact `solved` verdict.

    fp = P(report solved | truly unsolved)  -- a *false accept* (the poisoning direction:
         a non-solving sequence is treated as solved -> its strategy gets distilled).
    fn = P(report unsolved | truly solved)  -- a *false reject*.
    Flips are keyed by sha256(salt, task, sequence) so they are a fixed BIAS: the same
    (task, sequence) always gets the same verdict (a reused/distilled strategy stays poisoned).
    """

    def __init__(self, fp: float = 0.0, fn: float = 0.0, salt: str = "", exact: Optional[RewardOracle] = None):
        self.fp = float(fp)
        self.fn = float(fn)
        self.salt = str(salt)
        self.exact = exact or ExactRewardOracle()
        self.name = f"noisy_fp{self.fp:g}_fn{self.fn:g}"

    def _u(self, task: DesignTask, sequence: str) -> float:
        """Deterministic uniform in [0,1) per (salt, task, sequence)."""
        h = hashlib.sha256(f"{self.salt}|{task.task_id}|{task.target}|{sequence}".encode()).hexdigest()
        return int(h[:16], 16) / float(1 << 64)

    def evaluate(self, task: DesignTask, sequence: str) -> RewardVerdict:
        base = self.exact.evaluate(task, sequence)
        u = self._u(task, sequence)
        solved = base.exact_solved
        if base.exact_solved and u < self.fn:
            solved = False
        elif (not base.exact_solved) and u < self.fp:
            solved = True
        corrupted = solved != base.exact_solved
        reward = base.reward
        w = float(task.reward.w_solved)
        if corrupted and solved:        # false positive -> agent wrongly gets the solved bonus
            reward = base.reward + w
        elif corrupted and not solved:  # false negative -> strip the solved bonus it earned
            reward = max(0.0, base.reward - w)
        return RewardVerdict(solved, reward, base.bp_distance, base.exact_solved, corrupted, self.name)


class SurrogateRewardOracle(RewardOracle):
    """Structured residue: judge `solved` under a DIFFERENT (surrogate) energy model while
    ground truth stays the task's exact oracle. Folds in an isolated subprocess so the
    surrogate param set is honored (ViennaRNA's global param load is sticky in-process).

    NOTE: one subprocess fold per call -> slow in a tight loop; intended for calibration and
    modest sweeps, and it yields *bias* (a fixed different-model verdict), matching real
    surrogate error better than random flips.
    """

    def __init__(self, param_set: str = "turner1999", exact: Optional[RewardOracle] = None):
        self.param_set = param_set
        self.exact = exact or ExactRewardOracle()
        self.name = f"surrogate_{param_set}"

    def evaluate(self, task: DesignTask, sequence: str) -> RewardVerdict:
        from .isolated import isolated_fold_mfe

        base = self.exact.evaluate(task, sequence)
        struct, _ = isolated_fold_mfe(
            sequence, self.param_set, task.oracle.temperature, task.oracle.dangles
        )
        solved = bool(struct) and (struct == task.target)  # solved under the SURROGATE model
        corrupted = solved != base.exact_solved
        reward = base.reward
        w = float(task.reward.w_solved)
        if corrupted and solved:
            reward = base.reward + w
        elif corrupted and not solved:
            reward = max(0.0, base.reward - w)
        return RewardVerdict(solved, reward, base.bp_distance, base.exact_solved, corrupted, self.name)


def realized_residue(reward_oracle: RewardOracle, task: DesignTask, sequences) -> dict:
    """Measure the REALIZED residue of a reward oracle on a sequence sample: the fraction of
    sequences where its verdict disagrees with exact ground truth (this is 1 - R_det). Also
    breaks it into false-positive / false-negative rates (the FP rate is the poisoning one)."""
    n = fp = fn = disagree = 0
    exact = ExactRewardOracle()
    for s in sequences:
        gt = exact.evaluate(task, s).exact_solved
        v = reward_oracle.evaluate(task, s).solved
        n += 1
        if v != gt:
            disagree += 1
            if v and not gt:
                fp += 1
            elif gt and not v:
                fn += 1
    return {
        "reward_oracle": reward_oracle.name,
        "n": n,
        "residue": round(disagree / n, 6) if n else 0.0,      # 1 - R_det
        "r_det": round(1 - disagree / n, 6) if n else 1.0,
        "false_positive_rate": round(fp / n, 6) if n else 0.0,
        "false_negative_rate": round(fn / n, 6) if n else 0.0,
    }
