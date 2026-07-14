"""The canonical verifier == the harness-controlled, deterministic reward function.

This is the anti-reward-hacking boundary: the agent may fold sequences freely to explore,
but the *official* verdict and reward for a submitted sequence are computed HERE, by code
the agent does not control, against the exact ViennaRNA oracle. Given the same
(task, sequence) it always returns the same result -- that is R_det (reward determinism).

`solved` uses the strict uMFE criterion under the primary oracle AND (optionally) full
constraint satisfaction. The reward is a dense, shaped, deterministic signal so even a weak
policy gets gradient toward a solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .actuation.diagnosis import Diagnosis, diagnose
from .constraints import ConstraintReport, check_constraints
from .metrics import EnsembleMetrics, bp_distance, ensemble_metrics, gc_content
from .oracles.registry import build_panel, cross_oracle_consistency
from .oracles.vienna import ViennaRNAOracle
from .structure_utils import has_pseudoknot
from .task import DesignTask

_VALID = set("ACGU")


def _primary_from_task(task: DesignTask) -> ViennaRNAOracle:
    c = task.oracle
    return ViennaRNAOracle(
        name=f"primary_{c.param_set}_{int(c.temperature)}_d{c.dangles}",
        param_set=c.param_set,
        temperature=c.temperature,
        dangles=c.dangles,
    )


@dataclass
class VerificationResult:
    task_id: str
    sequence: str
    valid: bool
    solved: bool
    mfe_match: bool
    is_umfe: bool
    reward: float
    reward_breakdown: Dict[str, float]
    ensemble: Optional[Dict[str, object]]
    constraints: Dict[str, object]
    cross_oracle: Optional[Dict[str, object]]
    negative_design: Optional[Dict[str, object]]
    diagnosis: Optional[Dict[str, object]]
    gc_content: Optional[float]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "sequence": self.sequence,
            "valid": self.valid,
            "solved": self.solved,
            "mfe_match": self.mfe_match,
            "is_umfe": self.is_umfe,
            "reward": round(self.reward, 6),
            "reward_breakdown": {k: round(v, 6) for k, v in self.reward_breakdown.items()},
            "ensemble": self.ensemble,
            "constraints": self.constraints,
            "cross_oracle": self.cross_oracle,
            "negative_design": self.negative_design,
            "diagnosis": self.diagnosis,
            "gc_content": round(self.gc_content, 4) if self.gc_content is not None else None,
            "error": self.error,
        }


def _clean(seq: str) -> str:
    return seq.strip().upper().replace("T", "U")


def verify(
    task: DesignTask,
    sequence: str,
    *,
    compute_cross_oracle: bool = True,
    panel_kind: str = "default",
    primary: Optional[ViennaRNAOracle] = None,
) -> VerificationResult:
    """Score a candidate sequence against a task. Deterministic and self-contained."""
    seq = _clean(sequence)
    rc = task.reward

    # --- validity gate -------------------------------------------------
    if len(seq) != task.n:
        return VerificationResult(
            task_id=task.task_id, sequence=seq, valid=False, solved=False, mfe_match=False,
            is_umfe=False, reward=0.0, reward_breakdown={}, ensemble=None,
            constraints={}, cross_oracle=None, negative_design=None, diagnosis=None,
            gc_content=None, error=f"length {len(seq)} != target length {task.n}",
        )
    bad = sorted(set(seq) - _VALID)
    if bad:
        return VerificationResult(
            task_id=task.task_id, sequence=seq, valid=False, solved=False, mfe_match=False,
            is_umfe=False, reward=0.0, reward_breakdown={}, ensemble=None,
            constraints={}, cross_oracle=None, negative_design=None, diagnosis=None,
            gc_content=None, error=f"illegal base(s) {bad}; sequence must be over A/C/G/U",
        )

    # --- pseudoknot guard ----------------------------------------------
    # The ViennaRNA MFE oracle is pseudoknot-FREE: RNA.fold never emits '[]{}' and bp_distance /
    # string-match silently drop them, so an MFE verdict on a pseudoknot target is wrong EITHER way
    # (string-match under-reports; bp_distance over-reports -- a measured 29% false-positive rate on
    # Pseudobase++). Refuse to emit an MFE 'solved' here; a pseudoknot-aware verifier must be used
    # (rna_harness.benchmarks.rnainvbench.verify_pk, backed by pKiss).
    if has_pseudoknot(task.target):
        return VerificationResult(
            task_id=task.task_id, sequence=seq, valid=True, solved=False, mfe_match=False,
            is_umfe=False, reward=0.0, reward_breakdown={}, ensemble=None,
            constraints={}, cross_oracle=None, negative_design=None, diagnosis=None,
            gc_content=gc_content(seq),
            error="pseudoknot target ('[]{}<>' in target): not verifiable under the ViennaRNA MFE "
                  "oracle (pseudoknot-free). Use rna_harness.benchmarks.rnainvbench.verify_pk (pKiss).",
        )

    prim = primary or _primary_from_task(task)

    # --- structure + ensemble under the primary oracle -----------------
    em: EnsembleMetrics = ensemble_metrics(prim, seq, task.target)
    mfe_match = em.mfe_matches_target
    is_umfe = em.is_umfe

    # --- hybrid constraints --------------------------------------------
    creport: ConstraintReport = check_constraints(seq, task.constraints)
    constraints_ok = creport.all_satisfied

    struct_ok = is_umfe if rc.solved_criterion == "umfe" else mfe_match
    solved = struct_ok and (constraints_ok or not rc.require_constraints_for_solved)

    # --- cross-oracle robustness ---------------------------------------
    cross = None
    cross_frac = 0.0
    if compute_cross_oracle:
        panel = build_panel(panel_kind)
        cross = cross_oracle_consistency(seq, task.target, panel)
        cross_frac = float(cross["fraction"])

    # --- negative design ------------------------------------------------
    neg = None
    best_comp_prob = 0.0
    if task.constraints.competitors:
        comp_probs = {}
        for comp in task.constraints.competitors:
            try:
                p = prim.analyze(seq, comp).prob_of_target
            except Exception:
                p = None
            comp_probs[comp] = p
            if p is not None:
                best_comp_prob = max(best_comp_prob, p)
        neg = {"competitor_prob": comp_probs, "best_competitor_prob": best_comp_prob}

    # --- reward (dense, shaped, all deterministic) ---------------------
    n = max(task.n, 1)
    ed = em.ensemble_defect if em.ensemble_defect is not None else 1.0
    breakdown: Dict[str, float] = {}
    if solved:
        breakdown["solved"] = rc.w_solved
    elif mfe_match:
        breakdown["mfe_match"] = rc.w_mfe_match
    if is_umfe:
        breakdown["umfe_bonus"] = rc.w_umfe
    breakdown["shape"] = rc.w_shape * (1.0 - em.bp_distance_to_target / n)
    breakdown["ensemble_defect"] = rc.w_ensemble_defect * (1.0 - ed)
    if creport.checks:
        breakdown["constraints"] = rc.w_constraints * creport.satisfied_fraction
    if compute_cross_oracle:
        breakdown["cross_oracle"] = rc.w_cross_oracle * cross_frac
    if task.constraints.competitors:
        breakdown["negative_design"] = rc.w_negative * (1.0 - best_comp_prob)
    reward = float(sum(breakdown.values()))

    # --- diagnosis (T_trace) -------------------------------------------
    diag: Optional[Diagnosis] = None
    if not (mfe_match and constraints_ok):
        diag = diagnose(task.target, em.mfe_structure)

    return VerificationResult(
        task_id=task.task_id,
        sequence=seq,
        valid=True,
        solved=solved,
        mfe_match=mfe_match,
        is_umfe=is_umfe,
        reward=reward,
        reward_breakdown=breakdown,
        ensemble=em.as_dict(),
        constraints=creport.to_dict(),
        cross_oracle=cross,
        negative_design=neg,
        diagnosis=diag.to_dict() if diag else None,
        gc_content=gc_content(seq),
    )
