"""Four-coordinate determinism instrumentation: D_sym, R_det, T_trace (A lives in
actuation/instrument.py).

This turns the harness into a MEASUREMENT INSTRUMENT for the user's philosophy, not just a
designer. For RNA all three coordinates are high by construction -- the point is to (a)
show they are high (the positive control the catalysis harness lacked) and (b) provide the
knobs to deliberately degrade them in ablations (noisy reward / lossy observation) and show
that learning breaks, reproducing the low-alpha_f failure in a cheap, controlled setting.

  * D_sym  (data determinism)   : is the observation an exact, reproducible symbol?
  * R_det  (reward determinism) : is the reward exact/self-checkable, no learned surrogate?
  * T_trace(trace determinism)  : does a failure localize to specific positions/steps?
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev
from typing import Dict, List, Optional

from .actuation.diagnosis import diagnose
from .oracles.vienna import ViennaRNAOracle
from .task import DesignTask
from .verifier import verify


@dataclass
class Coordinate:
    name: str
    value: float          # in [0, 1]; 1 = fully deterministic/localizable
    detail: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "value": round(self.value, 6), "detail": self.detail}


def estimate_d_sym(task: DesignTask, sequences: List[str], repeats: int = 3) -> Coordinate:
    """Observation determinism: fold each sequence ``repeats`` times; D_sym = fraction of
    sequences whose MFE structure is identical across repeats (== 1.0 for an exact oracle)."""
    orc = ViennaRNAOracle(
        param_set=task.oracle.param_set, temperature=task.oracle.temperature, dangles=task.oracle.dangles
    )
    stable = 0
    for s in sequences:
        structs = {orc.fold_mfe(s)[0] for _ in range(max(1, repeats))}
        stable += int(len(structs) == 1)
    val = stable / len(sequences) if sequences else 1.0
    return Coordinate("D_sym", val, {"n_sequences": len(sequences), "repeats": repeats, "n_stable": stable})


def estimate_r_det(task: DesignTask, sequences: List[str], repeats: int = 3) -> Coordinate:
    """Reward determinism: recompute the official reward ``repeats`` times; R_det = fraction
    of sequences whose reward is bit-stable. Also flags self-checkability (ViennaRNA reward
    needs no external budget -> alpha_f ~ 1, no error-bearing surrogate)."""
    stable = 0
    for s in sequences:
        rewards = {round(verify(task, s, compute_cross_oracle=False).reward, 9) for _ in range(max(1, repeats))}
        stable += int(len(rewards) == 1)
    val = stable / len(sequences) if sequences else 1.0
    return Coordinate(
        "R_det",
        val,
        {
            "n_sequences": len(sequences),
            "repeats": repeats,
            "n_stable": stable,
            "self_checkable": True,
            "surrogate_in_reward_path": False,
        },
    )


def estimate_t_trace(task: DesignTask, sequences: List[str]) -> Coordinate:
    """Trace determinism: over the *failing* sequences, T_trace = fraction whose failure
    localizes to a concrete, non-empty mismatch set (exact per-position credit)."""
    orc = ViennaRNAOracle(
        param_set=task.oracle.param_set, temperature=task.oracle.temperature, dangles=task.oracle.dangles
    )
    failing = 0
    localizable = 0
    for s in sequences:
        struct, _ = orc.fold_mfe(s)
        if struct == task.target:
            continue
        failing += 1
        d = diagnose(task.target, struct)
        if d.mismatched_positions:
            localizable += 1
    val = (localizable / failing) if failing else 1.0
    return Coordinate("T_trace", val, {"n_failing": failing, "n_localizable": localizable})


def four_coordinate_report(
    task: DesignTask, sequences: List[str], actuation: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """Compute D_sym/R_det/T_trace over a set of candidate sequences for a task.
    ``actuation`` (the A coordinate) is computed separately over episodes and passed in."""
    seqs = [s for s in sequences if s]
    coords = {
        "D_sym": estimate_d_sym(task, seqs).to_dict(),
        "R_det": estimate_r_det(task, seqs).to_dict(),
        "T_trace": estimate_t_trace(task, seqs).to_dict(),
    }
    if actuation is not None:
        coords["A"] = actuation
    return {"task_id": task.task_id, "family": task.family, "coordinates": coords}
