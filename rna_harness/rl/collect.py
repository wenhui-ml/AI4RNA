"""Episode collection + RFT/GRPO export.

Two collection modes:
  * collect_group  -- one-shot GRPO groups: K samples per task/prompt, group-relative
    advantages. The classical generator acts as the (zero-credit) policy; later the
    openclaude agent / a local model produces the K samples instead.
  * repair_episode -- a multi-step repair trajectory instrumented for actuation fidelity.
    feedback_mode selects how the diagnosis is used, which powers the killer ablations:
      - 'diagnosis'       : edit the diagnosed positions (loop closed)   -> high A
      - 'random'          : model ignores the directive, edits at random  -> A_model -> 0
      - 'silent_fallback' : harness drops the directive before the actor  -> A_harness -> 0
    If solve-rate does not fall under 'random'/'silent_fallback', the loop is not using the
    signal (the A->0 / 2509.21403 null signature).
"""
from __future__ import annotations

import json
import os
import random
from statistics import mean, pstdev
from typing import Dict, List, Optional

from ..actuation.diagnosis import diagnose
from ..actuation.instrument import ActuationLogger
from ..generators.base import (
    frozen_motif_positions,
    gc_target_prob,
    init_sequence,
    structured_mutate,
)
from ..generators.base import Generator
from ..metrics import bp_distance
from ..oracles.vienna import ViennaRNAOracle
from ..task import DesignTask
from ..verifier import verify
from .prompts import render_repair_prompt, render_task_prompt
from .schema import RLSample


def _sample_from_verify(task, seq, prompt, generator, group_id, step, cross) -> RLSample:
    v = verify(task, seq, compute_cross_oracle=cross)
    return RLSample(
        task_id=task.task_id,
        task_family=task.family,
        target=task.target,
        prompt=prompt,
        response=seq,
        reward=v.reward,
        solved=v.solved,
        generator=generator,
        reward_breakdown=v.reward_breakdown,
        mfe_match=v.mfe_match,
        is_umfe=v.is_umfe,
        ensemble_defect=(v.ensemble or {}).get("ensemble_defect") if v.ensemble else None,
        cross_oracle_consistency=(v.cross_oracle or {}).get("fraction") if v.cross_oracle else None,
        constraints_satisfied=v.constraints.get("all_satisfied") if v.constraints else None,
        step=step,
        group_id=group_id,
        meta={"reward_error": v.error} if v.error else {},
    )


def add_group_advantages(samples: List[RLSample]) -> List[RLSample]:
    """Fill GRPO group-relative advantages: a_i = (r_i - mean_g) / (std_g or 1).

    Only samples with a real group_id are grouped; samples with group_id=None are left with
    advantage=None (they are NOT a group and must not be pooled with unrelated singletons)."""
    groups: Dict[str, List[RLSample]] = {}
    for s in samples:
        if s.group_id is None:
            continue
        groups.setdefault(s.group_id, []).append(s)
    for g, members in groups.items():
        rs = [m.reward for m in members]
        mu = mean(rs) if rs else 0.0
        sd = pstdev(rs) if len(rs) > 1 else 0.0
        denom = sd if sd > 1e-8 else 1.0
        for m in members:
            m.advantage = (m.reward - mu) / denom
    return samples


def collect_group(
    task: DesignTask,
    generator: Generator,
    k: int = 8,
    budget: int = 200,
    seed: int = 0,
    compute_cross_oracle: bool = False,
) -> List[RLSample]:
    """One GRPO group of K one-shot samples for a task (K independent generator runs)."""
    prompt = render_task_prompt(task)
    gid = f"{task.content_hash()}:{seed}"
    samples: List[RLSample] = []
    for i in range(k):
        res = generator.design(task, budget=budget, seed=seed * 1000 + i)
        s = _sample_from_verify(
            task, res.best_sequence, prompt, generator.name, gid, step=0, cross=compute_cross_oracle
        )
        s.meta["gen_evals"] = res.n_evals
        samples.append(s)
    add_group_advantages(samples)
    return samples


def repair_episode(
    task: DesignTask,
    max_steps: int = 40,
    seed: int = 0,
    feedback_mode: str = "diagnosis",
    compute_cross_oracle: bool = False,
):
    """Run an instrumented multi-step repair episode. Returns (samples, actuation, solved)."""
    if feedback_mode not in ("diagnosis", "random", "silent_fallback"):
        raise ValueError(feedback_mode)
    rng = random.Random(seed)
    orc = ViennaRNAOracle(
        param_set=task.oracle.param_set, temperature=task.oracle.temperature, dangles=task.oracle.dangles
    )
    spec = task.constraints
    p_gc = gc_target_prob(spec)
    frozen = frozen_motif_positions(spec)
    n = task.n

    logger = ActuationLogger()
    samples: List[RLSample] = []

    seq = init_sequence(task.target, spec, rng, p_gc)
    struct, _ = orc.fold_mfe(seq)
    cur_d = bp_distance(struct, task.target)
    solved = struct == task.target

    for step in range(max_steps):
        if solved:
            break
        diag = diagnose(task.target, struct)
        directive_positions = diag.diagnosed_position_set()  # the TRUE diagnosis (always the ground truth)

        # choose which positions the actor actually edits, per feedback mode
        if feedback_mode == "diagnosis":
            pool = [i for i in directive_positions if i not in frozen] or [i for i in range(n) if i not in frozen]
            read = True
        else:
            # 'random' (model ignores) and 'silent_fallback' (harness drops) both edit at random
            pool = [i for i in range(n) if i not in frozen]
            read = feedback_mode != "silent_fallback"  # in silent_fallback the directive never reached the actor
        if not pool:
            break
        i = rng.choice(pool)

        prompt = render_repair_prompt(task, seq, diag.directive)
        seq2 = structured_mutate(seq, i, task.target, spec, rng, p_gc, frozen)
        struct2, _ = orc.fold_mfe(seq2)
        d2 = bp_distance(struct2, task.target)

        logger.log(directive_positions, seq, seq2, cur_d, d2, read=read, executed=True)

        s = _sample_from_verify(task, seq2, prompt, f"repair:{feedback_mode}", None, step, compute_cross_oracle)
        s.actuation = logger.steps[-1].to_dict()
        s.meta["d_before"] = cur_d
        s.meta["d_after"] = d2
        samples.append(s)

        # accept greedily (improving or sideways) so the trajectory is a genuine repair
        if d2 <= cur_d:
            seq, struct, cur_d = seq2, struct2, d2
        if struct == task.target:
            solved = True

    return samples, logger.actuation_fidelity(), solved


# --------------------------------------------------------------------------- export
def export_rft(samples: List[RLSample], path: str, solved_only: bool = True, min_reward: Optional[float] = None) -> int:
    """Write an SFT/RFT (rejection-sampling) dataset: kept (prompt, completion) pairs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    seen = set()
    n = 0
    with open(path, "w") as f:
        for s in samples:
            if solved_only and not s.solved:
                continue
            if min_reward is not None and s.reward < min_reward:
                continue
            key = (s.prompt, s.response)
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps({
                "prompt": s.prompt,
                "completion": s.response,
                "messages": [
                    {"role": "user", "content": s.prompt},
                    {"role": "assistant", "content": s.response},
                ],
                "reward": round(s.reward, 6),
                "solved": s.solved,
                "task_id": s.task_id,
                "task_family": s.task_family,
            }) + "\n")
            n += 1
    return n


def export_grpo(samples: List[RLSample], path: str) -> int:
    """Write a GRPO dataset: per-sample rows with group_id + group-relative advantage."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps({
                "prompt": s.prompt,
                "response": s.response,
                "reward": round(s.reward, 6),
                "advantage": round(s.advantage, 6) if s.advantage is not None else None,
                "group_id": s.group_id,
                "solved": s.solved,
                "task_id": s.task_id,
                "task_family": s.task_family,
            }) + "\n")
            n += 1
    return n


def export_full(samples: List[RLSample], path: str) -> int:
    """Write the full trajectory records (all fields) for analysis / four-coordinate reports."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")
            n += 1
    return n
