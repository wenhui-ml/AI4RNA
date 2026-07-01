"""Online verifiable-reward function for RL training (TRL/verl-compatible).

This exposes the canonical ViennaRNA verifier as a reward callable so an ONLINE GRPO/PPO
trainer can score policy completions directly against exact ground truth -- no reward model,
no surrogate. This is the operational meaning of R_det=high: the training reward IS the
oracle.

TRL usage:
    from rna_harness.rl.reward_fn import rna_reward
    trainer = GRPOTrainer(model=..., reward_funcs=[rna_reward], train_dataset=ds, ...)
The dataset should carry a `target` column (dot-bracket); optionally `constraints` (a dict)
and `oracle` (a dict). The reward fn reads them from kwargs, TRL-style.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from ..constraints import ConstraintSpec
from ..task import DesignTask, OracleConfig, RewardConfig
from ..verifier import verify

_STRUCT_LINE = re.compile(r"^[().]{4,}$", re.MULTILINE)
_SEQ_RE = re.compile(r"[ACGUacgu]{4,}")


def parse_sequence(text: str, n: int) -> Optional[str]:
    """Extract an n-nt RNA sequence from text (robust to prose). Local copy to avoid a
    circular import with rna_harness.llm.generator (which imports rl.prompts)."""
    if not text:
        return None
    up = text.upper().replace("T", "U")
    runs = _SEQ_RE.findall(up)
    if not runs:
        return None
    exact = [r for r in runs if len(r) == n]
    if exact:
        return exact[-1]
    longest = max(runs, key=len)
    return longest if len(longest) == n else (longest[:n] if len(longest) > n else None)


def _extract_target_from_prompt(prompt: str) -> Optional[str]:
    """Best-effort: the longest dot-bracket-only line in the prompt is the target."""
    cands = _STRUCT_LINE.findall(prompt or "")
    return max(cands, key=len) if cands else None


def _completion_text(comp) -> str:
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[-1], dict):
        return comp[-1].get("content", "")
    return str(comp)


def rna_reward(
    prompts: Sequence,
    completions: Sequence,
    target: Optional[Sequence[str]] = None,
    constraints: Optional[Sequence[dict]] = None,
    oracle: Optional[Sequence[dict]] = None,
    reward: Optional[Sequence[dict]] = None,
    **kwargs,
) -> List[float]:
    """Return a deterministic reward per (prompt, completion). Invalid/absent sequence -> 0.

    `target` (preferred) or a dot-bracket line parsed from the prompt defines the task.
    `reward` (optional) is a per-item RewardConfig dict; defaults to RewardConfig().
    """
    n = len(completions)
    rewards: List[float] = []
    for i in range(n):
        tgt = None
        if target is not None and i < len(target):
            tgt = target[i]
        if not tgt:
            tgt = _extract_target_from_prompt(prompts[i] if i < len(prompts) else "")
        if not tgt:
            rewards.append(0.0)
            continue
        cspec = ConstraintSpec.from_dict(constraints[i]) if (constraints and i < len(constraints)) else ConstraintSpec()
        oconf = OracleConfig.from_dict(oracle[i]) if (oracle and i < len(oracle)) else OracleConfig()
        rconf = RewardConfig(**reward[i]) if (reward and i < len(reward) and isinstance(reward[i], dict)) else RewardConfig()
        task = DesignTask(task_id=f"rl_{i}", target=tgt, constraints=cspec, oracle=oconf, reward=rconf)
        seq = parse_sequence(_completion_text(completions[i]), len(tgt))
        if not seq:
            rewards.append(0.0)
            continue
        v = verify(task, seq, compute_cross_oracle=False)
        rewards.append(float(v.reward))
    return rewards


def rna_solved_reward(prompts, completions, target=None, constraints=None, oracle=None, **kwargs) -> List[float]:
    """Sparse variant: 1.0 iff solved (uMFE + constraints), else 0.0. Useful as a second
    reward_func alongside the dense one, or for pass@k evaluation."""
    out = []
    n = len(completions)
    for i in range(n):
        tgt = (target[i] if target and i < len(target)
               else _extract_target_from_prompt(prompts[i] if i < len(prompts) else "")) or ""
        seq = parse_sequence(_completion_text(completions[i]), len(tgt)) if tgt else None
        if not (tgt and seq):
            out.append(0.0)
            continue
        cspec = ConstraintSpec.from_dict(constraints[i]) if (constraints and i < len(constraints)) else ConstraintSpec()
        oconf = OracleConfig.from_dict(oracle[i]) if (oracle and i < len(oracle)) else OracleConfig()
        task = DesignTask(task_id=f"rl_{i}", target=tgt, constraints=cspec, oracle=oconf)
        out.append(1.0 if verify(task, seq, compute_cross_oracle=False).solved else 0.0)
    return out
