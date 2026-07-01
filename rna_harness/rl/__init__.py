"""RL data layer: prompts, trajectory schema, episode collection, RFT/GRPO export.

All rewards come from the canonical verifier (deterministic ground truth). Data is
model-agnostic (prompt/response/reward) so trajectories from any driving model -- Opus-4.8,
a local Gemma, or the classical generator -- feed the same RFT then GRPO training.
"""
from __future__ import annotations

from .prompts import render_task_prompt, render_repair_prompt  # noqa: F401
from .schema import RLSample  # noqa: F401
from .reward_fn import rna_reward, rna_solved_reward  # noqa: F401
from .collect import (  # noqa: F401
    collect_group,
    repair_episode,
    add_group_advantages,
    export_rft,
    export_grpo,
)

__all__ = [
    "render_task_prompt",
    "render_repair_prompt",
    "RLSample",
    "collect_group",
    "repair_episode",
    "add_group_advantages",
    "export_rft",
    "export_grpo",
]
