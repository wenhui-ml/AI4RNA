"""Meta-learning layer: cross-case self-improvement by EXTENDING openclaude's memdir.

The harness accumulates checkable RNA design strategies distilled from solved cases into
openclaude's native memory format, so the live agent recalls them on later cases (via
CLAUDE_COWORK_MEMORY_PATH_OVERRIDE -> getAutoMemPath). We then measure A_memory: whether a
recalled strategy applicable to a new case is actually adopted -- the cross-case analog of
the single-case actuation-fidelity instrument. This tests Paper 1's core claim directly: a
growing playbook is worthless unless it actually controls behavior and lifts capability.
"""
from __future__ import annotations

from .memstore import MemStore, Strategy  # noqa: F401
from .strategies import (  # noqa: F401
    distill_strategies,
    merge_strategies,
    applicable_strategies,
    memory_adoption,
)

__all__ = [
    "MemStore",
    "Strategy",
    "distill_strategies",
    "merge_strategies",
    "applicable_strategies",
    "memory_adoption",
]
