"""Benchmark task loaders."""
from __future__ import annotations

from .eterna100 import load_eterna100, eterna100_tasks  # noqa: F401
from .hybrid import synthesize_hybrid_tasks  # noqa: F401
from .openset import random_designable_tasks  # noqa: F401

__all__ = ["load_eterna100", "eterna100_tasks", "synthesize_hybrid_tasks", "random_designable_tasks"]
