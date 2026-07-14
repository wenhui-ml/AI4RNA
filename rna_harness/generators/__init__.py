"""Sequence generators.

These are the MODEL-INDEPENDENT performance floor and a zero-credit RLVR data source.
The LLM/agent path is NOT here -- it lives in the openclaude agent. Keeping a strong
classical floor is a hard design discipline: the LLM must beat it to justify itself, and
if any driving model is weak the harness never drops below this floor.
"""
from __future__ import annotations

from .base import GenerationResult, Generator  # noqa: F401
from .rnainverse import RNAinverseGenerator  # noqa: F401
from .local_search import LocalSearchGenerator  # noqa: F401
from .desirna import DesiRNAGenerator, desirna_available  # noqa: F401

__all__ = ["GenerationResult", "Generator", "RNAinverseGenerator", "LocalSearchGenerator",
           "DesiRNAGenerator", "desirna_available"]
