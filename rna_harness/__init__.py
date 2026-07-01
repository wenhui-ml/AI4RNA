"""rna_harness — the deterministic science/verifier/RL-data substrate for openclaude_rna.

This package is deliberately the "dumb but exact" ground-truth layer: it wraps ViennaRNA
folding oracles, computes checkable metrics/constraints, produces a canonical
harness-controlled reward (so an agent cannot grade its own homework), and emits
RL-ready (data, reward, trace) records. All *agentic* intelligence lives in the openclaude
TypeScript agent, which calls into this layer via a subprocess bridge (see cli.py).

Design compass (two InceptLabs papers):
  * Oracle Residue Thesis -> choose a cheap/exact/self-checkable oracle (ViennaRNA DP),
    the high-alpha_f "learnable region"; never let an error-bearing surrogate be the truth.
  * Actuation Fidelity   -> a self-improving loop only learns if the correct diagnosis
    controls the next action; the harness must instrument this (see actuation/).

Red line: everything here verifies *design success under a computational folding oracle*.
No claim of in-vivo function is ever made or supported.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .structure_utils import (  # noqa: F401
    base_pairs,
    pair_table,
    positional_mismatches,
)

__all__ = ["__version__"]
