"""Canonical prompts.

These render the task/state as an instruction. The SAME renderer feeds both the RL dataset
and the LLM/agent path, so training data matches what the policy sees at inference. No
model-specific tricks (model-portability discipline).
"""
from __future__ import annotations

from ..constraints import ConstraintSpec
from ..task import DesignTask


def _describe_constraints(spec: ConstraintSpec) -> str:
    if spec.is_empty():
        return "None (pure inverse folding)."
    parts = []
    if spec.gc_min is not None or spec.gc_max is not None:
        lo = spec.gc_min if spec.gc_min is not None else 0.0
        hi = spec.gc_max if spec.gc_max is not None else 1.0
        parts.append(f"GC content must be within [{lo:.2f}, {hi:.2f}]")
    if spec.forbidden:
        parts.append("must NOT contain any of: " + ", ".join(spec.forbidden))
    if spec.motifs:
        parts.append("; ".join(f"positions {p}..{p+len(m)-1} must read '{m}'" for p, m in spec.motifs))
    if spec.iupac_mask is not None:
        parts.append("each position must match the IUPAC mask provided")
    if spec.coding_aa is not None:
        parts.append(f"must translate (frame {spec.coding_frame}) to peptide '{spec.coding_aa}'")
    if spec.competitors:
        parts.append(f"must AVOID folding into {len(spec.competitors)} competitor structure(s) (negative design)")
    return "\n  - " + "\n  - ".join(parts)


def render_task_prompt(task: DesignTask) -> str:
    """One-shot design instruction for a task."""
    oc = task.oracle
    return (
        f"Design an RNA sequence of length {task.n} over the alphabet {{A,C,G,U}} whose "
        f"minimum-free-energy secondary structure under ViennaRNA "
        f"(params={oc.param_set}, T={oc.temperature}C, dangles={oc.dangles}) is EXACTLY the "
        f"target below (dot-bracket).\n\n"
        f"Target ({task.n} nt):\n{task.target}\n\n"
        f"Constraints: {_describe_constraints(task.constraints)}\n\n"
        f"Return ONLY the {task.n}-nucleotide RNA sequence, no explanation."
    )


def render_repair_prompt(task: DesignTask, sequence: str, diagnosis_directive: str) -> str:
    """Repair instruction given a current (failing) sequence and a folding-mismatch directive."""
    return (
        f"{render_task_prompt(task)}\n\n"
        f"Your current attempt:\n{sequence}\n\n"
        f"Folding diagnosis (exact mismatch localization):\n{diagnosis_directive}\n\n"
        f"Return an improved {task.n}-nucleotide sequence that fixes the diagnosed positions "
        f"while preserving the parts that already fold correctly."
    )
