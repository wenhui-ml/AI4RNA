"""Synthesize hybrid-constraint design tasks from base (structure-only) tasks.

The report's headline seam is "arbitrary hybrid-constraint amortized generation": target
structure + fixed motif + coding + GC + forbidden subsequences, in any combination. We
generate a small, reproducible battery of constraint profiles on top of any base target.

Constraints are placed to be *structurally compatible where possible* (e.g. motifs in
unpaired runs, whose base identity is otherwise free) so the tasks are hard-but-fair rather
than trivially impossible. Joint satisfiability with the folding target is itself part of
the challenge and is decided by the exact verifier.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

from ..constraints import ConstraintSpec
from ..structure_utils import loops_and_stems
from ..task import DesignTask


def _first_unpaired_run(target: str, k: int) -> Optional[int]:
    """Start index of the first run of >= k unpaired ('.') positions, or None."""
    run = 0
    for i, c in enumerate(target):
        if c == ".":
            run += 1
            if run >= k:
                return i - k + 1
        else:
            run = 0
    return None


DEFAULT_PROFILES = ("gc_mid", "gc_high", "motif", "forbidden", "coding", "combo")


def synthesize_hybrid_tasks(
    base: DesignTask, profiles: Optional[List[str]] = None
) -> List[DesignTask]:
    """Return constrained variants of ``base``. Skips profiles that don't fit the target."""
    profiles = list(profiles or DEFAULT_PROFILES)
    out: List[DesignTask] = []
    tgt = base.target

    for prof in profiles:
        spec: Optional[ConstraintSpec] = None
        if prof == "gc_mid":
            spec = ConstraintSpec(gc_min=0.4, gc_max=0.6)
        elif prof == "gc_high":
            spec = ConstraintSpec(gc_min=0.6, gc_max=0.8)
        elif prof == "forbidden":
            spec = ConstraintSpec(forbidden=["AAAA", "CCCC", "GGGG", "UUUU"])
        elif prof == "motif":
            pos = _first_unpaired_run(tgt, 4)
            if pos is None:
                continue  # no unpaired run long enough
            spec = ConstraintSpec(motifs=[(pos, "GAAA")])  # a common stable tetraloop-like motif
        elif prof == "coding":
            # Require a short peptide in a mostly-unpaired region; genuinely couples
            # base identity with structure where the frame overlaps stems.
            spec = ConstraintSpec(coding_aa="MA", coding_frame=0)
        elif prof == "combo":
            pos = _first_unpaired_run(tgt, 4)
            motifs = [(pos, "GAAA")] if pos is not None else []
            spec = ConstraintSpec(gc_min=0.45, gc_max=0.65, forbidden=["AAAAA", "GGGGG"], motifs=motifs)
        else:
            raise ValueError(f"unknown hybrid profile {prof!r}")

        if spec is None:
            continue
        t = copy.deepcopy(base)
        t.task_id = f"{base.task_id}|hybrid:{prof}"
        t.constraints = spec
        t.family = f"hybrid:{prof}"
        t.meta = dict(base.meta or {})
        t.meta["hybrid_profile"] = prof
        t.meta["base_task_id"] = base.task_id
        out.append(t)
    return out
