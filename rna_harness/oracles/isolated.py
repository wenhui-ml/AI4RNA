"""Isolated (spawn-subprocess) folding for the multi-oracle panel.

WHY THIS EXISTS: ViennaRNA's `params_load_RNA_*()` mutates a process-global C parameter set
that is STICKY -- the first set loaded in a process wins, and later loads are silently
ignored. So folding several distinct param sets in one process silently collapses them all to
the first-loaded model (verified: turner2004/turner1999/andronescu2007/langdon2018 all
returned identical energies in-process). That made the "multi-oracle" panel fake.

Fix: fold each param set in a FRESH spawned process (maxtasksperchild=1) so its load is the
first-and-only load -> genuinely different energy models. fork() would inherit the parent's
already-loaded (sticky) state, so we must use the 'spawn' context.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import List, Tuple


def _fold_worker(spec: Tuple[str, str, float, int]) -> Tuple[str, float]:
    """Runs in a fresh spawned process: load the requested param set FIRST, then fold."""
    import RNA

    seq, param_set, temperature, dangles = spec
    from .vienna import _PARAM_LOADERS

    _PARAM_LOADERS[param_set]()  # first (and only) load in this fresh process -> honored
    md = RNA.md()
    md.temperature = float(temperature)
    md.dangles = int(dangles)
    fc = RNA.fold_compound(seq.upper().replace("T", "U"), md)
    struct, energy = fc.mfe()
    return struct, float(energy)


def isolated_fold_mfe(sequence: str, param_set: str, temperature: float = 37.0, dangles: int = 2) -> Tuple[str, float]:
    """Fold one (seq, param_set) in a fresh process. Correct energy model, guaranteed."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        return pool.apply(_fold_worker, ((sequence, param_set, temperature, dangles),))


def isolated_fold_many(specs: List[Tuple[str, str, float, int]], max_procs: int = 8) -> List[Tuple[str, float]]:
    """Fold a list of (seq, param_set, temperature, dangles) each in a fresh process.

    maxtasksperchild=1 guarantees every fold runs in a brand-new process, so each honors its
    own param-set load (no sticky-state contamination across the panel)."""
    if not specs:
        return []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(len(specs), max_procs), maxtasksperchild=1) as pool:
        return pool.map(_fold_worker, specs)
