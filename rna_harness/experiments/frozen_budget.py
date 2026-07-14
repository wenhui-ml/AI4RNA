"""Frozen-search-budget attribution: does memory (retrieval warm-start) reduce bp-distance to the
target at a FIXED refinement budget -- isolating LEARNED EXPERIENCE from STRONGER SEARCH?

Three seed modes, ONE identical downstream repair budget:
  priors      : canonical no-search assembly (motif_kb prior fill)       [no memory, no search]
  cold_search : per-unit inverse_fold assembly (search builds the seed)  [no memory, uses search]
  memory      : retrieval warm-start (library lookup builds the seed)    [memory,    no search]

Every arm then gets the SAME `parallel_finish(restarts, iters)` + the SAME `close_pair` cap, so any
gap in the outcome is attributable to the SEED source, not to extra compute. The comparison
memory-vs-cold_search asks: can retrieved experience reach the same refined quality WITHOUT the
per-unit search cold_search spent? priors is the floor (no memory, no search) that isolates the
value of retrieval OVER generic biophysical priors.

Primary metric: best bp-distance to target (continuous -- shows structural progress before a full
solve). Secondary: solved (binary). RED LINE: designs under the folding oracle only; the library is
built from our own verified solves + priors, never from sample solutions (see retrieval.py).
"""
from __future__ import annotations

from typing import Optional

import RNA

from ..structure_utils import pair_table
from ..mcp.motif_kb import decompose_structure
from ..mcp.finisher import _fold, parallel_finish, parallel_inverse_fold
from .retrieval import RetrievalLibrary, _GC
from .self_revise import op_close_pair


def _cold_search_seed(target: str, pur: int, tries: int) -> str:
    """No-memory baseline seed: solve each leaf unit's sub_target by inverse_fold FROM SCRATCH, place
    it, and assign junction/seam pairs G-C (mirrors `decompose_solve`'s assembly, minus its finish so
    the downstream refine is applied uniformly). This is where cold_search spends its search budget."""
    n = len(target)
    pt = pair_table(target)
    dec = decompose_structure(target)
    seq = ["A"] * n
    covered: set[int] = set()
    n_search_ops = 0
    for u in dec.get("units", []):
        a, b = u["positions"]
        r = parallel_inverse_fold(u["sub_target"], n_restarts=pur, tries=tries)
        n_search_ops += 1
        us = r.get("seq") or ""
        if len(us) == (b - a + 1):
            for k, ch in enumerate(us):
                seq[a + k] = ch
            covered.update(range(a, b + 1))
    tog = 0
    for i in range(n):
        j = pt[i]
        if j > i and (i not in covered or j not in covered):
            a1, a2 = _GC if tog % 2 == 0 else _GC[::-1]
            seq[i], seq[j] = a1, a2
            tog += 1
    return "".join(seq), n_search_ops


def _refine(seed: str, target: str, restarts: int, iters: int, close_cap: int, use_close: bool = True):
    """The FROZEN, identical-across-arms local repair: bounded parallel_finish. When `use_close` is
    set, a bounded close_pair endgame runs if we are near a solve (that STRONG endgame masks the
    seed's effect, so the budget-sweep experiment turns it OFF and measures pure-hillclimb bp@budget,
    the clean seed-driven convergence signal)."""
    r = parallel_finish(seed, target, n_restarts=restarts, iters=iters)
    best_seq = r.get("seq") or seed
    best_bp = r.get("bp_distance")
    if best_bp is None:
        best_bp = RNA.bp_distance(_fold(seed), target)
    if use_close and not r.get("solved") and best_bp <= 6:
        cp = op_close_pair(best_seq, target, budget={"iters": close_cap})
        if cp.get("bp_distance", 99) < best_bp:
            best_seq, best_bp = cp["seq"], cp["bp_distance"]
    solved = best_bp == 0 and _fold(best_seq) == target
    return best_seq, best_bp, bool(solved)


def build_seed(target: str, seed_mode: str, lib: Optional[RetrievalLibrary] = None,
               pur: int = 6, tries: int = 30):
    """Build ONE mode's warm-start seed WITHOUT refining (so a budget sweep can reuse it). Returns
    (seed, telemetry). memory/priors spend no folding search; cold_search spends per-unit inverse_fold."""
    if seed_mode in ("memory", "memory_clamp"):
        if lib is None:
            raise ValueError(f"seed_mode={seed_mode!r} requires a library")
        seed, cov, meta = lib.warm_start(target, clamp_hits=(seed_mode == "memory_clamp"))
        ops = 0
    elif seed_mode == "priors":
        seed, cov, meta = RetrievalLibrary().warm_start(target)   # empty lib -> canonical priors only
        ops = 0
    elif seed_mode == "cold_search":
        seed, ops = _cold_search_seed(target, pur, tries)
        cov, meta = None, {"assembled_bp": RNA.bp_distance(_fold(seed), target)}
    else:
        raise ValueError(f"unknown seed_mode {seed_mode!r}")
    tel = {"coverage": cov, "seed_search_ops": ops,
           "seed_bp": RNA.bp_distance(_fold(seed), target), "assembled_bp": meta.get("assembled_bp")}
    return seed, tel


def solve_frozen(target: str, seed_mode: str, lib: Optional[RetrievalLibrary] = None,
                 restarts: int = 4, iters: int = 20000, pur: int = 6, tries: int = 30,
                 close_cap: int = 4000) -> dict:
    """Solve `target` under one seed mode and the frozen refine budget. Returns the primary bp metric,
    the pre-refine seed_bp (so we see how much of the gap the seed itself explains), and coverage."""
    seed, tel = build_seed(target, seed_mode, lib, pur, tries)
    best_seq, best_bp, solved = _refine(seed, target, restarts, iters, close_cap)
    return {"mode": seed_mode, "target_len": len(target), **tel,
            "bp_distance": best_bp, "solved": solved, "seq": best_seq if solved else None}
