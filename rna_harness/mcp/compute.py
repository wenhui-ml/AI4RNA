"""CPU-parallel RNA compute: batch folding + multi-restart inverse-folding search.

Why this exists: the agent loop is LLM-latency-bound, and the box has many idle cores. Instead
of the agent folding one candidate per LLM round-trip (or hand-writing a slow single-threaded
search in Bash), it calls ONE tool that fans out across up to RNA_MAX_CORES (default 160) worker
processes -- folding thousands of candidates or running a full multi-restart search in seconds.
That cuts the number of expensive LLM round-trips per episode, so even the serial with_memory arm
speeds up.

Determinism preserved: the exact ViennaRNA Turner2004 model is the ground truth (this uses the
same default md: T=37, dangles=2, built-in Turner2004 -- no param_load, so fork-safe and NOT
subject to the sticky-param bug). Seeds are explicit, so parallel_search is reproducible.

Core budget: pools are capped at RNA_MAX_CORES (default 160) to leave headroom for other users
on the shared 256-core host.
"""
from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

MAX_CORES = max(1, int(os.environ.get("RNA_MAX_CORES", "160")))
_WC = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]
# NOTE: these run with a default (fork) pool -- fast, cheap. They must be invoked from a CLEAN
# process (see compute_cli.py), NEVER directly inside the async FastMCP server (fork + a live
# asyncio loop deadlocks). The MCP tools shell out to compute_cli, which is such a clean process.


def _valid_target(target: str) -> bool:
    from ..structure_utils import pair_table

    try:
        pair_table(target)  # validates characters AND bracket balance in the caller
        return True
    except Exception:
        return False


def _md():
    import RNA

    md = RNA.md()
    md.temperature = 37.0
    md.dangles = 2  # matches the verifier's Turner2004/T37/dangles2 oracle
    return md


# --- batch fold -------------------------------------------------------------------------
def _fold_worker(arg: Tuple[str, str]) -> Dict:
    import RNA

    seq, target = arg
    s = seq.upper().replace("T", "U")
    fc = RNA.fold_compound(s, _md())
    st, en = fc.mfe()
    d = RNA.bp_distance(st, target) if target else None
    return {"seq": s, "mfe": st, "energy": round(float(en), 2),
            "bp_distance": d, "solved": (d == 0) if target else None}


def batch_fold_impl(sequences: List[str], target: str = "", max_return: int = 20) -> Dict:
    """Fold every sequence's MFE in parallel (<= MAX_CORES procs). If a target is given, score
    each by base-pair distance and return the closest `max_return`. One call screens a whole
    candidate pool instead of one-fold-per-LLM-turn."""
    seqs = [s for s in sequences if s][:4000]  # hard cap
    if not seqs:
        return {"error": "no sequences", "n": 0}
    if target and not _valid_target(target):
        return {"error": f"invalid dot-bracket target: {target!r}", "n": 0}
    args = [(s, target) for s in seqs]
    nproc = min(MAX_CORES, len(args))
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        results = list(ex.map(_fold_worker, args, chunksize=max(1, len(args) // (nproc * 4) or 1)))
    if target:
        results.sort(key=lambda r: (r["bp_distance"], r["energy"]))
        n_solved = sum(1 for r in results if r["solved"])
        return {"n": len(results), "n_solved": n_solved, "used_cores": nproc,
                "best": results[:max_return]}
    return {"n": len(results), "used_cores": nproc, "folds": results[:max_return]}


# --- parallel inverse-folding search ----------------------------------------------------
def _design_worker(arg: Tuple[str, int, int, bool]) -> Dict:
    import RNA

    from ..structure_utils import pair_table

    target, seed, steps, gc_closing = arg
    rng = random.Random(seed)
    n = len(target)
    pt = pair_table(target)
    pairs = [(i, j) for i, j in enumerate(pt) if j > i]
    md = _md()

    def score(seq_str: str) -> int:
        st, _ = RNA.fold_compound(seq_str, md).mfe()
        return RNA.bp_distance(st, target)

    seq = [rng.choice("ACGU") for _ in range(n)]
    for i, j in pairs:  # seed every target pair with a Watson-Crick/wobble pair
        a, b = rng.choice(_WC)
        seq[i], seq[j] = a, b
    if gc_closing:
        for i, j in pairs:  # bias closing pairs toward strong G-C
            if rng.random() < 0.5:
                seq[i], seq[j] = ("G", "C") if (i + j) % 2 == 0 else ("C", "G")
    cur = "".join(seq)
    cd = score(cur)
    best, bestd = cur, cd
    T = 2.0
    for _ in range(steps):
        T *= 0.999
        s2 = list(seq)
        if pairs and rng.random() < 0.7:
            i, j = rng.choice(pairs)
            a, b = rng.choice(_WC)
            s2[i], s2[j] = a, b
        else:
            p = rng.randrange(n)
            if pt[p] >= 0:
                i, j = p, pt[p]
                a, b = rng.choice(_WC)
                s2[i], s2[j] = a, b
            else:
                s2[p] = rng.choice("ACGU")
        cand = "".join(s2)
        d = score(cand)
        if d < cd or rng.random() < math.exp(-(d - cd) / max(T, 1e-3)):
            seq, cd = s2, d
            if d < bestd:
                best, bestd = cand, d
            if bestd == 0:
                break
    return {"seq": best, "bp_distance": bestd, "seed": seed}


def parallel_search_impl(target: str, n_restarts: int = 0, steps: int = 3000,
                         gc_closing: bool = True, base_seed: int = 0) -> Dict:
    """Run many independent simulated-annealing restarts in parallel (<= MAX_CORES) to design a
    sequence whose MFE structure equals `target`. Returns the best sequence found. One call
    replaces a slow hand-written single-threaded search; confirm the result with RnaVerify."""
    if not _valid_target(target):
        return {"error": f"invalid dot-bracket target: {target!r}"}
    n_restarts = n_restarts if n_restarts > 0 else MAX_CORES
    nproc = min(MAX_CORES, n_restarts)
    args = [(target, base_seed + k, steps, gc_closing) for k in range(n_restarts)]
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        results = list(ex.map(_design_worker, args))
    results.sort(key=lambda r: r["bp_distance"])
    best = results[0]
    return {
        "target": target,
        "best_sequence": best["seq"],
        "bp_distance": best["bp_distance"],
        "solved": best["bp_distance"] == 0,  # MFE-match under this model; CONFIRM with RnaVerify
        "n_restarts": n_restarts,
        "used_cores": nproc,
        "n_solved_restarts": sum(1 for r in results if r["bp_distance"] == 0),
    }
