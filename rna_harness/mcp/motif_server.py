"""Stdio MCP server exposing the curated RNA motif knowledge base to the openclaude agent.

Run (openclaude spawns this via --mcp-config):
    python -m rna_harness.mcp.motif_server

Tools appear to the agent as:
    mcp__rna_motifs__lookup_hairpin_loop
    mcp__rna_motifs__search_motifs
    mcp__rna_motifs__analyze_target

Design note: this is a DETERMINISTIC, offline knowledge base (no network). It supplies known
structural-motif priors ("what cap is stable for a 4-nt loop?", "what are the hard features of
this target?"). It is NOT the scorer -- the agent must still verify any assembled sequence with
RnaVerify (the ground-truth folding oracle). Knowledge in; reward stays exact.
"""
from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from .motif_kb import KB_VERSION, _autonomous, analyze_structure, recommend_hairpin, search_motifs

mcp = FastMCP("rna_motifs")


def _guard_target(target: str):
    """Kill Mode-A target drift: if the harness pinned the canonical target (RNA_TASK_TARGET) and
    the agent passed something that is NEITHER the exact target NOR a contiguous SUBSTRING of it
    (a legitimate decomposition sub_target), return a loud actionable error instead of computing
    against the agent's mis-typed target. Returns None when the target is acceptable."""
    pinned = os.environ.get("RNA_TASK_TARGET")
    target = (target or "").strip()          # tolerate a stray trailing newline/space
    pinned = (pinned or "").strip()
    if not pinned or target == pinned or (len(target) < len(pinned) and target in pinned):
        return None
    return (f"[TARGET MISMATCH] you passed a {len(target)}-nt target, but the task target is "
            f"{len(pinned)} nt. Do NOT retype the dot-bracket from memory (mis-counting long dot/paren "
            f"runs is the #1 failure). Copy the task target VERBATIM; for a decomposition unit pass a "
            f"contiguous SUBSTRING of it.")


def _tmo(target: str, base: float = 2.2, lo: int = 90, hi: int = 200) -> int:
    """Length-aware compute timeout. hi=200 stays STRICTLY under compute_cli's 240s SIGALRM
    watchdog, so the watchdog never pre-empts a running search and lose its best-so-far
    (raising hi above 240 regressed long targets -- they self-SIGKILL and return an error)."""
    return max(lo, min(hi, round(base * len(target))))


def lookup_hairpin_loop(loop_len: int) -> str:
    """Look up known stable hairpin-CAP motifs for a hairpin loop of `loop_len` unpaired
    nucleotides. Returns recommended loop sequences (e.g. GAAA/UUCG for 4-nt loops) and the
    preferred closing base pair. These are CANDIDATES to try -- always confirm with RnaVerify.

    Args:
        loop_len: number of unpaired nucleotides in the hairpin loop (>=3 for a valid hairpin).
    """
    rec = recommend_hairpin(int(loop_len))
    if not rec.get("valid", True):
        return f"[{KB_VERSION}] loop_len={loop_len}: {rec['note']}"
    lines = [
        f"[{KB_VERSION}] Hairpin loop of {rec['loop_len']} nt:",
        f"  recommended cap motif(s) (try in order): {', '.join(rec.get('motifs', []))}",
        f"  preferred closing pair: {rec.get('closing_pair')}",
        f"  expected stability: {rec.get('stability')}",
        f"  {rec.get('note')}",
        "  (Prior only -- assemble the sequence and confirm with RnaVerify.)",
    ]
    return "\n".join(lines)


def search_motifs(query: str) -> str:
    """Keyword-search the RNA structural-motif glossary (tetraloop, GNRA, UNCG, lonely pair,
    internal loop, multiloop, pseudoknot, wobble, GC clamp, coaxial stacking, kissing loop...).
    Returns matching motif descriptions with their design implications.

    Args:
        query: free-text keywords, e.g. "stable tetraloop" or "pseudoknot" or "lonely pair".
    """
    from .motif_kb import search_motifs as _search

    hits = _search(query)
    out = [f"[{KB_VERSION}] motif search for {query!r}:"]
    for h in hits:
        out.append(f"  - {h['name']}: {h['summary']}")
    return "\n".join(out)


def _format_target_analysis(res: dict) -> str:
    """Render the analyze_structure() dict into the agent-facing text. Kept as a module-level
    helper (not inlined in the tool) so it is directly unit-testable without the MCP transport."""
    if "error" in res:
        return f"[{KB_VERSION}] {res['error']}"
    if _autonomous():
        # FACTS-ONLY render: structural geometry + a neutral partition. NO recommended motifs, NO
        # "use G-C closing", NO DECOMPOSE->STITCH strategy, NO advice. The agent decides the method.
        out = [
            f"[{KB_VERSION}] structural facts ({res['length']} nt, {res['n_base_pairs']} pairs, "
            f"{res['n_hairpins']} hairpin(s)):",
            f"  stem lengths (stacked pairs): {res['stem_lengths']}",
            "  structural features:",
        ]
        for f in res.get("structural_notes", []):
            out.append(f"    * {f}")
        out.append("  hairpins (positions 0-indexed):")
        for h in res["hairpins"]:
            out.append(f"    * loop@{h.get('loop_positions')} len={h['loop_len']} "
                       f"closing_pair@{h['closing_pair_positions']}")
        dec = res.get("decomposition") or {}
        if "error" not in dec and (dec.get("n_units", 0) >= 2 or dec.get("n_junctions", 0) >= 1):
            out.append(f"  structural partition: {dec['n_units']} stem-loop region(s), "
                       f"{dec.get('n_junctions', 0)} multiloop junction(s), "
                       f"{dec.get('n_seam_positions', 0)} unpaired/junction nt:")
            for i, u in enumerate(dec.get("units", [])):
                out.append(f"    region[{i}] positions@{u['positions']} len={u['len']} "
                           f"substructure={u['sub_target']}")
            for jn in dec.get("junctions", []):
                out.append(f"    junction closing_pair@{jn['closing_pair_positions']} "
                           f"branches={jn['branches']}")
        return "\n".join(out)
    out = [
        f"[{KB_VERSION}] target analysis ({res['length']} nt, {res['n_base_pairs']} pairs, "
        f"{res['n_hairpins']} hairpin(s)):",
        f"  stem lengths (stacked pairs): {res['stem_lengths']}",
        "  hard-feature flags:",
    ]
    for f in res["hard_feature_flags"]:
        out.append(f"    * {f}")
    out.append("  per-hairpin plan (positions are 0-indexed):")
    for h in res["hairpins"]:
        cp = h["closing_pair_positions"]
        motifs = ", ".join(h.get("recommended_motifs") or []) or "(none)"
        out.append(
            f"    * loop@{h.get('loop_positions')} len={h['loop_len']} "
            f"closing_pair@{cp} -> use {h.get('closing_pair')} closing, cap motif: {motifs}"
        )
    # Decomposition plan: surfaced for COMPOSITIONAL targets (>=2 units or a multiloop junction).
    # analyze_structure() always computes this, and general_advice tells the agent to iterate the
    # units/sub_targets -- so it MUST appear in the tool output for that advice to be actionable.
    dec = res.get("decomposition") or {}
    if "error" not in dec and (dec.get("n_units", 0) >= 2 or dec.get("n_junctions", 0) >= 1):
        out.append(
            f"  decomposition plan ({dec['n_units']} design unit(s), "
            f"{dec.get('n_junctions', 0)} junction(s), {dec.get('n_seam_positions', 0)} seam nt) "
            "-- design each unit's sub_target INDEPENDENTLY, then assemble + repair seams:"
        )
        for i, u in enumerate(dec.get("units", [])):
            out.append(
                f"    unit[{i}] positions@{u['positions']} len={u['len']} "
                f"sub_target={u['sub_target']}"
            )
        for jn in dec.get("junctions", []):
            out.append(
                f"    junction closing_pair@{jn['closing_pair_positions']} "
                f"branches={jn['branches']}"
            )
        if dec.get("strategy"):
            out.append(f"    strategy: {dec['strategy']}")
    out.append(f"  advice: {res['general_advice']}")
    return "\n".join(out)


def analyze_target(dot_bracket: str) -> str:
    """Analyze a TARGET secondary structure (dot-bracket) and return, per hairpin, a
    recommended cap motif + closing pair, flags for the classic hard features (lonely
    pairs, triloops, pseudoknots), AND -- for long/multiloop targets -- a decomposition plan
    (independent stem-loop design units + their sub_targets + junction seams). Use this FIRST on a
    hard target to get a design plan, then build the sequence and run the verifier-guided repair
    loop with RnaVerify.

    Args:
        dot_bracket: the target structure in dot-bracket notation (e.g. "((((....))))").
    """
    g = _guard_target(dot_bracket)
    if g:
        return g
    return _format_target_analysis(analyze_structure(dot_bracket))


def _run_compute(payload: dict, timeout: float) -> str:
    """Shell out to a CLEAN process for the parallel pool (a fork pool cannot run inside this
    async MCP server). Returns the child's JSON stdout, or a JSON error."""
    import json as _json
    import os
    import signal
    import subprocess
    import sys

    # start_new_session so compute_cli is its own process-group leader; on timeout we kill the
    # ENTIRE group (compute_cli + its ProcessPoolExecutor workers). subprocess.run(timeout=) would
    # kill only compute_cli and orphan its ~N pool workers -> the process leak that melted the box.
    p = subprocess.Popen(
        [sys.executable, "-m", "rna_harness.mcp.compute_cli", _json.dumps(payload)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)  # reap compute_cli AND all pool workers
        except Exception:
            pass
        try:
            p.communicate(timeout=5)
        except Exception:
            pass
        return _json.dumps({"error": f"compute timed out after {timeout}s"})
    if p.returncode != 0 or not (out or "").strip():
        return _json.dumps({"error": "compute subprocess failed", "stderr": (err or "")[-400:]})
    return out


def batch_fold(sequences: list[str], target: str = "") -> str:
    """Fold MANY candidate sequences' MFE structures IN PARALLEL across up to 160 CPU cores in a
    single call. If `target` (dot-bracket) is given, each candidate is scored by base-pair
    distance to it and the closest are returned (solved = bp_distance 0). Use this to screen a
    whole pool of candidates at once instead of folding one-per-turn.

    Args:
        sequences: list of RNA/DNA sequences to fold (T is treated as U). Up to 4000.
        target: optional target dot-bracket to score each candidate against.
    """
    return _run_compute({"cmd": "batch_fold", "sequences": sequences, "target": target}, timeout=120)


def parallel_search(target: str, n_restarts: int = 0, steps: int = 3000) -> str:
    """Run a MANY-RESTART simulated-annealing inverse-folding search IN PARALLEL across up to 160
    CPU cores to design a sequence whose MFE structure equals `target`. Returns the best sequence
    found in one call (seconds), replacing a slow hand-written single-threaded search. ALWAYS
    confirm the returned sequence with RnaVerify (this uses the same Turner2004 model but RnaVerify
    is the official scorer).

    Args:
        target: target secondary structure in dot-bracket notation.
        n_restarts: parallel restarts (0 = use all allotted cores, ~160).
        steps: annealing steps per restart (default 3000).
    """
    g = _guard_target(target)
    if g:
        return g
    return _run_compute(
        {"cmd": "parallel_search", "target": target, "n_restarts": n_restarts, "steps": steps},
        timeout=_tmo(target),
    )


def inverse_fold_search(target: str, warm: str = "", n_restarts: int = 0, tries: int = 40) -> str:
    """STRONG inverse-folding search: run MANY parallel restarts of ViennaRNA's native
    RNA.inverse_fold to design a sequence whose MFE structure equals `target`. Much stronger than
    parallel_search at closing the last pairs. Use it per DECOMPOSED UNIT (short sub_target) or on
    the whole target. Optionally warm-start from an assembled candidate via `warm`. ALWAYS confirm
    the returned sequence with RnaVerify.

    Args:
        target: target dot-bracket (or a unit's sub_target).
        warm: optional starting sequence (e.g. an assembled decomposition) to refine.
        n_restarts: parallel restarts (0 = all cores).
        tries: inverse_fold attempts per restart (default 40).
    """
    g = _guard_target(target)
    if g:
        return g
    r = _run_compute({"cmd": "inverse_fold_search", "target": target, "warm": warm,
                      "n_restarts": n_restarts, "tries": tries}, timeout=_tmo(target))
    import json as _j
    d = _j.loads(r)
    if "error" in d:
        return f"[solver] {d['error']}"
    return (f"[inverse_fold_search] solved={d.get('solved')} bp_distance={d.get('bp_distance')}\n"
            f"sequence: {d.get('seq')}\n(confirm with RnaVerify)")


def finish(candidate: str, target: str, n_restarts: int = 0, iters: int = 20000) -> str:
    """FINISHER: take a NEAR-MISS candidate (bp_distance > 0) and run a diagnosis-guided
    simulated-annealing hill-climb (focused on the mispaired positions, with plateau escapes) to
    close the last wrong pairs into an exact solve. Use this when a design is ALMOST right
    (bp_distance 1-3) and RnaVerify shows a few residual mismatches. Confirm with RnaVerify.

    Args:
        candidate: the near-miss sequence to polish.
        target: target dot-bracket.
        n_restarts: parallel hill-climb restarts (0 = all cores).
        iters: iterations per restart (default 20000).
    """
    g = _guard_target(target)
    if g:
        return g
    r = _run_compute({"cmd": "finish", "candidate": candidate, "target": target,
                      "n_restarts": n_restarts, "iters": iters}, timeout=_tmo(target))
    import json as _j
    d = _j.loads(r)
    if "error" in d:
        return f"[finish] {d['error']}"
    return (f"[finish] solved={d.get('solved')} bp_distance={d.get('bp_distance')}\n"
            f"sequence: {d.get('seq')}\n(confirm with RnaVerify)")


def desirna_solve(target: str) -> str:
    """Design a sequence for `target` by calling the external DesiRNA solver (replica-exchange Monte
    Carlo; SOTA on Eterna100-V2). Returns the best sequence VERIFIED by RnaVerify -- use this for
    hard/long targets where the built-in search stalls. If DesiRNA is not installed on this host,
    returns a clear message and you should fall back to parallel_search / inverse_fold_search / finish.

    Args:
        target: target dot-bracket."""
    err = _guard_target(target)
    if err:
        return err
    from ..generators.desirna import DesiRNAGenerator, desirna_available
    from ..task import DesignTask
    if not desirna_available():
        return ("DesiRNA is not installed on this host. Set DESIRNA_CMD to its launcher to enable it; "
                "meanwhile use parallel_search / inverse_fold_search / finish.")
    r = DesiRNAGenerator().design(DesignTask(task_id="mcp", target=(target or "").strip(), family="mcp"))
    if not r.meta.get("available", True):
        return str(r.meta.get("hint", "DesiRNA unavailable."))
    if r.solved:
        return f"DesiRNA SOLVED (bp_distance=0).\nsequence: {r.best_sequence}\n(confirm with RnaVerify)"
    return (f"DesiRNA best bp_distance={r.best_bp_distance} over {r.n_evals} candidate(s).\n"
            f"sequence: {r.best_sequence or '(none)'}\n(confirm with RnaVerify; polish with finish)")


# --- tool registration (gated by RNA_AUTONOMOUS) -----------------------------------------
# Neutral (facts-only) docstrings used when RNA_AUTONOMOUS=1, so even the tool SCHEMA descriptions
# carry no solving strategy. Flag-OFF keeps each function's ORIGINAL docstring => unchanged behavior.
_NEUTRAL_DOCS = {
    "analyze_target":
        "Parse a target dot-bracket and return structural FACTS: length, base-pair count, per-hairpin "
        "loop and closing-pair positions and loop lengths, stem (stacked-pair) lengths, presence and "
        "positions of crossing pairs / length-1 stems / 3-nt loops, and a geometric partition of the "
        "structure into stem-loop regions (positions + substructure substrings) and multiloop-closing "
        "pairs.\n\nArgs:\n    dot_bracket: the target structure in dot-bracket notation.",
    "parallel_search":
        "Run a many-restart simulated-annealing search for a sequence whose ViennaRNA MFE structure "
        "(Turner2004) equals `target`. Returns the best sequence found. Confirm with RnaVerify.\n\n"
        "Args:\n    target: target dot-bracket.\n    n_restarts: parallel restarts (0 = all cores).\n"
        "    steps: annealing steps per restart.",
    "finish":
        "Run a simulated-annealing local search that mutates the positions where `candidate` mis-folds "
        "relative to `target`, seeking an exact MFE match. Returns the best sequence found. Confirm "
        "with RnaVerify.\n\nArgs:\n    candidate: starting sequence.\n    target: target dot-bracket.\n"
        "    n_restarts: parallel restarts (0 = all cores).\n    iters: iterations per restart.",
    "batch_fold":
        "Fold many candidate sequences' MFE structures in parallel. If `target` is given, score each by "
        "base-pair distance to it and return the closest.\n\nArgs:\n    sequences: sequences to fold "
        "(up to 4000).\n    target: optional target dot-bracket to score against.",
}


def _register_tools() -> None:
    auto = _autonomous()
    # neutral capabilities/facts -- registered in BOTH modes
    always = [analyze_target, parallel_search, finish, batch_fold]
    # strategy/advice tools + external/native solvers that "do the solving for you" -- ONLY in
    # crutched (flag-off) mode, so they never contaminate the autonomous-A_theta experiments.
    # desirna_solve is the PLAN-A accessibility surface: the agent calls the SOTA tool, Claude-Code-style.
    crutch_only = [lookup_hairpin_loop, search_motifs, inverse_fold_search, desirna_solve]
    for fn in always:
        if auto and fn.__name__ in _NEUTRAL_DOCS:
            fn.__doc__ = _NEUTRAL_DOCS[fn.__name__]
        mcp.tool()(fn)
    if not auto:
        for fn in crutch_only:
            mcp.tool()(fn)


_register_tools()


def main() -> None:
    mcp.run()  # default stdio transport -- openclaude connects over stdin/stdout


if __name__ == "__main__":
    main()
