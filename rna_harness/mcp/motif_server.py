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

from mcp.server.fastmcp import FastMCP

from .motif_kb import KB_VERSION, analyze_structure, recommend_hairpin, search_motifs

mcp = FastMCP("rna_motifs")


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def analyze_target(dot_bracket: str) -> str:
    """Analyze a TARGET secondary structure (dot-bracket) and return, per hairpin, a
    recommended cap motif + closing pair, plus flags for the classic hard features (lonely
    pairs, triloops, pseudoknots). Use this FIRST on a hard target to get a design plan, then
    build the sequence and run the verifier-guided repair loop with RnaVerify.

    Args:
        dot_bracket: the target structure in dot-bracket notation (e.g. "((((....))))").
    """
    res = analyze_structure(dot_bracket)
    if "error" in res:
        return f"[{KB_VERSION}] {res['error']}"
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
    out.append(f"  advice: {res['general_advice']}")
    return "\n".join(out)


def main() -> None:
    mcp.run()  # default stdio transport -- openclaude connects over stdin/stdout


if __name__ == "__main__":
    main()
