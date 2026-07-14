"""Mechanical structural features of a target dot-bracket. ZERO biology -- pure combinatorics.

These are the router's inputs. We do NOT hand-encode expert difficulty rules; the mapping from these
features to (difficulty, cost, best-tool) is LEARNED from the OUTCOMES of running tools. A few of
these (has_pseudoknot, n_lonely_pairs, len) also serve as engineering cold-start heuristics -- still
mechanical, not biological.
"""
from __future__ import annotations

from typing import Dict

from ..structure_utils import has_pseudoknot
from ..mcp.motif_kb import analyze_structure, decompose_structure

_PK = set("[]{}<>")


def features(target: str) -> Dict[str, float]:
    """Extract a fixed mechanical feature vector from an (extended) dot-bracket target. Robust to
    pseudoknots: nested-skeleton features are computed on the target with pseudoknot brackets removed,
    plus explicit pseudoknot counts."""
    n = len(target)
    pk = has_pseudoknot(target)
    n_pk_br = sum(1 for c in target if c in _PK)
    nested = "".join("." if c in _PK else c for c in target)   # valid nested dot-bracket

    ana = analyze_structure(nested)
    dec = decompose_structure(nested)
    if "error" in ana or "error" in dec:
        # malformed nested skeleton (should not happen after pk-strip on valid input); degrade
        # explicitly to length + pseudoknot features rather than silently returning all-zeros.
        ana = {"stem_lengths": [0], "n_base_pairs": 0, "n_hairpins": 0, "hairpins": []}
        dec = {"n_units": 0, "n_junctions": 0, "junctions": []}
    stems = ana.get("stem_lengths") or [0]
    n_pairs = ana.get("n_base_pairs", 0)
    n_hairpins = ana.get("n_hairpins", 0)
    n_units = dec.get("n_units", 0)
    n_junc = dec.get("n_junctions", 0)
    max_deg = max((j["branches"] for j in dec.get("junctions", [])), default=0)
    n_lonely = sum(1 for L in stems if L == 1)
    n_triloops = sum(1 for h in ana.get("hairpins", []) if h.get("loop_len") == 3)

    depth = mx = 0                                             # max nesting depth of round brackets
    for c in nested:
        if c == "(":
            depth += 1
            mx = max(mx, depth)
        elif c == ")":
            depth -= 1

    return {
        "len": float(n),
        "n_pairs": float(n_pairs),
        "frac_paired": (2.0 * n_pairs / n) if n else 0.0,
        "has_pseudoknot": float(int(pk)),
        "n_pk_brackets": float(n_pk_br),
        "n_units": float(n_units),
        "n_junctions": float(n_junc),
        "max_junction_degree": float(max_deg),
        "n_hairpins": float(n_hairpins),
        "max_stem": float(max(stems)),
        "min_stem": float(min(stems)),
        "mean_stem": (sum(stems) / len(stems)) if stems else 0.0,
        "n_lonely_pairs": float(n_lonely),
        "n_triloops": float(n_triloops),
        "nesting_depth": float(mx),
    }


FEATURE_NAMES = list(features("((((....))))").keys())   # stable ordering for vectorisation
