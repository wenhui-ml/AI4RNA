"""Curated, deterministic RNA structural-motif knowledge base.

Pure data + pure functions (no MCP, no ViennaRNA dependency beyond the light `pair_table`
parser) so it is unit-testable on its own. `motif_server.py` wraps these as MCP tools.

CONTRACT: every recommendation here is a *prior* -- a known-good candidate to TRY. The
deterministic folding oracle (RnaVerify) is the ground truth; the agent must verify. This
keeps the MCP consistent with the determinism thesis (knowledge in, but reward stays exact).

Facts are standard RNA biophysics (Turner-rules-era thermodynamics + classic tetraloop
literature): GNRA and UNCG tetraloops are the exceptionally stable hairpin caps; hairpin
loops are strongest closed by a G-C/C-G pair; loops < 3 nt cannot close a hairpin.
"""
from __future__ import annotations

from typing import Dict, List

from ..structure_utils import pair_table

KB_VERSION = "rna-motif-kb/1.0"

# --- Hairpin-cap motifs, the highest-leverage knowledge for inverse folding --------------
# GNRA: G-N-R-A (N=any, R=purine A/G). UNCG: U-N-C-G (N=any). Both are unusually stable
# tetraloops; GAAA and UUCG are the canonical "super-stable" caps. Prefer a C-G closing pair.
GNRA_TETRALOOPS = ["GAAA", "GCAA", "GAGA", "GUGA", "GGAA", "GUAA", "GCGA", "GAAA"]
UNCG_TETRALOOPS = ["UUCG", "CUUG", "UACG", "UGCG"]

# Length-specific hairpin-loop guidance. `motifs` are ordered by recommended preference.
_HAIRPIN_BY_LEN: Dict[int, Dict] = {
    3: {
        "motifs": ["UUU", "GAA", "GUA", "CUU"],
        "closing_pair": "C-G",
        "note": "Triloops are strained; ViennaRNA gives a stability bonus to only a small "
                "set of triloop sequences. Close with a strong C-G pair and TRY these, but "
                "expect to verify -- a triloop hairpin is often the hard feature in a target.",
        "stability": "low-moderate",
    },
    4: {
        "motifs": ["GAAA", "UUCG", "GCAA", "CUUG"],
        "closing_pair": "C-G",
        "note": "Tetraloops are the sweet spot. GAAA (GNRA family) and UUCG (UNCG family) are "
                "the two exceptionally stable caps; both are most stable on a C-G closing "
                "pair (G or U of the loop stacks on the closing pair). Use these to NUCLEATE "
                "the intended hairpin.",
        "stability": "very high",
    },
    5: {
        "motifs": ["GAAAC", "AAAAA", "GCAAC"],
        "closing_pair": "G-C",
        "note": "5-nt loops have no dramatic special bonus. Keep the loop free of "
                "self-complementary bases (A/U-rich, avoid GC pairs inside the loop) and use "
                "a G-C closing pair. A common trick: a GNRA core (GAAA) plus one spacer.",
        "stability": "moderate",
    },
    6: {
        "motifs": ["AAAAAA", "GAAAAC", "UUUUUU"],
        "closing_pair": "G-C",
        "note": "6-nt loops: keep unstructured and A-rich to avoid internal pairing; strong "
                "G-C closing pair. If the target allows, a longer loop is easier than a "
                "3-nt loop -- flexibility is your friend here.",
        "stability": "moderate",
    },
}


def recommend_hairpin(loop_len: int) -> Dict:
    """Recommended cap motif(s) for a hairpin loop of the given length. Candidates to VERIFY."""
    if loop_len < 3:
        return {
            "loop_len": loop_len,
            "valid": False,
            "note": "A hairpin loop must span at least 3 unpaired nucleotides (Turner model). "
                    f"A {loop_len}-nt loop cannot close a standard hairpin -- the target may "
                    "be undesignable here, or the pairing must be read differently.",
            "kb_version": KB_VERSION,
        }
    if loop_len in _HAIRPIN_BY_LEN:
        rec = dict(_HAIRPIN_BY_LEN[loop_len])
        rec.update({"loop_len": loop_len, "valid": True, "kb_version": KB_VERSION})
        return rec
    # generic guidance for longer loops
    return {
        "loop_len": loop_len,
        "valid": True,
        "motifs": ["A" * loop_len],
        "closing_pair": "G-C",
        "note": f"{loop_len}-nt loop: no special cap motif. Keep it unstructured (A-rich) to "
                "avoid spurious internal pairing, and use a strong G-C closing pair. Verify "
                "there is no competing hairpin forming inside the loop.",
        "stability": "moderate",
        "kb_version": KB_VERSION,
    }


# --- General structural-motif glossary (keyword search) ----------------------------------
_MOTIF_GLOSSARY: Dict[str, Dict] = {
    "gnra tetraloop": {
        "aliases": ["gnra", "tetraloop", "gaaa"],
        "summary": "G-N-R-A hairpin tetraloop (N=any, R=A/G). Exceptionally stable; GAAA and "
                   "GCAA are the canonical picks. Best on a C-G closing pair. Use to nucleate "
                   "a hairpin cheaply.",
    },
    "uncg tetraloop": {
        "aliases": ["uncg", "uucg"],
        "summary": "U-N-C-G hairpin tetraloop. UUCG is one of the most thermostable RNA caps "
                   "known; ideal on a C-G closing pair. Alternative to GNRA when GNRA clashes "
                   "with a constraint.",
    },
    "gc clamp": {
        "aliases": ["gc clamp", "closing pair", "helix closing", "gc closing"],
        "summary": "Placing G-C/C-G pairs at helix ends (especially the loop-closing pair) "
                   "stabilizes the intended helix and suppresses frayed/alternative folds. The "
                   "single highest-yield negative-design move.",
    },
    "wobble": {
        "aliases": ["gu wobble", "g-u", "wobble pair"],
        "summary": "G-U wobble pair: weaker than Watson-Crick, 2 H-bonds. Useful to DEStabilize "
                   "a helix deliberately or to satisfy a GC-window constraint without breaking "
                   "pairing entirely.",
    },
    "lonely pair": {
        "aliases": ["lonely pair", "isolated pair", "single base pair", "lonely"],
        "summary": "An isolated (unstacked) base pair. Thermodynamically fragile and a frequent "
                   "cause of hard/undesignable targets. Give it maximal support (G-C) and check "
                   "designability; sometimes the target simply has no MFE solution here.",
    },
    "internal loop": {
        "aliases": ["internal loop", "interior loop", "symmetric loop", "asymmetric loop"],
        "summary": "Unpaired bases on BOTH strands between two helices. Small symmetric internal "
                   "loops (esp. 1x1, 2x2) are common hard features. Strengthen the flanking "
                   "helices with G-C and avoid accidental pairing across the loop.",
    },
    "bulge": {
        "aliases": ["bulge", "bulge loop"],
        "summary": "Unpaired bases on ONE strand only. Usually mild; keep bulge bases from "
                   "pairing with anything and keep flanking helices strong.",
    },
    "multiloop": {
        "aliases": ["multiloop", "junction", "three-way junction", "four-way junction"],
        "summary": "A loop where >=3 helices meet. Coaxial stacking of adjacent helices matters; "
                   "strong closing pairs on each stem and A-rich junction linkers help the "
                   "intended topology win.",
    },
    "coaxial stacking": {
        "aliases": ["coaxial", "stacking"],
        "summary": "Adjacent helices across a junction stack end-to-end, stabilizing the fold. "
                   "Exploit by making the stacking helices' terminal pairs G-C.",
    },
    "kissing loop": {
        "aliases": ["kissing loop", "loop-loop", "kissing complex"],
        "summary": "Two hairpin loops pair with each other (a tertiary motif). NOT captured by "
                   "the default ViennaRNA MFE secondary-structure model -- do not rely on it for "
                   "MFE-match design unless explicitly modeling it.",
    },
    "pseudoknot": {
        "aliases": ["pseudoknot", "pknot"],
        "summary": "Crossing base pairs. The default ViennaRNA MFE oracle does NOT fold "
                   "pseudoknots; a pseudoknotted target string will not be an MFE structure "
                   "under the standard model. Flag as out-of-model.",
    },
    "poly-a linker": {
        "aliases": ["poly-a", "linker", "spacer", "unpaired region"],
        "summary": "Long single-stranded stretches: use A (or U) runs to keep them unpaired; "
                   "avoid G/C runs that can form spurious hairpins.",
    },
}


def search_motifs(query: str) -> List[Dict]:
    """Keyword search of the motif glossary. Returns matching entries (name + summary)."""
    q = (query or "").strip().lower()
    if not q:
        return [{"name": k, **v} for k, v in _MOTIF_GLOSSARY.items()]
    hits = []
    for name, entry in _MOTIF_GLOSSARY.items():
        hay = name + " " + " ".join(entry.get("aliases", [])) + " " + entry["summary"].lower()
        if any(tok and tok in hay for tok in q.split()):
            hits.append({"name": name, "summary": entry["summary"], "aliases": entry.get("aliases", [])})
    return hits or [{"name": "(no match)", "summary": f"No glossary entry matched {query!r}. "
                     "Try: tetraloop, gnra, uncg, lonely pair, internal loop, multiloop, "
                     "pseudoknot, wobble, gc clamp."}]


# --- Structure analysis: find hairpins & hard features, recommend caps -------------------
def _hairpin_loops(target: str):
    pt = pair_table(target)
    out = []
    for i in range(len(target)):
        j = pt[i]
        if j > i and all(pt[k] == -1 for k in range(i + 1, j)):
            out.append((i, j, j - i - 1))
    return out


def _stem_lengths(target: str) -> List[int]:
    """Length (in stacked pairs) of each maximal helix. Length-1 stems are 'lonely pairs'."""
    pt = pair_table(target)
    n = len(target)
    seen = set()
    lengths = []
    for i in range(n):
        j = pt[i]
        if j > i and (i, j) not in seen:
            # walk inward while stacked
            L = 0
            a, b = i, j
            while a < b and pt[a] == b:
                seen.add((a, b))
                L += 1
                a += 1
                b -= 1
            lengths.append(L)
    return lengths


def analyze_structure(dot_bracket: str) -> Dict:
    """Parse a target dot-bracket, enumerate hairpin loops with recommended cap motifs, and
    flag the classic hard features (lonely pairs, triloops, pseudoknots). Actionable priors;
    the agent must still verify the assembled sequence with RnaVerify."""
    try:
        pt = pair_table(dot_bracket)
    except ValueError as e:
        return {"error": f"invalid dot-bracket: {e}", "kb_version": KB_VERSION}

    n = len(dot_bracket)
    has_pk = any(c in dot_bracket for c in "[]{}<>")
    hairpins = _hairpin_loops(dot_bracket)
    stems = _stem_lengths(dot_bracket)
    n_lonely = sum(1 for L in stems if L == 1)

    hairpin_recs = []
    for i, j, L in hairpins:
        rec = recommend_hairpin(L)
        hairpin_recs.append({
            "closing_pair_positions": [i, j],  # 0-indexed
            "loop_positions": [i + 1, j - 1] if L > 0 else None,
            "loop_len": L,
            "recommended_motifs": rec.get("motifs"),
            "closing_pair": rec.get("closing_pair"),
            "note": rec.get("note"),
        })

    flags = []
    if has_pk:
        flags.append("PSEUDOKNOT: crossing pairs present -- the default ViennaRNA MFE oracle "
                     "does not fold pseudoknots; this target is out-of-model for standard MFE-match.")
    if n_lonely:
        flags.append(f"{n_lonely} LONELY PAIR(S): isolated unstacked pairs are fragile and a "
                     "common cause of hard/undesignable targets -- give them G-C support.")
    triloops = [h for h in hairpin_recs if h["loop_len"] == 3]
    if triloops:
        flags.append(f"{len(triloops)} TRILOOP(S) (3-nt hairpin): strained; often the hard "
                     "feature. Close with C-G and verify a stable triloop sequence.")

    return {
        "length": n,
        "n_hairpins": len(hairpins),
        "n_base_pairs": sum(1 for k in range(n) if pt[k] > k),
        "stem_lengths": stems,
        "hairpins": hairpin_recs,
        "hard_feature_flags": flags or ["none obvious"],
        "general_advice": "Nucleate each hairpin with its recommended cap on a G-C/C-G closing "
                          "pair, keep unpaired linkers A-rich, then fold and run the "
                          "verifier-guided repair loop (edit exactly the diagnosed positions).",
        "kb_version": KB_VERSION,
    }
