"""Structural-retrieval warm-start for RNA inverse design (AlphaFold-1-style template reuse).

WHY. In a nearly-solved, clean-oracle domain the scientific question is not "does the ceiling
rise" but "does accumulated experience let us REACH the ceiling at LOWER search cost." AlphaFold-1
reaches a structure fast by retrieving similar solved structures (templates/MSA) and refining from
them instead of searching from scratch. The inverse-design analog: given a TARGET secondary
structure, retrieve previously-solved targets that share structural ELEMENTS, transport their
(verified) sub-sequences as a WARM-START, then let a fixed-budget local repair finish the seams.

TWO-LEVEL LIBRARY.
  L2 (unit level, the workhorse): key = a leaf design unit's dot-bracket shape (`sub_target` from
     `decompose_structure`). value = verified sub-sequences that fold to that shape in isolation.
     Retrieval is an EXACT dict lookup (O(1)); units recur across targets even when whole targets
     do not, so coverage is high on the compositional bulk. This is more principled than AF
     templates because RNA secondary structure decomposes exactly and cheaply.
  L1 (whole level, coarse): an element-bag fingerprint (multiset of unit signatures + junction
     degrees); distance = 1 - |intersection|/|union|, which is exactly "how much of the
     decomposition transfers." Used for telemetry / nearest-neighbour diagnostics here; grafting a
     whole solved sequence across differing lengths is left to a later, alignment-careful version.

RED LINES (enforced in code).
  * Never ingest an Eterna sample solution; the library is built only from our OWN verified solves
    (self-generated) and canonical biophysical priors (motif_kb) / offline-synthesised units.
  * Leakage guard: a target placed in `_eval_guard` is refused ingestion, so we never "retrieve"
    the answer to a target we are evaluating on. Build from a TRAIN split, evaluate on HELD-OUT.
  * Oracle gate: `add_solution` only stores a (target, seq) whose fold EQUALS the target; a bad
    retrieval can therefore only waste a little refinement, never manufacture a false "solved".

A retrieved unit is a PRIOR to try, exactly as in `motif_kb`; ViennaRNA remains the ground truth.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import RNA

from ..structure_utils import pair_table
from ..mcp.motif_kb import decompose_structure, recommend_hairpin
from ..mcp.finisher import _fold, parallel_inverse_fold

LIB_VERSION = "rna-retrieval/1.0"
_BASES = "ACGU"
_GC = ("G", "C")


# --------------------------------------------------------------------------- structural signatures
def element_bag(target: str) -> Dict[str, int]:
    """Multiset of structural-element signatures: leaf-unit dot-bracket shapes + 'J<deg>' junctions.
    This is the L1 fingerprint AND the basis of the transfer distance."""
    dec = decompose_structure(target)
    bag: Dict[str, int] = {}
    for u in dec.get("units", []):
        k = u["sub_target"]
        bag[k] = bag.get(k, 0) + 1
    for j in dec.get("junctions", []):
        k = f"J{j['branches']}"
        bag[k] = bag.get(k, 0) + 1
    return bag


def bag_distance(b1: Dict[str, int], b2: Dict[str, int]) -> float:
    """1 - (multiset intersection / union). 0.0 == identical decomposition, 1.0 == disjoint.
    Measures the fraction of structural elements that could transfer between two targets."""
    keys = set(b1) | set(b2)
    inter = sum(min(b1.get(k, 0), b2.get(k, 0)) for k in keys)
    union = sum(max(b1.get(k, 0), b2.get(k, 0)) for k in keys)
    return 1.0 - (inter / union if union else 1.0)


def _canonical_unit_seq(sub_target: str) -> str:
    """A generic, no-search PRIOR sequence for a design unit: alternating G-C/C-G stems, A-rich
    unpaired, and a stable GAAA tetraloop cap on any length-4 hairpin loop. This is the motif_kb
    biophysical prior expressed as a concrete fill; used when retrieval has no hit for a unit."""
    n = len(sub_target)
    pt = pair_table(sub_target)
    s = ["A"] * n
    tog = 0
    for i in range(n):
        j = pt[i]
        if j > i:
            a, b = _GC if tog % 2 == 0 else _GC[::-1]
            s[i], s[j] = a, b
            tog += 1
    # cap hairpin loops with a recommended stable motif when the length matches a known cap
    for i in range(n):
        j = pt[i]
        if j > i and all(pt[k] == -1 for k in range(i + 1, j)):  # hairpin: i..j encloses only loop
            loop_len = j - i - 1
            rec = recommend_hairpin(loop_len)
            motifs = rec.get("motifs") or []
            cap = next((m for m in motifs if len(m) == loop_len), None)
            if cap:
                for k, ch in enumerate(cap):
                    s[i + 1 + k] = ch
    return "".join(s)


# --------------------------------------------------------------------------- the library
class RetrievalLibrary:
    """Persistent structural-retrieval memory. Grows by ingesting VERIFIED solves; serves warm-starts
    by exact unit-shape lookup with canonical-prior fallback."""

    def __init__(self) -> None:
        self.units: Dict[str, List[dict]] = {}   # sub_target -> [{subseq, iso_bp, freq, source}]
        self.wholes: List[dict] = []             # [{target, seq, bag, source}]
        self._eval_guard: set[str] = set()

    # -- leakage guard -------------------------------------------------------
    def forbid(self, targets) -> None:
        """Refuse to ingest these targets (the current evaluation set), preventing answer leakage."""
        self._eval_guard |= set(targets)

    # -- ingestion (oracle-gated) --------------------------------------------
    def add_solution(self, target: str, seq: str, source: str = "solve", verify: bool = True) -> dict:
        if target in self._eval_guard:
            return {"added": False, "reason": "eval-leakage-guard"}
        if not seq or len(seq) != len(target):
            return {"added": False, "reason": "length-mismatch"}
        if verify and _fold(seq) != target:
            return {"added": False, "reason": "not-a-verified-solve"}
        dec = decompose_structure(target)
        added_units = 0
        for u in dec.get("units", []):
            a, b = u["positions"]
            sub = u["sub_target"]
            subseq = seq[a:b + 1]
            if len(subseq) != len(sub):          # bug guard: positions must line up with the shape
                continue
            iso_bp = RNA.bp_distance(_fold(subseq), sub)
            self._add_unit(sub, subseq, iso_bp, source)
            added_units += 1
        self.wholes.append({"target": target, "seq": seq, "bag": element_bag(target), "source": source})
        return {"added": True, "units": added_units}

    def _add_unit(self, sub: str, subseq: str, iso_bp: int, source: str) -> None:
        lst = self.units.setdefault(sub, [])
        for e in lst:
            if e["subseq"] == subseq:
                e["freq"] += 1
                e["iso_bp"] = min(e["iso_bp"], iso_bp)
                return
        lst.append({"subseq": subseq, "iso_bp": iso_bp, "freq": 1, "source": source})

    def _best_unit(self, sub: str) -> Optional[str]:
        """The retrieved sequence for a unit shape: cleanest isolated fold first, then most frequent."""
        lst = self.units.get(sub)
        if not lst:
            return None
        return sorted(lst, key=lambda e: (e["iso_bp"], -e["freq"]))[0]["subseq"]

    # -- serving -------------------------------------------------------------
    def warm_start(self, target: str, clamp_hits: bool = False) -> Tuple[str, float, dict]:
        """Assemble a retrieval seed for `target`. Each leaf unit is filled from the library if its
        shape is known, else from a canonical prior; junction/seam pairs get alternating G-C, unpaired
        stay A (mirrors `decompose_solve`'s assembly). Returns (seed_seq, unit_coverage, meta).
        Pure lookup + prior fill: NO folding search is spent here.

        clamp_hits: GC-clamp the terminal (outermost) pair of each RETRIEVED (hit) unit. A retrieved
        sub-sequence was tuned to a DIFFERENT whole-context and can compose worse than a stiff prior
        (the compositional-fidelity effect); clamping the boundary pair is the negative-design move
        (motif_kb 'gc clamp') that makes transferred experience robust to the new context. The AF1
        lesson: use the template as a prior, adapt its boundary, don't transplant verbatim."""
        n = len(target)
        pt = pair_table(target)
        dec = decompose_structure(target)
        seq = ["A"] * n
        covered: set[int] = set()
        n_units = len(dec.get("units", []))
        n_hit = 0
        for u in dec.get("units", []):
            a, b = u["positions"]
            sub = u["sub_target"]
            best = self._best_unit(sub)
            is_hit = best is not None and len(best) == (b - a + 1)
            if not is_hit:
                best = _canonical_unit_seq(sub)
            if len(best) == (b - a + 1):
                for k, ch in enumerate(best):
                    seq[a + k] = ch
                covered.update(range(a, b + 1))
                if is_hit:
                    n_hit += 1
                    if clamp_hits and pt[a] == b:   # clamp the transplanted unit's terminal pair
                        seq[a], seq[b] = "G", "C"
        tog = 0
        for i in range(n):
            j = pt[i]
            if j > i and (i not in covered or j not in covered):
                a1, a2 = _GC if tog % 2 == 0 else _GC[::-1]
                seq[i], seq[j] = a1, a2
                tog += 1
        assembled = "".join(seq)
        cov = (n_hit / n_units) if n_units else 0.0
        meta = {"n_units": n_units, "n_hit": n_hit, "assembled_bp": RNA.bp_distance(_fold(assembled), target)}
        return assembled, cov, meta

    def nearest_whole(self, target: str) -> Optional[dict]:
        """Closest ingested whole target by element-bag transfer distance (L1 diagnostic)."""
        if not self.wholes:
            return None
        bt = element_bag(target)
        ranked = sorted(self.wholes, key=lambda w: bag_distance(bt, w["bag"]))
        best = ranked[0]
        return {"target": best["target"], "distance": round(bag_distance(bt, best["bag"]), 3),
                "source": best.get("source")}

    # -- persistence ---------------------------------------------------------
    def stats(self) -> dict:
        return {"version": LIB_VERSION, "n_unit_shapes": len(self.units),
                "n_unit_seqs": sum(len(v) for v in self.units.values()),
                "n_wholes": len(self.wholes)}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"version": LIB_VERSION, "units": self.units, "wholes": self.wholes}, f)

    @classmethod
    def load(cls, path: str) -> "RetrievalLibrary":
        lib = cls()
        with open(path) as f:
            d = json.load(f)
        lib.units = d.get("units", {})
        lib.wholes = d.get("wholes", [])
        return lib


# --------------------------------------------------------------------------- offline bootstrap
def synthesize_units(shapes: List[str], n_restarts: int = 4, tries: int = 30) -> RetrievalLibrary:
    """One-time offline build: design each canonical unit SHAPE from scratch and store the winners.
    Self-generated (no answers read), so it may seed the library legitimately. Verified per shape."""
    lib = RetrievalLibrary()
    for sub in shapes:
        r = parallel_inverse_fold(sub, n_restarts=n_restarts, tries=tries)
        s = r.get("seq")
        if s and _fold(s) == sub:
            lib._add_unit(sub, s, 0, "synthetic")
    return lib
