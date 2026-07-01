"""Hybrid-constraint specification + checkable predicates.

Every predicate is exact and deterministic (this is what makes the reward honest). A
constraint is either satisfied or not; there is no learned/approximate constraint here.

Supported constraints (the "hybrid" in hybrid-constraint design):
  * target secondary structure (checked by the oracle, not here)
  * GC-content window
  * forbidden subsequences (e.g. restriction sites, homopolymer runs)
  * fixed motifs at fixed positions (IUPAC-aware)
  * coding constraint: sequence must translate to a given amino-acid string in a frame
  * negative design: named competitor structures the design should avoid
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .structure_utils import find_motif, _iupac_match

# Standard genetic code over RNA codons -> single-letter amino acid ('*' = stop).
CODON_TABLE: Dict[str, str] = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate(seq: str, frame: int = 0) -> str:
    s = seq.upper().replace("T", "U")
    out = []
    for k in range(frame, len(s) - 2, 3):
        out.append(CODON_TABLE.get(s[k : k + 3], "X"))
    return "".join(out)


@dataclass
class ConstraintSpec:
    """A bundle of hybrid constraints. All fields optional; empty = pure inverse folding."""

    gc_min: Optional[float] = None
    gc_max: Optional[float] = None
    forbidden: List[str] = field(default_factory=list)          # substrings that must NOT appear
    motifs: List[Tuple[int, str]] = field(default_factory=list)  # (0-based pos, IUPAC literal)
    iupac_mask: Optional[str] = None                            # per-position IUPAC allowed set (len n)
    coding_aa: Optional[str] = None                            # required translation
    coding_frame: int = 0
    competitors: List[str] = field(default_factory=list)        # dot-bracket structures to avoid
    allow_gu: bool = True

    def is_empty(self) -> bool:
        return (
            self.gc_min is None
            and self.gc_max is None
            and not self.forbidden
            and not self.motifs
            and self.iupac_mask is None
            and self.coding_aa is None
            and not self.competitors
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "gc_min": self.gc_min,
            "gc_max": self.gc_max,
            "forbidden": list(self.forbidden),
            "motifs": [[p, m] for p, m in self.motifs],
            "iupac_mask": self.iupac_mask,
            "coding_aa": self.coding_aa,
            "coding_frame": self.coding_frame,
            "competitors": list(self.competitors),
            "allow_gu": self.allow_gu,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, object]]) -> "ConstraintSpec":
        if not d:
            return cls()
        return cls(
            gc_min=d.get("gc_min"),
            gc_max=d.get("gc_max"),
            forbidden=list(d.get("forbidden", []) or []),
            motifs=[(int(p), str(m)) for p, m in (d.get("motifs", []) or [])],
            iupac_mask=d.get("iupac_mask"),
            coding_aa=d.get("coding_aa"),
            coding_frame=int(d.get("coding_frame", 0) or 0),
            competitors=list(d.get("competitors", []) or []),
            allow_gu=bool(d.get("allow_gu", True)),
        )


@dataclass
class ConstraintReport:
    checks: Dict[str, bool]
    details: Dict[str, object]

    @property
    def all_satisfied(self) -> bool:
        return all(self.checks.values())

    @property
    def satisfied_fraction(self) -> float:
        if not self.checks:
            return 1.0
        return sum(self.checks.values()) / len(self.checks)

    def to_dict(self) -> Dict[str, object]:
        return {
            "all_satisfied": self.all_satisfied,
            "satisfied_fraction": round(self.satisfied_fraction, 4),
            "checks": self.checks,
            "details": self.details,
        }


def check_constraints(sequence: str, spec: ConstraintSpec) -> ConstraintReport:
    """Evaluate all sequence-level constraints (structure is checked separately by the oracle)."""
    seq = sequence.upper().replace("T", "U")
    checks: Dict[str, bool] = {}
    details: Dict[str, object] = {}

    if spec.gc_min is not None or spec.gc_max is not None:
        gc = sum(c in "GC" for c in seq) / max(len(seq), 1)
        lo = spec.gc_min if spec.gc_min is not None else 0.0
        hi = spec.gc_max if spec.gc_max is not None else 1.0
        checks["gc_window"] = lo <= gc <= hi
        details["gc_content"] = round(gc, 4)
        details["gc_window"] = [lo, hi]

    if spec.forbidden:
        # IUPAC-aware: a forbidden pattern may contain ambiguity codes (N, R, Y, ...).
        bad = [m for m in spec.forbidden if find_motif(seq, m.upper().replace("T", "U")) >= 0]
        checks["no_forbidden"] = not bad
        details["forbidden_found"] = bad

    if spec.motifs:
        ok = True
        misplaced = []
        for pos, motif in spec.motifs:
            m = motif.upper().replace("T", "U")
            seg = seq[pos : pos + len(m)]
            good = len(seg) == len(m) and all(_iupac_match(seg[k], m[k]) for k in range(len(m)))
            if not good:
                ok = False
                misplaced.append({"pos": pos, "motif": motif, "found": seg})
        checks["fixed_motifs"] = ok
        details["motif_violations"] = misplaced

    if spec.iupac_mask is not None:
        mask = spec.iupac_mask.upper()
        ok = len(mask) == len(seq) and all(_iupac_match(seq[i], mask[i]) for i in range(len(seq)))
        checks["iupac_mask"] = ok
        if not ok:
            details["iupac_violations"] = [
                i for i in range(min(len(mask), len(seq))) if not _iupac_match(seq[i], mask[i])
            ]

    if spec.coding_aa is not None:
        aa = spec.coding_aa.upper()
        got = translate(seq, spec.coding_frame)[: len(aa)]
        checks["coding"] = got == aa
        details["coding_expected"] = aa
        details["coding_got"] = got

    return ConstraintReport(checks=checks, details=details)
