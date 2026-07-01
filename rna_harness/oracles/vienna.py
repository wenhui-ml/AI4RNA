"""ViennaRNA-backed oracle.

Confirmed API (ViennaRNA 2.7.2):
    md = RNA.md(); md.temperature=37.0; md.dangles=2; md.noGU=0
    fc = RNA.fold_compound(seq, md)
    struct, mfe   = fc.mfe()
    _, gfe        = fc.pf()                 # ensemble free energy; enables pr/bpp/defect
    p_target      = fc.pr_structure(target) # Boltzmann prob of a given structure
    defect        = fc.ensemble_defect(target)
    RNA.params_load_RNA_Turner2004() / RNA.params_load_RNA_Turner1999()  # global param set

IMPORTANT (verified): ViennaRNA's `params_load_RNA_*()` mutates a process-global C param set
that is STICKY -- the FIRST set loaded in a process wins and later loads are silently
ignored. So this class is only reliable for ONE param set per process (in practice the
default turner2004 primary, which is the first load in any fresh worker). Folding MULTIPLE
distinct param sets in one process silently collapses them to the first -- do NOT do that
in-process; use oracles/isolated.py (spawn subprocess per param set), which is what
cross_oracle_consistency() uses for genuine multi-oracle diversity.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import RNA

from .base import FoldResult, Oracle

_PARAM_LOADERS = {
    "turner2004": RNA.params_load_RNA_Turner2004,   # canonical nearest-neighbour
    "turner1999": RNA.params_load_RNA_Turner1999,   # older Turner set
}

# Machine-learned parameter sets that ship with ViennaRNA. These are *genuinely
# independent* energy models (not just perturbations of Turner), so they give real
# cross-oracle diversity for free -- no external install needed.
if hasattr(RNA, "params_load_RNA_Andronescu2007"):
    _PARAM_LOADERS["andronescu2007"] = RNA.params_load_RNA_Andronescu2007  # BL* model
if hasattr(RNA, "params_load_RNA_Langdon2018"):
    _PARAM_LOADERS["langdon2018"] = RNA.params_load_RNA_Langdon2018  # EternaFold params


class ViennaRNAOracle(Oracle):
    """A ViennaRNA folding model at a fixed configuration.

    Different (param_set, temperature, dangles, noGU) tuples give genuinely
    different energy models; a design that folds correctly across several of them is
    robust rather than overfit to one engine's idiosyncrasies -- this is the cheap
    "multi-oracle cross-validation" seam.
    """

    def __init__(
        self,
        name: str = "vienna_t2004",
        param_set: str = "turner2004",
        temperature: float = 37.0,
        dangles: int = 2,
        no_gu: bool = False,
        no_gu_closure: bool = False,
        max_bp_span: int = 0,
    ) -> None:
        if param_set not in _PARAM_LOADERS:
            raise ValueError(
                f"unknown param_set {param_set!r}; have {sorted(_PARAM_LOADERS)}"
            )
        self.name = name
        self.param_set = param_set
        self.temperature = float(temperature)
        self.dangles = int(dangles)
        self.no_gu = bool(no_gu)
        self.no_gu_closure = bool(no_gu_closure)
        self.max_bp_span = int(max_bp_span)

    # -- internal ---------------------------------------------------------
    def _md(self) -> "RNA.md":
        md = RNA.md()
        md.temperature = self.temperature
        md.dangles = self.dangles
        md.noGU = 1 if self.no_gu else 0
        md.noGUclosure = 1 if self.no_gu_closure else 0
        if self.max_bp_span > 0:
            md.max_bp_span = self.max_bp_span
        return md

    def _new_fc(self, sequence: str) -> "RNA.fold_compound":
        _PARAM_LOADERS[self.param_set]()  # (re)install global param set
        return RNA.fold_compound(sequence.upper().replace("T", "U"), self._md())

    # -- Oracle API -------------------------------------------------------
    def fold_mfe(self, sequence: str) -> Tuple[str, float]:
        fc = self._new_fc(sequence)
        struct, energy = fc.mfe()
        return struct, float(energy)

    def analyze(self, sequence: str, target: Optional[str] = None) -> FoldResult:
        # ViennaRNA does NOT raise on a length-mismatched target; it returns garbage
        # sentinels (pr_structure=0.0, ensemble_defect=-1.0). Fail loudly instead.
        if target is not None and len(target) != len(sequence):
            raise ValueError(
                f"target length {len(target)} != sequence length {len(sequence)}"
            )
        fc = self._new_fc(sequence)
        mfe_struct, mfe_e = fc.mfe()
        # pf() must run after mfe() on the same fc to enable ensemble queries.
        _, gfe = fc.pf()
        res = FoldResult(
            sequence=sequence,
            mfe_structure=mfe_struct,
            mfe_energy=float(mfe_e),
            ensemble_free_energy=float(gfe),
            oracle_name=self.name,
        )
        # p(mfe) as a designability/robustness signal.
        try:
            res.mfe_frequency = float(fc.pr_structure(mfe_struct))
        except Exception:
            res.mfe_frequency = None
        if target is not None:
            try:
                res.prob_of_target = float(fc.pr_structure(target))
            except Exception:
                res.prob_of_target = None
            try:
                ed = float(fc.ensemble_defect(target))
                res.ensemble_defect = ed if 0.0 <= ed <= 1.0 else None  # -1.0 = invalid sentinel
            except Exception:
                res.ensemble_defect = None
        return res

    def bpp_matrix(self, sequence: str):
        """Full base-pair probability matrix (1-indexed tuple-of-tuples)."""
        fc = self._new_fc(sequence)
        fc.mfe()
        fc.pf()
        return fc.bpp()

    def eval_structure(self, sequence: str, structure: str) -> float:
        fc = self._new_fc(sequence)
        return float(fc.eval_structure(structure))

    def subopt_structures(self, sequence: str, delta_dekacal: int = 0):
        """Ground-state structure set within ``delta`` of the MFE.

        NOTE: ViennaRNA's subopt delta is in DEKA-CAL (10*cal/mol), so ``delta=0``
        enumerates exactly the MFE ground states and ``delta=100`` is within 1 kcal/mol.
        """
        fc = self._new_fc(sequence)
        fc.mfe()
        return [(sol.structure, float(sol.energy)) for sol in fc.subopt(int(delta_dekacal))]

    def is_mfe_solved(self, sequence: str, target: str) -> bool:
        """Eterna100 criterion: MFE structure string-equals the target (may be degenerate)."""
        struct, _ = self.fold_mfe(sequence)
        return struct == target

    def is_umfe(self, sequence: str, target: str) -> bool:
        """Stricter: target is the *unique* MFE ground state (no degenerate co-optima)."""
        structs = {s for s, _ in self.subopt_structures(sequence, 0)}
        return (target in structs) and (len(structs) == 1)

    def __repr__(self) -> str:
        return (
            f"ViennaRNAOracle(name={self.name!r}, param_set={self.param_set!r}, "
            f"T={self.temperature}, dangles={self.dangles}, noGU={self.no_gu})"
        )
