"""The multi-oracle panel.

The *primary* oracle defines the canonical, harness-controlled reward. Its configuration
(turner2004, 37 C, dangles=2) matches installed ViennaRNA 2.7.2 and therefore the
Eterna100-V2 benchmark semantics (see IMPLEMENTATION_SPEC.md).

The rest of the panel are genuinely different energy models (turner1999, andronescu2007,
and langdon2018 == EternaFold parameters -- all shipped with ViennaRNA, no external
install) plus temperature perturbations. A design that solves the target across many of
them is *robust* rather than overfit to one engine's idiosyncrasies -- this is the cheap
"multi-oracle cross-validation" seam the report calls for. Cross-oracle agreement is a
robustness signal, NOT the official reward: different energy models can legitimately
disagree on the MFE, so only the primary decides "solved" for scoring.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from .vienna import ViennaRNAOracle, _PARAM_LOADERS


def primary_oracle() -> ViennaRNAOracle:
    """Canonical oracle = installed 2.7.2 defaults = Eterna100-V2 semantics."""
    return ViennaRNAOracle(
        name="vienna_t2004_37_d2",
        param_set="turner2004",
        temperature=37.0,
        dangles=2,
    )


def _maybe(name, **kw) -> List[ViennaRNAOracle]:
    """Include an oracle only if its param set is available in this ViennaRNA build."""
    ps = kw.get("param_set", "turner2004")
    return [ViennaRNAOracle(name=name, **kw)] if ps in _PARAM_LOADERS else []


def build_panel(kind: str = "default") -> List[ViennaRNAOracle]:
    """Return a list of oracles. The first element is always the primary."""
    prim = primary_oracle()
    if kind == "primary":
        return [prim]
    if kind == "default":
        panel = [prim]
        panel += _maybe("vienna_t1999_37_d2", param_set="turner1999", temperature=37.0, dangles=2)
        panel += _maybe("vienna_andronescu2007_37_d2", param_set="andronescu2007", temperature=37.0, dangles=2)
        panel += _maybe("vienna_eternafold_37_d2", param_set="langdon2018", temperature=37.0, dangles=2)
        return panel
    if kind == "robust":
        # default + temperature perturbations on the primary energy model
        panel = build_panel("default")
        panel.append(ViennaRNAOracle(name="vienna_t2004_25_d2", param_set="turner2004", temperature=25.0, dangles=2))
        panel.append(ViennaRNAOracle(name="vienna_t2004_45_d2", param_set="turner2004", temperature=45.0, dangles=2))
        panel.append(ViennaRNAOracle(name="vienna_t2004_37_d0", param_set="turner2004", temperature=37.0, dangles=0))
        return panel
    raise ValueError(f"unknown panel kind {kind!r}; have {sorted(PANELS)}")


PANELS = {"primary", "default", "robust"}


def cross_oracle_consistency(
    sequence: str, target: str, panel: Sequence[ViennaRNAOracle]
) -> Dict[str, object]:
    """Fraction of the panel whose MFE structure equals the target exactly.

    Returns per-oracle solve flags + the aggregate fraction. This is a robustness /
    over-fitting signal: a design that only the primary solves is fragile.
    """
    per_oracle = {}
    n_solved = 0
    for orc in panel:
        struct, energy = orc.fold_mfe(sequence)
        ok = struct == target
        per_oracle[orc.name] = {"solved": ok, "mfe": struct, "energy": round(energy, 3)}
        n_solved += int(ok)
    frac = n_solved / len(panel) if panel else 0.0
    return {"fraction": frac, "n_solved": n_solved, "n_oracles": len(panel), "per_oracle": per_oracle}
