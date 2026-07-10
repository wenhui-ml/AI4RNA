"""Decomposition of a target into independent stem-loop units + seam (P1a)."""
from rna_harness.mcp.motif_kb import analyze_structure, decompose_structure
from rna_harness.mcp.motif_server import _format_target_analysis


def _balanced(db: str) -> bool:
    d = 0
    for c in db:
        d += (c == "(") - (c == ")")
        if d < 0:
            return False
    return d == 0


def test_coverage_units_plus_seam_equals_n():
    db = "((((....))))...((((....))))"  # two hairpins on an external loop
    d = decompose_structure(db)
    assert sum(u["len"] for u in d["units"]) + d["n_seam_positions"] == len(db)


def test_multiloop_splits_into_branches():
    # a closing stem enclosing a 3-branch multiloop -> 3 short units, 1 junction
    db = "(((" + "((((....))))" * 3 + ")))"
    d = decompose_structure(db)
    assert d["n_units"] == 3
    assert d["n_junctions"] == 1
    assert max(u["len"] for u in d["units"]) < len(db)          # actually broken up
    assert all(_balanced(u["sub_target"]) for u in d["units"])   # each unit is a valid structure


def test_simple_hairpin_is_one_unit():
    d = decompose_structure("((((....))))")
    assert d["n_units"] == 1 and d["n_junctions"] == 0


def test_buried_multiloop_is_found_through_the_stem():
    # multiloop sits UNDER a long stem; the walk must descend the stem to find it
    db = "((((((" + "((((....))))" * 2 + "))))))"
    d = decompose_structure(db)
    assert d["n_units"] == 2          # not 1: the stem is descended, multiloop found
    assert max(u["len"] for u in d["units"]) < len(db)


def test_analyze_target_surfaces_decomposition_for_compositional_target():
    # REGRESSION: general_advice tells the agent to iterate `decomposition.units`, so the
    # formatted analyze_target output MUST actually contain the units + their sub_targets.
    db = "(((" + "((((....))))" * 3 + ")))"   # multiloop: 3 units + 1 junction
    text = _format_target_analysis(analyze_structure(db))
    assert "decomposition plan" in text
    assert "3 design unit" in text
    assert "unit[0]" in text and "unit[2]" in text
    assert "sub_target=" in text            # the per-unit target the agent must design
    assert "junction closing_pair@" in text


def test_analyze_target_omits_decomposition_for_simple_hairpin():
    # a single stem-loop has nothing to decompose -> keep the output clean (no unit noise)
    text = _format_target_analysis(analyze_structure("((((....))))"))
    assert "decomposition plan" not in text
