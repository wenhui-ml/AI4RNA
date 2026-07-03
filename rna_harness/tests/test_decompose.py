"""Decomposition of a target into independent stem-loop units + seam (P1a)."""
from rna_harness.mcp.motif_kb import decompose_structure


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
