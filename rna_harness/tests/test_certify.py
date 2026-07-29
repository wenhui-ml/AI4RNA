"""Acceptance tests for the user-facing certificate.

The product promise is not "finds good sequences" -- it is "never hands a non-expert an answer it cannot
stand behind". So most of these are NEGATIVE tests, and the counter-examples are real: the population where
the single reference oracle accepts a sequence and the independent panel rejects it measured 43 of 43 false
positives on the crossing benchmark.
"""
from __future__ import annotations

import pytest

from rna_harness.cognitive.certify import (BUDGETS, Certificate, certify_design, certify_sequence,
                                           classify)

HAIRPIN = "((((....))))"
PK = "((((....[[[[))))....]]]]"


# ---- input handling: a non-expert will paste something malformed ---------------------------------
@pytest.mark.parametrize("bad,why", [
    ("", "empty"),
    ("((((....)))", "unbalanced"),
    ("AUGCAUGC", "not dot-bracket"),
    ("((((....))))]", "unbalanced bracket family"),
])
def test_malformed_target_is_refused_not_computed(bad, why):
    c = certify_design(bad)
    assert c.verdict == "invalid_input", why
    assert c.sequence is None
    assert c.cost_usd == 0.0                      # refused before spending anything


def test_classify_routes_by_pseudoknot():
    assert classify(HAIRPIN)[0] == "nested"
    assert classify(PK)[0] == "pseudoknot"


def test_sequence_length_mismatch_is_invalid_not_unsolved():
    c = certify_sequence("ACGU", HAIRPIN)
    assert c.verdict == "invalid_input"           # not 'not_found': we cannot judge it at all


def test_non_acgu_sequence_is_refused():
    assert certify_sequence("X" * len(HAIRPIN), HAIRPIN).verdict == "invalid_input"


# ---- the nested path: exact oracle, so every rung the answer passes is reported ------------------
def test_nested_certificate_reports_every_strictness_rung():
    """`match`, `umfe` and `pfe` are different questions and a non-expert cannot know which they meant.
    Archived survival on Eterna is 1.000 / 0.902 / 0.066, so reporting only `match` answers the easiest one."""
    import RNA
    seq = "GGGGAAAACCCC"
    assert RNA.fold(seq)[0] == HAIRPIN, "test fixture must actually fold to the target"
    c = certify_sequence(seq, HAIRPIN)
    assert c.verdict == "certified"
    assert c.rungs_passed[0] == "match"                  # match is a precondition for the stricter rungs
    assert c.strictness == c.rungs_passed[-1]            # the reported rung is the STRICTEST one passed
    assert set(c.rungs_passed) <= {"match", "umfe", "pfe"}


def test_nested_wrong_sequence_is_not_found_not_cannot_certify():
    """On a nested target the oracle is exact and unique, so `cannot_certify` must NOT arise -- either the
    fold equals the target or it does not. Conflating the two would hide the pseudoknot signal."""
    c = certify_sequence("A" * len(HAIRPIN), HAIRPIN)
    assert c.verdict == "not_found"
    assert c.checked_by.startswith("ViennaRNA")


# ---- the pseudoknot path: the whole reason the third verdict exists ------------------------------
def _panel_ready():
    try:
        from rna_harness.benchmarks.rnainvbench import pkiss_available
        from rna_harness.oracles_pk import rnastructure_available
        return pkiss_available() and rnastructure_available()
    except Exception:                                                     # noqa: BLE001
        return False


@pytest.mark.skipif(not _panel_ready(), reason="needs pKiss + RNAstructure")
def test_reference_only_sequence_is_never_certified():
    """THE headline negative test. A sequence the reference oracle accepts and the independent predictors
    reject must come back `cannot_certify`, never `certified`. That population was 43/43 false positives."""
    import json
    import os
    root = os.path.join(os.path.dirname(__file__), "..", "runs", "ceiling", "agent")
    pk = os.path.join(root, "pk_desirna.json")
    pan = os.path.join(root, "desirna_panel_pool.json")
    if not (os.path.exists(pk) and os.path.exists(pan)):
        pytest.skip("archived crossing rows not present")
    rows = {int(r["task_id"].split(":")[1]): r for r in json.load(open(pk))["rows"]}
    panel = {int(r["task_id"].split(":")[1]): r for r in json.load(open(pan))["rows"]}
    fp = [i for i in panel
          if panel[i].get("forms_T_pkiss")
          and not (panel[i].get("forms_T_probknot") and panel[i].get("forms_T_shapeknots"))]
    assert fp, "expected a non-empty reference-only population in the archive"
    for i in fp[:3]:
        c = certify_sequence(rows[i]["best_pk_seq"], rows[i]["target"])
        assert c.verdict != "certified", f"issued a false certificate for task {i}"


@pytest.mark.skipif(_panel_ready(), reason="only meaningful when the panel is unavailable")
def test_pseudoknot_without_panel_refuses_rather_than_guessing():
    c = certify_design(PK, budget="cheap")
    assert c.verdict == "cannot_certify"
    assert "panel" in c.reason.lower()


# ---- contract shape -------------------------------------------------------------------------------
def test_every_certificate_carries_the_disclaimer_and_a_reason():
    for c in (certify_design(""), certify_sequence("A" * len(HAIRPIN), HAIRPIN)):
        assert isinstance(c, Certificate)
        assert c.reason, "a non-expert needs to be told WHY"
        assert "in vivo" in c.disclaimer            # the red line travels with every answer
        d = c.to_dict()
        assert set(("verdict", "checked_by", "cost_usd", "wait_s", "disclaimer")) <= set(d)


def test_budget_names_are_closed_and_ordered():
    assert set(BUDGETS) == {"cheap", "standard", "best_effort"}
    assert BUDGETS["cheap"][0] == 5
    for name in BUDGETS:
        assert list(BUDGETS[name]) == sorted(BUDGETS[name])
    assert certify_design(HAIRPIN, budget="nonsense").verdict == "invalid_input"
