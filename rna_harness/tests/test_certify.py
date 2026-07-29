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


@pytest.mark.skipif(not _panel_ready(), reason="needs pKiss + RNAstructure")
def test_knot_confirmed_is_a_weaker_claim_than_certified_and_says_so():
    """The calibrated verdict. Exact-and-unanimous has 0.8% sensitivity on known natural structures, so it
    refuses nearly everything including correct designs; crossing-stem recall by >=1 oracle reaches 84.0% at
    0.4% false accept per draw, and 3.1% [0.1, 16.2] on a matched best-of-pool null. It is reported as a
    claim about the KNOT only -- composing it with an exact nested-skeleton check was refuted (ViennaRNA
    recovers the nested reduction from 20 of 186 natives)."""
    from rna_harness.cognitive.certify import KNOT_OPERATING_POINT, _knot_recall
    assert KNOT_OPERATING_POINT["sensitivity_on_known_structures"] > 0.8
    assert "CROSSING STEMS only" in KNOT_OPERATING_POINT["scope"]
    kr = _knot_recall("G" * len(PK), PK)
    assert set(kr) >= {"per_oracle", "confirmed_by", "knot_confirmed", "n_cross"}
    assert kr["n_cross"] > 0                       # PK really does have crossing pairs to recall


# ---- rendering for someone who does not read dot-bracket -----------------------------------------
def test_hairpin_is_drawn_with_the_right_bases_in_the_right_places():
    from rna_harness.cognitive.certify import sketch
    art = sketch("((((((....))))))", "GCCGGCGAAAGCCGGC")
    assert art
    lines = art.split("\n")
    assert lines[0].split() == ["G", "A", "A", "A"], "the loop bases, in order"
    # innermost pair first: seq[5]='C' with seq[10]='G'
    assert lines[2].startswith("      C") and lines[2].rstrip().endswith("G")
    # outermost pair last, just above the 5'/3' line
    assert lines[-2].startswith("      G") and lines[-2].rstrip().endswith("C")
    assert lines[-1].strip().startswith("5'") and lines[-1].strip().endswith("3'")
    left = lines[-1].index("\u256f")
    assert lines[-2].index("G") == left, "the 5' corner must sit under the stem column"


def test_shapes_it_cannot_draw_honestly_return_nothing():
    """Guessing a layout for a shape this cannot draw would misinform; the caller falls back to words."""
    from rna_harness.cognitive.certify import sketch
    assert sketch("((((....)))).....(((.....)))") is None      # two hairpins
    assert sketch("((((..(((....))).))))") is None             # bulge
    assert sketch("((((....[[[[))))....]]]]") is None          # pseudoknot
    assert sketch("....") is None                              # nothing paired


def test_a_mismatched_sequence_is_ignored_rather_than_drawn_wrong():
    from rna_harness.cognitive.certify import sketch
    art = sketch("((((((....))))))", "ACGU")                   # too short
    assert art and "\u25cf" in art and "A" not in art.split("\n")[0]


def test_display_speaks_chinese_and_drops_the_energy_model_jargon():
    from rna_harness.cognitive.certify import certify_sequence, to_display
    d = to_display(certify_sequence("GGGGAAAACCCC", "((((....))))"), "zh")
    assert "turner2004" not in d["checked_by"] and "dangles" not in d["checked_by"]
    assert "turner2004" in d["checked_by_raw"], "the exact configuration stays available"
    assert d["plain"] and "碱基" in d["plain"]
    assert d["criterion"]["means"] and not d["criterion"]["means"].startswith("而且")


# ---- the audit block: a certificate that cannot be reproduced is not a record --------------------
def test_every_certificate_carries_provenance_and_a_timestamp():
    from rna_harness.cognitive.certify import SCHEMA_VERSION, certify_sequence
    c = certify_sequence("GGGGAAAACCCC", HAIRPIN)
    assert c.provenance.get("schema") == SCHEMA_VERSION
    assert c.provenance.get("viennarna"), "the verdict depends on it, so it belongs in the answer"
    assert c.generated_utc.endswith("Z")


def test_version_probe_never_records_an_error_message_as_a_version():
    """pKiss does not understand --version; recording its complaint as the version would be worse than
    recording the path."""
    from rna_harness.cognitive.certify import provenance
    for k, v in provenance().items():
        assert "unknown option" not in str(v).lower()
        assert "usage:" not in str(v).lower()


def test_the_two_commands_promise_different_things_and_say_so():
    """MEASURED: the same seed does NOT give the same design -- DesiRNA's budget is wall-clock, not
    iterations, so two runs at seed 3 gave GGCCGUUGGGCC and GCUCGUAGGAGC. A single `reproduce` field would
    have been a promise this cannot keep. What IS exactly reproducible is the claim the certificate makes
    about the sequence it names, because folding is deterministic."""
    from rna_harness.cognitive.certify import certify_sequence
    c = certify_sequence("GGGGAAAACCCC", HAIRPIN)
    assert "--sequence" in c.verify_cmd, "the deterministic command re-checks THIS sequence"
    assert "--sequence" not in c.rerun_search, "the stochastic command re-runs the search"
    assert "wall-clock" in c.reproducibility and "not seed-reproducible" in c.reproducibility


def test_verifying_the_same_sequence_twice_gives_the_same_verdict():
    from rna_harness.cognitive.certify import certify_sequence
    a = certify_sequence("GGGGAAAACCCC", HAIRPIN)
    b = certify_sequence("GGGGAAAACCCC", HAIRPIN)
    assert (a.verdict, a.strictness, a.rungs_passed) == (b.verdict, b.strictness, b.rungs_passed)


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
