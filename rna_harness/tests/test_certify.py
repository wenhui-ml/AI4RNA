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
    """The calibrated verdict, and the numbers pinned here are the HELD-OUT ones.

    Exact-and-unanimous has 0.8% sensitivity on known natural structures, so it refuses nearly everything
    including correct designs. Crossing-stem recall by >=1 oracle reaches 78.5% [68.8, 86.3] on a held-out
    half, and -- the null that actually matters -- accepted 0 of 183 decoys designed for the target's
    PSEUDOKNOT-FREE skeleton, which get the easy half right by construction. An earlier 84.0% was in-sample
    on a different denominator (250 natives vs the 186 whose sequence can be recovered) and is not quotable;
    this test exists so it cannot drift back in. Composing the rule with an exact nested-skeleton check was
    refuted (ViennaRNA recovers the nested reduction from 20 of 186 natives)."""
    from rna_harness.cognitive.certify import KNOT_OPERATING_POINT, ORACLE_KAPPA, _knot_recall
    op = KNOT_OPERATING_POINT
    assert op["sensitivity_on_known_structures"] == 0.785, "held out, not the in-sample 0.840"
    lo, hi = op["sensitivity_ci95"]
    assert lo < op["sensitivity_on_known_structures"] < hi, "a rate without its interval is not a result"
    assert op["false_accept_nested_decoys"] == 0.0 and op["false_accept_nested_ci95"][1] < 0.05
    assert "CROSSING STEMS only" in op["scope"]
    # "three independent predictors" is not what the data says, and the certificate must not imply it
    assert ORACLE_KAPPA["probknot~shapeknots"] > 0.7 > ORACLE_KAPPA["pkiss~probknot"]
    assert ORACLE_KAPPA["effective_independent_sources"] == 2
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


def test_calibrated_rule_is_restricted_to_the_topologies_it_was_validated_on():
    """The crossing-stem rule may not be used on H-type, and the reason is an outside measurement.

    A fourth predictor that never entered the pipeline (RNAPKplex) backs 60.8% of our kissing designs against
    its own 54.8% [38.7, 70.2] ceiling on kissing natives -- inside the interval. On H-type it backs 29.9%
    against a 71.1% [62.7, 78.6] ceiling, and a paired comparison over the same 135 targets prefers the
    NATIVE sequence 65 times to our 20 (p=1.03e-06). Raising k does not fix it (29.9% -> 26.5% at k>=2) and a
    global energy margin is refuted because it separates the two families in OPPOSITE directions. So the fix
    is a restriction, not a patch, and this test is what stops the restriction quietly eroding.
    """
    from rna_harness.cognitive.certify import (CALIBRATED_RULE_TOPOLOGIES, TOPOLOGY_OPERATING_POINTS,
                                               _topology_point, calibrated_rule_applies)
    H = ".((((..[[[)))).....]]]..."                       # one crossing, second helix inside then outside
    KISS = ".((((....[[[[[[))))......].]]]]]."             # two crossings
    assert not calibrated_rule_applies(H), "H-type must fall back to the strict panel"
    assert calibrated_rule_applies(KISS)
    assert "H-type" not in CALIBRATED_RULE_TOPOLOGIES
    # the certificate must carry the operating point for THIS family, never a blended average
    p = _topology_point(H)
    assert p["topology"] == "H-type" and "panel-unanimous" in p["rule"]
    assert p["external_agreement"] > 0.9 and p["external_native_ceiling"] < p["external_agreement"]
    k = _topology_point(KISS)
    lo, hi = k["external_ceiling_ci95"]
    assert lo <= k["external_agreement"] <= hi, "kissing passes because it sits inside the outsider's own CI"
    # an unmeasured family must say so rather than borrow someone else's numbers
    assert "note" in _topology_point("((((....))))")
    assert set(TOPOLOGY_OPERATING_POINTS) >= {"kissing / chain", "H-type"}


def test_swap_isomer_preserves_the_base_multiset_exactly():
    """The isomer null is only a control BECAUSE composition is preserved by construction.

    Every earlier composition control in this project was a SELECTION -- a GC band, then GC matched per
    target -- and each one leaked: the band left a 3.2-point gap inside it, and the per-target matching left
    only 29 usable strata. Transposing the two partners of a target pair cannot change the multiset at all,
    which is what makes this null able to say something the others could not.

    If a refactor ever breaks that invariant, the null silently becomes a composition comparison again and
    would still produce a plausible-looking number. This test is the thing standing between those two.
    """
    import random
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs" / "ceiling"))
    from kinetic_nulls import swap_isomer
    from rna_harness.benchmarks.kinetic_suite import gc_percent, pair_set

    seq = "GGGCAUAGCUUCGGCUAUGCCC"
    tgt = "((((((((....)))))))).."
    pairs = sorted(pair_set(tgt))
    rng = random.Random(7)
    changed = 0
    for _ in range(40):
        iso = swap_isomer(seq, pairs, rng, 0.6)
        assert sorted(iso) == sorted(seq), "transposition changed the multiset -- the control is void"
        assert gc_percent(iso) == gc_percent(seq)
        assert len(iso) == len(seq)
        changed += iso != seq
    assert changed > 0, "a control that never varies the arrangement tests nothing"


def test_ensemble_entropy_tiebreaker_direction_and_failure_mode():
    """The tie-breaker must prefer the LESS determined fold, and must never let an unevaluable sequence win.

    Our designs fold about three times more definitely than natural pseudoknots (mean positional entropy
    0.087 vs 0.299), which is why a probability-based judge has no mass left for the crossing pairs. The
    tie-break therefore takes the HIGHER entropy among equal-grade candidates. Two ways this could break
    silently: the comparison could face the wrong way, or a sequence the partition function chokes on could
    come back as the winner.
    """
    from rna_harness.cognitive.certify import _ensemble_entropy
    structured = _ensemble_entropy("GGCGCGGCACCGUCCGCGGAACAAACGG")
    unpaired = _ensemble_entropy("A" * 28)
    assert structured > unpaired >= 0.0, "a sequence with competing pairings must score above one with none"
    assert _ensemble_entropy("") == float("-inf"), "an unevaluable sequence must never displace an incumbent"


def test_intervention_numbers_are_wired_and_monotone():
    """The certificate must carry the error rate measured against the MOLECULE, and carry its threshold too.

    `false_accept_on_random_sequences` = 0.4% is measured against random sequences, a null so weak that
    surviving it means very little. The mutate-map-rescue figure asks the same question of pairs a real
    molecule declined to form. Four ways this could silently break: the field stops being emitted; it stops
    moving with the grade (the whole point is that it does, unlike the random-sequence number); the
    threshold-conditional point estimate gets quoted without its sweep, which an earlier version of this
    block did; or the per-design ladder claim gets restated at the strength of the per-pair one, which is a
    different and much stronger measurement.
    """
    from rna_harness.cognitive.certify import knot_grade, _INTERVENTION
    rates = []
    for k in (1, 2, 3):
        g = knot_grade({"confirmed_by": ["pkiss", "probknot", "shapeknots"][:k]})
        assert g["intervention"]["assay"].startswith("mutate-map-rescue")
        assert g["intervention"]["n_crossing_pairs"] > 0
        rates.append(g["pair_absent_rate_vs_reality"])
    assert rates == sorted(rates, reverse=True), (
        f"more agreeing oracles must mean fewer pairs the molecule refuses to form, got {rates}")
    assert rates[-1] > 0.004 * 10, (
        "the reality-facing rate is an order of magnitude above the random-sequence rate; if it has "
        "collapsed to the same scale, the wrong number is being quoted")

    fa = _INTERVENTION["false_accept_vs_reality_at_k3"]
    lo, hi = fa["ci95_cluster_bootstrap"]
    assert lo < fa["value"] < hi
    sweep = _INTERVENTION["pair_absent_threshold_sweep_k3"]
    assert sweep[fa["threshold"]] == fa["value"], "the headline number must be the sweep entry at its own cut"
    assert max(sweep.values()) > 3 * min(sweep.values()), (
        "the sweep exists to show the point estimate is threshold-driven; if it has flattened, the "
        "warning attached to the number is no longer true")

    tf = _INTERVENTION["per_pair_ordering_threshold_free"]
    lad = _INTERVENTION["certificate_ladder_ordered_in_vitro"]
    assert tf["p"] < lad["p"], (
        "the per-pair result is measured on more clusters than the per-design one and should be the "
        "stronger of the two; if they have swapped, one is being reported at the other's strength")
    assert tf["p"] > 1e-10, (
        "a p below 1e-10 on 74 clusters is the signature of the pseudo-replication this field was "
        "corrected for -- pooled-pair statistics must never come back into this slot")
    assert "pooled_mannwhitney_RETRACTED" in tf, "the retraction must stay visible, not be tidied away"
    assert lad["pre_registered"] is False, "the per-design test was post-hoc and must keep saying so"

    assert _INTERVENTION["negative_control"]["median_rescue"] < \
        _INTERVENTION["negative_control"]["vs_complementary"], \
        "the negative control must sit BELOW real pairs, or the assay proves nothing"
    assert "52-100 nt" in _INTERVENTION["scope"], "the shipped scope once said 52 nt; it is 52-100"
