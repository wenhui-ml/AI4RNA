"""Tests for the intent layer.

The design rule under test is that the model never does arithmetic: it emits element types and integer
lengths, and deterministic code assembles the dot-bracket. So most of these tests need no model at all --
which is the point. The model's half is exercised with a stub, so the parse/validate path is covered without
a network call or an API key.
"""
from __future__ import annotations

import pytest

from rna_harness.cognitive.intent import (MIN_LOOP, MIN_STEM, Spec, assemble, design_from_request,
                                          translate)


class _Stub:
    """Returns a fixed payload, as a well-behaved (or misbehaving) model would."""

    def __init__(self, payload, error=None):
        self.payload, self.error = payload, error

    def chat(self, messages, **kw):
        class R:
            pass
        r = R()
        r.text, r.error = self.payload, self.error
        return r


GOOD = ('{"elements":[{"type":"hairpin","stem":6,"loop":4}],"budget":"cheap",'
        '"interpretation":"a single hairpin: 6 base pairs closing a 4-nucleotide loop"}')


# ---- the deterministic half: exact, and the model is not involved -------------------------------
def test_hairpin_is_assembled_exactly():
    db, err = assemble(Spec(elements=[{"type": "hairpin", "stem": 6, "loop": 4}]))
    assert err is None
    assert db == "((((((....))))))"
    assert db.count("(") == db.count(")") == 6


def test_chain_of_hairpins_with_a_spacer():
    db, err = assemble(Spec(elements=[{"type": "hairpin", "stem": 4, "loop": 4},
                                      {"type": "spacer", "len": 5},
                                      {"type": "hairpin", "stem": 3, "loop": 5}]))
    assert err is None
    assert db == "((((....)))).....(((.....)))"


def test_internal_loop_is_asymmetric_and_balanced():
    db, err = assemble(Spec(elements=[{"type": "internal_loop", "outer_stem": 4, "left": 2,
                                       "right": 1, "inner_stem": 3, "loop": 4}]))
    assert err is None
    assert db == "((((..(((....))).))))"
    assert db.count("(") == db.count(")")


@pytest.mark.parametrize("el,why", [
    ({"type": "hairpin", "stem": 6, "loop": 2}, "loop below the closing minimum"),
    ({"type": "hairpin", "stem": 1, "loop": 4}, "a lonely pair"),
    ({"type": "nonsense"}, "unknown element"),
    ({"type": "hairpin", "stem": 6}, "missing field"),
])
def test_physically_impossible_requests_are_refused_with_a_reason(el, why):
    """Domain constraints the caller does not know are enforced here, not discovered as a mystery failure."""
    db, err = assemble(Spec(elements=[el]))
    assert db is None and err, why
    assert len(err) > 20, "the refusal has to explain itself"


def test_out_of_scope_shapes_say_so_instead_of_guessing():
    for t in ("multiloop", "pseudoknot", "kissing"):
        db, err = assemble(Spec(elements=[{"type": t}]))
        assert db is None
        assert "dot-bracket directly" in err


def test_minimums_are_the_physical_ones():
    assert MIN_LOOP == 3          # a hairpin loop shorter than 3 nt cannot close
    assert MIN_STEM == 2          # a 1-bp stem is a lonely pair


# ---- the model's half, stubbed --------------------------------------------------------------------
def test_dot_bracket_input_never_calls_a_model():
    sp = translate("((((....))))", client=None)      # client=None would construct a real one if reached
    assert sp.raw_target == "((((....))))"
    assert sp.error is None


def test_json_is_recovered_from_a_chatty_reply():
    sp = translate("x", client=_Stub("Sure! ```json\n" + GOOD + "\n``` hope that helps"))
    assert sp.error is None and len(sp.elements) == 1 and sp.budget == "cheap"


@pytest.mark.parametrize("payload", ["I think you want a hairpin.", '{"elements": [oops]}'])
def test_a_model_that_does_not_return_a_spec_is_an_error_not_a_guess(payload):
    sp = translate("x", client=_Stub(payload))
    assert sp.error and not sp.elements


def test_model_failure_is_reported_not_swallowed():
    sp = translate("x", client=_Stub("", error="HTTP 401"))
    assert sp.error and "401" in sp.error


def test_dry_run_shows_the_interpretation_without_spending():
    sp, target, cert = design_from_request("anything", dry_run=True, client=_Stub(GOOD))
    assert target == "((((((....))))))"
    assert cert is None, "dry run must not spend"
    assert sp.interpretation, "the caller has to be able to check the model understood them"


def test_the_model_cannot_smuggle_an_invalid_structure_past_the_validator():
    bad = '{"elements":[{"type":"hairpin","stem":3,"loop":1}],"interpretation":"tiny loop"}'
    sp, target, cert = design_from_request("x", client=_Stub(bad))
    assert cert is None and target is None
    assert str(MIN_LOOP) in sp.error
