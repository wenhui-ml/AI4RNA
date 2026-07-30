"""Tests for the agent loop.

The loop's value is that it acts on REAL observations, so the primitives are tested for whether their
observations are actionable -- a message that does not tell the caller what to do next costs a step, which
happened during development and is asserted against here. The model's half runs on a stub: no network, no key.
"""
from __future__ import annotations

import json

import pytest

from rna_harness.cognitive.agent import (MAX_STEPS, check_feasible, place_constraint, propose_structure,
                                         render, run)

HAIRPIN6 = "((((((....))))))"


class _Scripted:
    """Replays a fixed list of actions, so a whole trajectory is testable without a model."""

    def __init__(self, actions):
        self.actions, self.i = list(actions), 0

    def chat(self, messages, **kw):
        class R:
            pass
        r = R()
        r.error = None
        r.text = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return r


# ---- primitives: exact, and their observations must be actionable --------------------------------
def test_structure_comes_from_elements_never_from_a_written_string():
    o = propose_structure([{"type": "hairpin", "stem": 6, "loop": 4}])
    assert o["ok"] and o["target"] == HAIRPIN6 and o["n"] == 16
    assert o["plain"], "the caller is told what the shape IS, not only its bracket string"


def test_impossible_element_is_refused_with_a_reason():
    o = propose_structure([{"type": "hairpin", "stem": 6, "loop": 1}])
    assert not o["ok"] and "3" in o["error"]


def test_constraints_are_additive_and_disagreements_are_refused():
    a = place_constraint(HAIRPIN6, "GGGG", 0)
    assert a["ok"] and a["restraint"].startswith("GGGG")
    b = place_constraint(HAIRPIN6, "AAAA", 12, existing=a["restraint"])
    assert b["ok"] and b["pinned_total"] == 8, "the first pin must survive the second"
    c = place_constraint(HAIRPIN6, "CCCC", 0, existing=a["restraint"])
    assert not c["ok"] and c["clashes"], "a silent overwrite would answer a question nobody asked"


def test_a_motif_that_does_not_fit_says_where_it_could_go():
    o = place_constraint(HAIRPIN6, "GGGGGGGGGGGGGGGGGGGG", 0)
    assert not o["ok"] and "valid range" in o["error"]


def test_feasibility_is_exact_and_names_the_offending_positions():
    """This is the primitive that saves the whole budget: if two pinned positions must pair and cannot, no
    amount of search will succeed, and the agent needs the POSITIONS to do anything about it."""
    r = place_constraint(HAIRPIN6, "GGGG", 0)["restraint"]
    r = place_constraint(HAIRPIN6, "AAAA", 12, existing=r)["restraint"]
    f = check_feasible(HAIRPIN6, r)
    assert not f["ok"] and f["conflicts"]
    c = f["conflicts"][0]
    assert {c["i"], c["j"]} == {0, 15} and "cannot" in c["why"]
    assert "no amount of search budget" in f["note"]


def test_compatible_pins_are_reported_compatible():
    r = place_constraint(HAIRPIN6, "GGGG", 0)["restraint"]
    r = place_constraint(HAIRPIN6, "CCCC", 12, existing=r)["restraint"]
    assert check_feasible(HAIRPIN6, r)["ok"]


def test_checking_before_pinning_anything_says_what_to_do_next():
    """An observation that does not tell the agent what to do next costs a step. It cost one in development."""
    o = check_feasible(HAIRPIN6, None)
    assert o["ok"] and ("place_constraint" in o["note"] and "design" in o["note"])


# ---- GC content and loop caps: steer vs screen, and the bugs the live runs exposed ---------------
def test_gc_content_rejects_impossible_percentages():
    from rna_harness.cognitive.agent import set_gc_content
    assert not set_gc_content(150)["ok"]
    assert not set_gc_content(-1)["ok"]
    assert not set_gc_content("half")["ok"]


def test_gc_target_becomes_a_symmetric_acgu_split():
    from rna_harness.cognitive.agent import set_gc_content
    g = set_gc_content(50, 5)
    assert g["acgu_content"] == "25,25,25,25"
    assert set_gc_content(60)["acgu_content"] == "20,30,30,20"


def test_gc_is_checked_exactly_and_reports_the_measurement():
    from rna_harness.cognitive.agent import _gc_check, set_gc_content
    g = set_gc_content(50, 10)
    assert _gc_check("GCAUGCAU", g)["held"] is True and _gc_check("GCAUGCAU", g)["measured"] == 50.0
    assert _gc_check("GGGGCCCC", g)["held"] is False
    assert _gc_check(None, g)["tolerance"] == 10.0, "the no-sequence branch must still carry the tolerance"
    assert _gc_check("GCAU", None) is None


def test_loop_cap_suggests_and_does_not_decide():
    from rna_harness.cognitive.agent import recommend_loop_cap
    o = recommend_loop_cap(HAIRPIN6)
    assert o["ok"] and o["loops"][0]["start"] == 6 and o["loops"][0]["len"] == 4
    assert "GAAA" in o["loops"][0]["motifs"]
    assert "place_constraint" in o["how"], "knowledge in, hard pin out -- the KB never decides"
    assert not recommend_loop_cap("....")["ok"]


def test_a_stale_restraint_is_refused_rather_than_crashing_or_resetting():
    """Both of these came out of a live run. check_feasible indexed past the end and raised; place_constraint
    silently started over, which drops pins the caller still expects."""
    from rna_harness.cognitive.agent import check_feasible, place_constraint
    o = check_feasible(HAIRPIN6, "GGGG")
    assert not o["ok"] and "different shape" in o["error"]
    p = place_constraint(HAIRPIN6, "AA", 0, existing="G" * 30)
    assert not p["ok"] and "different shape" in p["error"]


# ---- the loop ------------------------------------------------------------------------------------
def test_function_requests_never_reach_the_solver():
    s = run("设计一个能结合某某蛋白的 RNA", client=_Scripted(["{}"]))
    assert s.done and s.outcome == "refused"
    assert s.steps and s.steps[0].action == "refuse"
    assert s.total_cost_usd == 0.0


def test_the_agent_cannot_declare_success_itself():
    """`done` records that the agent stopped; only a certificate from the exact verifier says certified."""
    s = run("x", client=_Scripted(['{"thought":"t","action":"done","args":{"reason":"I solved it"}}']))
    assert s.done and s.certificate is None and s.outcome != "certified"


def test_an_unknown_action_is_reported_and_the_loop_continues():
    s = run("x", client=_Scripted(['{"action":"hallucinated_tool","args":{}}',
                                   '{"action":"done","args":{"reason":"stop"}}']))
    assert any("unknown action" in (st.observation.get("error") or "") for st in s.steps)
    assert s.done


def test_a_reply_with_no_action_ends_the_loop_rather_than_guessing():
    s = run("x", client=_Scripted(["I think you want a hairpin."]))
    assert not s.done and "did not return an action" in s.outcome


def test_json_is_taken_from_the_last_object_not_the_first_brace():
    """A long `thought` containing braces used to swallow the action; the parser takes the last valid one."""
    reply = 'Considering {a:1} and {b:2}...\n{"thought":"t","action":"done","args":{"reason":"ok"}}'
    s = run("x", client=_Scripted([reply]))
    assert s.done and s.outcome == "ok"


def test_design_without_a_target_is_refused_before_spending():
    s = run("x", client=_Scripted(['{"action":"design","args":{"budget":"cheap"}}',
                                   '{"action":"done","args":{"reason":"stop"}}']))
    assert s.steps[0].observation["error"].startswith("no target")
    assert s.total_cost_usd == 0.0


def test_the_trace_shows_what_was_tried():
    s = run("x", client=_Scripted(['{"thought":"build it","action":"propose_structure",'
                                   '"args":{"elements":[{"type":"hairpin","stem":6,"loop":4}]}}',
                                   '{"action":"done","args":{"reason":"stop"}}']))
    txt = render(s)
    assert HAIRPIN6 in txt and "build it" in txt


def test_step_cap_is_bounded_and_cost_is_capped_separately():
    assert MAX_STEPS >= 6, "a request needing a restructure takes about six steps"
    s = run("x", client=_Scripted(['{"action":"check_feasible","args":{}}']))
    assert len(s.steps) <= MAX_STEPS


# ---- the SHAPE check: the gap that let the loop pay three times for an unreachable target ---------
def test_the_probe_never_runs_on_a_pseudoknot_because_it_crashes_on_one():
    """ViennaRNA's inverse_fold SEGFAULTS on crossing pairs ("Junk in structure at aux_structure"), and a
    segfault inside a primitive takes the whole agent down. So knots are excluded by name, not by luck."""
    from rna_harness.cognitive.agent import designable_probe
    o = designable_probe("((((....[[[[))))....]]]]")
    assert o["ok"] and o["designable"] is None
    assert "crash" in o["note"].lower() or "cannot represent" in o["note"]


def test_the_probe_separates_a_reachable_shape_from_an_unreachable_one():
    """MEASURED: on a 2-branch junction, 2 bp helices are undesignable and 4 bp helices are not. This is the
    check that would have saved $0.38 and 45 minutes on one benchmark item."""
    from rna_harness.cognitive.agent import designable_probe
    assert designable_probe(HAIRPIN6)["designable"] is True
    bad = designable_probe("((...((....))...((....))...))")     # every helix 2 bp -- the real failing item
    assert bad["designable"] is False
    assert "budget" in bad["note"], "it must say that paying more is the wrong move"


def test_min_helix_measures_the_shortest_stack_not_the_total_pairs():
    from rna_harness.cognitive.agent import _min_helix
    assert _min_helix(HAIRPIN6) == 6
    assert _min_helix("((...((....))...((....))...))") == 2
    assert _min_helix("(((((..((((....))))..((((....))))..)))))") == 4
    assert _min_helix("....") == 0


def test_every_feasibility_answer_carries_the_shape_verdict():
    """All three exits, including the error ones -- a caller that reads `shape` must never find it missing."""
    for target, restraint in ((HAIRPIN6, None), (HAIRPIN6, "GGGGNN" + "N" * 10), (HAIRPIN6, "GGG")):
        assert "shape" in check_feasible(target, restraint)


def test_a_failed_nested_design_says_where_the_fold_went_wrong():
    """A bare not_found gave the loop nothing to act on, and it responded the only way it could: by buying a
    bigger budget. The detail below is what makes a different decision possible."""
    from rna_harness.cognitive.certify import certify_sequence
    c = certify_sequence("CCAUAUCGAAAGAAAGGCGCAAGCGGUGG", "((...((....))...((....))...))")
    assert c.verdict == "not_found"
    assert c.detail["mfe_structure"] and c.detail["best_bp"] > 0
    assert c.detail["wrong_positions"] and c.detail["n_wrong"] == len(c.detail["wrong_positions"])


def test_the_same_problem_cannot_be_paid_for_a_third_time(monkeypatch):
    """Escalating cheap -> standard -> best_effort without changing anything is not a new idea, and the third
    call is the expensive one. The refusal has to name what to change, or it just wastes the step instead."""
    from rna_harness.cognitive import agent as A
    calls = []

    def fake_design(target, restraint=None, budget="cheap", seed=0, gc=None):
        calls.append(budget)
        return {"ok": True, "verdict": "not_found", "cost_usd": 0.01, "bp_away": 4}

    monkeypatch.setitem(A.PRIMITIVES, "design", fake_design)
    script = ['{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":6,"loop":4}]}}',
              '{"action":"design","args":{"budget":"cheap"}}',
              '{"action":"design","args":{"budget":"standard"}}',
              '{"action":"design","args":{"budget":"best_effort"}}',
              '{"action":"design","args":{"budget":"best_effort"}}']
    s = run("x", client=_Scripted(script), max_steps=5)
    assert calls == ["cheap", "standard"], f"paid more than twice for one problem: {calls}"
    refused = [st for st in s.steps if st.action == "design" and not st.observation.get("ok")]
    assert refused and "twice" in refused[0].observation["error"]
    assert "helices" in refused[0].observation["hint"] or "loop" in refused[0].observation["hint"]


def test_a_new_problem_gets_its_own_two_attempts(monkeypatch):
    """The guard keys on the PROBLEM, not on a global counter -- restructuring is exactly the behaviour we
    want, so it must not inherit the previous shape's spend."""
    from rna_harness.cognitive import agent as A
    calls = []

    def fake_design(target, restraint=None, budget="cheap", seed=0, gc=None):
        calls.append(target)
        return {"ok": True, "verdict": "not_found", "cost_usd": 0.01}

    monkeypatch.setitem(A.PRIMITIVES, "design", fake_design)
    script = ['{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":6,"loop":4}]}}',
              '{"action":"design","args":{"budget":"cheap"}}',
              '{"action":"design","args":{"budget":"standard"}}',
              '{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":6,"loop":6}]}}',
              '{"action":"design","args":{"budget":"cheap"}}']
    run("x", client=_Scripted(script), max_steps=5)
    assert len(calls) == 3 and len(set(calls)) == 2, calls
