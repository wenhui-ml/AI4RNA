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
