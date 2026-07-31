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
    """`done` records that the agent stopped; only a certificate from the exact verifier says certified.

    STRENGTHENED after a measured failure: a bare `done` with no prior observation is now refused outright,
    because the agent used it on step 1 with the reason "Design succeeded" and nothing had been run.
    """
    s = run("x", client=_Scripted(['{"thought":"t","action":"done","args":{"reason":"I solved it"}}']))
    assert not s.done, "a conclusion with nothing behind it is not a conclusion"
    assert s.certificate is None and s.outcome != "certified"
    assert all("I solved it" not in (st.observation.get("reason") or "") for st in s.steps)


def test_an_unknown_action_is_reported_and_the_loop_continues():
    s = run("x", client=_Scripted(['{"action":"hallucinated_tool","args":{}}',
                                   '{"action":"check_feasible","args":{}}']), max_steps=2)
    assert any("unknown action" in (st.observation.get("error") or "") for st in s.steps)
    assert len(s.steps) == 2, "an unknown action costs a step and the loop keeps going"


def test_a_reply_with_no_action_ends_the_loop_rather_than_guessing():
    s = run("x", client=_Scripted(["I think you want a hairpin."]))
    assert not s.done and "did not return an action" in s.outcome


def test_json_is_taken_from_the_last_object_not_the_first_brace():
    """A long `thought` containing braces used to swallow the action; the parser takes the last valid one."""
    reply = 'Considering {a:1} and {b:2}...\n{"thought":"t","action":"done","args":{"reason":"ok"}}'
    s = run("x", client=_Scripted([reply]), max_steps=1)
    assert s.steps[0].action == "done" and s.steps[0].args["reason"] == "ok"


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

    def fake_design(target, restraint=None, budget="cheap", seed=0, **kw):
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

    def fake_design(target, restraint=None, budget="cheap", seed=0, **kw):
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


# ---- pins survive a reshape: they come from the request, not from the agent ----------------------
def test_pins_are_carried_onto_a_new_shape_instead_of_being_silently_cleared():
    """Three benchmark items reached a conflict-free, designable shape and then ran out of steps re-issuing
    pins they had never chosen to drop. The pins belong to the REQUEST; a reshape does not retract them."""
    script = ['{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":5,"loop":6}]}}',
              '{"action":"place_constraint","args":{"motif":"GGGG","at":0}}',
              '{"action":"propose_structure","args":{"elements":[{"type":"spacer","len":4},'
              '{"type":"hairpin","stem":5,"loop":6}]}}',
              '{"action":"check_feasible","args":{}}']
    s = run("x", client=_Scripted(script), max_steps=4)
    reshape = s.steps[2]
    assert reshape.observation["ok"] and reshape.observation["n"] == 20
    assert reshape.observation["pins_carried_over"] == [{"motif": "GGGG", "at": 0}]
    assert s.restraint == "GGGG" + "N" * 16, s.restraint


def test_a_pin_that_no_longer_fits_is_reported_not_forgotten():
    script = ['{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":6,"loop":4}]}}',
              '{"action":"place_constraint","args":{"motif":"GGGG","at":12}}',
              '{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":2,"loop":4}]}}']
    s = run("x", client=_Scripted(script), max_steps=3)
    o = s.steps[2].observation
    assert o["ok"] and o["pins_dropped"] and o["pins_dropped"][0]["at"] == 12
    assert "re-place" in o["note"]
    assert s.restraint is None


def test_re_pinning_the_same_position_replaces_it_rather_than_stacking():
    script = ['{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":6,"loop":4}]}}',
              '{"action":"place_constraint","args":{"motif":"AAA","at":6}}',
              '{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":6,"loop":4}]}}']
    s = run("x", client=_Scripted(script), max_steps=3)
    assert s.steps[2].observation["pins_carried_over"] == [{"motif": "AAA", "at": 6}]


# ---- a conclusion needs an observation, and a substitution has to be said out loud ----------------
def test_done_is_refused_when_nothing_has_been_observed():
    """MEASURED: the agent's first action on one item was `done` with the reason "Design succeeded: MFE
    structure matches the target". It had run nothing. No false certificate can be issued -- only the oracle
    issues those -- but a sentence that reads like success is the answer to someone who cannot check."""
    s = run("x", client=_Scripted(['{"action":"done","args":{"reason":"Design succeeded"}}',
                                   '{"action":"propose_structure","args":{"elements":'
                                   '[{"type":"hairpin","stem":6,"loop":4}]}}']))
    first = s.steps[0]
    assert first.action == "done" and not first.observation["ok"]
    assert "nothing has been observed" in first.observation["error"]
    assert not s.done and "Design succeeded" not in s.outcome


def test_a_stopped_session_never_repeats_the_agents_claim_as_the_result():
    s = run("x", client=_Scripted(['{"action":"design","args":{"budget":"cheap"}}',
                                   '{"action":"done","args":{"reason":"I solved it"}}']))
    assert s.done and s.outcome.startswith("the agent stopped:")
    assert s.certificate is None


def test_delivering_a_different_shape_than_the_request_is_stated_not_hidden():
    """MEASURED: on a request that fixed the shape completely the agent returned a genuinely certified
    sequence for a different 40-nt shape. Every oracle was satisfied and the question was not answered.
    Reshaping is legitimate -- it is how a pin conflict gets resolved -- so this states it rather than
    forbidding it."""
    script = ['{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":6,"loop":4}]}}',
              '{"action":"check_feasible","args":{}}',
              '{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":4,"loop":8}]}}']
    s = run("x", client=_Scripted(script), max_steps=3)
    assert s.substituted and s.substituted["requested"] == HAIRPIN6
    assert s.substituted["delivered"] == "((((........))))"
    assert s.substituted["same_length"] is True
    assert "⚠" in render(s, "zh") and "requested" in render(s, "en")


def test_no_substitution_is_reported_when_the_shape_never_changed():
    s = run("x", client=_Scripted(['{"action":"propose_structure","args":{"elements":'
                                   '[{"type":"hairpin","stem":6,"loop":4}]}}',
                                   '{"action":"check_feasible","args":{}}']), max_steps=2)
    assert s.substituted is None


# ---- a structure the USER brought: the model selects it, it never writes one ----------------------
def test_the_model_chooses_which_structure_never_what_it_is():
    """The invariant this module is built on is that the model never writes dot-bracket -- counting brackets
    is precise computation. So a user-supplied structure is EXTRACTED by code and indexed by the model. There
    is no argument through which a bracket the user never wrote can enter."""
    from rna_harness.cognitive.agent import use_target, structures_in
    import inspect
    assert "structure" not in inspect.signature(use_target).parameters
    req = "我有一个结构 ((((...)))) 请给我序列"
    assert structures_in(req) == ["((((...))))"]
    o = use_target(req, 0)
    assert o["ok"] and o["target"] == "((((...))))" and o["kind"] == "nested"


def test_an_unbalanced_or_absent_structure_is_not_adopted():
    from rna_harness.cognitive.agent import structures_in, use_target
    assert structures_in("((((...))") == [], "an unbalanced string is not a structure"
    assert structures_in("....") == [], "nothing paired is not a structure to design for"
    o = use_target("给我一个发夹", 0)
    assert not o["ok"] and "propose_structure" in o["error"]


def test_an_out_of_range_index_lists_what_is_actually_available():
    from rna_harness.cognitive.agent import use_target
    o = use_target("A ((((...)))) B ((....))", 5)
    assert not o["ok"] and len(o["available"]) == 2


def test_adopting_the_users_structure_makes_it_the_requested_one():
    """If the agent later designs for something else, that is a substitution and must be reported against
    what the USER brought -- not against whatever the agent proposed afterwards."""
    script = ['{"action":"use_target","args":{"which":0}}',
              '{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":6,"loop":4}]}}',
              '{"action":"check_feasible","args":{}}']
    s = run("我有一个结构 ((((...))))..((((....)))) 请给我序列", client=_Scripted(script), max_steps=3)
    assert s.steps[0].observation["ok"]
    assert s.substituted and s.substituted["requested"] == "((((...))))..((((....))))"
    assert s.substituted["delivered"] == HAIRPIN6 and s.substituted["same_length"] is False


def test_a_structure_the_user_supplied_cannot_be_designed_away_from(monkeypatch):
    """MEASURED on natural bpRNA shapes: given a structure, two pins that must pair and cannot, and an
    explicit length lock, the agent SHIFTED the whole structure so the pins fell off the stem and returned a
    genuinely certified sequence -- for a structure the user never wrote. 2 of 8, and it had already named
    the conflict in all 8. Reshaping is fine when the agent built the shape; never when the user gave it."""
    from rna_harness.cognitive import agent as A
    calls = []
    monkeypatch.setitem(A.PRIMITIVES, "design",
                        lambda *a, **k: (calls.append(a[0]) or {"ok": True, "verdict": "certified",
                                                                "cost_usd": 0.0}))
    script = ['{"action":"use_target","args":{"which":0}}',
              '{"action":"propose_structure","args":{"elements":[{"type":"spacer","len":2},'
              '{"type":"hairpin","stem":4,"loop":3}]}}',
              '{"action":"design","args":{"budget":"cheap"}}',
              '{"action":"design","args":{"budget":"cheap"}}']
    s = run("我有一个结构 ((((...)))) 请给我序列", client=_Scripted(script), max_steps=4)
    assert calls == [], "it designed for a structure the user never gave"
    bad = [st for st in s.steps if st.action == "design"][0]
    assert "cannot be replaced" in bad.observation["error"]
    assert bad.observation["requested"] == "((((...))))"


def test_designing_for_the_users_own_structure_is_not_blocked(monkeypatch):
    """The guard must key on 'different from what was given', not on 'the user gave one'."""
    from rna_harness.cognitive import agent as A
    calls = []
    monkeypatch.setitem(A.PRIMITIVES, "design",
                        lambda *a, **k: (calls.append(a[0]) or {"ok": True, "verdict": "certified",
                                                                "cost_usd": 0.0}))
    s = run("我有一个结构 ((((...)))) 请给我序列",
            client=_Scripted(['{"action":"use_target","args":{"which":0}}',
                              '{"action":"design","args":{"budget":"cheap"}}']), max_steps=2)
    assert calls == ["((((...))))"] and s.outcome == "certified"


# ---- the deterministic opening: the model is not asked what it cannot choose ----------------------
def test_a_lone_structure_is_adopted_and_checked_without_a_model_call():
    """MEASURED: eight plain requests produced eight identical trajectories, use_target -> check_feasible ->
    design, at three model calls each. The first two had nothing to decide."""
    class _Counting(_Scripted):
        calls = 0

        def chat(self, messages, **kw):
            type(self).calls += 1
            return super().chat(messages, **kw)

    _Counting.calls = 0
    c = _Counting(['{"action":"check_feasible","args":{}}'])
    s = run("我有一个结构 ((((...)))) 请给我序列", client=c, max_steps=1)
    assert [st.action for st in s.steps[:2]] == ["use_target", "check_feasible"]
    assert all(st.observation.get("automatic") for st in s.steps[:2])
    assert _Counting.calls == 1, "the primed steps must not cost a model call"
    assert s.target == "((((...))))" and s.requested_by_user


def test_choosing_between_two_structures_stays_the_models_decision():
    """With more than one candidate, adopting one IS a decision, so the harness must not make it."""
    s = run("对比 ((((...)))) 和 ((....)) ，给第二个的序列",
            client=_Scripted(['{"action":"use_target","args":{"which":1}}']), max_steps=1)
    assert s.steps[0].action == "use_target" and not s.steps[0].observation.get("automatic")
    assert s.target == "((....))"


def test_priming_does_not_eat_the_step_budget():
    s = run("我有一个结构 ((((...)))) 请给我序列",
            client=_Scripted(['{"action":"check_feasible","args":{}}']), max_steps=3)
    assert len(s.steps) == 2 + 3, "2 primed + 3 the model was actually given"


def test_a_request_with_no_structure_is_not_primed():
    s = run("给我一个茎 6 对、环 4 nt 的发夹",
            client=_Scripted(['{"action":"propose_structure","args":{"elements":'
                              '[{"type":"hairpin","stem":6,"loop":4}]}}']), max_steps=1)
    assert s.steps[0].action == "propose_structure" and len(s.steps) == 1


def test_the_free_check_declines_by_length_instead_of_burning_its_timeout():
    """MEASURED on bpRNA, 6 structures per band at a 20 s cap: <=110 nt answers 6/6 at a 1.1 s median;
    111-150 answers 5/6; 151-200 answers 1/6; above 200 it answers 0/6 and always spends the full cap. Since
    the whole reason it runs before every design is that it is free, above the cutoff it must decline rather
    than make someone wait 20 s to be told nothing -- the primed opening runs it on every pasted structure."""
    import time
    from rna_harness.cognitive.agent import PROBE_MAX_N, designable_probe
    long_target = "(" * 200 + "." * 20 + ")" * 200
    t0 = time.time()
    o = designable_probe(long_target)
    assert o["designable"] is None and str(PROBE_MAX_N) in o["note"]
    assert time.time() - t0 < 1.0, "declining must be instant, not a timeout"
    assert o["min_helix"] == 200, "it can still report what it computes for free"


def test_primed_steps_are_numbered_in_order():
    s = run("我有一个结构 ((((...)))) 请给我序列",
            client=_Scripted(['{"action":"check_feasible","args":{}}']), max_steps=1)
    assert [st.n for st in s.steps] == [1, 2, 3]


# ---- the two measured capabilities, now reachable by a user --------------------------------------
def test_melting_temperature_is_a_check_and_never_a_steer():
    """MEASURED: the solver has NO folding-temperature option -- -tmin/-tmax are replica temperatures and
    -tm is a mutation heuristic. Asked for a Tm below what it naturally produces (median 83.1 C) its own
    answer landed 12 of 60 times against 54 for screening its pool. So this primitive declares a check."""
    from rna_harness.cognitive.agent import melting_point, set_melting_temperature, _tm_check
    o = set_melting_temperature(55, 5)
    assert o["ok"] and o["celsius"] == 55 and "checked" in o["note"]
    assert "cannot be steered" in o["note"], "it must say plainly that there is no steer, only a check"
    assert "partition function" in o["definition"], "the quantity has to be defined, not just named"
    assert not set_melting_temperature("hot")["ok"]
    assert not set_melting_temperature(500)["ok"]
    t = "((((((((((......))))))))))"
    hot = melting_point("GGGGGGGGGGAAAAAACCCCCCCCCC", t)
    assert 60 < hot < 100, hot
    assert _tm_check("GGGGGGGGGGAAAAAACCCCCCCCCC", t, {"celsius": 55, "tolerance": 5})["held"] is False
    assert _tm_check(None, t, {"celsius": 55, "tolerance": 5})["measured"] is None


def test_a_design_that_folds_but_melts_wrong_is_not_the_answer(monkeypatch):
    """The same discipline as the GC and pin checks: every declared requirement has to hold, or the loop is
    not done. A sequence that folds perfectly and melts 30 degrees off answers a question nobody asked."""
    from rna_harness.cognitive import agent as A
    monkeypatch.setitem(A.PRIMITIVES, "design",
                        lambda *a, **k: {"ok": True, "verdict": "certified", "cost_usd": 0.0,
                                         "sequence": "G" * 16, "constraint_held": None, "gc": None,
                                         "tm": {"held": False, "measured": 88.0, "target": 55,
                                                "tolerance": 5}})
    s = run("x", client=_Scripted([
        '{"action":"propose_structure","args":{"elements":[{"type":"hairpin","stem":6,"loop":4}]}}',
        '{"action":"set_melting_temperature","args":{"celsius":55,"tolerance":5}}',
        '{"action":"design","args":{"budget":"cheap"}}']), max_steps=3)
    assert s.tm and s.tm["celsius"] == 55
    assert s.outcome != "certified", "it folded, but it did not meet the stated requirement"


def test_a_set_requirement_is_refused_below_two():
    from rna_harness.cognitive.agent import design_set
    o = design_set("((((....))))", count=1)
    assert not o["ok"] and "not a set" in o["error"]


def test_the_set_check_is_exact_and_reports_both_bounds(monkeypatch):
    """A set-level requirement cannot be expressed by any per-sequence filter, which is exactly why the
    external solver has no formulation for it. The reported numbers must be the measured extremes, not a
    claim that the constraints were applied."""
    import RNA
    from rna_harness.cognitive import agent as A
    seqs = ["GGGGAAAACCCC", "AUAUAAAAUAUA", "CCGGAAAACCGG"]
    monkeypatch.setattr(A, "design_set", A.design_set)          # keep the real one; check its arithmetic
    ident = [100.0 * sum(1 for x, y in zip(seqs[i], seqs[j]) if x == y) / 12
             for i in range(3) for j in range(i + 1, 3)]
    dup = [RNA.duplexfold(seqs[i], seqs[j]).energy for i in range(3) for j in range(i + 1, 3)]
    assert max(ident) >= 0 and min(dup) < 0, "the exact quantities the primitive reports are computable here"


def test_the_new_requirements_are_in_the_action_space_the_model_is_told_about():
    """A capability that exists only in a benchmark script is not a capability a user has."""
    from rna_harness.cognitive.agent import PRIMITIVES, SYSTEM
    for name in ("set_melting_temperature", "design_set"):
        assert name in PRIMITIVES
        assert name in SYSTEM, f"{name} is callable but the model is never told it exists"


def test_knot_grade_reports_agreement_not_specificity():
    """The grade must say how many oracles agreed, and must NOT imply a specificity difference.

    k=1 and k=3 have the SAME measured random false-accept rate (0.004, runs/ceiling/calibrate.json). A
    certificate that let a reader infer "stricter grade => fewer false accepts" would be asserting something
    the calibration does not support, so the number is pinned here.
    """
    from rna_harness.cognitive.certify import KNOT_GRADES, knot_grade
    for k, want in ((3, "k=3"), (2, "k=2"), (1, "k=1")):
        g = knot_grade({"confirmed_by": ["a", "b", "c"][:k]})
        assert g["grade"] == want and g["agreeing_oracles"] == k
        assert g["false_accept_on_random_sequences"] == 0.004
        assert g["shipped_threshold"] == "k>=1"
    # sensitivity must DECREASE as the grade tightens -- that is the thing the grade actually costs
    sens = [KNOT_GRADES[k]["sensitivity_on_known_structures"] for k in (1, 2, 3)]
    assert sens == sorted(sens, reverse=True)
    # out-of-range k must not crash or silently invent a grade
    assert knot_grade({"confirmed_by": []})["grade"] == "k=1"
    assert knot_grade({})["grade"] == "k=1"
