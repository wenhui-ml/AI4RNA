"""Tests for actuation-fidelity reconstruction from a REAL openclaude agent trace.

The fixture is a genuine stream-json trace of Opus-4.8 solving Eterna100 #26 (stickshift)
via the RNA tools -- this pins the trace-parsing + A computation against real agent output
so the meta-loop instrument stays correct as the harness evolves.
"""
from __future__ import annotations

import json
import os

import pytest

from rna_harness.actuation.trace import (
    actuation_from_agent_trace,
    actuation_from_trace_file,
    extract_verify_chain,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "agent_trace_stickshift.jsonl")


def test_fixture_exists():
    assert os.path.exists(FIXTURE)


def test_extract_verify_chain_from_real_trace():
    with open(FIXTURE) as f:
        events = [json.loads(l) for l in f if l.strip()]
    chain = extract_verify_chain(events)
    assert len(chain) >= 3
    # every directive call carries a submitted sequence (bp_distance may be None for an
    # invalid-sequence attempt, which the actuation computation filters out)
    assert all(c["sequence"] for c in chain)
    assert any(c["bp_distance"] is not None for c in chain)
    # the agent eventually solved it
    assert any(c["solved"] for c in chain)


def test_actuation_on_real_agent_trace():
    a = actuation_from_trace_file(FIXTURE)
    assert a["n_directive_calls"] >= 3
    assert a["n_steps"] >= 1
    # Opus acted on the diagnosed positions on the live agent -> high A_model.
    assert a["A_model"] is not None and a["A_model"] >= 0.8
    assert a["A_harness"] == 1.0  # no silent fallback: directives reached the agent
    # per-step records are well-formed
    for s in a["repair_steps"]:
        assert set(("step", "selected", "adoption", "affected", "d_before", "d_after")) <= set(s)


def test_actuation_empty_and_malformed_traces():
    assert actuation_from_agent_trace([])["A"] is None
    # malformed events should not crash the parser
    a = actuation_from_agent_trace([{"type": "assistant"}, {"foo": "bar"}, {"message": {"content": "x"}}])
    assert a["n_directive_calls"] == 0


def test_silent_fallback_synthetic_trace():
    """Synthetic trace where the agent ignores the diagnosis -> A_model should be low."""
    # verify #1: seq AAAA..., diagnosis says positions [0,1,2,3] wrong, bp_distance 4
    # verify #2: agent changes only position 8 (NOT diagnosed) -> selected should be False
    def verify_event(seq, bp, mism, solved, tid):
        return [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "RnaVerify", "id": tid, "input": {"sequence": seq}}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": tid, "content": json.dumps({
                    "ok": True, "solved": solved, "sequence": seq,
                    "diagnosis": {"bp_distance": bp, "mismatched_positions": mism},
                    "ensemble": {"bp_distance_to_target": bp}})}]}},
        ]
    events = (
        verify_event("AAAAAAAAAAAA", 4, [0, 1, 2, 3], False, "t1")
        + verify_event("AAAAAAAACAAA", 4, [0, 1, 2, 3], False, "t2")  # edited pos 8, not diagnosed
    )
    a = actuation_from_agent_trace(events)
    assert a["n_steps"] == 1
    assert a["A_model"] == 0.0  # the edit missed the diagnosed positions
