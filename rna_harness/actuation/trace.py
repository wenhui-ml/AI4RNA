"""Compute actuation fidelity (A) from a REAL openclaude agent stream-json trace.

This closes the meta-learning-harness loop on the actual agent (not a Python simulation):
we reconstruct the agent's repair chain from its RnaVerify tool calls and measure whether
each next submitted sequence actually changed the positions the PREVIOUS RnaVerify diagnosed
as wrong (selected), and whether that improved the fold (affected). A -> 0 here would be the
Diagnosis-Actuation Gap on the live agent.

Trace event shapes (openclaude stream-json):
  {type:'assistant', message:{content:[{type:'tool_use', name, id, input:{sequence,target}}]}}
  {type:'user',      message:{content:[{type:'tool_result', tool_use_id, content:'<json>'}]}}
The tool_result content is the JSON emitted by rnaResultBlock (full verify result).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from .instrument import ActuationLogger

# tool calls that carry a target-relative diagnosis (the "directive" source)
_DIRECTIVE_TOOLS = {"RnaVerify", "RnaDiagnose"}


def _content_blocks(event: Dict) -> List[Dict]:
    msg = event.get("message", event)
    content = msg.get("content") if isinstance(msg, dict) else None
    return content if isinstance(content, list) else []


def _parse_result_json(block_content) -> Optional[Dict]:
    if isinstance(block_content, str):
        try:
            return json.loads(block_content)
        except Exception:
            return None
    if isinstance(block_content, list):  # some encoders wrap as [{type:'text',text:...}]
        for b in block_content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                try:
                    return json.loads(b["text"])
                except Exception:
                    continue
    if isinstance(block_content, dict):
        return block_content
    return None


def extract_verify_chain(events: List[Dict]) -> List[Dict]:
    """Ordered list of directive-bearing tool calls with their submitted sequence + result.

    Each item: {tool, sequence, bp_distance, mismatched_positions, solved}.
    """
    # 1) collect tool_use (id -> {tool, sequence}) in order; 2) match tool_result by id.
    uses: List[Dict] = []
    results: Dict[str, Dict] = {}
    for ev in events:
        for b in _content_blocks(ev):
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use" and b.get("name") in _DIRECTIVE_TOOLS:
                inp = b.get("input") or {}
                uses.append({"id": b.get("id"), "tool": b.get("name"), "sequence": (inp.get("sequence") or "").upper()})
            elif b.get("type") == "tool_result":
                res = _parse_result_json(b.get("content"))
                if res is not None:
                    results[b.get("tool_use_id")] = res

    chain: List[Dict] = []
    for u in uses:
        res = results.get(u["id"])
        if res is None:
            continue
        diag = res.get("diagnosis") or {}
        ens = res.get("ensemble") or {}
        bp = diag.get("bp_distance")
        if bp is None:
            bp = ens.get("bp_distance_to_target")
        if bp is None:
            bp = 0 if res.get("solved") else None
        seq = u["sequence"] or res.get("sequence", "")
        chain.append({
            "tool": u["tool"],
            "sequence": seq.upper(),
            "bp_distance": bp,
            "mismatched_positions": diag.get("mismatched_positions") or [],
            "solved": bool(res.get("solved")),
        })
    return chain


def actuation_from_agent_trace(events: List[Dict]) -> Dict:
    """Reconstruct A from the agent's repair chain. Consecutive directive-tool calls with a
    known sequence + bp_distance form (before -> after) steps; the directive is the PREVIOUS
    call's diagnosed mismatch set."""
    chain = [c for c in extract_verify_chain(events) if c["sequence"] and c["bp_distance"] is not None]
    logger = ActuationLogger()
    for k in range(1, len(chain)):
        prev, cur = chain[k - 1], chain[k]
        if len(prev["sequence"]) != len(cur["sequence"]):
            continue  # can't align edits across different lengths
        logger.log(
            directive_positions=prev["mismatched_positions"],
            before=prev["sequence"],
            after=cur["sequence"],
            d_before=prev["bp_distance"],
            d_after=cur["bp_distance"],
            read=True,       # the directive was in-context for the next turn
            executed=True,
        )
    out = logger.actuation_fidelity()
    out["n_directive_calls"] = len(chain)
    out["repair_steps"] = [s.to_dict() for s in logger.steps]
    return out


def actuation_from_trace_file(path: str) -> Dict:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return actuation_from_agent_trace(events)
