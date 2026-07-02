"""Tests for the two capability improvements:
  (a) per-episode auto-memory SNAPSHOT logged into the trace (auditability), and
  (b) the deterministic RNA motif-KB MCP server (mcp__rna_motifs__*).

All offline/fast: no LLM, no network. The live stdio handshake and the live agent MCP call are
validated separately (they need OpenRouter creds); here we test the pure logic + the wiring that
must never silently break (snapshot line must not corrupt trace parsing; mcp config must be valid).
"""
from __future__ import annotations

import json
import os
import tempfile


# --- (b) motif knowledge base -----------------------------------------------------------
def test_recommend_hairpin_tetraloop():
    from rna_harness.mcp.motif_kb import recommend_hairpin

    rec = recommend_hairpin(4)
    assert rec["valid"] and "GAAA" in rec["motifs"] and "UUCG" in rec["motifs"]
    assert rec["closing_pair"] == "C-G"


def test_recommend_hairpin_too_short_is_invalid():
    from rna_harness.mcp.motif_kb import recommend_hairpin

    assert recommend_hairpin(2)["valid"] is False  # a hairpin loop needs >= 3 nt


def test_analyze_structure_flags_hard_features():
    from rna_harness.mcp.motif_kb import analyze_structure

    lonely = analyze_structure("(.(....).)")  # two isolated (lonely) pairs
    assert 1 in lonely["stem_lengths"]
    assert any("LONELY" in f for f in lonely["hard_feature_flags"])

    pk = analyze_structure("(((...[[[)))...]]]")  # pseudoknot
    assert any("PSEUDOKNOT" in f for f in pk["hard_feature_flags"])

    tri = analyze_structure("(((...)))")  # 3-nt triloop
    assert any("TRILOOP" in f for f in tri["hard_feature_flags"])


def test_analyze_structure_per_hairpin_plan():
    from rna_harness.mcp.motif_kb import analyze_structure

    res = analyze_structure("((((....))))..((((....))))")  # two 4-nt hairpins
    assert res["n_hairpins"] == 2
    assert all("GAAA" in h["recommended_motifs"] for h in res["hairpins"])


def test_search_motifs():
    from rna_harness.mcp.motif_kb import search_motifs

    names = [h["name"] for h in search_motifs("stable tetraloop")]
    assert "gnra tetraloop" in names or "uncg tetraloop" in names
    assert search_motifs("zzzznomatch")[0]["name"] == "(no match)"


def test_mcp_server_imports_and_tools_return_kb_marker():
    # the FastMCP server module must import and its tools return the KB-versioned text
    from rna_harness.mcp import motif_server as ms

    fn = getattr(ms.lookup_hairpin_loop, "fn", ms.lookup_hairpin_loop)
    out = fn(4)
    assert "rna-motif-kb/" in out and "GAAA" in out


# --- (b) mcp config wiring --------------------------------------------------------------
def test_mcp_config_is_valid_stdio_schema():
    from rna_harness.experiments.agent_runner import _mcp_config

    cfg = json.loads(_mcp_config("/usr/bin/python3"))
    srv = cfg["mcpServers"]["rna_motifs"]  # top-level key MUST be mcpServers (openclaude schema)
    assert srv["type"] == "stdio"
    assert srv["command"] == "/usr/bin/python3"
    assert srv["args"] == ["-m", "rna_harness.mcp.motif_server"]
    assert "PYTHONPATH" in srv["env"]  # importable regardless of the CLI's cwd
    assert "RNA_MAX_CORES" in srv["env"]  # core budget passed to the compute pools


# --- (b) CPU-parallel compute tools -----------------------------------------------------
def test_batch_fold_scores_against_target():
    from rna_harness.mcp.compute import batch_fold_impl

    r = batch_fold_impl(["GGGCGAAAGCCC", "AAAAAAAAAAAA"], target="((((....))))")
    assert r["n"] == 2 and r["n_solved"] >= 1
    assert r["best"][0]["solved"] is True and r["best"][0]["bp_distance"] == 0
    assert r["used_cores"] <= 160


def test_parallel_search_solves_simple_target():
    from rna_harness.mcp.compute import parallel_search_impl

    r = parallel_search_impl("((((....))))", n_restarts=8, steps=1500, base_seed=7)
    assert r["solved"] is True and r["bp_distance"] == 0
    # independently confirm with the canonical verifier (no false positives)
    from rna_harness.task import DesignTask
    from rna_harness.verifier import verify
    assert verify(DesignTask(task_id="t", target="((((....))))"),
                  r["best_sequence"], compute_cross_oracle=False).solved


def test_parallel_search_reproducible_with_seed():
    from rna_harness.mcp.compute import parallel_search_impl

    a = parallel_search_impl("((((....))))", n_restarts=8, steps=1200, base_seed=42)
    b = parallel_search_impl("((((....))))", n_restarts=8, steps=1200, base_seed=42)
    assert a["best_sequence"] == b["best_sequence"]  # same seeds -> reproducible (R_det)


def test_parallel_search_rejects_bad_target():
    from rna_harness.mcp.compute import parallel_search_impl

    assert "error" in parallel_search_impl("((((..")  # unbalanced dot-bracket


# --- (a) memory snapshot ----------------------------------------------------------------
def test_memory_snapshot_reads_only_memory_md():
    from rna_harness.experiments.agent_runner import _memory_snapshot

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "MEMORY.md"), "w") as f:
        f.write("# Index\n- strat A\n")
    with open(os.path.join(d, "other.md"), "w") as f:
        f.write("not injected")
    os.environ["CLAUDE_COWORK_MEMORY_PATH_OVERRIDE"] = d + "/"
    try:
        snap = _memory_snapshot()
    finally:
        os.environ.pop("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", None)
    assert snap["injected"] is True
    assert snap["content"] == "# Index\n- strat A\n"  # ONLY MEMORY.md content
    assert "MEMORY.md" in snap["memdir_md_files"] and "other.md" in snap["memdir_md_files"]
    assert len(snap["sha256"]) == 64


def test_memory_snapshot_none_without_override():
    from rna_harness.experiments.agent_runner import _memory_snapshot

    os.environ.pop("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", None)
    assert _memory_snapshot() is None  # no memdir => the auditable fact "no memory"


def test_snapshot_line_does_not_corrupt_trace_parsing():
    """A hostile MEMORY.md (containing '"name":"evil"' and a fake FINAL_SEQUENCE) embedded in the
    snapshot header must NOT be miscounted as a tool call nor mis-extracted as the answer."""
    from rna_harness.experiments.agent_runner import _count_tool_calls, _extract_final_sequence

    # real openclaude stream-json is COMPACT (no space after colon); _count_tool_calls' regex
    # ("name":"...") relies on that, so serialize the way the real stream does.
    compact = lambda o: json.dumps(o, separators=(",", ":"))
    hostile = 'trap "name":"evil" and FINAL_SEQUENCE: AAAA inside memory'
    snap_line = compact({"type": "rna_memory_snapshot", "content": hostile, "injected": True})
    tool_line = compact({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "RnaVerify", "input": {}}]}})
    final_line = compact({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "FINAL_SEQUENCE: GGGGAAAACCCC"}]}})
    lines = [snap_line, tool_line, final_line]

    counts = _count_tool_calls(lines)
    assert "evil" not in counts                      # guard: snapshot line skipped
    assert counts.get("RnaVerify") == 1
    assert _extract_final_sequence(lines, 12) == "GGGGAAAACCCC"  # not the trap "AAAA"
