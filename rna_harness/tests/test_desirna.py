"""Tests for the incremental DesiRNA adapter (Plan A: agent calls the external SOTA solver).

Covers the two things we CAN test without a real DesiRNA install:
  * graceful absence (no crash, meta.available=False) -- the default state on this host;
  * the full ACTIVE pipeline via a MOCK DesiRNA (a shell script that echoes a sequence): the adapter
    must run it, parse the sequence, and RE-VERIFY with our own oracle -- including REJECTING a wrong
    sequence the mock returns (never trust the tool's self-report)."""
import RNA

from rna_harness.generators import RNAinverseGenerator
from rna_harness.generators.desirna import DesiRNAGenerator, desirna_available
from rna_harness.task import DesignTask

HAIRPIN = "((((....))))"


def _clear(monkeypatch):
    monkeypatch.delenv("DESIRNA_CMD", raising=False)
    monkeypatch.delenv("DESIRNA_ARGS", raising=False)


def test_input_format_and_argv_match_desirna():
    # lock the REAL DesiRNA interface (github.com/fryzjergda/DesiRNA)
    g = DesiRNAGenerator(replicas=8, timelimit_s=30, results_number=15)
    inp = g._build_input(DesignTask(task_id="t", target=HAIRPIN))
    assert ">name" in inp and ">seq_restr" in inp and ">sec_struct" in inp
    assert HAIRPIN in inp and ("N" * len(HAIRPIN)) in inp
    argv = g._argv(["python", "DesiRNA.py"], "input.txt", 7, param="2004")
    assert argv[:2] == ["python", "DesiRNA.py"]
    # -p 2004 aligns DesiRNA's energy model to our turner2004 oracle (DesiRNA defaults to 1999)
    # -seed is OFFSET BY ONE on purpose: DesiRNA reads `-seed 0` as "choose a seed at random"
    # (`original_seed = random.random()` when in_seed == 0), and 0 is this project's default everywhere, so
    # without the offset every seeded run silently asks the solver to randomise. This assertion is what
    # stops the offset being "tidied away" by someone who reads it as an off-by-one.
    for flag, val in [("-f", "input.txt"), ("-p", "2004"), ("-R", "8"), ("-t", "30"),
                      ("-r", "15"), ("-seed", "8")]:
        assert flag in argv and argv[argv.index(flag) + 1] == val
    assert "-s" not in argv, "no step cap unless the caller asks for one; the product budget is seconds"

    # A caller that needs an exactly reproducible run (an A/B of a code change) asks for a step budget.
    gs = DesiRNAGenerator(replicas=8, timelimit_s=600, results_number=15, steps=40)
    argv_s = gs._argv(["python", "DesiRNA.py"], "input.txt", 0, param="2004")
    assert argv_s[argv_s.index("-s") + 1] == "40"
    assert argv_s[argv_s.index("-seed") + 1] == "1", "our seed 0 must not reach DesiRNA as its randomise flag"


def test_hash_seed_is_pinned_for_the_subprocess(monkeypatch):
    """The solver's own dict/set iteration order is what made it irreproducible; pin it.

    DesiRNA walks dicts and sets of string keys in its design loop, and Python randomises string hashing
    per process, so two identical invocations explored different mutations. Measured 2026-08-08: with the
    hash seed free, two runs at the same seed shared 1 candidate out of 121 (Jaccard 0.004); with it pinned
    and a step budget, three runs returned byte-identical 200-candidate pools.
    """
    import subprocess
    seen = {}

    def fake_run(argv, **kw):
        seen.update(kw.get("env") or {})
        raise subprocess.TimeoutExpired(argv, 1)      # we only care about the env that was handed over

    monkeypatch.setattr(subprocess, "run", fake_run)
    g = DesiRNAGenerator(replicas=2, timelimit_s=1, results_number=2)
    g._run(["python", "DesiRNA.py"], DesignTask(task_id="t", target=HAIRPIN), 0)
    assert seen.get("PYTHONHASHSEED") == "0"


def test_absent_is_graceful(monkeypatch):
    _clear(monkeypatch)
    if desirna_available():
        import pytest; pytest.skip("DesiRNA present on host")
    r = DesiRNAGenerator().design(DesignTask(task_id="t", target=HAIRPIN))
    assert r.meta["available"] is False
    assert r.solved is False
    assert r.best_bp_distance == len(HAIRPIN)
    assert "DESIRNA_CMD" in r.meta.get("hint", "")


def test_mcp_tool_absent_message(monkeypatch):
    _clear(monkeypatch)
    if desirna_available():
        import pytest; pytest.skip("DesiRNA present on host")
    from rna_harness.mcp.motif_server import desirna_solve
    msg = desirna_solve(HAIRPIN)
    assert "not installed" in msg or "DESIRNA_CMD" in msg


def test_fake_desirna_end_to_end(monkeypatch, tmp_path):
    seq = RNAinverseGenerator().design(DesignTask(task_id="t", target=HAIRPIN), budget=6, seed=1).best_sequence
    assert RNA.fold(seq)[0] == HAIRPIN, "test setup: need a valid solution"
    fake = tmp_path / "fake_desirna.sh"
    fake.write_text(f"#!/usr/bin/env bash\necho 'candidate {seq}'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("DESIRNA_CMD", f"bash {fake}")
    monkeypatch.delenv("DESIRNA_ARGS", raising=False)
    assert desirna_available() is True
    r = DesiRNAGenerator().design(DesignTask(task_id="t", target=HAIRPIN))
    assert r.meta["available"] is True and r.meta["verified_by"] == "primary_oracle"
    assert r.solved is True and r.best_bp_distance == 0
    assert RNA.fold(r.best_sequence)[0] == HAIRPIN


def test_oracle_rejects_bad_tool_output(monkeypatch, tmp_path):
    # DesiRNA returns a WRONG sequence -> our oracle re-verification must NOT report solved.
    bad = "A" * len(HAIRPIN)
    fake = tmp_path / "fake_bad.sh"
    fake.write_text(f"#!/usr/bin/env bash\necho '{bad}'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("DESIRNA_CMD", f"bash {fake}")
    monkeypatch.delenv("DESIRNA_ARGS", raising=False)
    r = DesiRNAGenerator().design(DesignTask(task_id="t", target=HAIRPIN))
    assert r.meta["available"] is True
    assert r.solved is False          # never trust the tool's self-report
