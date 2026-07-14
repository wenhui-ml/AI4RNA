"""Lock the pseudoknot guard: the ViennaRNA MFE oracle is pseudoknot-FREE, so the harness must
REFUSE to emit an MFE 'solved' on any target containing '[]{}<>' (else it silently mis-scores --
we measured a 29% false-positive rate on Pseudobase++). A pseudoknot-aware verifier (pKiss) is
required instead. These tests guarantee the guard stays wired at the two exposure points and that
pseudoknot-free targets are entirely unaffected."""
import RNA

from rna_harness.generators.desirna import DesiRNAGenerator
from rna_harness.structure_utils import has_pseudoknot
from rna_harness.task import DesignTask
from rna_harness.verifier import verify

PK_TARGET = "((.((((.....[[[[))))..))......]]]]"


def test_has_pseudoknot_detects_all_bracket_families():
    assert has_pseudoknot("..[[..]]..")
    assert has_pseudoknot("(({{))}}")
    assert has_pseudoknot("((<<))>>")
    assert not has_pseudoknot("((((....))))")
    assert not has_pseudoknot("...(((...)))...")


def test_verifier_refuses_pseudoknot_target():
    # any valid ACGU sequence: the guard fires on the TARGET, not the sequence
    r = verify(DesignTask(task_id="pk", target=PK_TARGET), "A" * len(PK_TARGET),
               compute_cross_oracle=False)
    assert r.valid is True           # the sequence itself is valid
    assert r.solved is False         # but NO MFE 'solved' is emitted
    assert r.mfe_match is False
    assert r.error and "pseudoknot" in r.error.lower()
    assert r.reward == 0.0


def test_verifier_unaffected_on_pseudoknot_free_target():
    seq = "GGGGAAAACCCC"
    assert RNA.fold(seq)[0] == "((((....))))"  # a real solution
    r = verify(DesignTask(task_id="hp", target="((((....))))"), seq, compute_cross_oracle=False)
    assert r.valid and r.solved and r.mfe_match and r.error is None


def test_desirna_adapter_refuses_mfe_verdict_on_pseudoknot(monkeypatch, tmp_path):
    # a mock DesiRNA that returns SOME length-n ACGU sequence; the adapter must NOT call it solved,
    # because our MFE oracle cannot verify a pseudoknot (would be a false positive).
    seq = "G" * len(PK_TARGET)
    fake = tmp_path / "fake.sh"
    fake.write_text(f"#!/usr/bin/env bash\necho '{seq}'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("DESIRNA_CMD", f"bash {fake}")
    monkeypatch.delenv("DESIRNA_ARGS", raising=False)
    r = DesiRNAGenerator().design(DesignTask(task_id="pk", target=PK_TARGET))
    assert r.meta["available"] is True
    assert r.solved is False                       # never a false MFE 'solved' on a pseudoknot
    assert "pseudoknot" in (r.meta.get("hint", "").lower())
    assert "MFE oracle cannot verify" in r.meta.get("verified_by", "")
