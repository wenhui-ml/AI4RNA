"""Fast, deterministic tests for the rna_harness science spine.

Run: conda run -n rna python -m pytest rna_harness/tests -q
"""
from __future__ import annotations

import pytest

from rna_harness.actuation.diagnosis import diagnose
from rna_harness.actuation.instrument import ActuationLogger
from rna_harness.benchmarks import eterna100_tasks, synthesize_hybrid_tasks
from rna_harness.constraints import ConstraintSpec, check_constraints, translate
from rna_harness.determinism import estimate_d_sym, estimate_r_det, estimate_t_trace
from rna_harness.metrics import bp_distance, gc_content
from rna_harness.oracles.registry import build_panel, cross_oracle_consistency, primary_oracle
from rna_harness.rl.collect import add_group_advantages, collect_group, repair_episode
from rna_harness.rl.reward_fn import rna_reward, rna_solved_reward
from rna_harness.structure_utils import base_pairs, pair_table, positional_mismatches
from rna_harness.task import DesignTask
from rna_harness.verifier import verify

HAIRPIN = "((((....))))"
GOOD = "GGGGAAAACCCC"


# ---- structure utils ----
def test_pair_table_and_pairs():
    pt = pair_table(HAIRPIN)
    assert pt[0] == 11 and pt[11] == 0 and pt[4] == -1
    assert base_pairs(HAIRPIN) == [(0, 11), (1, 10), (2, 9), (3, 8)]


def test_pair_table_unbalanced_raises():
    with pytest.raises(ValueError):
        pair_table("((.")


# ---- oracle ----
def test_primary_oracle_folds_and_solves():
    orc = primary_oracle()
    struct, energy = orc.fold_mfe(GOOD)
    assert struct == HAIRPIN and energy < 0
    assert orc.is_mfe_solved(GOOD, HAIRPIN)


def test_cross_oracle_consistency():
    panel = build_panel("default")
    assert len(panel) >= 2
    res = cross_oracle_consistency(GOOD, HAIRPIN, panel)
    assert 0.0 <= res["fraction"] <= 1.0 and res["per_oracle"]


# ---- verifier (the canonical reward) ----
def test_verify_solved_good():
    t = DesignTask(task_id="t", target=HAIRPIN)
    v = verify(t, GOOD)
    assert v.valid and v.solved and v.mfe_match and v.is_umfe
    assert v.reward > 1.0 and v.diagnosis is None


def test_verify_unsolved_has_diagnosis():
    t = DesignTask(task_id="t", target=HAIRPIN)
    v = verify(t, "AAAAAAAAAAAA", compute_cross_oracle=False)
    assert not v.solved and v.diagnosis is not None
    assert v.diagnosis["missing_pairs"]


def test_verify_invalid_inputs():
    t = DesignTask(task_id="t", target=HAIRPIN)
    assert not verify(t, "GGGGAAAACCC").valid          # wrong length
    assert not verify(t, "GGGGAAAAXXXX").valid          # bad base


def test_verify_deterministic():
    t = DesignTask(task_id="t", target=HAIRPIN)
    r1 = verify(t, GOOD, compute_cross_oracle=False).reward
    r2 = verify(t, GOOD, compute_cross_oracle=False).reward
    assert r1 == r2


# ---- constraints ----
def test_constraints():
    spec = ConstraintSpec(gc_min=0.4, gc_max=0.6, forbidden=["AAAA"], motifs=[(0, "GG")])
    rep = check_constraints(GOOD, spec)
    assert rep.checks["fixed_motifs"] is True
    assert rep.checks["no_forbidden"] is False   # GOOD has AAAA
    assert not rep.all_satisfied


def test_coding_constraint():
    assert translate("AUGGCG", 0) == "MA"
    spec = ConstraintSpec(coding_aa="MA", coding_frame=0)
    assert check_constraints("AUGGCGAAAA", spec).checks["coding"] is True


# ---- metrics / diagnosis ----
def test_bp_distance_and_diagnose():
    assert bp_distance(HAIRPIN, HAIRPIN) == 0
    d = diagnose(HAIRPIN, "." * 12)
    assert d.bp_distance == 4 and len(d.missing_pairs) == 4 and d.mismatched_positions


def test_positional_mismatches():
    assert positional_mismatches(HAIRPIN, HAIRPIN) == []
    assert len(positional_mismatches(HAIRPIN, "." * 12)) == 8


# ---- four-coordinate determinism (positive control) ----
def test_determinism_high():
    t = DesignTask(task_id="t", target=HAIRPIN)
    seqs = [GOOD, "GCGCAAAAGCGC", "AAAAAAAAAAAA"]
    assert estimate_d_sym(t, seqs).value == 1.0
    assert estimate_r_det(t, seqs).value == 1.0
    assert estimate_t_trace(t, seqs).value == 1.0  # every failure localizes


# ---- actuation ----
def test_actuation_logger():
    log = ActuationLogger()
    # edit at a diagnosed position that improves -> selected, affected
    log.log([3], "AAAAAAAAAAAA", "AAAGAAAAAAAA", d_before=4, d_after=3, read=True)
    # edit at a non-diagnosed position -> not selected
    log.log([3], "AAAAAAAAAAAA", "CAAAAAAAAAAA", d_before=4, d_after=4, read=True)
    a = log.actuation_fidelity()
    assert a["A_model"] == 0.5 and a["A_harness"] == 1.0 and a["n_steps"] == 2


def test_actuation_silent_fallback():
    log = ActuationLogger()
    log.log([3], "AAAAAAAAAAAA", "CAAAAAAAAAAA", d_before=4, d_after=4, read=False)
    a = log.actuation_fidelity()
    assert a["A_harness"] == 0.0 and a["A"] == 0.0


# ---- benchmarks ----
def test_eterna100_loader():
    tasks = eterna100_tasks(max_len=30)
    assert len(tasks) >= 3
    assert all(set(t.target) <= set("().") for t in tasks)


def test_hybrid_synthesis():
    base = eterna100_tasks(ids=[1])[0]
    hyb = synthesize_hybrid_tasks(base, profiles=["gc_mid", "forbidden"])
    assert len(hyb) == 2 and all(h.family.startswith("hybrid:") for h in hyb)


# ---- RL data ----
def test_collect_group_and_advantages():
    from rna_harness.generators import RNAinverseGenerator

    t = eterna100_tasks(ids=[1])[0]
    samples = collect_group(t, RNAinverseGenerator(), k=4, budget=15, seed=0)
    assert len(samples) == 4
    add_group_advantages(samples)
    assert all(s.advantage is not None and s.group_id for s in samples)


def test_repair_episode_actuation_modes():
    t = eterna100_tasks(ids=[1])[0]
    _, a_diag, _ = repair_episode(t, max_steps=30, seed=0, feedback_mode="diagnosis")
    _, a_fb, _ = repair_episode(t, max_steps=30, seed=0, feedback_mode="silent_fallback")
    # diagnosis mode should target diagnosed positions more than silent_fallback
    if a_diag["A_model"] is not None and a_fb["A_model"] is not None:
        assert a_diag["A_model"] >= a_fb["A_model"]
    assert a_fb["A_harness"] == 0.0  # harness drops the directive


# ---- online reward fn ----
def test_reward_fn():
    t = DesignTask(task_id="t", target=HAIRPIN)
    from rna_harness.rl.prompts import render_task_prompt

    p = render_task_prompt(t)
    r = rna_reward([p, p], [GOOD, "junk"], target=[HAIRPIN, HAIRPIN])
    assert r[0] > 1.0 and r[1] == 0.0
    s = rna_solved_reward([p], [GOOD], target=[HAIRPIN])
    assert s[0] == 1.0
