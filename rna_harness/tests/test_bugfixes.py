"""Regression tests for the bugs found by the adversarial bug-hunt (Goal-1 closure)."""
from __future__ import annotations

import tempfile

import pytest

from rna_harness.constraints import ConstraintSpec, check_constraints
from rna_harness.oracles.registry import build_panel, cross_oracle_consistency, primary_oracle
from rna_harness.oracles.vienna import ViennaRNAOracle


# --- CRITICAL: multi-oracle panel must be GENUINELY diverse (sticky-param bug) -----------
def test_multi_oracle_real_diversity():
    """The 4 param sets are different energy models; cross_oracle must NOT collapse them
    (folding each in its own spawned process). Energies should differ across the panel."""
    seq = "GCGCGCGCGCGCAUAUAUAUAUAUGCGCGCGCGCGC"
    target = primary_oracle().fold_mfe(seq)[0]
    res = cross_oracle_consistency(seq, target, build_panel("default"))
    energies = {round(v["energy"], 2) for v in res["per_oracle"].values()}
    assert len(res["per_oracle"]) >= 3
    assert len(energies) >= 2, f"panel collapsed to one model (sticky bug not fixed): {energies}"


# --- length-mismatch must fail loudly, not return garbage --------------------------------
def test_analyze_length_guard():
    orc = primary_oracle()
    with pytest.raises(ValueError):
        orc.analyze("GGGGAAAACCCC", "((((....))))))))")  # target longer than seq


def test_diagnose_length_guard():
    from rna_harness.actuation.diagnosis import diagnose

    with pytest.raises(ValueError):
        diagnose("((((....))))", "((((..))))")  # different lengths


# --- IUPAC-aware forbidden --------------------------------------------------------------
def test_no_forbidden_iupac():
    spec = ConstraintSpec(forbidden=["GNNC"])  # N = any base
    rep = check_constraints("GAAC" + "A" * 8, spec)
    assert rep.checks["no_forbidden"] is False  # GAAC matches GNNC


# --- distill dedups identical (loop_len, motif) within a target -------------------------
def test_distill_dedup():
    from rna_harness.meta.strategies import distill_strategies

    tgt = "((((....))))((((....))))"   # two identical 4-nt loops
    seq = "GGGGGAAACCCC" + "GGGGGAAACCCC"  # both loops = GAAA
    hm = [s for s in distill_strategies(tgt, seq) if s.kind == "hairpin_motif"]
    assert len(hm) == 1 and hm[0].rule["motif"] == "GAAA"  # deduped, not two copies


def test_adopted_requires_all_loops():
    from rna_harness.meta.strategies import Strategy, memory_adoption

    tgt = "((((....))))((((....))))"
    s = Strategy(name="x", kind="hairpin_motif", trigger={"loop_len": 4}, rule={"motif": "GAAA"}, description="")
    both = "GGGGGAAACCCCGGGGGAAACCCC"   # both loops GAAA
    one = "GGGGGAAACCCCGGGGAAAACCCC"    # only first loop GAAA
    assert memory_adoption([s], tgt, both)["A_memory"] == 1.0
    assert memory_adoption([s], tgt, one)["A_memory"] == 0.0  # not ALL loops -> not adopted


# --- group advantages must not pool None group_ids -------------------------------------
def test_add_group_advantages_skips_none():
    from rna_harness.rl.collect import add_group_advantages
    from rna_harness.rl.schema import RLSample

    def mk(gid, r):
        return RLSample(task_id="t", task_family="f", target="((((....))))", prompt="p",
                        response="GGGGAAAACCCC", reward=r, solved=False, generator="g", group_id=gid)
    samples = [mk(None, 1.0), mk(None, 5.0), mk("g1", 0.0), mk("g1", 2.0)]
    add_group_advantages(samples)
    assert samples[0].advantage is None and samples[1].advantage is None  # None not pooled
    assert samples[2].advantage is not None and samples[3].advantage is not None


# --- rnainverse reproducible with a seed ------------------------------------------------
def test_rnainverse_reproducible():
    from rna_harness.generators import RNAinverseGenerator
    from rna_harness.task import DesignTask

    t = DesignTask(task_id="t", target="(((((......)))))")
    a = RNAinverseGenerator().design(t, budget=5, seed=123).best_sequence
    b = RNAinverseGenerator().design(t, budget=5, seed=123).best_sequence
    assert a == b  # same seed -> same result (RNA.init_rand seeded)


# --- local_search budget=0 reports a consistent distance -------------------------------
def test_local_search_budget_zero():
    from rna_harness.generators import LocalSearchGenerator
    from rna_harness.metrics import bp_distance
    from rna_harness.task import DesignTask

    t = DesignTask(task_id="t", target="((((....))))")
    r = LocalSearchGenerator().design(t, budget=0, seed=0)
    assert r.best_sequence and len(r.best_sequence) == t.n
    d = bp_distance(primary_oracle().fold_mfe(r.best_sequence)[0], t.target)
    assert r.best_bp_distance == d  # reported distance matches the returned sequence


# --- llm package importable (circular import fixed) ------------------------------------
def test_llm_package_importable():
    import importlib

    importlib.import_module("rna_harness.llm")           # was broken by the reward_fn cycle
    importlib.import_module("rna_harness.llm.generator")
