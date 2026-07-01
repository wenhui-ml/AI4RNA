"""Tests for the meta-learning layer (cross-case memdir self-improvement)."""
from __future__ import annotations

import tempfile

from rna_harness.meta import (
    MemStore,
    Strategy,
    applicable_strategies,
    distill_strategies,
    memory_adoption,
    merge_strategies,
)
from rna_harness.meta.meta_runner import hairpin_family, run_arm_classical

HAIRPIN = "((((....))))"
GAAA_SOL = "GGGGGAAACCCC"  # loop positions 4:8 -> "GAAA"


def test_distill_hairpin_and_gc():
    ss = distill_strategies(HAIRPIN, GAAA_SOL)
    kinds = {s.kind for s in ss}
    assert "hairpin_motif" in kinds and "gc_closing" in kinds
    hm = [s for s in ss if s.kind == "hairpin_motif"][0]
    assert hm.trigger == {"loop_len": 4} and hm.rule["motif"] == "GAAA"


def test_merge_picks_majority_motif():
    a = distill_strategies(HAIRPIN, "GGGGGAAACCCC")  # GAAA
    b = distill_strategies(HAIRPIN, "GGGGGAAACCCC")  # GAAA again
    c = distill_strategies(HAIRPIN, "GGGGUUCGCCCC")  # UUCG
    merged = merge_strategies(merge_strategies(a, b), c)
    hm = [s for s in merged if s.kind == "hairpin_motif"][0]
    assert hm.rule["motif"] == "GAAA"  # majority (2 vs 1)
    assert hm.support == 3


def test_applicable_and_adoption():
    ss = distill_strategies(HAIRPIN, GAAA_SOL)
    # applicable to a 4-loop hairpin
    assert len(applicable_strategies(ss, HAIRPIN)) >= 1
    # NOT applicable (a 3-loop) for the 4-loop motif
    assert all(s.kind != "hairpin_motif" for s in applicable_strategies(ss, "(((...)))"))
    adopt = memory_adoption(ss, HAIRPIN, GAAA_SOL)
    assert adopt["A_memory"] == 1.0
    # a solution that ignores the motif -> lower adoption
    adopt2 = memory_adoption(ss, HAIRPIN, "GGGGAAAACCCC")  # loop AAAA != GAAA
    assert adopt2["A_memory"] < 1.0


def test_memstore_roundtrip():
    d = tempfile.mkdtemp()
    store = MemStore(d)
    ss = distill_strategies(HAIRPIN, GAAA_SOL)
    store.write(ss)
    back = store.strategies()
    assert {s.kind for s in back} == {s.kind for s in ss}
    assert "RNA design strategies" in store.render_playbook()


def test_hairpin_family_designable():
    tasks = hairpin_family(6, seed=0)
    assert len(tasks) == 6
    assert all(set(t.target) <= set("().") for t in tasks)


def test_meta_classical_memory_helps_or_neutral():
    """with_memory should not be worse than no_memory on evals-to-solve (usually better)."""
    tasks = hairpin_family(8, seed=1)
    wm = run_arm_classical(tasks, "with_memory", budget=3000, seed=1)
    nm = run_arm_classical(tasks, "no_memory", budget=3000, seed=1)
    assert wm["solve_rate"] >= 0.8 and nm["solve_rate"] >= 0.8
    assert wm["A_memory"] is not None
    # learned memory should cut (or at least not increase) mean evals-to-solve
    if wm["mean_evals_solved"] is not None and nm["mean_evals_solved"] is not None:
        assert wm["mean_evals_solved"] <= nm["mean_evals_solved"] + 1.0
