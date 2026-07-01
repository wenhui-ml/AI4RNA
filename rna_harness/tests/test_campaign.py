"""Smoke + resumability tests for the phase-2 campaign runner (zero-credit classical)."""
from __future__ import annotations

import os
import tempfile

from rna_harness.benchmarks import eterna100_tasks, random_designable_tasks
from rna_harness.experiments.campaign import run_campaign


def _small_tasks():
    return sorted(eterna100_tasks(max_len=30), key=lambda t: t.n)[:3]


def test_campaign_runs_and_records_curves():
    out = tempfile.mkdtemp()
    tasks = _small_tasks()
    summ = run_campaign(tasks, executor="classical", arms=["with_memory", "no_memory"],
                        epochs=2, out_dir=out, classical_budget=1500, resume=False, max_workers=2)
    assert summ["n_episodes"] == len(tasks) * 2 * 2  # arms x epochs x tasks
    assert set(summ["curves"]) == {"with_memory", "no_memory"}
    for arm in ("with_memory", "no_memory"):
        assert len(summ["curves"][arm]) == 2  # two epochs
        assert all("solve_rate" in c for c in summ["curves"][arm])
    # artifacts present
    assert os.path.exists(os.path.join(out, "campaign_traces.jsonl"))
    assert os.path.exists(os.path.join(out, "campaign_summary.json"))
    assert os.path.exists(os.path.join(out, "campaign_state.json"))


def test_campaign_resumable():
    out = tempfile.mkdtemp()
    tasks = _small_tasks()
    s1 = run_campaign(tasks, executor="classical", arms=["with_memory"], epochs=2,
                      out_dir=out, classical_budget=1200, resume=False, max_workers=2)
    n1 = s1["n_episodes"]
    # resume: everything already done -> no new episodes
    s2 = run_campaign(tasks, executor="classical", arms=["with_memory"], epochs=2,
                      out_dir=out, classical_budget=1200, resume=True, max_workers=2)
    assert s2["n_episodes"] == n1  # resume skipped all completed episodes


def test_campaign_global_cap_stops_early():
    out = tempfile.mkdtemp()
    tasks = _small_tasks()
    summ = run_campaign(tasks, executor="classical", arms=["with_memory"], epochs=5,
                        out_dir=out, classical_budget=1000, global_episode_cap=4, resume=False, max_workers=2)
    assert summ["n_episodes"] <= 4  # cap honored


def test_open_random_designable_tasks():
    tasks = random_designable_tasks(4, len_range=(30, 50), seed=0)
    assert len(tasks) == 4
    # each target is designable by its witness sequence (never shown to solver)
    from rna_harness.verifier import verify
    for t in tasks:
        assert set(t.target) <= set("().")
        assert verify(t, t.meta["witness_seq"], compute_cross_oracle=False).mfe_match
