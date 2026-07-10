#!/usr/bin/env python3
"""RNA controlled actuation-severing ablation (single-variable A manipulation).

Reviewer (GPT-5.5) asked: fix ONE learned memory, adopt each remembered directive with
probability A (sever with 1-A), and watch oracle-calls-to-solve vs A. Memory CONTENT is
held fixed; only the adoption probability varies -> isolates actuation from strategy
content, turning RNA from supporting into causal evidence.

Design:
  * learn the correct playbook once on a TRAIN split (frozen thereafter);
  * TEST split is disjoint (different seed);
  * for A in {0,0.25,0.5,0.75,1.0}: warm-start each test task with per-directive
    Bernoulli(A) adoption of the frozen playbook; classical LocalSearch + exact ViennaRNA
    oracle; record evals-to-solve, solved, realized A_memory.
Runs on the RNA box (has harness + ViennaRNA); fully reproducible from this script.
"""
import json, random, statistics as st
import rna_harness.meta.meta_runner as M
from rna_harness.generators.base import frozen_motif_positions

A_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_TRAIN, N_TEST, REPS, BUDGET = 24, 40, 3, 3500

def severed_init(task, strategies, rng, A):
    """build_memory_init, but each remembered directive is adopted only w.p. A."""
    seq = list(M.init_sequence(task.target, task.constraints, rng, M.gc_target_prob(task.constraints)))
    frozen = frozen_motif_positions(task.constraints)
    for i, j, L in M._hairpin_loops(task.target):
        if any(p in frozen for p in range(i + 1, j)):
            continue
        for s in strategies:
            if s.kind == "hairpin_motif" and s.trigger.get("loop_len") == L:
                motif = s.rule.get("motif", "")
                if len(motif) == L and set(motif) <= set("ACGU") and rng.random() < A:  # <-- sever
                    seq[i + 1:j] = list(motif)
    if any(s.kind == "gc_closing" for s in strategies):
        for i, j in M._closing_pairs(task.target):
            if i not in frozen and j not in frozen and rng.random() < A:                # <-- sever
                seq[i], seq[j] = ("G", "C") if (i + j) % 2 == 0 else ("C", "G")
    return "".join(seq)

# ---- 1) learn + freeze the correct playbook on the TRAIN split ----
train = M.hairpin_family(N_TRAIN, seed=7)
gen = M.LocalSearchGenerator(restarts=8)
rng = random.Random(1234)
playbook = []
for t in train:
    r = gen.design(t, budget=BUDGET, seed=rng.randint(0, 10**6))
    v = M.verify(t, r.best_sequence, compute_cross_oracle=False)
    if v.solved:
        playbook = M.merge_strategies(playbook, M.distill_strategies(t.target, r.best_sequence))
print("[frozen playbook]", [(s.kind, s.trigger, s.rule) for s in playbook], flush=True)

# ---- 2) severing dose-response on the disjoint TEST split ----
test = M.hairpin_family(N_TEST, seed=0)
levels, per_task = [], []
for A in A_LEVELS:
    rows = []
    for t in test:
        for rep in range(REPS):
            r = gen.design(t, budget=BUDGET, seed=rng.randint(0, 10**6),
                           warm_starts=[severed_init(t, playbook, rng, A)])
            v = M.verify(t, r.best_sequence, compute_cross_oracle=False)
            adopt = M.memory_adoption(playbook, t.target, r.best_sequence)
            rows.append({"task": t.task_id, "A": A, "evals": r.n_evals, "solved": bool(v.solved),
                         "A_mem": adopt["A_memory"]})
    per_task += rows
    solved = [x for x in rows if x["solved"]]
    amem = [x["A_mem"] for x in rows if x["A_mem"] is not None]
    lv = {"A_target": A, "n": len(rows), "solve_rate": round(len(solved) / len(rows), 3),
          "mean_evals_solved": round(st.mean([x["evals"] for x in solved]), 1) if solved else None,
          "mean_evals_all": round(st.mean([x["evals"] for x in rows]), 1),
          "realized_A_mem": round(st.mean(amem), 3) if amem else None}
    levels.append(lv)
    print("  A=%.2f  realizedA=%s  solve=%.3f  evals_solved=%s" %
          (A, lv["realized_A_mem"], lv["solve_rate"], lv["mean_evals_solved"]), flush=True)

out = {"design": {"n_train": N_TRAIN, "n_test": N_TEST, "reps": REPS, "budget": BUDGET,
                  "playbook": [(s.kind, s.trigger, s.rule) for s in playbook]},
       "levels": levels, "per_task": per_task}
open("/data/wenhui/openclaude_rna/runs/meta/sever_ablation.json", "w").write(json.dumps(out, indent=2))
print("[wrote] runs/meta/sever_ablation.json", flush=True)
