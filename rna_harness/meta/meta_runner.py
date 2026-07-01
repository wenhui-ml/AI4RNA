"""The meta-learning harness loop: cross-case self-improvement over openclaude's memdir.

For a sequence of cases the harness (a) recalls the current playbook, (b) solves the case
(warm-started from remembered strategies), (c) measures A_memory (did the solution adopt the
recalled strategies), and (d) distills new strategies from the solution back into memory.

Three arms isolate whether accumulated memory actually helps (the cross-case analog of the
single-case actuation ablation, and a direct test of Paper 1's thesis):
  * with_memory   : accumulate + recall real learned strategies
  * no_memory     : never use/accumulate memory (control)
  * permuted_memory: a fixed WRONG playbook (the "random feedback" analog)

Executor: 'classical' (zero-credit local search, memory biases the warm start) or 'agent'
(the live openclaude agent; memory is written to a memdir the agent recalls via
CLAUDE_COWORK_MEMORY_PATH_OVERRIDE). Classical validates the mechanism for free; agent is
the real thing (credit-gated).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from statistics import mean
from typing import Dict, List, Optional

from ..benchmarks import eterna100_tasks
from ..generators.base import gc_target_prob, init_sequence
from ..generators.local_search import LocalSearchGenerator
from ..task import DesignTask
from ..verifier import verify
from .memstore import MemStore, Strategy
from .strategies import (
    _closing_pairs,
    _hairpin_loops,
    distill_strategies,
    memory_adoption,
    merge_strategies,
)


# --------------------------------------------------------------------------- task sources
def hairpin_family(n: int = 12, seed: int = 0) -> List[DesignTask]:
    """Designable multi-hairpin targets that all share 4-nt hairpin loops, so a learned
    'GAAA for 4-loops' + 'GC closings' strategy transfers strongly across cases."""
    rng = random.Random(seed)
    tasks: List[DesignTask] = []
    i = 0
    while len(tasks) < n:
        k = rng.choice([2, 2, 3, 3, 4])           # number of hairpins
        parts = []
        for _ in range(k):
            s = rng.choice([5, 6, 7, 8])           # stem length
            parts.append("(" * s + "." * 4 + ")" * s)
        spacer = "." * rng.choice([0, 1, 2])
        target = spacer.join(parts)
        tasks.append(DesignTask(task_id=f"hairpin_family:{i}", target=target, family="hairpin_family"))
        i += 1
    return tasks


def permuted_playbook() -> List[Strategy]:
    """A deliberately WRONG playbook: an unstable loop motif for 4-nt loops."""
    return [Strategy(name="rna-hairpin-4-cccc", kind="hairpin_motif",
                     trigger={"loop_len": 4}, rule={"motif": "CCCC"},
                     description="(permuted/wrong) use CCCC for 4-nt loops", support=1)]


# --------------------------------------------------------------------------- memory-biased init
def build_memory_init(task: DesignTask, strategies: List[Strategy], rng: random.Random) -> str:
    """Warm-start sequence: standard init, then overwrite hairpin loops with remembered
    motifs and set loop-closing pairs to G-C where the playbook says so."""
    seq = list(init_sequence(task.target, task.constraints, rng, gc_target_prob(task.constraints)))
    for i, j, L in _hairpin_loops(task.target):
        for s in strategies:
            if s.kind == "hairpin_motif" and s.trigger.get("loop_len") == L:
                motif = s.rule.get("motif", "")
                if len(motif) == L and set(motif) <= set("ACGU"):
                    seq[i + 1 : j] = list(motif)
    if any(s.kind == "gc_closing" for s in strategies):
        for i, j in _closing_pairs(task.target):
            seq[i], seq[j] = ("G", "C") if (i + j) % 2 == 0 else ("C", "G")
    return "".join(seq)


# --------------------------------------------------------------------------- classical arm
def run_arm_classical(tasks: List[DesignTask], arm: str, budget: int, seed: int) -> Dict:
    rng = random.Random(seed)
    gen = LocalSearchGenerator(restarts=8)
    strategies: List[Strategy] = permuted_playbook() if arm == "permuted_memory" else []
    rows: List[Dict] = []
    for t in tasks:
        warm = None
        if arm != "no_memory" and strategies:
            warm = [build_memory_init(t, strategies, rng)]
        res = gen.design(t, budget=budget, seed=rng.randint(0, 10**6), warm_starts=warm)
        v = verify(t, res.best_sequence, compute_cross_oracle=False)
        adopt = memory_adoption(strategies, t.target, res.best_sequence) if strategies else {"A_memory": None, "n_applicable": 0}
        rows.append({"task_id": t.task_id, "solved": v.solved, "evals": res.n_evals,
                     "A_memory": adopt["A_memory"], "n_applicable": adopt["n_applicable"]})
        # accumulate ONLY real learned strategies (with_memory)
        if arm == "with_memory" and v.solved:
            strategies = merge_strategies(strategies, distill_strategies(t.target, res.best_sequence))
    return _summarize_arm(arm, rows, len(strategies))


def _summarize_arm(arm: str, rows: List[Dict], n_strategies: int) -> Dict:
    solved = [r for r in rows if r["solved"]]
    a_vals = [r["A_memory"] for r in rows if r.get("A_memory") is not None]
    return {
        "arm": arm,
        "n_cases": len(rows),
        "solve_rate": round(len(solved) / max(len(rows), 1), 4),
        "mean_evals_all": round(mean(r["evals"] for r in rows), 1) if rows else 0,
        "mean_evals_solved": round(mean(r["evals"] for r in solved), 1) if solved else None,
        "A_memory": round(mean(a_vals), 4) if a_vals else None,
        "n_strategies_learned": n_strategies,
        "rows": rows,
    }


# --------------------------------------------------------------------------- agent arm (live)
def run_arm_agent(tasks: List[DesignTask], arm: str, model: str, out_dir: str,
                  max_budget_usd: float, timeout_s: int) -> Dict:
    """Live openclaude agent; memory is written to a memdir the agent recalls via
    CLAUDE_COWORK_MEMORY_PATH_OVERRIDE. Accumulates real strategies in with_memory."""
    from ..experiments.agent_runner import run_agent_on_task, _load_knowledge

    mem_dir = os.path.join(out_dir, f"memdir_{arm}")
    store = MemStore(mem_dir)
    store.clear()
    if arm == "permuted_memory":
        store.write(permuted_playbook())
    knowledge = _load_knowledge()
    rows: List[Dict] = []
    for t in tasks:
        # point the agent's auto-memory at our store for this arm (empty for no_memory)
        if arm == "no_memory":
            os.environ.pop("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", None)
        else:
            os.environ["CLAUDE_COWORK_MEMORY_PATH_OVERRIDE"] = mem_dir + os.sep
        rec = run_agent_on_task(t, model, out_dir, max_budget_usd, timeout_s, None, knowledge)
        seq = rec.get("final_sequence") or ""
        strategies = store.strategies()
        adopt = memory_adoption(strategies, t.target, seq) if (seq and strategies) else {"A_memory": None, "n_applicable": 0}
        rows.append({"task_id": t.task_id, "solved": rec["solved"], "evals": rec["n_rna_tool_calls"],
                     "A_memory": adopt["A_memory"], "n_applicable": adopt["n_applicable"],
                     "agent_actuation": rec.get("actuation", {})})
        if arm == "with_memory" and rec["solved"] and seq:
            store.write(merge_strategies(store.strategies(), distill_strategies(t.target, seq)))
    os.environ.pop("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", None)
    return _summarize_arm(arm, rows, len(store.strategies()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executor", choices=["classical", "agent"], default="classical")
    ap.add_argument("--source", choices=["hairpin_family", "eterna100"], default="hairpin_family")
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=70)
    ap.add_argument("--arms", nargs="*", default=["with_memory", "no_memory", "permuted_memory"])
    ap.add_argument("--budget", type=int, default=4000, help="classical local-search eval budget/case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="anthropic/claude-opus-4.8")
    ap.add_argument("--max-budget-usd", type=float, default=0.75)
    ap.add_argument("--timeout-s", type=int, default=400)
    ap.add_argument("--out", default="runs/meta")
    args = ap.parse_args(argv)

    if args.source == "hairpin_family":
        tasks = hairpin_family(args.n_tasks, args.seed)
    else:
        tasks = sorted(eterna100_tasks(max_len=args.max_len), key=lambda t: t.n)[: args.n_tasks]
    os.makedirs(args.out, exist_ok=True)

    print(f"[meta] executor={args.executor} source={args.source} n_tasks={len(tasks)} arms={args.arms}", flush=True)
    results = {}
    for arm in args.arms:
        if args.executor == "classical":
            r = run_arm_classical(tasks, arm, args.budget, args.seed)
        else:
            r = run_arm_agent(tasks, arm, args.model, args.out, args.max_budget_usd, args.timeout_s)
        results[arm] = r
        print(f"  {arm:16s} solve={r['solve_rate']:.3f}  mean_evals_solved={r['mean_evals_solved']}  "
              f"A_memory={r['A_memory']}  strategies_learned={r['n_strategies_learned']}", flush=True)

    summary = {"config": vars(args), "n_tasks": len(tasks),
               "arms": {a: {k: v for k, v in r.items() if k != "rows"} for a, r in results.items()},
               "detail": results}
    with open(os.path.join(args.out, "meta_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # headline: does learned memory help vs control?
    if "with_memory" in results and "no_memory" in results:
        wm, nm = results["with_memory"], results["no_memory"]
        print(f"\n[meta] MEMORY EFFECT: solve {nm['solve_rate']:.3f} -> {wm['solve_rate']:.3f}; "
              f"evals_solved {nm['mean_evals_solved']} -> {wm['mean_evals_solved']}; "
              f"A_memory(with)={wm['A_memory']}", flush=True)
    print(f"[meta] wrote {os.path.join(args.out, 'meta_summary.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
