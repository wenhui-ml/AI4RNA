"""Phase-2 main vehicle: a hardened, resumable meta-learning campaign.

Runs the harness over a task set for multiple epochs, accumulating strategies into an
openclaude-format memdir (self-improvement), collecting every trajectory (data, reward,
trace), and recording the SATURATION curve (solve-rate / oracle-calls-to-solve vs
accumulated experience). Meta-learning saturates at the FROZEN model's ceiling -- that
ceiling + the trace corpus are the phase-2 deliverable; RL (phase 3) is what moves it.

Hardening:
  * resumable  -- per-(arm,epoch,task) checkpoint; restart skips completed episodes.
  * cost cap   -- global episode cap + per-episode budget (agent).
  * retry      -- one retry on agent parse/exec failure.
  * trace store-- append-only campaign_traces.jsonl (+ full stream-json in traces/).
  * isolation  -- a task that errors is recorded as failed; the campaign continues.

Executors:
  * classical  -- zero-credit; memory biases the warm start (validates mechanics for free).
  * agent      -- live openclaude agent; memory recalled via CLAUDE_COWORK_MEMORY_PATH_OVERRIDE.

Verifier = ViennaRNA folding correctness (design folds to target). Mature algorithms
(local_search) serve as baseline + optional teacher/fallback whose solution is added to the
trace (imitation+improvement) -- never as a sequence-match answer key.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from statistics import mean
from typing import Dict, List, Optional

from ..benchmarks import eterna100_tasks, random_designable_tasks
from ..meta.memstore import MemStore, Strategy
from ..meta.meta_runner import build_memory_init, permuted_playbook
from ..meta.strategies import distill_strategies, memory_adoption, merge_strategies
from ..task import DesignTask
from ..verifier import verify


# --------------------------------------------------------------------------- state
def _load_state(path: str) -> Dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"done": {}, "curves": {}, "n_episodes": 0}


def _save_state(path: str, state: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- executors
def _solve_classical(task: DesignTask, store: MemStore, arm: str, budget: int, seed: int) -> Dict:
    import random as _r
    from ..generators.local_search import LocalSearchGenerator

    strategies = store.strategies()
    warm = None
    if arm != "no_memory" and strategies:
        warm = [build_memory_init(task, strategies, _r.Random(seed))]
    res = LocalSearchGenerator(restarts=8).design(task, budget=budget, seed=seed, warm_starts=warm)
    v = verify(task, res.best_sequence, compute_cross_oracle=False)
    adopt = memory_adoption(strategies, task.target, res.best_sequence) if strategies else {"A_memory": None}
    return {"solved": v.solved, "reward": v.reward, "oracle_calls": res.n_evals,
            "sequence": res.best_sequence, "A": None, "A_memory": adopt["A_memory"],
            "generator": "local_search", "trace_path": None}


def _solve_agent(task: DesignTask, store: MemStore, arm: str, model: str, out_dir: str,
                 budget_usd: float, timeout_s: int, knowledge: str) -> Dict:
    from .agent_runner import run_agent_on_task

    if arm == "no_memory":
        os.environ.pop("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", None)
    else:
        os.environ["CLAUDE_COWORK_MEMORY_PATH_OVERRIDE"] = store.dir + os.sep
    rec = run_agent_on_task(task, model, out_dir, budget_usd, timeout_s, None, knowledge)
    # one retry on parse/exec failure
    if not rec.get("final_sequence") or rec.get("rc", 0) != 0:
        rec = run_agent_on_task(task, model, out_dir, budget_usd, timeout_s, None, knowledge)
    os.environ.pop("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", None)
    seq = rec.get("final_sequence") or ""
    adopt = memory_adoption(store.strategies(), task.target, seq) if (seq and store.strategies()) else {"A_memory": None}
    return {"solved": rec["solved"], "reward": rec["reward"], "oracle_calls": rec["n_rna_tool_calls"],
            "sequence": seq, "A": rec.get("actuation", {}).get("A"), "A_memory": adopt["A_memory"],
            "generator": rec["generator"], "trace_path": rec.get("trace_path")}


def _teacher_solution(task: DesignTask, budget: int, seed: int) -> Optional[str]:
    """Mature-algorithm fallback (baseline solver). Returns a folding-correct sequence or None."""
    from ..generators.local_search import LocalSearchGenerator
    from ..generators.rnainverse import RNAinverseGenerator

    for gen in (RNAinverseGenerator(), LocalSearchGenerator(restarts=10)):
        res = gen.design(task, budget=budget, seed=seed)
        if verify(task, res.best_sequence, compute_cross_oracle=False).mfe_match:
            return res.best_sequence
    return None


# --------------------------------------------------------------------------- campaign
def run_campaign(
    tasks: List[DesignTask],
    executor: str = "classical",
    arms: Optional[List[str]] = None,
    epochs: int = 3,
    out_dir: str = "runs/campaign",
    classical_budget: int = 4000,
    agent_budget_usd: float = 0.75,
    agent_timeout_s: int = 400,
    model: str = "anthropic/claude-opus-4.8",
    global_episode_cap: int = 10_000,
    teacher: bool = False,
    teacher_budget: int = 3000,
    resume: bool = True,
    seed: int = 0,
) -> Dict:
    arms = arms or ["with_memory", "no_memory"]
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "campaign_state.json")
    state = _load_state(state_path) if resume else {"done": {}, "curves": {}, "n_episodes": 0}
    trace_fp = open(os.path.join(out_dir, "campaign_traces.jsonl"), "a")
    stores = {a: MemStore(os.path.join(out_dir, f"memdir_{a}")) for a in arms}
    knowledge = ""
    if executor == "agent":
        from .agent_runner import _load_knowledge
        knowledge = _load_knowledge()

    for arm in arms:
        # fresh memory for this arm only if not resuming into it
        if not resume or not any(k.startswith(f"{arm}:") for k in state["done"]):
            stores[arm].clear()
            if arm == "permuted_memory":
                stores[arm].write(permuted_playbook())
        state["curves"].setdefault(arm, [])
        for epoch in range(epochs):
            ep_rows = []
            for t in tasks:
                key = f"{arm}:{epoch}:{t.task_id}"
                if key in state["done"]:
                    ep_rows.append(state["done"][key])
                    continue
                if state["n_episodes"] >= global_episode_cap:
                    print(f"[campaign] global episode cap {global_episode_cap} reached; stopping.", flush=True)
                    trace_fp.close()
                    _finalize(state, state_path, out_dir)
                    return _summary(state, out_dir)
                try:
                    if executor == "classical":
                        rec = _solve_classical(t, stores[arm], arm, classical_budget, seed + epoch)
                    else:
                        rec = _solve_agent(t, stores[arm], arm, model, out_dir, agent_budget_usd, agent_timeout_s, knowledge)
                except Exception as e:  # noqa: BLE001  isolate per-task failures
                    rec = {"solved": False, "reward": 0.0, "oracle_calls": 0, "sequence": "",
                           "A": None, "A_memory": None, "generator": "error", "trace_path": None, "error": str(e)[:200]}

                teacher_seq = None
                if teacher and not rec["solved"]:
                    teacher_seq = _teacher_solution(t, teacher_budget, seed + epoch)

                # accumulate real learned strategies (with_memory), from agent OR teacher solution
                if arm == "with_memory":
                    learn_seq = rec["sequence"] if rec["solved"] else teacher_seq
                    if learn_seq and verify(t, learn_seq, compute_cross_oracle=False).mfe_match:
                        stores[arm].write(merge_strategies(stores[arm].strategies(), distill_strategies(t.target, learn_seq)))

                row = {"arm": arm, "epoch": epoch, "task_id": t.task_id, "family": t.family,
                       "n": t.n, "solved": rec["solved"], "reward": round(rec["reward"], 4),
                       "oracle_calls": rec["oracle_calls"], "A": rec["A"], "A_memory": rec["A_memory"],
                       "generator": rec["generator"], "final_sequence": rec["sequence"],
                       "teacher_used": bool(teacher_seq), "trace_path": rec["trace_path"]}
                trace_fp.write(json.dumps(row) + "\n")
                trace_fp.flush()
                state["done"][key] = row
                state["n_episodes"] += 1
                _save_state(state_path, state)
                ep_rows.append(row)

            solved = [r for r in ep_rows if r["solved"]]
            oc = [r["oracle_calls"] for r in solved if r["oracle_calls"]]
            am = [r["A_memory"] for r in ep_rows if r.get("A_memory") is not None]
            state["curves"][arm].append({
                "epoch": epoch, "n": len(ep_rows),
                "solve_rate": round(len(solved) / max(len(ep_rows), 1), 4),
                "mean_oracle_calls_solved": round(mean(oc), 2) if oc else None,
                "A_memory": round(mean(am), 4) if am else None,
                "n_strategies": len(stores[arm].strategies()),
            })
            _save_state(state_path, state)
            c = state["curves"][arm][-1]
            print(f"[campaign] arm={arm} epoch={epoch} solve={c['solve_rate']} "
                  f"oracle_calls={c['mean_oracle_calls_solved']} A_memory={c['A_memory']} "
                  f"strategies={c['n_strategies']}", flush=True)

    trace_fp.close()
    _finalize(state, state_path, out_dir)
    return _summary(state, out_dir)


def _finalize(state: Dict, state_path: str, out_dir: str) -> None:
    _save_state(state_path, state)


def _summary(state: Dict, out_dir: str) -> Dict:
    summ = {"n_episodes": state["n_episodes"], "curves": state["curves"]}
    with open(os.path.join(out_dir, "campaign_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executor", choices=["classical", "agent"], default="classical")
    ap.add_argument("--source", choices=["eterna100", "open_random"], default="eterna100")
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--arms", nargs="*", default=["with_memory", "no_memory"])
    ap.add_argument("--classical-budget", type=int, default=4000)
    ap.add_argument("--agent-budget-usd", type=float, default=0.75)
    ap.add_argument("--agent-timeout-s", type=int, default=400)
    ap.add_argument("--model", default="anthropic/claude-opus-4.8")
    ap.add_argument("--global-episode-cap", type=int, default=10000)
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rna_harness/runs/campaign")
    args = ap.parse_args(argv)

    if args.source == "eterna100":
        tasks = sorted(eterna100_tasks(max_len=args.max_len), key=lambda t: t.n)[: args.n_tasks]
    else:
        tasks = random_designable_tasks(args.n_tasks, len_range=(30, args.max_len), seed=args.seed)
    print(f"[campaign] executor={args.executor} source={args.source} tasks={len(tasks)} "
          f"epochs={args.epochs} arms={args.arms} teacher={args.teacher}", flush=True)
    summ = run_campaign(
        tasks, executor=args.executor, arms=args.arms, epochs=args.epochs, out_dir=args.out,
        classical_budget=args.classical_budget, agent_budget_usd=args.agent_budget_usd,
        agent_timeout_s=args.agent_timeout_s, model=args.model,
        global_episode_cap=args.global_episode_cap, teacher=args.teacher,
        resume=not args.no_resume, seed=args.seed,
    )
    # headline: saturation of with_memory vs no_memory
    for arm, curve in summ["curves"].items():
        pts = " ".join(f"e{c['epoch']}:{c['solve_rate']}/{c['mean_oracle_calls_solved']}" for c in curve)
        print(f"[campaign] {arm} saturation (epoch:solve/oracle_calls): {pts}", flush=True)
    print(f"[campaign] episodes={summ['n_episodes']} -> {args.out}/campaign_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
