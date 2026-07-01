"""Phase-2 main vehicle: a hardened, resumable, PARALLEL meta-learning campaign.

Runs the harness over a task set for multiple epochs, accumulating strategies into an
openclaude-format memdir (self-improvement), collecting every trajectory (data, reward,
trace), and recording the SATURATION curve (solve-rate / oracle-calls-to-solve vs
accumulated experience). Meta-learning saturates at the FROZEN model's ceiling -- that
ceiling + the trace corpus are the phase-2 deliverable; RL (phase 3) is what moves it.

Speed: EPOCH-BATCHED + PARALLEL. Within an epoch the memory is frozen and all tasks run
concurrently (classical: process pool over CPU cores; agent: bounded thread pool of
concurrent API sessions); memory is updated once at the epoch boundary from that epoch's
solved cases. This is standard batched meta-learning and uses the whole machine instead of
one core. (Set max_workers=1 for the older online/sequential intra-epoch behavior.)

Hardening: resumable per-(arm,epoch,task) checkpoint, global episode cost cap, per-task error
isolation, append-only trace store. Verifier = ViennaRNA folding correctness; mature
algorithms (local_search) are baseline + optional teacher/fallback whose solution is
distilled into memory (imitation+improvement), never a sequence-match answer key.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
    return {"done": {}, "n_episodes": 0}


def _save_state(path: str, state: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- workers
def _classical_worker(payload: Dict) -> Dict:
    """Process-pool worker: solve one task with frozen strategies (no shared store)."""
    import random as _r

    from ..generators.local_search import LocalSearchGenerator

    task = DesignTask.from_dict(payload["task"])
    arm = payload["arm"]
    strategies = [Strategy.from_dict(s) for s in payload["strategies"]]
    warm = None
    if arm != "no_memory" and strategies:
        warm = [build_memory_init(task, strategies, _r.Random(payload["seed"]))]
    res = LocalSearchGenerator(restarts=8).design(task, budget=payload["budget"], seed=payload["seed"], warm_starts=warm)
    v = verify(task, res.best_sequence, compute_cross_oracle=False)
    adopt = memory_adoption(strategies, task.target, res.best_sequence) if strategies else {"A_memory": None}
    teacher_seq = None
    if payload.get("teacher") and not v.solved:
        teacher_seq = _teacher_solution(task, payload["teacher_budget"], payload["seed"])
    return {"task_id": task.task_id, "family": task.family, "n": task.n, "solved": v.solved,
            "reward": round(v.reward, 4), "oracle_calls": res.n_evals, "A": None,
            "A_memory": adopt["A_memory"], "generator": "local_search",
            "final_sequence": res.best_sequence, "teacher_seq": teacher_seq, "trace_path": None}


def _teacher_solution(task: DesignTask, budget: int, seed: int) -> Optional[str]:
    from ..generators.local_search import LocalSearchGenerator
    from ..generators.rnainverse import RNAinverseGenerator

    for gen in (RNAinverseGenerator(), LocalSearchGenerator(restarts=10)):
        res = gen.design(task, budget=budget, seed=seed)
        if verify(task, res.best_sequence, compute_cross_oracle=False).mfe_match:
            return res.best_sequence
    return None


def _agent_worker(payload: Dict) -> Dict:
    """Thread worker: one live agent episode. Memory (env override) is set once per epoch by
    the caller, so concurrent threads share the same frozen memdir (no race)."""
    from .agent_runner import run_agent_on_task

    task = DesignTask.from_dict(payload["task"])
    rec = run_agent_on_task(task, payload["model"], payload["out_dir"], payload["budget_usd"],
                            payload["timeout_s"], None, payload["knowledge"])
    if not rec.get("final_sequence") or rec.get("rc", 0) != 0:  # one retry
        rec = run_agent_on_task(task, payload["model"], payload["out_dir"], payload["budget_usd"],
                                payload["timeout_s"], None, payload["knowledge"])
    seq = rec.get("final_sequence") or ""
    strategies = [Strategy.from_dict(s) for s in payload["strategies"]]
    adopt = memory_adoption(strategies, task.target, seq) if (seq and strategies) else {"A_memory": None}
    teacher_seq = None
    if payload.get("teacher") and not rec["solved"]:
        teacher_seq = _teacher_solution(task, payload["teacher_budget"], payload["seed"])
    return {"task_id": task.task_id, "family": task.family, "n": task.n, "solved": rec["solved"],
            "reward": round(rec["reward"], 4), "oracle_calls": rec["n_rna_tool_calls"],
            "A": rec.get("actuation", {}).get("A"), "A_memory": adopt["A_memory"],
            "generator": rec["generator"], "final_sequence": seq,
            "teacher_seq": teacher_seq, "trace_path": rec.get("trace_path")}


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
    max_workers: Optional[int] = None,
) -> Dict:
    arms = arms or ["with_memory", "no_memory"]
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 8) - 2) if executor == "classical" else 6
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "campaign_state.json")
    state = _load_state(state_path) if resume else {"done": {}, "n_episodes": 0}
    trace_fp = open(os.path.join(out_dir, "campaign_traces.jsonl"), "a")
    stores = {a: MemStore(os.path.join(out_dir, f"memdir_{a}")) for a in arms}
    knowledge = ""
    if executor == "agent":
        from .agent_runner import _load_knowledge
        knowledge = _load_knowledge()

    def record(arm, epoch, row):
        key = f"{arm}:{epoch}:{row['task_id']}"
        row = {"arm": arm, "epoch": epoch, **row}
        trace_fp.write(json.dumps(row) + "\n")
        trace_fp.flush()
        state["done"][key] = row
        state["n_episodes"] += 1

    for arm in arms:
        if not resume or not any(k.startswith(f"{arm}:") for k in state["done"]):
            stores[arm].clear()
            if arm == "permuted_memory":
                stores[arm].write(permuted_playbook())
        for epoch in range(epochs):
            pending = [t for t in tasks if f"{arm}:{epoch}:{t.task_id}" not in state["done"]]
            remaining = global_episode_cap - state["n_episodes"]
            if remaining <= 0:
                print(f"[campaign] global cap {global_episode_cap} reached; stopping.", flush=True)
                trace_fp.close()
                return _summary(state, out_dir, arms, epochs)
            pending = pending[:remaining]  # hard cap: never submit more than the remaining budget
            # freeze strategies for the epoch
            strat_dicts = [s.to_dict() for s in stores[arm].strategies()]
            if pending:
                if executor == "classical":
                    payloads = [{"task": t.to_dict(), "arm": arm, "strategies": strat_dicts,
                                 "budget": classical_budget, "seed": seed + epoch,
                                 "teacher": teacher, "teacher_budget": teacher_budget} for t in pending]
                    with ProcessPoolExecutor(max_workers=max_workers) as ex:
                        for fut in as_completed([ex.submit(_classical_worker, p) for p in payloads]):
                            record(arm, epoch, fut.result())
                            _save_state(state_path, state)
                else:  # agent: set memory env once per (arm,epoch), then thread-parallel
                    if arm == "no_memory":
                        os.environ.pop("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", None)
                    else:
                        os.environ["CLAUDE_COWORK_MEMORY_PATH_OVERRIDE"] = stores[arm].dir + os.sep
                    payloads = [{"task": t.to_dict(), "arm": arm, "strategies": strat_dicts,
                                 "model": model, "out_dir": out_dir, "budget_usd": agent_budget_usd,
                                 "timeout_s": agent_timeout_s, "knowledge": knowledge,
                                 "teacher": teacher, "teacher_budget": teacher_budget, "seed": seed + epoch}
                                for t in pending]
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        for fut in as_completed([ex.submit(_agent_worker, p) for p in payloads]):
                            record(arm, epoch, fut.result())
                            _save_state(state_path, state)
                    os.environ.pop("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", None)

            # epoch-boundary memory update (with_memory only): distill from this epoch's solutions
            if arm == "with_memory":
                new: List[Strategy] = []
                for t in tasks:
                    key = f"{arm}:{epoch}:{t.task_id}"
                    if key not in state["done"]:  # skipped by the global cap this epoch
                        continue
                    r = state["done"][key]
                    learn = r["final_sequence"] if r["solved"] else r.get("teacher_seq")
                    if learn and verify(t, learn, compute_cross_oracle=False).mfe_match:
                        new += distill_strategies(t.target, learn)
                if new:
                    stores[arm].write(merge_strategies(stores[arm].strategies(), new))
            _save_state(state_path, state)
            pt = _epoch_point(state, arm, epoch, tasks, len(stores[arm].strategies()))
            print(f"[campaign] arm={arm} epoch={epoch} solve={pt['solve_rate']} "
                  f"oracle_calls={pt['mean_oracle_calls_solved']} A_memory={pt['A_memory']} "
                  f"strategies={pt['n_strategies']}", flush=True)

    trace_fp.close()
    return _summary(state, out_dir, arms, epochs)


def _epoch_point(state, arm, epoch, tasks, n_strategies) -> Dict:
    rows = [state["done"][f"{arm}:{epoch}:{t.task_id}"] for t in tasks
            if f"{arm}:{epoch}:{t.task_id}" in state["done"]]
    solved = [r for r in rows if r["solved"]]
    oc = [r["oracle_calls"] for r in solved if r["oracle_calls"]]
    am = [r["A_memory"] for r in rows if r.get("A_memory") is not None]
    return {"epoch": epoch, "n": len(rows),
            "solve_rate": round(len(solved) / max(len(rows), 1), 4),
            "mean_oracle_calls_solved": round(mean(oc), 2) if oc else None,
            "A_memory": round(mean(am), 4) if am else None, "n_strategies": n_strategies}


def _summary(state: Dict, out_dir: str, arms: List[str], epochs: int) -> Dict:
    """Curves recomputed from state (no duplicate points on resume)."""
    # reconstruct tasks-per-(arm,epoch) from done keys
    curves: Dict[str, List[Dict]] = {}
    for arm in arms:
        pts = []
        for epoch in range(epochs):
            rows = [v for k, v in state["done"].items() if k.startswith(f"{arm}:{epoch}:")]
            if not rows:
                continue
            solved = [r for r in rows if r["solved"]]
            oc = [r["oracle_calls"] for r in solved if r["oracle_calls"]]
            am = [r["A_memory"] for r in rows if r.get("A_memory") is not None]
            pts.append({"epoch": epoch, "n": len(rows),
                        "solve_rate": round(len(solved) / max(len(rows), 1), 4),
                        "mean_oracle_calls_solved": round(mean(oc), 2) if oc else None,
                        "A_memory": round(mean(am), 4) if am else None})
        curves[arm] = pts
    summ = {"n_episodes": state["n_episodes"], "curves": curves}
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
    ap.add_argument("--max-workers", type=int, default=None)
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
          f"epochs={args.epochs} arms={args.arms} workers={args.max_workers or 'auto'} teacher={args.teacher}", flush=True)
    summ = run_campaign(
        tasks, executor=args.executor, arms=args.arms, epochs=args.epochs, out_dir=args.out,
        classical_budget=args.classical_budget, agent_budget_usd=args.agent_budget_usd,
        agent_timeout_s=args.agent_timeout_s, model=args.model,
        global_episode_cap=args.global_episode_cap, teacher=args.teacher,
        resume=not args.no_resume, seed=args.seed, max_workers=args.max_workers,
    )
    for arm, curve in summ["curves"].items():
        pts = " ".join(f"e{c['epoch']}:{c['solve_rate']}/{c['mean_oracle_calls_solved']}" for c in curve)
        print(f"[campaign] {arm} (epoch:solve/oracle_calls): {pts}", flush=True)
    print(f"[campaign] episodes={summ['n_episodes']} -> {args.out}/campaign_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
