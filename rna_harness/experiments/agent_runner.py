"""RL environment: drive the FULL openclaude agent on RNA design tasks and capture the
(data, reward, trace) triple.

This is the centerpiece: openclaude runs as its complete autonomous self (all native tools
+ the RNA specialization tools), driven by any OpenRouter model. The agent solves each task
however it likes; we capture the entire stream-json trajectory and score the final sequence
with the canonical verifier. The result is model-agnostic agentic RLVR data suitable for
distilling into a local policy (RFT then GRPO).

Model portability: pass --model any OpenRouter id (anthropic/claude-opus-4.8,
deepseek/deepseek-v4-pro, z-ai/glm-5.2, qwen/...). One flag, no code change.

Usage (needs OpenRouter creds in env; runs openclaude via `bun dist/cli.mjs`):
  python -m rna_harness.experiments.agent_runner --model anthropic/claude-opus-4.8 \
      --max-len 30 --n-tasks 5 --max-budget-usd 1.0 --workers 3 --out runs/agent_opus
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from ..benchmarks import eterna100_tasks, synthesize_hybrid_tasks
from ..task import DesignTask
from ..verifier import verify

_REPO = os.environ.get("RNA_HARNESS_DIR", "/data/wenhui/openclaude_rna")
_BUN = os.environ.get("BUN_BIN", os.path.expanduser("~/.bun/bin/bun"))
_CLI = os.path.join(_REPO, "dist", "cli.mjs")
_KNOWLEDGE = os.path.join(_REPO, "rna_harness", "knowledge", "rna_expert.md")
_FINAL_RE = re.compile(r"FINAL_SEQUENCE:\s*([ACGUacgu]+)")


def _load_knowledge() -> str:
    try:
        with open(_KNOWLEDGE) as f:
            return f.read()
    except Exception:
        return ""


def _build_prompt(task: DesignTask, knowledge: str) -> str:
    from ..rl.prompts import render_task_prompt

    return (
        f"{knowledge}\n\n---\n\n# Your task\n{render_task_prompt(task)}\n\n"
        f"Work autonomously. Use RnaVerify to check candidates and its diagnosis to guide "
        f"repairs; you may also use RnaFold/RnaDiagnose or write ViennaRNA code. "
        f"When RnaVerify reports solved=true (or you exhaust your budget), output a final "
        f"line EXACTLY of the form:\nFINAL_SEQUENCE: <your {task.n}-nt sequence>"
    )


def _extract_final_sequence(trace_lines: List[str], n: int) -> Optional[str]:
    """Extract the agent's submitted sequence from the LAST explicit `FINAL_SEQUENCE:` marker.

    Conservative by design: we only accept the agent's own declared answer. We do NOT scrape
    arbitrary n-mers from the trace, because tool-result blocks (e.g. RnaVerify echoing a
    sequence) would otherwise be misread as the agent's submission -> false positives that
    corrupt scoring. No marker -> None (counts as no submission)."""
    texts: List[str] = []
    for line in trace_lines:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        blob = json.dumps(ev)
        m = _FINAL_RE.search(blob.replace("\\n", "\n"))
        if m:
            texts.append(m.group(1).upper().replace("T", "U"))
    return texts[-1] if texts else None


def _count_tool_calls(trace_lines: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for line in trace_lines:
        for name in re.findall(r'"name":"([A-Za-z_]+)"', line):
            counts[name] = counts.get(name, 0) + 1
    return counts


def run_agent_on_task(
    task: DesignTask,
    model: str,
    out_dir: str,
    max_budget_usd: float = 1.0,
    timeout_s: int = 600,
    allowed_tools: Optional[List[str]] = None,
    knowledge: str = "",
    cross_oracle: bool = True,
) -> Dict:
    allowed_tools = allowed_tools or ["RnaVerify", "RnaFold", "RnaDiagnose", "RnaConstraints", "Bash"]
    prompt = _build_prompt(task, knowledge)
    env = dict(os.environ)
    env.update({
        "CLAUDE_CODE_USE_OPENAI": "1",
        "OPENAI_BASE_URL": env.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        "OPENAI_MODEL": model,
        "OPENCLAUDE_RNA": "1",
        "RNA_HARNESS_DIR": _REPO,
        "RNA_HARNESS_PYTHON": env.get("RNA_HARNESS_PYTHON", "/home/ubuntu/miniconda3/envs/rna/bin/python"),
    })
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", task.task_id)
    trace_path = os.path.join(out_dir, "traces", f"{safe_id}.jsonl")
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)

    cmd = [
        _BUN, _CLI, "-p", prompt,
        "--output-format", "stream-json", "--verbose",
        "--dangerously-skip-permissions",
        "--allowedTools", *allowed_tools,
        "--max-budget-usd", str(max_budget_usd),
        "--no-session-persistence",
    ]
    t0 = time.time()
    # Run bun in its OWN process group (start_new_session) so a timeout can kill the whole
    # tree -- openclaude spawns children (python bridge, etc.) that would otherwise orphan and
    # hang. subprocess.run's plain timeout only kills the direct child.
    proc = subprocess.Popen(
        cmd, env=env, cwd=out_dir, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # kill the whole group
        except Exception:
            proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except Exception:
            stdout, stderr = "", "TIMEOUT"
        rc = -1
    elapsed = round(time.time() - t0, 1)

    lines = [l for l in (stdout or "").splitlines() if l.strip()]
    with open(trace_path, "w") as f:
        f.write(stdout or "")

    final_seq = _extract_final_sequence(lines, task.n)
    v = verify(task, final_seq, compute_cross_oracle=cross_oracle) if final_seq else None
    tool_calls = _count_tool_calls(lines)
    # Actuation fidelity (A) reconstructed from the REAL agent tool-call chain.
    from ..actuation.trace import actuation_from_agent_trace

    events = []
    for l in lines:
        try:
            events.append(json.loads(l))
        except Exception:
            continue
    actuation = actuation_from_agent_trace(events)
    return {
        "task_id": task.task_id,
        "family": task.family,
        "n": task.n,
        "model": model,
        "final_sequence": final_seq,
        "solved": bool(v.solved) if v else False,
        "reward": v.reward if v else 0.0,
        "cross_oracle": (v.cross_oracle or {}).get("fraction") if v and v.cross_oracle else None,
        "tool_calls": tool_calls,
        "n_rna_tool_calls": sum(c for k, c in tool_calls.items() if k.startswith("Rna")),
        "actuation": actuation,
        "n_trace_events": len(lines),
        "trace_path": trace_path,
        "elapsed_s": elapsed,
        "rc": rc,
        "stderr_tail": (stderr or "")[-300:],
    }


def _agentic_rlvr_record(rec: Dict, task: DesignTask, knowledge: str) -> Dict:
    """One agentic RLVR sample: the whole trajectory + verifiable reward."""
    with open(rec["trace_path"]) as f:
        trace = f.read()
    return {
        "task_id": rec["task_id"],
        "task_family": rec["family"],
        "target": task.target,
        "prompt": _build_prompt(task, knowledge),
        "final_sequence": rec["final_sequence"],
        "reward": rec["reward"],
        "solved": rec["solved"],
        "generator": f"agent:{rec['model'].split('/')[-1]}",
        "trace_stream_json": trace,           # the full agentic trajectory (tool calls, reasoning)
        "tool_calls": rec["tool_calls"],
        "cross_oracle_consistency": rec["cross_oracle"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-opus-4.8")
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--max-budget-usd", type=float, default=1.0)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="runs/agent")
    args = ap.parse_args(argv)

    if not os.path.exists(_CLI):
        print(json.dumps({"error": f"openclaude not built at {_CLI}; run `bun run build`"}))
        return 2

    tasks = eterna100_tasks(ids=args.ids, max_len=args.max_len)
    if args.hybrid:
        base = tasks[: max(1, args.n_tasks)]
        tasks = []
        for b in base:
            tasks += synthesize_hybrid_tasks(b, profiles=["gc_high", "motif"])
    tasks = sorted(tasks, key=lambda t: t.n)[: args.n_tasks]
    os.makedirs(args.out, exist_ok=True)
    knowledge = _load_knowledge()

    print(f"[agent] model={args.model} tasks={len(tasks)} workers={args.workers}", flush=True)
    records: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_agent_on_task, t, args.model, args.out, args.max_budget_usd, args.timeout_s, None, knowledge): t
            for t in tasks
        }
        for fut in as_completed(futs):
            t = futs[fut]
            rec = fut.result()
            records.append(rec)
            print(f"  [{rec['n']:3d}nt] {t.meta.get('name', t.task_id)[:26]:26s} "
                  f"solved={str(rec['solved']):5s} reward={rec['reward']:.2f} "
                  f"rna_calls={rec['n_rna_tool_calls']} ({rec['elapsed_s']}s)", flush=True)

    # export agentic RLVR + summary
    by_id = {t.task_id: t for t in tasks}
    rlvr_path = os.path.join(args.out, "agentic_rlvr.jsonl")
    with open(rlvr_path, "w") as f:
        for rec in records:
            f.write(json.dumps(_agentic_rlvr_record(rec, by_id[rec["task_id"]], knowledge)) + "\n")
    solved = sum(r["solved"] for r in records)
    a_vals = [r["actuation"]["A"] for r in records if r.get("actuation", {}).get("A") is not None]
    am_vals = [r["actuation"]["A_model"] for r in records if r.get("actuation", {}).get("A_model") is not None]
    summary = {
        "model": args.model,
        "n_tasks": len(records),
        "solved": solved,
        "solve_rate": round(solved / max(len(records), 1), 4),
        "mean_reward": round(sum(r["reward"] for r in records) / max(len(records), 1), 4),
        "mean_rna_calls": round(sum(r["n_rna_tool_calls"] for r in records) / max(len(records), 1), 2),
        # actuation fidelity on the REAL agent (only tasks that needed >=1 repair step count)
        "agent_A": round(sum(a_vals) / len(a_vals), 4) if a_vals else None,
        "agent_A_model": round(sum(am_vals) / len(am_vals), 4) if am_vals else None,
        "n_tasks_with_repair": len(a_vals),
        "records": records,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[agent] solve_rate={summary['solve_rate']} mean_reward={summary['mean_reward']} "
          f"-> {rlvr_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
