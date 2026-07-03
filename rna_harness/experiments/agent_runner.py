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
import hashlib
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


def _build_prompt(task: DesignTask, knowledge: str, enable_mcp: bool = True) -> str:
    from ..rl.prompts import render_task_prompt

    mcp_hint = (
        " You also have MCP tools. For a design PLAN: mcp__rna_motifs__analyze_target (per-hairpin "
        "plan + hard-feature flags), mcp__rna_motifs__lookup_hairpin_loop (stable cap motifs), "
        "mcp__rna_motifs__search_motifs (glossary). For COMPUTE (fast, many CPU cores in parallel "
        "in ONE call): mcp__rna_motifs__parallel_search runs a many-restart search that often solves "
        "the target outright -- prefer it over hand-writing a search loop; and "
        "mcp__rna_motifs__batch_fold screens thousands of candidates at once. Always confirm the "
        "final sequence with RnaVerify (the official scorer)."
        if enable_mcp else ""
    )
    return (
        f"{knowledge}\n\n---\n\n# Your task\n{render_task_prompt(task)}\n\n"
        f"Work autonomously. Use RnaVerify to check candidates and its diagnosis to guide "
        f"repairs; you may also use RnaFold/RnaDiagnose or write ViennaRNA code.{mcp_hint}\n\n"
        f"OUTPUT PROTOCOL (critical -- read carefully):\n"
        f"1. Your sequence MUST be EXACTLY {task.n} nucleotides. Count every base. A sequence that "
        f"is even 1 nt too short or too long is an AUTOMATIC FAILURE, no matter how good the shape.\n"
        f"2. As SOON as you have any plausible candidate, output a line EXACTLY of the form:\n"
        f"   FINAL_SEQUENCE: <your {task.n}-nt sequence>\n"
        f"   Then keep improving it and RE-EMIT an updated 'FINAL_SEQUENCE:' line every time you "
        f"find a better candidate. Always keep your current BEST as the most recent FINAL_SEQUENCE "
        f"line -- if you run low on time, your best work is then already on record (the LAST "
        f"FINAL_SEQUENCE line is what gets scored).\n"
        f"3. Stop once RnaVerify reports solved=true."
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
        if isinstance(ev, dict) and ev.get("type") == "rna_memory_snapshot":
            continue  # our injected memory-audit header, not agent output
        blob = json.dumps(ev)
        m = _FINAL_RE.search(blob.replace("\\n", "\n"))
        if m:
            texts.append(m.group(1).upper().replace("T", "U"))
    return texts[-1] if texts else None


def _count_tool_calls(trace_lines: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for line in trace_lines:
        if '"rna_memory_snapshot"' in line:
            continue  # skip our memory-audit header so its JSON isn't miscounted as tool calls
        for name in re.findall(r'"name":"([A-Za-z_]+)"', line):
            counts[name] = counts.get(name, 0) + 1
    return counts


def _best_effort_candidate(trace_lines: List[str], task: DesignTask, declared: Optional[str]) -> Optional[Dict]:
    """Best right-length sequence the agent ACTUALLY tested (any candidate passed to a tool call),
    scored by the verifier. The strict `solved` metric still requires a DECLARED FINAL_SEQUENCE, but
    this captures true capability even when the declared answer was lost to a wall-clock timeout --
    i.e. 'the agent found a working design it never delivered'."""
    cands = set()
    if declared:
        cands.add(declared.upper().replace("T", "U"))
    for line in trace_lines:
        if '"rna_memory_snapshot"' in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if not (isinstance(ev, dict) and ev.get("type") == "assistant"):
            continue
        for b in (ev.get("message", {}) or {}).get("content", []) or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                for s in re.findall(r"[ACGUacgu]{6,}", json.dumps(b.get("input") or {})):
                    su = s.upper().replace("T", "U")
                    if len(su) == task.n:
                        cands.add(su)
    if not cands:
        return None
    try:
        import RNA
        md = RNA.md(); md.temperature = 37.0; md.dangles = 2
        best = None
        for s in cands:
            st, _ = RNA.fold_compound(s, md).mfe()
            bp = RNA.bp_distance(st, task.target)
            if best is None or bp < best[1]:
                best = (s, bp)
    except Exception:
        return None
    seq, bp = best
    bv = verify(task, seq, compute_cross_oracle=False)
    return {"sequence": seq, "bp_distance": bp, "solved": bool(bv.solved),
            "reward": bv.reward, "n_candidates_tested": len(cands)}


_MEMORY_ENTRYPOINT = "MEMORY.md"  # openclaude injects ONLY <memdir>/MEMORY.md into the prompt


def _memory_snapshot() -> Optional[Dict]:
    """Capture the EXACT auto-memory openclaude injects into the agent's system prompt for this
    episode, so memory recall is auditable from the trace alone (it otherwise leaves no
    footprint -- the system prompt is not in the stream-json). Per the openclaude source
    (src/memdir/paths.ts, src/utils/claudemd.ts) only <memdir>/MEMORY.md is injected (individual
    strategy files are NOT), truncated at 200 lines / 25000 bytes. Returns None when no memdir
    override is active (e.g. the no_memory arm), which is itself the auditable fact 'no memory'."""
    memdir = os.environ.get("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE")
    if not memdir:
        return None
    mem_path = os.path.join(memdir, _MEMORY_ENTRYPOINT)
    content, exists = "", os.path.exists(mem_path)
    if exists:
        try:
            with open(mem_path, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = ""
    injected = content[:25000]  # mirror openclaude's MAX_ENTRYPOINT_BYTES cap
    try:
        files = sorted(fn for fn in os.listdir(memdir) if fn.endswith(".md"))
    except Exception:
        files = []
    return {
        "memdir": memdir,
        "memory_md_path": mem_path,
        "injected": exists,
        "sha256": hashlib.sha256(injected.encode("utf-8")).hexdigest(),
        "n_chars": len(injected),
        "n_lines": (injected.count("\n") + 1) if injected else 0,
        "content": injected,
        "memdir_md_files": files,
        "note": "openclaude injects ONLY MEMORY.md into the agent system prompt; other files in "
                "memdir are present but not auto-injected by default.",
    }


def _mcp_config(python_bin: str) -> str:
    """Inline --mcp-config JSON connecting the deterministic RNA motif knowledge base as a real
    MCP server. openclaude's stdio schema has no `cwd` field and agent_runner launches the CLI
    with cwd=out_dir, so PYTHONPATH makes the package importable regardless of cwd. Tools appear
    to the agent as mcp__rna_motifs__{lookup_hairpin_loop,search_motifs,analyze_target}."""
    return json.dumps({
        "mcpServers": {
            "rna_motifs": {
                "type": "stdio",
                "command": python_bin,
                "args": ["-m", "rna_harness.mcp.motif_server"],
                # RNA_MAX_CORES caps the batch_fold/parallel_search pools (leave headroom for
                # other users on the shared host); PYTHONPATH makes the package importable.
                "env": {"PYTHONPATH": _REPO, "RNA_MAX_CORES": os.environ.get("RNA_MAX_CORES", "160")},
            }
        }
    })


def run_agent_on_task(
    task: DesignTask,
    model: str,
    out_dir: str,
    max_budget_usd: float = 1.0,
    timeout_s: int = 600,
    allowed_tools: Optional[List[str]] = None,
    knowledge: str = "",
    cross_oracle: bool = True,
    enable_mcp: bool = True,
) -> Dict:
    allowed_tools = allowed_tools or ["RnaVerify", "RnaFold", "RnaDiagnose", "RnaConstraints", "Bash"]
    prompt = _build_prompt(task, knowledge, enable_mcp=enable_mcp)
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
    ]
    if enable_mcp:
        # connect the RNA motif KB as a real MCP server. mcp__rna_motifs is added as an explicit
        # server-level allow rule (redundant under --dangerously-skip-permissions but clarifies
        # intent and works even without bypass); --strict-mcp-config keeps it a clean, single,
        # deterministic server (no merged .mcp.json/settings servers).
        allowed_tools = allowed_tools + ["mcp__rna_motifs"]
        cmd += ["--mcp-config", _mcp_config(env["RNA_HARNESS_PYTHON"]), "--strict-mcp-config"]
    cmd += [
        "--allowedTools", *allowed_tools,
        "--max-budget-usd", str(max_budget_usd),
        "--no-session-persistence",
    ]
    # Snapshot the auto-memory the agent will actually see THIS episode (auditability); taken
    # before launch because the online loop may rewrite MEMORY.md between episodes.
    mem_snap = _memory_snapshot()
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
        # in-trace memory footprint: the exact MEMORY.md injected into the agent's system prompt
        # (all values JSON-encoded so this synthetic line is never miscounted as a tool call).
        if mem_snap is not None:
            f.write(json.dumps({"type": "rna_memory_snapshot", **mem_snap}) + "\n")
        f.write(stdout or "")

    declared = _extract_final_sequence(lines, task.n)
    final_seq = declared
    v = verify(task, final_seq, compute_cross_oracle=cross_oracle) if final_seq else None
    best_eff = _best_effort_candidate(lines, task, declared)
    # Salvage (A_harness closure): if the agent TESTED a verified-solved sequence but the declared
    # FINAL_SEQUENCE was missing/unsolved (lost to a wall-clock timeout or a format slip), deliver
    # that oracle-verified candidate as the answer. This closes the delivery seam -- the same
    # Diagnosis->Actuation gap that pins A_harness in the failure literature -- without inventing
    # anything: the sequence is one the agent itself produced and the oracle confirmed. Audited via
    # the `salvaged` flag so declared-vs-recovered wins stay distinguishable.
    salvaged = False
    if best_eff and best_eff.get("solved") and not (v and v.solved):
        final_seq = best_eff["sequence"]
        v = verify(task, final_seq, compute_cross_oracle=cross_oracle)
        salvaged = True
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
    # compact memory footprint for state/summary (full MEMORY.md content stays in the trace file)
    mem_compact = {k: val for k, val in mem_snap.items() if k != "content"} if mem_snap else None
    return {
        "task_id": task.task_id,
        "family": task.family,
        "n": task.n,
        "model": model,
        "final_sequence": final_seq,
        "solved": bool(v.solved) if v else False,
        "reward": v.reward if v else 0.0,
        "delivered": bool(declared),                        # did the agent itself emit a FINAL_SEQUENCE?
        "salvaged": salvaged,                               # answer recovered from a tested candidate (A_harness)
        "best_effort": best_eff,                            # best right-length candidate it tested
        "capability_solved": bool(best_eff and best_eff["solved"]),  # solved incl. undelivered work
        "cross_oracle": (v.cross_oracle or {}).get("fraction") if v and v.cross_oracle else None,
        "tool_calls": tool_calls,
        "n_rna_tool_calls": sum(c for k, c in tool_calls.items() if k.startswith("Rna")),
        "n_mcp_calls": sum(c for k, c in tool_calls.items() if k.startswith("mcp__")),
        "memory_snapshot": mem_compact,
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
    n = max(len(records), 1)
    cap_solved = sum(1 for r in records if r.get("capability_solved"))
    delivered = sum(1 for r in records if r.get("delivered"))
    # "found but not delivered": the agent tested a verified solution it never emitted as FINAL
    lost = sum(1 for r in records if r.get("capability_solved") and not r.get("solved"))
    a_vals = [r["actuation"]["A"] for r in records if r.get("actuation", {}).get("A") is not None]
    am_vals = [r["actuation"]["A_model"] for r in records if r.get("actuation", {}).get("A_model") is not None]
    summary = {
        "model": args.model,
        "n_tasks": len(records),
        "solved": solved,
        "solve_rate": round(solved / n, 4),                       # STRICT: delivered + verified
        "capability_solved": cap_solved,                          # incl. best-effort undelivered
        "capability_rate": round(cap_solved / n, 4),
        "found_not_delivered": lost,                              # solved-but-lost (should be ~0 now)
        "delivered": delivered,
        "mean_reward": round(sum(r["reward"] for r in records) / n, 4),
        "mean_rna_calls": round(sum(r["n_rna_tool_calls"] for r in records) / n, 2),
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
