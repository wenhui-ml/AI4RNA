"""P1 — de-scaffolding FLOOR vs CEILING, matched single-shot, multi-repeat (Paper C).

Measures how much of realized capability is the SCAFFOLD (hand-coded tools + prompt ladder +
expert knowledge) vs the model's own autonomous loop. ONE process = one (COND, TASKSET, REPEAT):

  COND=ceiling : crutched  -> RNA_AUTONOMOUS off, full prompt ladder, expert knowledge  (flag-off byte-path)
  COND=floor   : autonomous-> RNA_AUTONOMOUS=1,  minimal prompt,     knowledge=""        (neutral facts-only)

Held constant across both so the ONLY difference is the scaffold axis:
  * EMPTY memory (CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 + empty memdir override) — memory is P2, not here.
  * single attempt per task (EPOCHS=1, no meta accumulation, no skill distillation).
  * same per-task wall budget (TIMEOUT), cross_oracle=False, MCP on, WebSearch/WebFetch stay disabled.

"seeds": OpenRouter/openclaude expose no seed knob, so REPEAT=1..3 are INDEPENDENT REPETITIONS
(LLM sampling stochasticity), each in its own OUT dir; their spread gives the CI. Report as
"independent repetitions", NOT "RNG seeds".

Denominator fix (the P0 methodological catch): solve-rate = solved / len(taskset). For full100 that is
the CLEAN autonomous floor (NOT 65 crutch-solved + k), unlike the earlier auto_frozen tail-only run.

Env: COND=ceiling|floor  TASKSET=tail35|full100  REPEAT=1  [M2_MODEL, TIMEOUT, WORKERS, OUT]
Guarded __main__ (fork-bomb safe). Crutched refs (context, not matched): memON=65, meta=76.
"""
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "/data/wenhui/openclaude_rna")
os.environ["PYTHONPATH"] = "/data/wenhui/openclaude_rna"
os.environ.setdefault("RNA_MAX_CORES", "16")

# the 35 hard-tail = Eterna100-V2 ids unsolved by the crutched baseline (same set the meta ladder used)
TAIL35 = [7, 9, 16, 22, 35, 50, 52, 57, 61, 64, 65, 68, 71, 72, 73, 74, 76, 78, 79, 80, 81, 83,
          85, 86, 87, 89, 90, 91, 92, 94, 96, 97, 98, 99, 100]

COND = os.environ.get("COND", "floor").lower()
assert COND in ("floor", "ceiling"), COND
TASKSET = os.environ.get("TASKSET", "tail35").lower()
assert TASKSET in ("tail35", "full100"), TASKSET
REPEAT = int(os.environ.get("REPEAT", "1"))
MODEL = os.environ.get("M2_MODEL", "z-ai/glm-5.2")
TIMEOUT = int(os.environ.get("TIMEOUT", "1500"))
WORKERS = int(os.environ.get("WORKERS", "3"))
OUT = os.environ.get("OUT", f"/data/wenhui/openclaude_rna/rna_harness/runs/p1_{COND}_{TASKSET}_r{REPEAT}")
MEMDIR = os.path.join(OUT, "memdir")

# --- scaffold axis: set ONCE at process level (stable for all workers -> no env race) ---
if COND == "floor":
    os.environ["RNA_AUTONOMOUS"] = "1"      # propagates to the MCP subprocess via agent_runner env-copy
    MINIMAL, USE_KNOWLEDGE = True, False
else:
    os.environ.pop("RNA_AUTONOMOUS", None)  # flag-off = today's crutched harness, byte-identical path
    MINIMAL, USE_KNOWLEDGE = False, True
# memory held empty & inert for BOTH arms (memory effects are P2)
os.environ["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
os.environ["CLAUDE_COWORK_MEMORY_PATH_OVERRIDE"] = MEMDIR + os.sep


def _tok_estimate(task_id):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
    p = os.path.join(OUT, "traces", f"{safe}.jsonl")
    if not os.path.exists(p):
        return 0
    return sum(len(json.dumps(json.loads(l).get("message", {}))) for l in open(p)
               if l.strip() and l.lstrip().startswith("{")) // 4


def main():
    from rna_harness.benchmarks.eterna100 import eterna100_tasks
    from rna_harness.experiments.agent_runner import run_agent_on_task, _load_knowledge

    os.makedirs(os.path.join(OUT, "traces"), exist_ok=True)
    os.makedirs(MEMDIR, exist_ok=True)
    open(os.path.join(MEMDIR, "MEMORY.md"), "w").write("# (no accumulated memory)\n")

    knowledge = _load_knowledge() if USE_KNOWLEDGE else ""
    ids = TAIL35 if TASKSET == "tail35" else list(range(1, 101))
    tasks = eterna100_tasks(ids=ids)
    denom = len(tasks)
    print(f"[P1 {COND}/{TASKSET}/r{REPEAT}] RNA_AUTONOMOUS={os.environ.get('RNA_AUTONOMOUS','<off>')} "
          f"minimal_prompt={MINIMAL} knowledge={'full' if USE_KNOWLEDGE else 'none'} | {denom} tasks | "
          f"model={MODEL} timeout={TIMEOUT} workers={WORKERS} | denom={denom} (raw solve-rate, no baseline add)",
          flush=True)

    solved = {}                       # task_id -> winning seq
    pertask = {}
    solved_json = os.path.join(OUT, "solved.json")
    if os.path.exists(solved_json):   # resume within THIS repeat only (independent OUT dir per repeat)
        try:
            solved.update(json.load(open(solved_json)))
        except Exception:
            pass

    def write_summary(provisional=True):
        n_solved = len(solved)
        json.dump({"paper": "C", "phase": "P1", "cond": COND, "taskset": TASKSET, "repeat": REPEAT,
                   "model": MODEL, "autonomous": COND == "floor", "minimal_prompt": MINIMAL,
                   "knowledge": "full" if USE_KNOWLEDGE else "none", "empty_memory": True,
                   "n_tasks": denom, "solved": n_solved, "solve_rate": round(n_solved / denom, 4),
                   "provisional": provisional, "timeout_s": TIMEOUT,
                   "records": [{"task_id": tid, "n": v.get("n"), "solved": tid in solved,
                                "elapsed_s": v.get("secs"), "tool_calls": v.get("tool_calls"),
                                "best_effort": v.get("best_effort"), "tokens_est": v.get("tokens_est")}
                               for tid, v in pertask.items()]},
                  open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    def attempt(t):
        rec = run_agent_on_task(t, MODEL, OUT, 1.5, TIMEOUT, None, knowledge,
                                cross_oracle=False, enable_mcp=True, minimal_prompt=MINIMAL)
        return t, rec

    remaining = [t for t in tasks if t.task_id not in solved]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(attempt, t): t for t in remaining}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                _, rec = fut.result()
            except Exception as e:
                rec = {"solved": False, "final_sequence": "", "n_rna_tool_calls": None,
                       "elapsed_s": None, "best_effort": None}
                print(f"    [{t.task_id}] EXC {e}", flush=True)
            ok = bool(rec.get("solved"))
            pt = pertask.setdefault(t.task_id, {})
            pt.update({"n": t.n, "tool_calls": rec.get("n_rna_tool_calls"), "secs": rec.get("elapsed_s"),
                       "best_effort": rec.get("best_effort"), "tokens_est": _tok_estimate(t.task_id)})
            if ok:
                solved[t.task_id] = rec.get("final_sequence") or ""
                json.dump(solved, open(solved_json, "w"))
            done += 1
            print(f"    [{t.n:>4}nt] {t.task_id.split(':')[-1][:24]:24s} {'SOLVED' if ok else '.':6s} "
                  f"{len(solved)}/{denom} ({done}/{len(remaining)} done) calls={rec.get('n_rna_tool_calls')} "
                  f"{rec.get('elapsed_s')}s", flush=True)
            write_summary(provisional=True)

    write_summary(provisional=False)
    print(f"\n===== P1 {COND}/{TASKSET}/r{REPEAT} DONE: {len(solved)}/{denom} "
          f"= {round(100*len(solved)/denom,1)}%  ({round(time.time()-t0,1)}s) =====", flush=True)


if __name__ == "__main__":
    main()
