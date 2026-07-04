"""Offline HONEST re-scoring of an agent_runner run (measurement integrity, Tier-0).

Reads an existing run's traces and re-derives capability WITHOUT re-driving the agent:
for each task we extract *every* correct-length sequence the agent actually produced
(declared FINAL_SEQUENCE lines of any length + every A/C/G/U run of length == n inside
its tool-call inputs), canonical-`verify()` each, and report three separate numbers:

  delivered_solved    : the agent's own last FINAL_SEQUENCE canonically solves
  capability_solved   : ANY correct-length candidate the agent tested canonically solves
  found_not_delivered : capability_solved and NOT delivered_solved  (delivery/format leak)

This is pure measurement -- it does not modify the agent's answer, it only scores what the
agent already did with the deterministic oracle. It exists because the shipped
`_best_effort_candidate` verifies only the single min-bp_distance candidate, so a genuine
solved-but-mis-delivered task (e.g. Eterna100 "Fractile": a working 105-nt design emitted as
a length-113 FINAL_SEQUENCE) is scored 0 and lost from the capability metric.

Usage:
  python -m rna_harness.experiments.rescore_traces --run rna_harness/runs/glm52_decomp_hardcore31
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional

from ..benchmarks import eterna100_tasks
from ..task import DesignTask
from ..verifier import verify

_FINAL_RE = re.compile(r"FINAL_SEQUENCE:\s*([ACGUacgu]+)")
_ACGU_RE = re.compile(r"[ACGUacgu]{6,}")


def _clean(s: str) -> str:
    return s.upper().replace("T", "U")


def _load_lines(trace_path: str) -> List[str]:
    try:
        with open(trace_path) as f:
            return [l for l in f.read().splitlines() if l.strip()]
    except Exception:
        return []


def _declared_sequences(lines: List[str]) -> List[str]:
    """Every FINAL_SEQUENCE the agent emitted (any length), in order; last = what was scored."""
    out: List[str] = []
    for line in lines:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if isinstance(ev, dict) and ev.get("type") == "rna_memory_snapshot":
            continue
        blob = json.dumps(ev).replace("\\n", "\n")
        for m in _FINAL_RE.finditer(blob):
            out.append(_clean(m.group(1)))
    return out


def _correct_length_candidates(lines: List[str], n: int) -> List[str]:
    """Every A/C/G/U run of length exactly n that appears in an assistant tool_use input.

    Mirrors _best_effort_candidate's extraction surface (candidates the agent ACTUALLY passed
    to a tool) so we only ever credit sequences the agent produced -- never scraped noise."""
    cands = set()
    for line in lines:
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
                for s in _ACGU_RE.findall(json.dumps(b.get("input") or {})):
                    su = _clean(s)
                    if len(su) == n:
                        cands.add(su)
    return sorted(cands)


def rescore_task(task: DesignTask, trace_path: str, max_candidates: int = 400) -> Dict:
    lines = _load_lines(trace_path)
    declared = _declared_sequences(lines)
    last_declared = declared[-1] if declared else None
    last_declared_len = len(last_declared) if last_declared else None

    # what the harness actually scored: the LAST declared line, whatever its length
    delivered_solved = False
    if last_declared:
        dv = verify(task, last_declared, compute_cross_oracle=False)
        delivered_solved = bool(dv.solved)

    # capability: canonical-verify EVERY correct-length candidate (incl. a correct-length
    # declared line, if any), report the best
    cands = set(_correct_length_candidates(lines, task.n))
    for d in declared:
        if len(d) == task.n:
            cands.add(d)
    cand_list = sorted(cands)[:max_candidates]

    best = None  # (reward, solved, mfe_match, is_umfe, bp, seq)
    any_solved = False
    solved_seq = None
    for s in cand_list:
        v = verify(task, s, compute_cross_oracle=False)
        bp = v.ensemble.get("bp_distance_to_target") if v.ensemble else None
        key = (v.reward, v.solved)
        if best is None or v.reward > best[0]:
            best = (v.reward, v.solved, v.mfe_match, v.is_umfe, bp, s)
        if v.solved and not any_solved:
            any_solved = True
            solved_seq = s

    capability_solved = any_solved
    found_not_delivered = bool(capability_solved and not delivered_solved)
    return {
        "task_id": task.task_id,
        "n": task.n,
        "n_correct_len_candidates": len(cand_list),
        "n_declared": len(declared),
        "last_declared_len": last_declared_len,
        "length_mismatch": bool(last_declared and last_declared_len != task.n),
        "delivered_solved": delivered_solved,
        "capability_solved": capability_solved,
        "found_not_delivered": found_not_delivered,
        "best_reward": round(best[0], 4) if best else None,
        "best_bp": best[4] if best else None,
        "best_mfe_match": best[2] if best else None,
        "best_is_umfe": best[3] if best else None,
        "solved_sequence": solved_seq,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing summary.json + traces/")
    ap.add_argument("--out", default=None, help="where to write rescored.json (default: <run>/rescored.json)")
    args = ap.parse_args(argv)

    summ_path = os.path.join(args.run, "summary.json")
    summary = json.load(open(summ_path))
    recs = summary.get("records", [])
    # index every eterna100 task by task_id (default RewardConfig -> solved_criterion='mfe_match')
    all_tasks = {t.task_id: t for t in eterna100_tasks()}

    out_recs: List[Dict] = []
    for r in sorted(recs, key=lambda z: z.get("n", 0)):
        tid = r.get("task_id")
        task = all_tasks.get(tid)
        if task is None:
            continue
        tp = r.get("trace_path")
        if not (tp and os.path.exists(tp)):
            tp = os.path.join(args.run, "traces",
                              re.sub(r"[^A-Za-z0-9_.-]", "_", tid) + ".jsonl")
        rr = rescore_task(task, tp)
        rr["prev_solved"] = bool(r.get("solved"))
        rr["prev_capability_solved"] = bool(r.get("capability_solved"))
        rr["prev_best_bp"] = (r.get("best_effort") or {}).get("bp_distance")
        out_recs.append(rr)

    n = len(out_recs)
    prev_solved = sum(1 for r in out_recs if r["prev_solved"])
    prev_cap = sum(1 for r in out_recs if r["prev_capability_solved"])
    deliv = sum(1 for r in out_recs if r["delivered_solved"])
    cap = sum(1 for r in out_recs if r["capability_solved"])
    fnd = sum(1 for r in out_recs if r["found_not_delivered"])
    lenmis = sum(1 for r in out_recs if r["length_mismatch"])
    agg = {
        "run": args.run,
        "n_tasks": n,
        "prev_solved": prev_solved,
        "prev_capability_solved": prev_cap,
        "honest_delivered_solved": deliv,
        "honest_capability_solved": cap,
        "found_not_delivered": fnd,
        "n_length_mismatch_final": lenmis,
        "records": out_recs,
    }
    out_path = args.out or os.path.join(args.run, "rescored.json")
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"=== HONEST RE-SCORE: {args.run} ===")
    print(f"tasks={n}")
    print(f"  prev (shipped summary):   solved={prev_solved}  capability={prev_cap}")
    print(f"  honest delivered_solved:  {deliv}")
    print(f"  honest capability_solved: {cap}   (+{cap - prev_cap} vs shipped)")
    print(f"  found_not_delivered:      {fnd}   (solved in-trace, lost at delivery)")
    print(f"  final line wrong length:  {lenmis}")
    print()
    hdr = f"{'n':>4} {'prevSolv':8} {'deliv':6} {'capab':6} {'FND':4} {'bestbp':6} {'umfe':5} {'declLen':7} :: task"
    print(hdr)
    for r in out_recs:
        flag = "  <<< FOUND-NOT-DELIVERED" if r["found_not_delivered"] else ""
        print(f"{r['n']:>4} {str(r['prev_solved'])[:1]:8} {str(r['delivered_solved'])[:1]:6} "
              f"{str(r['capability_solved'])[:1]:6} {str(r['found_not_delivered'])[:1]:4} "
              f"{str(r['best_bp']):6} {str(r['best_is_umfe'])[:1]:5} {str(r['last_declared_len']):7} :: "
              f"{r['task_id'].replace('eterna100:','')}{flag}")
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
