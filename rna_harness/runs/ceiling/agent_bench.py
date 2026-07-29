#!/usr/bin/env python3
"""Run the agent loop against the generated suite and score it on what actually distinguishes a loop.

A single "solve rate" would hide the thing worth knowing. Three classes with three different correct
behaviours, and one column that matters more than the score:

    direct          pins compatible with the obvious shape  -> must certify, cheaply
    restructurable  pins conflict, length not fixed         -> must certify by CHANGING THE PROBLEM
    impossible      pins conflict, length fixed             -> must REFUSE, and name the conflict

    avoided vs recovered   on a restructurable item, did the agent see the conflict coming and route around
                           it, or hit it and change course? Only the second is Claude Code's mechanism. The
                           one live trajectory this project has is an avoidance, so "the loop works" is so
                           far a claim about anticipation, not about recovery.

Detection is mechanical, not judged: the naive shape has n = 2*stem + loop, so a first proposal longer than
that is an anticipation, and a first proposal equal to it followed by a conflict observation and a second
proposal is a recovery.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
OUT = os.path.join(HERE, "agent_bench")


def classify_trajectory(item, sess) -> dict:
    """What the agent actually did, from the trace alone."""
    naive_n = 2 * item["shape"]["stem"] + item["shape"]["loop"]
    proposals = [s for s in sess.steps if s.action == "propose_structure" and s.observation.get("ok")]
    saw_conflict = any(s.action == "check_feasible" and s.observation.get("conflicts")
                       for s in sess.steps)
    failed_design = any(s.action == "design" and s.observation.get("verdict")
                        not in ("certified", "knot_confirmed") for s in sess.steps)
    n_design = sum(1 for s in sess.steps if s.action == "design")
    first_n = proposals[0].observation["n"] if proposals else None
    route = "none"
    if proposals:
        if first_n > naive_n:
            route = "anticipated"                       # never built the shape that conflicts
        elif len(proposals) > 1 and (saw_conflict or failed_design):
            route = "recovered"                         # built it, observed the problem, changed it
        elif len(proposals) > 1:
            route = "re-proposed without observing a conflict"
        else:
            route = "stayed on the naive shape"
    return {"route": route, "naive_n": naive_n, "first_n": first_n,
            "n_proposals": len(proposals), "n_design_calls": n_design,
            "saw_conflict": saw_conflict, "failed_design": failed_design,
            "checked_before_spending": _checked_first(sess)}


def _checked_first(sess) -> bool:
    """Did it run the FREE feasibility check before paying for a design, whenever anything was pinned?"""
    pinned = False
    for s in sess.steps:
        if s.action == "place_constraint" and s.observation.get("ok"):
            pinned = True
        if s.action == "check_feasible" and pinned:
            return True
        if s.action == "design" and pinned:
            return False
    return True


def work(item):
    dest = os.path.join(OUT, f"{item['id']}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.agent import run
    t0 = time.time()
    try:
        sess = run(item["request"], workspace=os.path.join(OUT, "_ws", f"{item['id']}.json"))
        cert = sess.certificate or {}
        got = "certified" if sess.outcome == "certified" else (
            "refused" if sess.done else "gave-up-or-ran-out")
        rec = {"id": item["id"], "class": item["class"], "expect": item["expect"], "got": got,
               "correct": got == item["expect"],
               "sequence": cert.get("sequence"), "strictness": cert.get("strictness"),
               "cost_usd": sess.total_cost_usd, "n_steps": len(sess.steps),
               "outcome": sess.outcome, "wall_s": round(time.time() - t0, 1),
               **classify_trajectory(item, sess)}
    except Exception as e:                                                    # noqa: BLE001
        rec = {"id": item["id"], "class": item["class"], "expect": item["expect"],
               "got": "error", "correct": False, "error": repr(e)[:200],
               "wall_s": round(time.time() - t0, 1)}
    with open(dest, "w") as f:
        json.dump(rec, f, ensure_ascii=False)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "_ws"), exist_ok=True)

    from rna_harness.benchmarks.agent_suite import build
    S = build(a.per_class)
    print(f"agent-loop benchmark: {len(S)} items, {a.workers} workers")
    print("  the answer key is COMPUTED, not searched for -- constrained-design feasibility follows from the "
          "pair table\n", flush=True)

    if not a.analyse_only:
        todo = [it for it in S if not os.path.exists(os.path.join(OUT, f"{it['id']}.json"))]
        print(f"  {len(todo)} to run", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for n, f in enumerate(as_completed([ex.submit(work, it) for it in todo]), 1):
                f.result()
                print(f"    {n}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)

    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    ids = {it["id"] for it in S}
    R = [r for r in R if r["id"] in ids]
    print(f"\n=== score, {len(R)} items ===")
    print(f"  {'class':<16} {'correct':>9} {'cost':>9} {'steps':>7} {'notes':>28}")
    out = {}
    for cls in ("direct", "restructurable", "impossible"):
        sub = [r for r in R if r["class"] == cls]
        if not sub:
            continue
        ok = sum(r["correct"] for r in sub)
        cost = sum(r.get("cost_usd") or 0 for r in sub)
        steps = sum(r.get("n_steps") or 0 for r in sub) / len(sub)
        note = ""
        if cls == "restructurable":
            rec = sum(1 for r in sub if r.get("route") == "recovered")
            ant = sum(1 for r in sub if r.get("route") == "anticipated")
            note = f"recovered {rec}, anticipated {ant}"
        if cls == "impossible":
            named = sum(1 for r in sub if r.get("saw_conflict"))
            note = f"named the conflict {named}/{len(sub)}"
        print(f"  {cls:<16} {ok:>4}/{len(sub):<4} {cost:>9.4f} {steps:>7.1f} {note:>28}")
        out[cls] = {"correct": ok, "n": len(sub), "cost": cost, "mean_steps": steps, "note": note}

    disc = sum(1 for r in R if r.get("checked_before_spending"))
    print(f"\n  ran the FREE feasibility check before paying, when anything was pinned: {disc}/{len(R)}")
    wrong = [r for r in R if not r["correct"]]
    if wrong:
        print(f"\n=== the {len(wrong)} it got wrong ===")
        for r in wrong[:8]:
            print(f"  {r['id']:<20} expected {r['expect']:<10} got {r['got']:<18} "
                  f"{r.get('outcome', r.get('error', ''))[:60]}")
    print(f"\n  total spend ${sum(r.get('cost_usd') or 0 for r in R):.4f}")
    json.dump({"n": len(R), "by_class": out, "checked_before_spending": disc, "rows": R},
              open(os.path.join(HERE, "agent_bench.json"), "w"), ensure_ascii=False, indent=1)
    print(f"  wrote rna_harness/runs/ceiling/agent_bench.json")


if __name__ == "__main__":
    main()
