#!/usr/bin/env python3
"""Run the agent loop on structures this project did not design, and price the loop against a direct call.

Two questions, and the second is the one that decides whether the loop is worth having:

  does it work at all on natural shapes    pinned / conflicted, scored against an exactly computed key
  what does the loop COST                  plain, where the same target is also handed straight to the
                                           ladder. If the agent solves fewer of them, or costs much more,
                                           then for a user who already has a dot-bracket the loop is
                                           overhead wearing a nicer interface.
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
OUT = os.path.join(HERE, "wild_bench")


def work(item):
    dest = os.path.join(OUT, f"{item['id']}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.agent import run
    t0 = time.time()
    try:
        sess = run(item["request"], workspace=os.path.join(OUT, "_ws", f"{item['id']}.json"))
        cert = sess.certificate or {}
        got = "certified" if sess.outcome == "certified" else ("refused" if sess.done else "ran-out")
        rec = {"id": item["id"], "class": item["class"], "source_id": item["source_id"],
               "n": len(item["target"]), "expect": item["expect"], "got": got,
               "correct": (None if item["expect"] is None else got == item["expect"]),
               "sequence": cert.get("sequence"), "cost_usd": sess.total_cost_usd,
               "llm_cost_usd": sess.llm_cost_usd, "n_steps": len(sess.steps),
               "outcome": sess.outcome, "wall_s": round(time.time() - t0, 1),
               "adopted_target": any(s.action == "use_target" and s.observation.get("ok")
                                     for s in sess.steps),
               "substituted": sess.substituted,
               "named_conflict": any(s.action == "check_feasible" and s.observation.get("conflicts")
                                     for s in sess.steps)}
    except Exception as e:                                                    # noqa: BLE001
        rec = {"id": item["id"], "class": item["class"], "source_id": item["source_id"],
               "expect": item["expect"], "got": "error", "correct": False, "error": repr(e)[:200],
               "wall_s": round(time.time() - t0, 1)}
    if item["class"] == "plain":
        # THE BASELINE, on the same target, same verifier: what the user would get without the agent.
        from rna_harness.cognitive.certify import certify_design
        b0 = time.time()
        c = certify_design(item["target"], budget="standard")
        rec["baseline"] = {"verdict": c.verdict, "cost_usd": c.cost_usd,
                           "wall_s": round(time.time() - b0, 1)}
    with open(dest, "w") as f:
        json.dump(rec, f, ensure_ascii=False)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    global OUT
    if a.tag:
        OUT = os.path.join(HERE, f"wild_bench_{a.tag}")
    os.makedirs(os.path.join(OUT, "_ws"), exist_ok=True)

    from rna_harness.benchmarks.wild_suite import build
    S = build(a.per_class)
    print(f"wild benchmark: {len(S)} items, shapes from bpRNA -- not designed here")
    print(f"  selection: {json.dumps(S[0].get('selection'), ensure_ascii=False)}\n", flush=True)

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

    print(f"\n=== scored against an exactly computed key ===")
    out = {}
    for cls in ("pinned", "conflicted"):
        sub = [r for r in R if r["class"] == cls]
        if not sub:
            continue
        ok = sum(1 for r in sub if r.get("correct"))
        note = (f"named the conflict {sum(1 for r in sub if r.get('named_conflict'))}/{len(sub)}"
                if cls == "conflicted" else "")
        print(f"  {cls:<12} {ok:>2}/{len(sub):<3} correct   "
              f"${sum(r.get('cost_usd') or 0 for r in sub):.4f}   {note}")
        out[cls] = {"correct": ok, "n": len(sub)}

    plain = [r for r in R if r["class"] == "plain"]
    if plain:
        ag_ok = sum(1 for r in plain if r["got"] == "certified")
        bl_ok = sum(1 for r in plain if (r.get("baseline") or {}).get("verdict") == "certified")
        ag_c = sum(r.get("cost_usd") or 0 for r in plain)
        bl_c = sum((r.get("baseline") or {}).get("cost_usd") or 0 for r in plain)
        ag_w = sum(r.get("wall_s") or 0 for r in plain)
        bl_w = sum((r.get("baseline") or {}).get("wall_s") or 0 for r in plain)
        print(f"\n=== what the loop COSTS, same targets, same verifier ===")
        print(f"  {'':<22} {'solved':>8} {'$':>10} {'wall s':>9}")
        print(f"  {'agent loop':<22} {ag_ok:>4}/{len(plain):<3} {ag_c:>10.4f} {ag_w:>9.0f}")
        print(f"  {'direct ladder call':<22} {bl_ok:>4}/{len(plain):<3} {bl_c:>10.4f} {bl_w:>9.0f}")
        out["plain"] = {"agent_solved": ag_ok, "baseline_solved": bl_ok, "n": len(plain),
                        "agent_cost": ag_c, "baseline_cost": bl_c,
                        "agent_wall_s": ag_w, "baseline_wall_s": bl_w}

    ad = sum(1 for r in R if r.get("adopted_target"))
    sb = [r for r in R if r.get("substituted")]
    print(f"\n  adopted the user's own structure: {ad}/{len(R)}")
    print(f"  designed for a DIFFERENT shape than the user gave: {len(sb)}/{len(R)}"
          + (f"   {[r['id'] for r in sb]}" if sb else ""))
    wrong = [r for r in R if r.get("correct") is False]
    if wrong:
        print(f"\n=== the {len(wrong)} it got wrong ===")
        for r in wrong[:10]:
            print(f"  {r['id']:<14} n={r.get('n')} expected {r['expect']:<10} got {r['got']:<10} "
                  f"{str(r.get('outcome', r.get('error', '')))[:66]}")
    print(f"\n  total spend ${sum(r.get('cost_usd') or 0 for r in R):.4f}")
    dest = os.path.join(HERE, f"wild_bench{('_' + a.tag) if a.tag else ''}.json")
    json.dump({"n": len(R), "selection": S[0].get("selection"), "summary": out, "rows": R},
              open(dest, "w"), ensure_ascii=False, indent=1)
    print(f"  wrote {os.path.relpath(dest, REPO)}")


if __name__ == "__main__":
    main()
