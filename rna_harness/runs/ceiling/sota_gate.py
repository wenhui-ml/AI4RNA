#!/usr/bin/env python3
"""Wrap SOTA in the panel screen. The single most informative experiment available.

DesiRNA reaches 43/60 under pKiss and 0/60 under the three-oracle panel (desirna_seqs/
panel_always_sota.json), at 1.6x our operator's budget. Our gated operator reaches 5.50/60 per execution
and 22/60 ever. Reading that as "we beat SOTA 22 to 0" would be wrong: DesiRNA optimises its own energy
model and has never been asked for the panel, exactly as our operator had never been asked for ShapeKnots.

The experiment that decides what we are actually claiming. DesiRNA already emits ~200 candidates per call
(not the 20 its -r flag suggests) and this tool has always kept ONE, screening the rest with pKiss alone
and stopping at the first hit. Turning on panel_screen puts every candidate to the full conjunction.

  * If DesiRNA+screen jumps from 0/60 to something large, the contribution is the SCREEN, not our
    operator -- and that is the bigger result, because improving SOTA is worth more than beating it.
  * If it does not, that is clean evidence the operator itself matters.

Both outcomes beat the current 22-vs-0, which is almost certainly an artefact of asymmetry.

This is also the cheapest unused INFORMATION SOURCE, which is the day's one reliable pattern: re-analysing
existing signals has gone 0/6 on this branch, adding a new signal source 2/2. The screen adds no new
oracle and no new search -- it just stops throwing away 199 of 200 candidates.

Sharded across top-level processes: DesiRNA runs its own replica pool.
"""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
os.environ.setdefault("DESIRNA_CMD",
                      "/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py")
from rna_harness.benchmarks.rnainvbench import load_pseudobase                      # noqa: E402
from rna_harness.cognitive.tools import DesirnaTool                                  # noqa: E402
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,       # noqa: E402
                                    shapeknots_pairs)

OUT = os.path.join(HERE, "sota")
TL = int(os.environ.get("TL", "60"))
REPL = int(os.environ.get("REPL", "4"))
NSEED = int(os.environ.get("NSEED", "4"))
ARMS = ("plain", "panel")


def work(item):
    arm, tid, target, seed = item
    tool = DesirnaTool(timelimit_s=TL, replicas=REPL, panel_screen=(arm == "panel"))
    t0 = time.time()
    try:
        r = tool.attempt(target, seed=seed)
        wall = time.time() - t0
        adj = None
        if r.seq:                       # independent adjudication -- never trust the arm's own verdict
            adj = {"pkiss": bool(forms_target(pkiss_pairs(r.seq), target)),
                   "probknot": bool(forms_target(probknot_pairs(r.seq), target)),
                   "shapeknots": bool(forms_target(shapeknots_pairs(r.seq), target))}
            adj["unanimous"] = all(adj.values())
        st = r.stats or {}
        return {"arm": arm, "id": tid, "seed": seed, "n": len(target), "claimed": bool(r.solved),
                "seq": r.seq, "panel": adj, "wall_s": round(wall, 1),
                "cpu_s": round(wall * REPL + (r.oracle_s or 0.0), 1),
                "oracle_s": r.oracle_s, "n_candidates": st.get("n_candidates", 0),
                "n_screened": st.get("n_screened", 0), "n_unanimous": st.get("n_unanimous", 0),
                "oracle_calls": st.get("oracle_calls")}
    except Exception as e:
        return {"arm": arm, "id": tid, "seed": seed, "error": repr(e)[:220],
                "cpu_s": round((time.time() - t0) * REPL, 1)}


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = load_pseudobase(limit=60)
    items = [(arm, t.task_id, t.target, 900 + s)
             for s in range(NSEED) for arm in ARMS for t in tasks]
    key = lambda it: f"{it[0]}_s{it[3]}_{it[1].split(':')[-1]}"
    done = {f[:-5] for f in os.listdir(OUT) if f.endswith(".json")}
    todo = [it for it in items if key(it) not in done]
    shard = int(os.environ.get("SHARD", "-1"))
    nshard = int(os.environ.get("NSHARD", "1"))
    if shard >= 0:
        todo = [it for k, it in enumerate(todo) if k % nshard == shard]
    print(f"{len(tasks)} targets x {NSEED} seeds x {len(ARMS)} arms = {len(items)}; "
          f"{len(todo)} to run in this shard  (DesiRNA -t {TL}s x {REPL} replicas)", flush=True)
    t0 = time.time()
    for i, it in enumerate(todo, 1):
        rec = work(it)
        json.dump(rec, open(os.path.join(OUT, key(it) + ".json"), "w"))
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el/60:.1f} min  eta {el/i*(len(todo)-i)/60:.1f} min", flush=True)
    print("done")


if __name__ == "__main__":
    main()
