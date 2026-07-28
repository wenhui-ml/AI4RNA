#!/usr/bin/env python3
"""Does putting ShapeKnots in the loop raise the panel-unanimous rate? Full 60 targets, 8 executions.

Why this and not more allocation work. Measured on the committed sweep (bottleneck.py): of 530 in-loop
solves, pKiss agreed 530/530 and ProbKnot agreed 530/530, ShapeKnots agreed on 48. Every one of the 482
failures is ShapeKnots ALONE -- pKiss kills 0 and ProbKnot kills 0 of otherwise-passing solutions. The
operator drives its in-loop oracles to 100% and a single out-of-loop oracle rejects 90.9%. And ShapeKnots
is the CHEAPEST of the three at these lengths (sk_cost.py: 64.8 ms vs pKiss 210.1, ratio 0.31), so it was
excluded for no cost reason.

Three arms, same targets and seeds, paired:

  off_150   gate off, max_pkiss = 150   the configuration behind the reported 3.2/60 per execution
  off_400   gate off, max_pkiss = 400   COMPUTE-MATCHED control: does more pKiss budget alone help?
  on_400    gate on,  max_pkiss = 400   the change

off_400 is the arm that makes the claim falsifiable. If extra pKiss budget alone lifts the panel rate, the
gate is not what did the work. The prior measurement says it will not -- panel-YES and panel-NO attempts
are statistically identical in every resource (Mann-Whitney p = 0.88 on ProbKnot calls, 0.83 on pKiss) --
so off_400 should sit on top of off_150.

Every returned sequence is adjudicated by the full three-oracle panel regardless of what the operator
believed, so a gated arm cannot score itself.

Prediction, from 9.1% per candidate and ~37 pKiss calls per candidate: 400 calls buys ~11 candidates,
1 - (1-0.091)^11 = 65% for the 42 targets that reach an in-loop solve. Discounted by the fact that 21 of
those 42 never passed ShapeKnots in 15 stored executions, the honest expectation is 17-20 of 60 per
execution against 3.2 now, at roughly 3.8x the per-target compute.
"""
import json, os, sys, time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase                      # noqa: E402
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,       # noqa: E402
                                    shapeknots_pairs, rnastructure_available)

OUT = os.path.join(HERE, "gate")
NSEED = int(os.environ.get("NSEED", "8"))
ARMS = {"off_150": dict(sk_gate=False, max_pkiss=150),
        "off_400": dict(sk_gate=False, max_pkiss=400),
        "on_400":  dict(sk_gate=True,  max_pkiss=400, max_shapeknots=60)}


def _init():
    os.environ.setdefault("DESIRNA_CMD",
                          "/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py")
    rnastructure_available()


def work(item):
    arm, tid, target, seed = item
    from rna_harness.cognitive.skills.pk_decompose_panel import DecomposePKPanelTool
    kw = dict(skel_wall_s=12.0, repair_wall_s=90.0, max_repair_iters=800, panel_gate=3,
              max_probknot=120, deterministic=True)
    kw.update(ARMS[arm])
    t0 = time.time()
    try:
        r = DecomposePKPanelTool(**kw).attempt(target, seed=seed)
        st = r.stats or {}
        # independent adjudication: never trust the arm's own verdict
        adj = None
        if r.seq:
            adj = {"pkiss": bool(forms_target(pkiss_pairs(r.seq), target)),
                   "probknot": bool(forms_target(probknot_pairs(r.seq), target)),
                   "shapeknots": bool(forms_target(shapeknots_pairs(r.seq), target))}
            adj["unanimous"] = all(adj.values())
        return {"arm": arm, "id": tid, "seed": seed, "claimed": bool(r.solved), "seq": r.seq,
                "panel": adj, "n_pkiss": st.get("n_pkiss", 0), "n_probknot": st.get("n_probknot", 0),
                "n_shapeknots": st.get("n_shapeknots", 0), "n_sk_reject": st.get("n_sk_reject", 0),
                "oracle_s": st.get("oracle_s", 0.0), "s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"arm": arm, "id": tid, "seed": seed, "error": repr(e)[:250], "n_pkiss": 0}


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = load_pseudobase(limit=60)
    items = [(arm, t.task_id, t.target, 300 + s)
             for s in range(NSEED) for arm in ARMS for t in tasks]
    key = lambda it: f"{it[0]}_s{it[3]}_{it[1].split(':')[-1]}"
    done = {f[:-5] for f in os.listdir(OUT) if f.endswith(".json")}
    todo = [it for it in items if key(it) not in done]
    print(f"{len(tasks)} targets x {NSEED} seeds x {len(ARMS)} arms = {len(items)} attempts, "
          f"{len(todo)} to run", flush=True)
    for a, kw in ARMS.items():
        print(f"  {a:8} {kw}")

    nw = int(os.environ.get("NW", "40"))
    t0 = time.time()
    with mp.get_context("fork").Pool(nw, initializer=_init) as pool:
        for i, rec in enumerate(pool.imap_unordered(work, todo, chunksize=1), 1):
            json.dump(rec, open(os.path.join(
                OUT, f"{rec['arm']}_s{rec['seed']}_{rec['id'].split(':')[-1]}.json"), "w"))
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el/60:.1f} min  eta {el/i*(len(todo)-i)/60:.1f} min",
                      flush=True)
    print("done")


if __name__ == "__main__":
    main()
