#!/usr/bin/env python3
"""Put Eterna100 on the same S(B) surface as the crossing benchmark.

metric.py placed the three pseudoknot systems on one curve. Eterna needs the same treatment, and the
reason is not symmetry -- it is that the archived 64/100 is a ONE-SHOT number. rows_partial.json holds
about 1.35 attempts per target, so it measures "solved on the first try", not S_inf = #{i : p_i > 0}, the
quantity that actually answers "how many of these can the system reach at all".

Design, deliberately the mirror of the gate experiment:

  remc_plain   RemcTool, which verifies at the default strictness 'match'
  remc_gate    StrictGate(RemcTool, 'umfe') -- resample when the solution is not the UNIQUE MFE

Adjudication is independent and always records both predicates, so no arm can score itself. Budget is
CPU-seconds (wall x replicas), the same currency metric.py uses for the pseudoknot side.

An expectation worth stating before the run, because it differs sharply from the pseudoknot case: umfe
rejects only 9 of 64 archived solutions (14%), where ShapeKnots rejected 482 of 530 (91%). So the GATE
should buy little here. What this run is really for is S_inf and the shape of S(B) -- multi-seed
measurement of a ceiling that has only ever been measured one-shot.

Pseudoknotted Eterna targets are excluded: verify() judges those with pKiss and ignores `strictness`, so
'umfe' and 'match' would be the same predicate and the comparison would be vacuous on them.
"""
import json, os, sys, time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.eterna100 import load_eterna100 as L          # noqa: E402
from rna_harness.cognitive.tools import RemcTool, verify                   # noqa: E402
from rna_harness.cognitive.strict_gate import StrictGate                   # noqa: E402
from rna_harness.structure_utils import has_pseudoknot                     # noqa: E402

OUT = os.path.join(HERE, "eterna_sb")
WALL = float(os.environ.get("WALL", "20"))
REPL = int(os.environ.get("REPL", "2"))
NSEED = int(os.environ.get("NSEED", "8"))
TRIES = int(os.environ.get("TRIES", "4"))
ARMS = ("remc_plain", "remc_gate")


def work(item):
    arm, tid, target, seed = item
    inner = RemcTool(wall_s=WALL, n_replicas=REPL)
    tool = inner if arm == "remc_plain" else StrictGate(inner, strictness="umfe",
                                                        max_attempts=TRIES,
                                                        budget_s=WALL * TRIES * 1.5)
    t0 = time.time()
    try:
        r = tool.attempt(target, seed=seed)
        wall = time.time() - t0
        adj = None
        if r.seq:
            m, _, _ = verify(r.seq, target, "match")
            u, _, _ = verify(r.seq, target, "umfe")
            adj = {"match": bool(m), "umfe": bool(u)}
        st = r.stats or {}
        return {"arm": arm, "id": tid, "seed": seed, "n": len(target), "claimed": bool(r.solved),
                "seq": r.seq, "adj": adj, "wall_s": round(wall, 2),
                "cpu_s": round(wall * REPL, 2),
                "tries": st.get("gate_attempts", 1), "rejects": st.get("gate_rejects", 0)}
    except Exception as e:
        return {"arm": arm, "id": tid, "seed": seed, "error": repr(e)[:200],
                "cpu_s": round((time.time() - t0) * REPL, 2)}


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = [t for t in L() if not has_pseudoknot(t["target"])]
    print(f"Eterna100: {len(tasks)} nested targets (pseudoknotted ones excluded), "
          f"lengths {min(len(t['target']) for t in tasks)}-{max(len(t['target']) for t in tasks)}")
    print(f"  REMC wall {WALL}s x {REPL} replicas, {NSEED} seeds, gate up to {TRIES} tries")
    items = [(arm, f"eterna100:{t['id']}:{t['name']}", t["target"], 400 + s)
             for s in range(NSEED) for arm in ARMS for t in tasks]
    key = lambda it: f"{it[0]}_s{it[3]}_{it[1].split(':')[1]}"
    done = {f[:-5] for f in os.listdir(OUT) if f.endswith(".json")}
    todo = [it for it in items if key(it) not in done]
    est = sum(WALL * REPL * (TRIES if it[0] == "remc_gate" else 1) for it in todo) / 3600
    print(f"  {len(items)} attempts, {len(todo)} to run, worst case {est:.0f} core-h", flush=True)

    # REMC spawns its OWN replica pool, and experiments/guards.py raises if a pool is launched inside a
    # worker (the reparenting fork-bomb guard). So this cannot use a Pool: parallelism comes from running
    # several SHARDS as separate top-level processes, each iterating its slice sequentially while REMC
    # uses its replicas underneath. Set SHARD/NSHARD; the launcher below starts them.
    shard = int(os.environ.get("SHARD", "-1"))
    nshard = int(os.environ.get("NSHARD", "1"))
    if shard >= 0:
        todo = [it for k, it in enumerate(todo) if k % nshard == shard]
        print(f"  shard {shard}/{nshard}: {len(todo)} attempts", flush=True)
    t0 = time.time()
    for i, it in enumerate(todo, 1):
        rec = work(it)
        json.dump(rec, open(os.path.join(
            OUT, f"{rec['arm']}_s{rec['seed']}_{rec['id'].split(':')[1]}.json"), "w"))
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el/60:.1f} min  eta {el/i*(len(todo)-i)/60:.1f} min",
                  flush=True)
    print("done")


if __name__ == "__main__":
    main()
