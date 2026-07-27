#!/usr/bin/env python3
"""Step 2: the allocation frontier, measured. Predictions are frozen in PREREG.md before this ran.

Five arms over ids 30-59, at four budgets, paired across 15 execution seeds. Only the allocation rule
differs; targets, seeds and operator are identical.

  uniform   b_i = B/30 for all 30                                    (the null)
  delta     top m by Delta (N=8 draws), b_i = B/m, rest 0            (the policy)
  random    a random m targets, b_i = B/m                            (falsifier P5)
  length    the m shortest targets, b_i = B/m                        (falsifier P6)
  oracle    only the 10 targets ever solved in the stored sweep      (the skyline; uses outcome
            b_i = B/10                                                knowledge, so an upper bound,
                                                                       not a policy)

m is read from cure.json, fitted on ids 0-29: m = 6, 13, 18, 25 at B/B0 = 0.10, 0.20, 0.50, 1.00.

The budget is enforced as `max_pkiss = b_i` with `deterministic=True`, so the realised cost is a counted
quantity and not a function of machine load. This matters: under the wall-clock regime the same policy
would spend differently depending on how busy the box is, and no cost comparison would mean anything.

Writes one JSON per (cell, arm, frac) so a crash costs one shard.
"""
import json, os, random, sys, time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase                  # noqa: E402
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,   # noqa: E402
                                    shapeknots_pairs, rnastructure_available)

OUT = os.path.join(HERE, "alloc")
FRACS = [0.10, 0.20, 0.50, 1.00]
NCELLS = int(os.environ.get("NCELLS", "15"))
ARMS = ["uniform", "delta", "random", "length", "oracle"]
_RNASTR = None


def panel(seq, target):
    pk = forms_target(pkiss_pairs(seq), target)
    pb = forms_target(probknot_pairs(seq), target) if _RNASTR else None
    sk = forms_target(shapeknots_pairs(seq), target) if _RNASTR else None
    return {"pkiss": bool(pk), "probknot": None if pb is None else bool(pb),
            "shapeknots": None if sk is None else bool(sk),
            "unanimous": (bool(pk) and bool(pb) and bool(sk)) if _RNASTR else None}


def _init():
    global _RNASTR
    os.environ.setdefault("DESIRNA_CMD",
                          "/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py")
    _RNASTR = rnastructure_available()


def work(item):
    """One (cell, arm, frac, target) cell of the design."""
    cell, arm, frac, tid, target, cap = item
    from rna_harness.cognitive.skills.pk_decompose_panel import DecomposePKPanelTool
    tool = DecomposePKPanelTool(skel_wall_s=12.0, repair_wall_s=90.0, max_repair_iters=800,
                               panel_gate=3, max_probknot=120,
                               deterministic=True, max_pkiss=int(cap))
    t0 = time.time()
    try:
        r = tool.attempt(target, seed=cell)
        seq, st, solved = r.seq, (r.stats or {}), bool(r.solved)
        adj = panel(seq, target) if (solved and seq) else None
        return {"cell": cell, "arm": arm, "frac": frac, "id": tid, "cap": int(cap),
                "solved": solved, "seq": seq, "panel": adj,
                "n_pkiss": st.get("n_pkiss", 0), "s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"cell": cell, "arm": arm, "frac": frac, "id": tid, "cap": int(cap),
                "error": repr(e)[:250], "n_pkiss": 0, "s": round(time.time() - t0, 1)}


def main():
    os.makedirs(OUT, exist_ok=True)
    cure = json.load(open(os.path.join(HERE, "cure.json")))
    pk = {r["id"]: r for r in json.load(open(os.path.join(HERE, "pk_ev.json")))["rows"]}
    held = cure["held"]
    B0 = cure["B0"]
    ndraw = cure["ndraw"]
    m_of = {round(f, 2): int(m) for f, _B, _es, _eu, _er, m in cure["frontier"]}
    delta = {t: min(pk[t]["draws_norm"][:ndraw]) for t in held}
    nlen = {t: pk[t]["n"] for t in held}
    ever = [t for t in held if pk[t]["panel_win"] > 0]
    tgt = {t.task_id: t.target for t in load_pseudobase(limit=60)}

    by_delta = sorted(held, key=lambda t: (delta[t], nlen[t], t))
    by_len = sorted(held, key=lambda t: (nlen[t], t))
    print(f"ids {held[0]}..{held[-1]}  n={len(held)}  B0={B0:.0f}  Delta from N={ndraw}")
    print(f"  m per frac (from ids 0-29): " + ", ".join(f"{f}->{m_of[f]}" for f in sorted(m_of)))
    print(f"  oracle arm funds {len(ever)} ever-solved targets")

    items, planned = [], {}
    for cell in range(NCELLS):
        for frac in FRACS:
            B = B0 * frac
            m = m_of[round(frac, 2)]
            rng = random.Random(10_000 + cell * 97 + int(frac * 100))
            sets = {"uniform": held,
                    "delta": by_delta[:m],
                    "random": rng.sample(held, m),
                    "length": by_len[:m],
                    "oracle": ever}
            for arm in ARMS:
                S = sets[arm]
                cap = max(1, int(B / max(1, len(S))))
                planned[(cell, arm, round(frac, 2))] = {"m": len(S), "cap": cap,
                                                        "budget": cap * len(S), "ids": list(S)}
                for t in S:
                    items.append((cell, arm, frac, t, tgt[t], cap))

    done = set()
    for f in os.listdir(OUT):
        if f.endswith(".json"):
            done.add(f[:-5])
    todo = [it for it in items
            if f"c{it[0]}_{it[1]}_f{int(it[2]*100):03d}_{it[3].split(':')[-1]}" not in done]
    print(f"  {len(items)} design cells, {len(items)-len(todo)} already on disk, {len(todo)} to run")
    print(f"  planned pKiss ceiling: {sum(it[5] for it in todo):,} calls "
          f"(~{sum(it[5] for it in todo)*0.391/3600:.1f} core-h)", flush=True)

    nw = int(os.environ.get("NW", "40"))
    t0 = time.time()
    with mp.get_context("fork").Pool(nw, initializer=_init) as pool:
        for i, rec in enumerate(pool.imap_unordered(work, todo, chunksize=1), 1):
            name = f"c{rec['cell']}_{rec['arm']}_f{int(rec['frac']*100):03d}_{rec['id'].split(':')[-1]}"
            json.dump(rec, open(os.path.join(OUT, name + ".json"), "w"))
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"    {i}/{len(todo)}  {el/60:.1f} min  eta {el/i*(len(todo)-i)/60:.1f} min",
                      flush=True)
    json.dump({"B0": B0, "fracs": FRACS, "ncells": NCELLS, "arms": ARMS, "ndraw": ndraw,
               "m_of": {str(k): v for k, v in m_of.items()},
               "planned": {f"{c}|{a}|{f}": v for (c, a, f), v in planned.items()}},
              open(os.path.join(HERE, "alloc_plan.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
