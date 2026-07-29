#!/usr/bin/env python3
"""The one untested cell: does reading DesiRNA's DISCARDED candidates find pfe-robust designs?

Why this cell and not another. Two things I spent today re-deriving already existed in this repo, and the
audit that found them also found the gap:

  * "read the discards" (best-of-pool vs one fixed candidate) -- runs/axis1_oracle_panel (pkiss 180/251,
    probknot 213/251, shapeknots 99/251, unanimous 73/251) against runs/axis1_desirna_fixedcand (174, 88,
    29 of 250). ProbKnot 88 -> 213 and ShapeKnots 29 -> 99 IS the discard-reading effect, measured for
    reviewer question R3-3.2 long before today.
  * the Eterna strictness ladder -- runs/verifier_frontier, a Paper D central experiment: survival
    match 1.000, umfe 0.902, pfe 0.066 over n = 61.

But `verifier_frontier` re-judges ONE solution per target. It never asks whether ANOTHER candidate from the
same solver run clears the stricter bar. Crossing that with the strictness ladder leaves exactly one cell
empty, and it is the cell with the most room:

    umfe   90.2% of single solutions already survive  -> only ~10% of targets have anything to gain
    pfe     6.6% survive                              -> 93.4% fail, and that is the headroom

Three reasons this is the right cell:

  1. Largest remaining headroom of any strictness level.
  2. The target is known reachable. runs/ceiling/robustness.py measured RNA.inverse_pf_fold reaching median
     P(target) = 0.8077 with 86% still MFE-matching, against our designs' 0.0594 -- 13.59x, p = 6.5e-12.
     So pfe-clearing designs exist and are findable. The open question is whether they are ALREADY sitting
     in DesiRNA's discarded pool.
  3. pfe is not a bar we chose for convenience: it is a declared strictness level in cognitive/tools.py and
     the centre of an existing experiment. That answers the "you picked the bar" objection, which is the one
     objection to today's panel-screen result that I cannot otherwise rebut.

FALSIFIER, stated first. If best-of-pool pfe survival is also near 6.6%, then DesiRNA's whole candidate pool
is thermodynamically fragile, discard-reading does not generalise to this criterion, and the panel-screen
result is specific to the three-oracle configuration rather than a mechanism.

Measured per target: the pfe verdict for DesiRNA's top-1, the pfe verdict for the best of all candidates,
P(target) for each, and the cost of the screen. Adjudication is independent of DesiRNA throughout.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
os.environ.setdefault("DESIRNA_CMD",
                      "/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py")
from rna_harness.benchmarks.eterna100 import load_eterna100                       # noqa: E402
from rna_harness.cognitive.cost import CPU_USD_PER_SEC                             # noqa: E402
from rna_harness.structure_utils import has_pseudoknot                             # noqa: E402
from rna_harness.task import DesignTask, OracleConfig                              # noqa: E402

OUT = os.path.join(HERE, "pfe_pool")
_T2004 = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
PFE_MIN = float(os.environ.get("COGCORE_PFE_MIN", "0.90"))


def judge(seq, target):
    """(match, umfe, pfe, P(target)) for one sequence -- computed here, never taken from the solver."""
    import RNA
    fc = RNA.fold_compound(seq)
    mfe_db, mfe_e = fc.mfe()
    match = (mfe_db == target)
    umfe = False
    if match:
        # Count co-optimals exactly the way cognitive/tools.py does, so 'umfe' here means the same
        # predicate it means everywhere else on this branch. A first version of this used subopt_cb with a
        # counting callback and got umfe = False on sequences with P(target) = 0.976, which is impossible --
        # ViennaRNA fires the callback once more with a NULL structure as a terminator, so the count was
        # always one too high and the predicate could never be satisfied.
        n_coopt = sum(1 for s in fc.subopt(1) if abs(s.energy - mfe_e) < 1e-6)
        umfe = (n_coopt == 1)
    fc2 = RNA.fold_compound(seq)
    fc2.pf()
    pT = float(fc2.pr_structure(target))
    return bool(match), bool(umfe), bool(pT >= PFE_MIN), pT


def work(item):
    tid, target, seed, tl, repl, nres = item
    from rna_harness.generators.desirna import DesiRNAGenerator, desirna_available
    if not desirna_available():
        return {"id": tid, "error": "desirna unavailable"}
    t0 = time.time()
    try:
        res = DesiRNAGenerator(replicas=repl, timelimit_s=tl, results_number=nres).design(
            DesignTask(task_id=str(tid), target=target, oracle=_T2004), seed=seed)
        design_s = time.time() - t0
        top = res.best_sequence or None
        cands = [c for c in (getattr(res, "candidates", None) or []) if c]
        seen, uniq = set(), []
        for c in ([top] if top else []) + cands:
            if c and c not in seen and len(c) == len(target):
                seen.add(c); uniq.append(c)
        g0 = time.time()
        # top-1 verdict, which is what verifier_frontier scored
        v_top = judge(top, target) if top else (False, False, False, 0.0)
        # best of pool on each rung, plus the best P(target) anywhere in the pool
        any_m = any_u = any_p = False
        bestP, bestP_seq, n_screen = -1.0, None, 0
        first_pfe = None
        for c in uniq:
            n_screen += 1
            m, u, p, pT = judge(c, target)
            any_m |= m; any_u |= u
            if p and not any_p:
                any_p = True; first_pfe = n_screen
            any_p |= p
            if pT > bestP:
                bestP, bestP_seq = pT, c
        screen_s = time.time() - g0
        return {"id": tid, "n": len(target), "seed": seed, "n_cands": len(uniq),
                "top": {"match": v_top[0], "umfe": v_top[1], "pfe": v_top[2], "pT": v_top[3]},
                "pool": {"match": any_m, "umfe": any_u, "pfe": any_p, "bestP": bestP,
                         "first_pfe_rank": first_pfe},
                "design_s": round(design_s, 1), "screen_s": round(screen_s, 1),
                "n_screened": n_screen, "seq_top": top, "seq_bestP": bestP_seq}
    except Exception as e:
        return {"id": tid, "error": repr(e)[:220], "design_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=60)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--results", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=200)   # pfe/subopt cost grows with n; declared cap
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    T = [(p["id"], p["target"]) for p in load_eterna100(version="V2")
         if not has_pseudoknot(p["target"]) and len(p["target"]) <= a.max_len]
    items = [(i, t, 500 + s, a.timelimit, a.replicas, a.results)
             for s in range(a.seeds) for i, t in T]
    key = lambda it: f"p{it[0]}_s{it[2]}"
    done = {f[:-5] for f in os.listdir(OUT) if f.endswith(".json")}
    todo = [it for it in items if key(it) not in done]
    if a.shard >= 0:
        todo = [it for k, it in enumerate(todo) if k % a.nshard == a.shard]
    print(f"pfe_pool: {len(T)} nested Eterna targets <= {a.max_len} nt x {a.seeds} seeds "
          f"= {len(items)}; {len(todo)} in this shard", flush=True)
    print(f"  DesiRNA -t {a.timelimit} -R {a.replicas} -r {a.results}; PFE_MIN = {PFE_MIN}")
    print(f"  baseline to beat (verifier_frontier, top-1 only): pfe survival 0.066 over n = 61\n", flush=True)
    t0 = time.time()
    for i, it in enumerate(todo, 1):
        rec = work(it)
        json.dump(rec, open(os.path.join(OUT, key(it) + ".json"), "w"))
        if i % 5 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el/60:.1f} min  eta {el/i*(len(todo)-i)/60:.1f} min", flush=True)
    print("done")


if __name__ == "__main__":
    main()
