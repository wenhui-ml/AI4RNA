"""E1 -- designability certificate for the pseudoknot residual.

v1 found the bp<=2 residual is designability-limited, not search-limited (DesiRNA and pkdecomp both plateau
at the same bp=1 pair). This TESTS that claim rigorously: throw HEAVY multi-seed pKiss-guided search
(DecomposePKTool with a big budget, several seeds) at each of the 71 DesiRNA-residual targets and record the
BEST bp any run reaches. A target that reaches bp=0 was merely search-limited (recovered); one that stays
bp>0 across all seeds at Nx budget is a designability-wall candidate. Honest capability ceiling. CPU+pKiss
only (no LLM) -- safe to run in parallel with the LLM experiments."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.rnainvbench import load_pseudobase, pk_pairs, pkiss_available
from ..cognitive.skills.pk_decompose import DecomposePKTool

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_designability")


def _residual_ids():
    p = os.path.join(os.path.dirname(__file__), "..", "runs", "pk_desirna", "pk_desirna.json")
    rows = json.load(open(p))["rows"]
    res = [r for r in rows if not r["pk_verified_solved"]]
    res.sort(key=lambda r: (r["pk_best_bp"] if r["pk_best_bp"] is not None else 99, r["n"]))
    return [(int(r["task_id"].split(":")[1]), r["pk_best_bp"]) for r in res]


def _one(idx, target, seed, skel_wall, repair_wall, max_iters, skel_restarts):
    tool = DecomposePKTool(skel_wall_s=skel_wall, repair_wall_s=repair_wall,
                           max_repair_iters=max_iters, skel_restarts=skel_restarts)
    r = tool.attempt(target, seed=seed)
    return idx, seed, {"solved": bool(r.solved), "bp": r.bp, "cost": r.cost_usd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="independent restarts per target")
    ap.add_argument("--skel-wall-s", type=float, default=12.0)
    ap.add_argument("--repair-wall-s", type=float, default=150.0, help="HEAVY budget per attempt")
    ap.add_argument("--max-iters", type=int, default=1500)
    ap.add_argument("--skel-restarts", type=int, default=12)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    if not pkiss_available():
        raise SystemExit("pKiss unavailable")
    os.makedirs(a.outdir, exist_ok=True)
    tasks = {int(t.task_id.split(":")[1]): t.target for t in load_pseudobase()}
    residual = _residual_ids()
    print(f"E1 designability certificate: {len(residual)} residual targets x {a.seeds} seeds, "
          f"heavy budget (repair={a.repair_wall_s}s, iters={a.max_iters}, restarts={a.skel_restarts})", flush=True)

    jobs = [(idx, tasks[idx], s) for idx, _bp in residual for s in range(a.seeds)]
    best = {idx: {"desirna_bp": bp, "n": len(tasks[idx]), "pk_pairs": len(pk_pairs(tasks[idx])),
                  "best_bp": 99, "recovered": False} for idx, bp in residual}
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, idx, tgt, s, a.skel_wall_s, a.repair_wall_s, a.max_iters, a.skel_restarts):
                (idx, s) for idx, tgt, s in jobs}
        for fut in as_completed(futs):
            idx, seed, r = fut.result()
            done += 1
            if r["bp"] is not None and r["bp"] < best[idx]["best_bp"]:
                best[idx]["best_bp"] = r["bp"]
            if r["solved"]:
                best[idx]["recovered"] = True
            json.dump(best, open(os.path.join(a.outdir, "designability.json"), "w"), indent=1)
            if r["solved"]:
                print(f"  RECOVERED pseudobase:{idx} (was DesiRNA-residual bp={best[idx]['desirna_bp']}) "
                      f"-> heavy pkdecomp solved (seed {seed})  [{done}/{len(jobs)}]", flush=True)

    recovered = [i for i, v in best.items() if v["recovered"]]
    stuck = [i for i, v in best.items() if not v["recovered"]]
    from collections import Counter
    stuck_hist = dict(sorted(Counter(best[i]["best_bp"] for i in stuck).items()))
    print(f"\n=== E1 DESIGNABILITY CERTIFICATE ({len(residual)} residual, {a.seeds} seeds, heavy search) ===", flush=True)
    print(f"  RECOVERED by heavy search (were search-limited): {len(recovered)}/{len(residual)}  {sorted(recovered)}", flush=True)
    print(f"  STILL STUCK (designability-wall candidates):     {len(stuck)}/{len(residual)}", flush=True)
    print(f"    stuck best-bp histogram: {stuck_hist}", flush=True)
    print(f"  => implied genuine PK ceiling: {180 + len(recovered)}/251 solvable by pkdecomp+heavy "
          f"(vs DesiRNA 180/251); remaining {len(stuck)} resist ALL search at {a.seeds}x heavy budget.", flush=True)
    json.dump({"config": vars(a), "recovered": sorted(recovered), "stuck": sorted(stuck),
               "stuck_bp_hist": stuck_hist, "best": best},
              open(os.path.join(a.outdir, "designability.json"), "w"), indent=1)
    print(f"  wrote {os.path.join(a.outdir, 'designability.json')}", flush=True)


if __name__ == "__main__":
    main()
