"""Head-to-head: is DesiRNA (a Turner-MFE tool) still dominant OUTSIDE the single-target-MFE box?
Test bed = RNAInvBench Pseudobase++ pseudoknot targets.

For each target we run DesiRNA (it can only see the pseudoknot-FREE reduction, since its energy model
== ViennaRNA MFE) and then verify its designs TWO ways:
  * MFE-oracle 'solved'  -> the BOGUS verdict our current harness would report (it silently drops the
    '[]{}' pseudoknot pairs and scores against the nested skeleton);
  * pKiss  'solved'      -> the REAL, pseudoknot-aware verdict.
The gap between them is the false-positive rate: how often "DesiRNA solved it" is an artifact of the
oracle not representing pseudoknots. Expected: MFE-solved >> pk-solved ~ 0  => DesiRNA is out of box.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.rnainvbench import (load_pseudobase, pk_bp_distance, pk_pairs, pkiss_available,
                                      pkiss_fold, _round_only)
from ..generators.desirna import DesiRNAGenerator, desirna_available
from ..metrics import bp_distance
from ..task import DesignTask

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "pk_desirna")


def _run_desirna(task_dict: dict, replicas: int, timelimit: int, seed: int) -> dict:
    """Worker: DesiRNA on one pseudoknot target. Returns best MFE-seq + all candidates for PK re-check."""
    t = DesignTask.from_dict(task_dict)
    r = DesiRNAGenerator(replicas=replicas, timelimit_s=timelimit, results_number=40).design(t, seed=seed)
    return {"task_id": t.task_id, "n": t.n, "target": t.target,
            "n_pk_pairs": t.meta.get("n_pk_pairs"),
            "mfe_best_seq": r.best_sequence, "mfe_bp": r.best_bp_distance,  # bp vs CORRUPTED target
            "candidates": (r.candidates or [])[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="number of Pseudobase++ targets")
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--timelimit", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pk-check-cands", type=int, default=25, help="how many DesiRNA candidates to PK-verify")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if not desirna_available():
        raise SystemExit("DESIRNA_CMD not set.")
    if not pkiss_available():
        raise SystemExit("pKiss not found -- set PKISS_CMD or install (conda -n pk pkiss).")
    tasks = load_pseudobase(limit=a.n)
    print(f"PK head-to-head on {len(tasks)} Pseudobase++ targets; DesiRNA R={a.replicas} -t={a.timelimit}s "
          f"(workers={a.workers}); then pKiss re-verify. Oracle-vs-PK false-positive test.", flush=True)

    # stage 1: DesiRNA in parallel (MFE design)
    des = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_run_desirna, t.to_dict(), a.replicas, a.timelimit, 1): t.task_id for t in tasks}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                des[r["task_id"]] = r
            except Exception as e:
                print(f"  desirna crash {futs[fut]}: {e}", flush=True)

    # stage 2: pKiss re-verify (fast, sequential) -- MFE-solved (bogus) vs PK-solved (real)
    rows, mfe_solved, pk_solved, false_pos = [], 0, 0, 0
    for t in tasks:
        d = des.get(t.task_id)
        if not d:
            continue
        # (a) what our MFE harness would REPORT: does DesiRNA's design match the ROUND-ONLY reduction?
        mfe_says_solved = (d["mfe_bp"] == 0)
        # (b) REAL pseudoknot-aware check over DesiRNA's best + top candidates (benefit of the doubt)
        cands = [d["mfe_best_seq"]] + [c for c in d["candidates"] if c != d["mfe_best_seq"]]
        best_pk_bp, best_pk_seq = None, None
        for s in cands[: a.pk_check_cands]:
            if not s:
                continue
            folded = pkiss_fold(s, timeout_s=60.0)
            if folded is None:
                continue
            bp = pk_bp_distance(folded, t.target)
            if best_pk_bp is None or bp < best_pk_bp:
                best_pk_bp, best_pk_seq = bp, s
                if bp == 0:
                    break
        pk_says_solved = (best_pk_bp == 0)
        mfe_solved += int(mfe_says_solved)
        pk_solved += int(pk_says_solved)
        false_pos += int(mfe_says_solved and not pk_says_solved)
        rows.append({"task_id": t.task_id, "n": t.n, "n_pk_pairs": d["n_pk_pairs"],
                     "mfe_reported_solved": mfe_says_solved, "mfe_bp_vs_reduction": d["mfe_bp"],
                     "pk_verified_solved": pk_says_solved, "pk_best_bp": best_pk_bp})
        print(f"  {t.task_id:>14} n={t.n:>3} pk_pairs={d['n_pk_pairs']:>2} | "
              f"MFE-oracle says solved={mfe_says_solved} | pKiss REAL bp={best_pk_bp} solved={pk_says_solved}"
              f"{'  <-- FALSE POSITIVE' if (mfe_says_solved and not pk_says_solved) else ''}", flush=True)
        json.dump({"config": vars(a), "n_tasks": len(rows),
                   "mfe_reported_solved": mfe_solved, "pk_verified_solved": pk_solved,
                   "false_positives": false_pos, "rows": rows},
                  open(os.path.join(OUT, "pk_desirna.json"), "w"), indent=1)

    N = len(rows)
    fp_rate = 100 * false_pos / max(1, mfe_solved)
    print(f"\n=== RESULT on {N} Pseudobase++ pseudoknot targets ===", flush=True)
    print(f"  MFE-oracle REPORTED solved : {mfe_solved}/{N}  (INFLATED -- scored vs the pseudoknot-free "
          f"reduction; the MFE model silently drops '[]{{}}')", flush=True)
    print(f"  pKiss REAL solved          : {pk_solved}/{N}  (pseudoknot-aware verification, ground truth)",
          flush=True)
    print(f"  ORACLE false positives     : {false_pos}/{mfe_solved}  ({fp_rate:.0f}% of MFE 'solved' claims "
          f"are NOT real pseudoknot solves)", flush=True)
    print(f"  Interpretation: the false-positive is a HARNESS/ORACLE artifact (wrong verifier for the "
          f"problem type), NOT DesiRNA failing -- DesiRNA enforces the '[]' pairing constraints and "
          f"pKiss confirms {pk_solved}/{N} real solves. LESSON: match the VERIFIER to the problem; the "
          f"is_pseudoknot guard must refuse an MFE 'solved' on any target containing '[]{{}}'.", flush=True)


if __name__ == "__main__":
    main()
