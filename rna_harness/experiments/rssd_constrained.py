"""E2 -- constrained pseudoknot design: is 180/251 an UNDER-estimate because we dropped the IUPAC restraints?

The full-251 baseline ran DesiRNA with all-N sequence restraints (`_build_input` hardcodes 'N'*n), i.e. it
threw away the Pseudobase IUPAC constraints. Real constraints shrink the search space -- they could (a) let
DesiRNA solve some of the 71 residual it missed unconstrained, or (b) BREAK some of the 180 it solved (the
pinned bases conflict). E2 runs CONSTRAINED DesiRNA (real seq_restr) on the residual + a sample of the solved,
pKiss-verifies, and reports the net Delta-solve vs the unconstrained baseline. Honest either way."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.rnainvbench import load_pseudobase, verify_pk, pkiss_available
from ..generators.desirna import DesiRNAGenerator, desirna_available
from ..task import DesignTask, OracleConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_constrained")
_T2004 = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)


class ConstrainedDesiRNA(DesiRNAGenerator):
    """Thread the REAL IUPAC restraint into DesiRNA's input instead of all-N."""
    def _build_input(self, task: DesignTask) -> str:
        n = len(task.target)
        restr = (task.meta or {}).get("seq_restr") or ("N" * n)
        return f">name\nDesign\n>seq_restr\n{restr}\n>sec_struct\n{task.target}\n"


def _baseline():
    rows = json.load(open(os.path.join(os.path.dirname(__file__), "..", "runs",
                                       "pk_desirna", "pk_desirna.json")))["rows"]
    return {int(r["task_id"].split(":")[1]): bool(r["pk_verified_solved"]) for r in rows}


def _one(task, tl, replicas, seed, constrained):
    gen = (ConstrainedDesiRNA if constrained else DesiRNAGenerator)(replicas=replicas, timelimit_s=tl,
                                                                    results_number=20)
    res = gen.design(task, seed=seed)
    idx = int(task.task_id.split(":")[1])
    best = res.best_sequence or None
    v = verify_pk(best, task.target) if best else {"solved": False, "bp": None}
    solved, bp = bool(v.get("solved")), v.get("bp")
    for c in (getattr(res, "candidates", None) or []):        # pKiss-check the candidate pool too
        if solved:
            break
        vc = verify_pk(c, task.target)
        if vc.get("solved"):
            solved, bp = True, 0
    return idx, {"solved": solved, "bp": bp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit-s", type=int, default=300)
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sample-solved", type=int, default=20, help="also re-run this many baseline-SOLVED (check for breakage)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    if not (pkiss_available() and desirna_available()):
        raise SystemExit("need pKiss + DesiRNA")
    os.makedirs(a.outdir, exist_ok=True)
    tasks = {int(t.task_id.split(":")[1]): t for t in load_pseudobase(with_constraints=True)}
    base = _baseline()
    residual = sorted([i for i, s in base.items() if not s])
    solved_ids = sorted([i for i, s in base.items() if s])[: a.sample_solved]
    run_ids = residual + solved_ids
    print(f"E2 constrained design: {len(residual)} residual + {len(solved_ids)} baseline-solved (breakage check); "
          f"CONSTRAINED DesiRNA -t{a.timelimit_s} R{a.replicas}", flush=True)

    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, tasks[i], a.timelimit_s, a.replicas, a.seed, True): i for i in run_ids}
        for fut in as_completed(futs):
            idx, r = fut.result()
            out[idx] = r
            json.dump(out, open(os.path.join(a.outdir, "constrained.json"), "w"), indent=0)
            if idx in residual and r["solved"]:
                print(f"  RESIDUAL CRACKED by constraints: pseudobase:{idx}", flush=True)

    newly = [i for i in residual if out.get(i, {}).get("solved")]
    broke = [i for i in solved_ids if not out.get(i, {}).get("solved")]
    kept = len(solved_ids) - len(broke)
    print(f"\n=== E2 CONSTRAINED DESIGN ===", flush=True)
    print(f"  residual cracked by adding IUPAC constraints: {len(newly)}/{len(residual)}  {newly}", flush=True)
    print(f"  baseline-solved that BROKE under constraints:  {len(broke)}/{len(solved_ids)}  {broke}", flush=True)
    net = len(newly) - round(len(broke) * (71 / max(1, a.sample_solved)))   # rough net if extrapolated
    print(f"  => constrained solve on residual: +{len(newly)}; sampled-solved kept {kept}/{len(solved_ids)}", flush=True)
    print(f"  interpretation: >0 cracked => 180/251 UNDER-counts real designability; breakage => constraints "
          f"can also hurt naive runs. (net extrapolation rough: {net:+d})", flush=True)
    json.dump({"config": vars(a), "residual_cracked": newly, "solved_broke": broke, "results": out},
              open(os.path.join(a.outdir, "constrained.json"), "w"), indent=1)
    print(f"  wrote {os.path.join(a.outdir, 'constrained.json')}", flush=True)


if __name__ == "__main__":
    main()
