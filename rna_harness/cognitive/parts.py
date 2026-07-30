"""DECOMPOSE -> design each unit -> PIN what worked -> repair the seams.

WHY THIS AND NOT A BIGGER BUDGET. On Eterna100 the shipped cascade solves 84/100 and 16 targets survive every
tier up to 180 s. That residual has been attacked from the budget side repeatedly and the whole axis is
measured out: early stop 1.156x, ladder reshaping 1.097x, multi-seed probing null, learned abstention caught
0 of 16, retrieval-as-pin net -21, warm-start mean 0. Paying more, or allocating more cleverly, is finished.
What is left is to change what the search is asked to do.

THE OBSERVATION THAT MAKES IT PLAUSIBLE, computed not assumed: 13 of those 16 targets decompose into two or
more independent stem-loop units, and the units are small. Snowflake 3 is 393 nt and becomes 29 units of at
most 10 nt; Snowflake 4 is 370 nt and becomes 24 units of at most 11 nt. The solver never has to solve the
long problem at once. The other 3 (Mat-Elements 106 nt, Variation of a crop circle 88 nt) are a single unit
and decomposition has nothing to offer them -- they are excluded and said so, not quietly dropped.

WHY IT MIGHT STILL FAIL, which is the point of measuring it. Units that fold correctly alone need not fold
correctly together: two identical units can cross-pair and collapse, and the seams between units are exactly
the positions no unit search ever looked at. So this is not "decompose and win" -- it is decompose, design,
PIN THE PART THAT VERIFIABLY WORKED, and hand the seams to the constrained solver. Every stage is checked by
the same exact oracle that judges everything else here, and the observation reports each stage separately so
a failure says WHERE it failed.

THE RESULT: THIS DOES NOT WORK, AND THE REASON IS WORTH MORE THAN THE ATTEMPT. Paired against the shipped
cascade on the 14 decomposable targets, same oracle, same solver, comparable wall clock (85 min against 89):

    whole-target cascade    1/14      decompose + design + pin + repair    0/14

Not a near miss -- the closest the parts arm came was 20 base pairs away and the median was 94. So the
question is which assumption broke, and it is the load-bearing one. Design every unit independently, verify
each folds to its own sub-target exactly, place them at their positions, and ask what the assembly folds to:

    42 of 145 individually-correct units -- 29% -- still hold their own shape once embedded

Snowflake 3 keeps 5 of 29. Hoglafractal keeps 1 of 18. The units are not independent under the Turner model
at these lengths: a unit's neighbours change what it folds into, so pinning a verified unit pins a sequence
that is no longer a solution. Repair cannot recover from that, because the seams are the only free positions
and the damage is inside the pinned regions.

This closes the decomposition axis for these targets rather than leaving it as an untested hope. It also says
what a working version would have to do -- design each unit IN ITS CONTEXT rather than alone -- which is the
whole-target problem again, which is where we started.

ONE MORE THING THE RUN SHOWED: the cascade solved Snowflake 4 here, and the archived frontier lists it among
the targets no tier ever solved. DesiRNA is not deterministic, so this residual set moves by at least +/-1
between runs and a single-run comparison on it cannot resolve a one-target difference.

WHAT THIS IS NOT. It is not the agent deciding anything. An earlier attempt surfaced the decomposition plan
to a free-form agent and measured 1 of 28 solved: the agent SAW the units and did not execute the loop. That
isolated the bottleneck as behaviour rather than plumbing, so the mechanism is built as a primitive here and
tested deterministically first. Whether the agent then chooses it is a separate question, measured separately.
"""
from __future__ import annotations

import subprocess
import sys
import time
from typing import Dict, List, Optional

UNIT_TRIES = 8          # inverse-fold restarts per unit before falling back to the external solver
MIN_UNITS = 2           # below this there is nothing to decompose and the honest answer is to say so


def _inverse_fold_many(sub_targets: List[str], tries: int = UNIT_TRIES, timeout_s: float = 300.0):
    """Design every unit in ONE isolated child process.

    One spawn rather than one per unit, and isolated for the same reason designable_probe is: ViennaRNA's
    inverse_fold is C and segfaults on input it dislikes, which would take the whole run down with it. `-I`
    also keeps the current directory off sys.path.
    """
    # A PER-UNIT DEADLINE, and the partial result is always printed. Without it one slow unit loses the
    # whole batch: Gladius decomposes into 6 units of which one is 147 nt, and eight inverse-fold restarts on
    # that one unit ran past the batch timeout -- taking the five units that had already succeeded with it.
    code = ("import sys, json, time, RNA\n"
            "targets = json.loads(sys.argv[1]); tries = int(sys.argv[2]); per = float(sys.argv[3])\n"
            "out = []\n"
            "for t in targets:\n"
            "    got, d = None, time.time() + per\n"
            "    for _ in range(tries):\n"
            "        if time.time() > d: break\n"
            "        s, _d = RNA.inverse_fold(None, t)\n"
            "        if RNA.fold(s)[0] == t:\n"
            "            got = s; break\n"
            "    out.append(got)\n"
            "print(json.dumps(out))\n")
    import json as _json
    try:
        per_unit = max(2.0, timeout_s / max(1, len(sub_targets)) * 0.8)
        r = subprocess.run([sys.executable, "-I", "-c", code, _json.dumps(sub_targets), str(tries),
                            str(per_unit)],
                           capture_output=True, text=True, timeout=timeout_s)
        return _json.loads(r.stdout.strip())
    except Exception:                                                     # noqa: BLE001
        return [None] * len(sub_targets)


def design_by_parts(target: str, restraint: Optional[str] = None, repair_budget: str = "standard",
                    seed: int = 0, unit_tries: int = UNIT_TRIES) -> Dict:
    """Returns the same shape as design(), plus a stage-by-stage account of what the parts contributed."""
    from ..mcp.motif_kb import decompose_structure
    from .certify import BUDGETS, certify_sequence
    from .cost import solver_cost_usd

    t0 = time.time()
    dec = decompose_structure(target)
    if dec.get("error"):
        return {"ok": False, "error": dec["error"]}
    units = dec.get("units") or []
    if len(units) < MIN_UNITS:
        return {"ok": False, "cost_usd": 0.0, "wall_s": round(time.time() - t0, 2),
                "n_units": len(units),
                "error": (f"this target decomposes into {len(units)} unit(s), so there is nothing to design "
                          f"separately"),
                "hint": "decomposition cannot help here; the whole target has to be searched as one"}

    # ---- stage 1: each unit on its own, verified exactly ------------------------------------------
    seqs = _inverse_fold_many([u["sub_target"] for u in units], tries=unit_tries)
    solved = [(u, s) for u, s in zip(units, seqs) if s]
    unit_report = [{"positions": u["positions"], "len": u["len"], "solved": bool(s)}
                   for u, s in zip(units, seqs)]

    # ---- stage 2: pin what verifiably worked, leave the seams free --------------------------------
    # A caller's own pin outranks a unit sequence: the unit is our idea, the pin is the request. Any unit
    # that would overwrite one is dropped whole rather than half-applied.
    r = list(restraint) if (restraint and len(restraint) == len(target)) else ["N"] * len(target)
    pinned_units, dropped_units = 0, 0
    for u, s in solved:
        a, b = u["positions"]
        if any(r[a + k] != "N" and r[a + k] != s[k] for k in range(len(s))):
            dropped_units += 1
            continue
        for k, ch in enumerate(s):
            r[a + k] = ch
        pinned_units += 1
    seed_restraint = "".join(r)
    covered = sum(1 for c in seed_restraint if c != "N")

    if covered == 0:
        return {"ok": True, "verdict": "not_found", "cost_usd": 0.0,
                "wall_s": round(time.time() - t0, 2), "n_units": len(units), "units_solved": 0,
                "units": unit_report,
                "note": "no unit could be designed on its own, so there was nothing to pin and no repair to run"}

    # ---- stage 3: the seams, by constrained search on the WHOLE target ----------------------------
    from ..experiments.rssd_constrained import ConstrainedDesiRNA
    from ..generators.desirna import desirna_available
    from ..task import DesignTask, OracleConfig
    if not desirna_available():
        return {"ok": False, "error": "the external solver is not installed", "n_units": len(units),
                "units_solved": pinned_units}
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    cost, screened, best, best_bp = 0.0, 0, None, None
    for tl in BUDGETS.get(repair_budget, BUDGETS["standard"]):
        a0 = time.time()
        res = ConstrainedDesiRNA(replicas=10, timelimit_s=tl, results_number=50).design(
            DesignTask(task_id="parts", target=target, oracle=oc, meta={"seq_restr": seed_restraint}),
            seed=seed)
        cost += solver_cost_usd(time.time() - a0, 10)
        pool, seen = [], set()
        for c in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
            if c and len(c) == len(target) and c not in seen:
                seen.add(c)
                pool.append(c)
        for c in pool:
            screened += 1
            cert = certify_sequence(c, target)
            if cert.verdict in ("certified", "knot_confirmed"):
                return {"ok": True, "verdict": cert.verdict, "sequence": c, "cost_usd": cost,
                        "wall_s": round(time.time() - t0, 1), "n_units": len(units),
                        "units_solved": pinned_units, "units_dropped_for_pin_clash": dropped_units,
                        "seed_coverage": round(covered / len(target), 3), "n_screened": screened,
                        "units": unit_report, "certificate": cert.to_dict(),
                        "note": (f"{pinned_units}/{len(units)} units designed and pinned "
                                 f"({covered}/{len(target)} positions), the rest repaired by search")}
            bp = (cert.detail or {}).get("best_bp")
            if bp is not None and (best_bp is None or bp < best_bp):
                best, best_bp = c, bp
    return {"ok": True, "verdict": "not_found", "sequence": best, "cost_usd": cost,
            "wall_s": round(time.time() - t0, 1), "n_units": len(units), "units_solved": pinned_units,
            "units_dropped_for_pin_clash": dropped_units, "seed_coverage": round(covered / len(target), 3),
            "n_screened": screened, "bp_away": best_bp, "units": unit_report,
            "note": (f"{pinned_units}/{len(units)} units were designed and verified individually and pinned "
                     f"({covered}/{len(target)} positions); the seams did not close -- closest miss "
                     f"{best_bp} base pairs")}
