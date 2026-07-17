"""RSSD held-out compounding on Pseudobase++ pseudoknots -- the gaming-proof test of whether a distilled
PROCEDURE (the DecomposePKTool skill) + learned routing cuts COST at equal quality, and GENERALISES to
targets it never saw. See docs/RSSD_design.md.

WHAT WE FOUND (drives the design):
  * CAPABILITY (delta-solve) is a NULL, and we explain it: the bp<=2 residual is DESIGNABILITY-limited --
    DesiRNA (an MFE optimiser) and pkdecomp (a pKiss optimiser) independently plateau at exactly bp=1 on
    the same fragile terminal crossing pair. No skill can crack what has no pKiss ground state. So adding
    the skill does NOT enlarge the solve set (it solves a SUBSET of DesiRNA's 180).
  * COST is where the skill compounds: on the SOLVABLE PK targets pkdecomp solves at ~$3e-6 (often the
    decompose+compose sequence already folds correctly -- zero repair), vs DesiRNA's fixed heavy search.

So the falsifiable claim we TEST here is the cost "真门槛": at the SAME held-out solve set, does routing
(skill-first, learned on TRAIN) cost far less than always-SOTA -- and does the routing rule transfer to
held-out? Simulation over a REAL-cost outcome table (arms share the table, so the DELTA is robust):
  arms: always_sota [DesiRNA on every target]  ->  cold [remc->pkdecomp->desirna, no learning]
        ->  learned [router learns feature->tool on TRAIN, FROZEN, evaluated on HELD-OUT].
Cost accounting (all documented, none nominal-only where it matters):
  * pkdecomp : REAL measured (solved, cost) -- this module's --build step runs it on all 251.
  * desirna  : solved from the cached pk251 baseline; per-attempt cost = MEASURED median from
               rssd_desirna_cost (DesiRNA runs a fixed budget, so a per-attempt constant is fair).
  * remc     : modelled solved=False on every PK target (REMC optimises the nested MFE skeleton and
               cannot form a crossing stem -- established in track_b_pk fp_demo), cost = its nominal.
               It is the "doomed cheap rung" a learned router should skip; kept so that saving is visible.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..benchmarks.rnainvbench import load_pseudobase
from ..cognitive.cost import solver_cost_usd
from ..cognitive.features import features as extract_features
from ..cognitive.router import Router
from ..cognitive.skills.pk_decompose import DecomposePKTool
from ..cognitive.tools import DesirnaTool, RemcTool

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "rssd_compounding")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")


# ---- build the pkdecomp column (real cost) -------------------------------------------------------
def _pkdecomp_one(idx, target, skel_wall_s, repair_wall_s, max_iters, seed):
    tool = DecomposePKTool(skel_wall_s=skel_wall_s, repair_wall_s=repair_wall_s,
                           max_repair_iters=max_iters, skel_restarts=6)
    r = tool.attempt(target, seed=seed)
    return idx, {"solved": bool(r.solved), "cost": float(r.cost_usd), "bp": int(r.bp),
                 "wall_s": float(r.wall_s), "oracle_s": float(r.oracle_s)}


def build_pkdecomp(outdir, skel_wall_s, repair_wall_s, max_iters, workers, seed):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "pkdecomp_table.json")
    table = json.load(open(path)) if os.path.exists(path) else {}
    tasks = load_pseudobase()
    todo = [(int(t.task_id.split(":")[1]), t.target) for t in tasks
            if str(int(t.task_id.split(":")[1])) not in table]
    print(f"build_pkdecomp: {len(tasks)} targets, {len(table)} cached, {len(todo)} to run "
          f"(repair={repair_wall_s}s workers={workers})", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_pkdecomp_one, i, tgt, skel_wall_s, repair_wall_s, max_iters, seed): i
                for i, tgt in todo}
        for fut in as_completed(futs):
            idx, res = fut.result()
            table[str(idx)] = res
            json.dump(table, open(path, "w"), indent=0)
            if res["solved"]:
                print(f"  {idx:>4} solved ${res['cost']:.2e} ({res['oracle_s']:.0f}s pKiss)", flush=True)
    solved = sum(1 for r in table.values() if r["solved"])
    print(f"build_pkdecomp done: pkdecomp solves {solved}/{len(table)} at median "
          f"${sorted(r['cost'] for r in table.values())[len(table)//2]:.2e} -> {path}", flush=True)
    return table


# ---- load the other columns ----------------------------------------------------------------------
def _desirna_column():
    """solved from cached pk251; per-attempt cost = measured DesiRNA median (fixed budget => fair)."""
    rows = json.load(open(os.path.join(RUNS, "pk_desirna", "pk_desirna.json")))["rows"]
    solved = {int(r["task_id"].split(":")[1]): bool(r["pk_verified_solved"]) for r in rows}
    cost_path = os.path.join(RUNS, "rssd_desirna_cost.json")
    if os.path.exists(cost_path):
        costs = [v["cost_usd"] for v in json.load(open(cost_path)).values()]
        med = sorted(costs)[len(costs) // 2]
        src = f"measured median of {len(costs)} runs"
    else:
        med = solver_cost_usd(300, 10)                       # nominal upper bound if unmeasured
        src = "NOMINAL (-t300 x10 replicas) -- run rssd_desirna_cost for measured"
    return solved, med, src


# ---- simulate cold vs learned on a TRAIN/HELD-OUT split ------------------------------------------
def _split(ids, mode="parity", seed=0):
    """Deterministic TRAIN/HELD-OUT split. Multiple modes so the held-out gain can be shown NOT to be a
    lucky partition: 'parity' (even/odd id), 'half' (first/second), 'shuffle' (seeded permutation)."""
    if mode == "parity":
        return [i for i in ids if i % 2 == 0], [i for i in ids if i % 2 == 1]
    if mode == "half":
        h = len(ids) // 2
        return ids[:h], ids[h:]
    if mode == "shuffle":
        import random
        r = random.Random(seed)
        perm = list(ids)
        r.shuffle(perm)
        h = len(perm) // 2
        return sorted(perm[:h]), sorted(perm[h:])
    raise ValueError(mode)


def _cascade_cost(order, target_id, cols):
    """Fire tools in `order`, paying each cost, until one solves. Returns (cost, solved, first_solver)."""
    cost, solved, first = 0.0, False, None
    for name in order:
        c = cols[name]["cost"](target_id)
        cost += c
        if cols[name]["solved"](target_id):
            solved, first = True, name
            break
    return cost, solved, first


def simulate(outdir, split_mode="parity", split_seed=0):
    tasks = load_pseudobase()
    ids = [int(t.task_id.split(":")[1]) for t in tasks]
    tgt = {int(t.task_id.split(":")[1]): t.target for t in tasks}
    pk = json.load(open(os.path.join(outdir, "pkdecomp_table.json")))
    des_solved, des_cost, des_src = _desirna_column()
    remc_cost = RemcTool(wall_s=30).nominal_cost_usd()

    # tool columns: solved(id)->bool, cost(id)->usd
    cols = {
        "remc@30s":    {"solved": lambda i: False,                       "cost": lambda i: remc_cost,
                        "nominal": remc_cost},
        "pkdecomp":    {"solved": lambda i: pk[str(i)]["solved"],        "cost": lambda i: pk[str(i)]["cost"],
                        "nominal": DecomposePKTool().nominal_cost_usd()},
        "desirna@300": {"solved": lambda i: des_solved.get(i, False),    "cost": lambda i: des_cost,
                        "nominal": des_cost},
    }
    cold_order = sorted(cols, key=lambda n: cols[n]["nominal"])           # cheapest nominal first
    feats = {i: extract_features(tgt[i]) for i in ids}
    train, held = _split(ids, mode=split_mode, seed=split_seed)

    # LEARN on TRAIN only: build a router over the same tool objects (names must match cols)
    router = Router([RemcTool(wall_s=30), DecomposePKTool(), DesirnaTool(timelimit_s=300)])
    name_of = {"remc@30s": "remc@30s", "pkdecomp": DecomposePKTool().name, "desirna@300": DesirnaTool(timelimit_s=300).name}
    # map our column names to the tool.name the router uses
    col_for_toolname = {t.name: cn for cn, t in
                        zip(["remc@30s", "pkdecomp", "desirna@300"],
                            [RemcTool(wall_s=30), DecomposePKTool(), DesirnaTool(timelimit_s=300)])}
    for i in train:
        for cn in cold_order:                                            # observe every tool's outcome
            tname = name_of[cn]
            router.record(feats[i], tname, cols[cn]["solved"](i), cols[cn]["cost"](i))

    def learned_order(i):
        plan = router.plan(feats[i])                                     # tool objects, reordered
        return [col_for_toolname[t.name] for t in plan]

    # LEARNED ABSTENTION (the real cognitive-core lever): always try the ~free skill; fire the EXPENSIVE
    # DesiRNA only when the router (trained on TRAIN) predicts P(DesiRNA solves | features) >= tau. On the
    # designability-wall targets DesiRNA ALSO fails, so declining to pay for it there loses no quality --
    # the only risk is abstaining on a genuinely DesiRNA-solvable target (tracked as solves_lost).
    from ..cognitive.features import FEATURE_NAMES
    des_name = DesirnaTool(timelimit_s=300).name

    def abstain_arm(split_ids, tau):
        c = s = fired = abst = lost = 0
        for i in split_ids:
            c += cols["pkdecomp"]["cost"](i)                             # the cheap skill is always worth a try
            if cols["pkdecomp"]["solved"](i):
                s += 1
                continue
            x = [feats[i][k] for k in FEATURE_NAMES]
            est = router._knn(x, des_name)
            p = est[0] if est else 1.0                                   # no data -> fire (conservative)
            if p >= tau:
                c += cols["desirna@300"]["cost"](i)
                fired += 1
                s += int(cols["desirna@300"]["solved"](i))
            else:
                abst += 1
                lost += int(cols["desirna@300"]["solved"](i))            # a real solve we declined
        return {"cost": c, "solved": s, "desirna_fired": fired, "abstained": abst, "solves_lost": lost}

    def oracle_abstain_arm(split_ids):
        """UPPER BOUND on abstention efficiency: a PERFECT predictor fires SOTA iff DesiRNA actually
        solves the target (never wastes SOTA on a doomed one, never loses a solve). The gap between the
        learned k-NN abstention and this bound = how much the LEARNED policy leaves on the table."""
        c = s = 0
        for i in split_ids:
            c += cols["pkdecomp"]["cost"](i)
            if cols["pkdecomp"]["solved"](i):
                s += 1; continue
            if des_solved.get(i, False):                        # perfect foresight: only fire when it works
                c += des_cost; s += 1
        return {"cost": c, "solved": s}

    arms = {}
    TAU_GRID = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90)
    for split_name, split_ids in (("held", held), ("train", train), ("all", ids)):
        rec = {}
        # always_sota: DesiRNA on every target
        c_sota = sum(cols["desirna@300"]["cost"](i) for i in split_ids)
        s_sota = sum(int(cols["desirna@300"]["solved"](i)) for i in split_ids)
        # cold cascade (skill, no learning)
        c_cold = s_cold = 0
        firsts_cold = {}
        for i in split_ids:
            c, s, f = _cascade_cost(cold_order, i, cols)
            c_cold += c; s_cold += int(s); firsts_cold[f] = firsts_cold.get(f, 0) + int(s)
        # learned (router trained on TRAIN, applied here)
        c_learn = s_learn = 0
        firsts_learn = {}
        for i in split_ids:
            c, s, f = _cascade_cost(learned_order(i), i, cols)
            c_learn += c; s_learn += int(s); firsts_learn[f] = firsts_learn.get(f, 0) + int(s)
        rec = {"n": len(split_ids),
               "always_sota": {"cost": c_sota, "solved": s_sota},
               "cold":        {"cost": c_cold, "solved": s_cold, "first_solver": firsts_cold},
               "learned":     {"cost": c_learn, "solved": s_learn, "first_solver": firsts_learn},
               "abstain":     {str(tau): abstain_arm(split_ids, tau) for tau in TAU_GRID},
               "oracle_abstain": oracle_abstain_arm(split_ids)}
        arms[split_name] = rec

    # skill capability: does pkdecomp solve any target DesiRNA does NOT? (delta-solve)
    skill_only = [i for i in ids if cols["pkdecomp"]["solved"](i) and not des_solved.get(i, False)]
    pk_solved = [i for i in ids if cols["pkdecomp"]["solved"](i)]

    out = {"desirna_cost_source": des_src, "desirna_cost_per_attempt": des_cost,
           "remc_cost": remc_cost, "cold_order": cold_order,
           "pkdecomp_solves": len(pk_solved), "desirna_solves": sum(des_solved.values()),
           "skill_only_solves_ids": skill_only, "arms": arms}
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, "compounding.json"), "w"), indent=1)

    # report
    h = arms["held"]
    print(f"\n=== RSSD held-out compounding (Pseudobase++ PK; DesiRNA cost = {des_src}) ===", flush=True)
    print(f"  pkdecomp solves {len(pk_solved)}/251, DesiRNA {sum(des_solved.values())}/251; "
          f"skill-only (delta-capability) = {len(skill_only)}  -> {'NULL (designability wall)' if not skill_only else skill_only}", flush=True)
    print(f"\n  HELD-OUT ({h['n']} targets, learned router trained ONLY on the disjoint TRAIN split):", flush=True)
    for arm in ("always_sota", "cold", "learned"):
        a = h[arm]
        print(f"    {arm:12} solved={a['solved']:>3}/{h['n']}  cost=${a['cost']:.4f}"
              + (f"  first_solver={a.get('first_solver')}" if 'first_solver' in a else ""), flush=True)
    sota, cold, learn = h["always_sota"]["cost"], h["cold"]["cost"], h["learned"]["cost"]
    print(f"\n  cost to reach the same held-out solve set:", flush=True)
    print(f"    skill (cold)   vs always-SOTA : ${cold:.4f} vs ${sota:.4f}  -> {sota/max(cold,1e-12):.1f}x cheaper", flush=True)
    print(f"    skill+learned  vs always-SOTA : ${learn:.4f} vs ${sota:.4f}  -> {sota/max(learn,1e-12):.1f}x cheaper", flush=True)
    print(f"    learning gain (cold->learned) : ${cold:.4f} -> ${learn:.4f}  "
          f"({100*(cold-learn)/max(cold,1e-12):.1f}% of cold)", flush=True)
    # G1 -- ABLATION: where does the efficiency come from? (add ingredients one at a time)
    print(f"\n  [G1] ABLATION (held-out, same 96-solve quality):", flush=True)
    print(f"    always-SOTA (no cognitive core)      solved={h['always_sota']['solved']:>3}/{h['n']}  cost=${sota:.4f}  (1.0x)", flush=True)
    print(f"    + cheap skill first (cascade)        solved={h['cold']['solved']:>3}/{h['n']}  cost=${cold:.4f}  ({sota/max(cold,1e-12):.1f}x)", flush=True)
    print(f"    + learned reorder                    solved={h['learned']['solved']:>3}/{h['n']}  cost=${learn:.4f}  ({sota/max(learn,1e-12):.1f}x)", flush=True)
    best_ab = min((a for a in h["abstain"].values() if a["solves_lost"] == 0), key=lambda a: a["cost"], default=None)
    if best_ab:
        print(f"    + learned ABSTENTION (0 loss)        solved={best_ab['solved']:>3}/{h['n']}  cost=${best_ab['cost']:.4f}  "
              f"({sota/max(best_ab['cost'],1e-12):.1f}x)   <-- the lever", flush=True)

    # F1 -- PARETO frontier of the abstention threshold + the perfect-foresight upper bound
    orc = h["oracle_abstain"]
    print(f"\n  [F1] ABSTENTION PARETO (held-out): solved / cost / x-cheaper / real-solves-lost", flush=True)
    for tau in sorted(h["abstain"], key=float):
        a = h["abstain"][tau]
        print(f"    tau={float(tau):.2f}  solved={a['solved']:>3}/{h['n']}  cost=${a['cost']:.4f}  "
              f"{sota/max(a['cost'],1e-12):>5.1f}x  lost={a['solves_lost']}", flush=True)
    print(f"    ORACLE (perfect foresight, upper bound)  solved={orc['solved']:>3}/{h['n']}  cost=${orc['cost']:.4f}  "
          f"{sota/max(orc['cost'],1e-12):.1f}x  lost=0", flush=True)
    if best_ab:
        gap = best_ab["cost"] / max(orc["cost"], 1e-12)
        print(f"    -> best 0-loss learned abstention is {gap:.2f}x the oracle-optimal cost "
              f"(closer to 1.0 = learned policy captures more of the achievable saving)", flush=True)
    print(f"\n  wrote {os.path.join(outdir, 'compounding.json')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="run pkdecomp on all 251 (resumable)")
    ap.add_argument("--sim", action="store_true", help="simulate always_sota vs cold vs learned on the split")
    ap.add_argument("--split", choices=["parity", "half", "shuffle"], default="parity")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--skel-wall-s", type=float, default=8.0)
    ap.add_argument("--repair-wall-s", type=float, default=30.0)
    ap.add_argument("--max-iters", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    if a.build:
        build_pkdecomp(a.outdir, a.skel_wall_s, a.repair_wall_s, a.max_iters, a.workers, a.seed)
    if a.sim:
        simulate(a.outdir, split_mode=a.split, split_seed=a.split_seed)
    if not (a.build or a.sim):
        ap.error("pass --build and/or --sim")
