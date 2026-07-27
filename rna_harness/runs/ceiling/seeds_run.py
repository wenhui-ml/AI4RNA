#!/usr/bin/env python3
"""The efficiency fix, validated: allocate independent TRAJECTORIES, not call depth.

Why. alloc_run.py varied per-target call depth and lost at low budget. per_target.py found the reason,
and it was a design error on my side, not a property of the problem:

  * The cap is never binding. Across every arm and budget the operator used 43-50% of its allowance and
    only 16-33% of attempts reached the cap. It self-terminates, so DEPTH IS NOT PURCHASABLE.
  * With deterministic=True the outcome is a function of (target, seed): a target's win count is nearly
    constant across a 12x budget range, and wins per seed are under-dispersed relative to Poisson
    (var 0.38 vs mean 0.53 at 20% budget, 0.60 vs 1.73 at 100%).
  * On Delta ranks 1-6 the stored NON-deterministic sweep produced 5 wins at comparable call counts where
    the deterministic run produced 1 -- the difference is trajectory diversity, not compute.

So the scarce resource is independent trajectories. alloc_run.py gave every arm the same 15 seeds and
never varied it: it held the scarce resource fixed and varied the free one.

The model. A trajectory on target i succeeds with probability p_i, and within one replicate a target
counts once, so allocating b_i trajectories gives

    E[solves] = sum_i [ 1 - (1 - p_i)^{b_i} ]

which is CONCAVE in b_i -- water-filling is exact, and p_i = 0 is the cure fraction, so none of the
Weibull/cure machinery from cure.py is needed. Cost per trajectory is measured, not assumed.

Declared before running. k (how many of the lowest-Delta targets to fund) and the per-target trajectory
count are chosen on ids 0-29 ONLY, from the stored sweep's per-target win rates. Run with no GO env var
to print the declaration and stop; GO=1 to execute on ids 30-59.
"""
import json, os, random, sys, time
import multiprocessing as mp
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase                  # noqa: E402
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,   # noqa: E402
                                    shapeknots_pairs, rnastructure_available)

OUT = os.path.join(HERE, "seeds")
BUDGET = 2101          # calls/replicate: what uniform actually spent at full budget (per_target.py)
MAXPK = 122            # keep the cap high -- truncating it cuts the panel-closing phase
NREP = 15
_RNASTR = None


def _init():
    global _RNASTR
    os.environ.setdefault("DESIRNA_CMD",
                          "/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py")
    _RNASTR = rnastructure_available()


def panel(seq, target):
    pk = forms_target(pkiss_pairs(seq), target)
    pb = forms_target(probknot_pairs(seq), target) if _RNASTR else None
    sk = forms_target(shapeknots_pairs(seq), target) if _RNASTR else None
    return {"pkiss": bool(pk), "probknot": None if pb is None else bool(pb),
            "shapeknots": None if sk is None else bool(sk),
            "unanimous": (bool(pk) and bool(pb) and bool(sk)) if _RNASTR else None}


def work(item):
    rep, arm, tid, target, seed = item
    from rna_harness.cognitive.skills.pk_decompose_panel import DecomposePKPanelTool
    tool = DecomposePKPanelTool(skel_wall_s=12.0, repair_wall_s=90.0, max_repair_iters=800,
                               panel_gate=3, max_probknot=120,
                               deterministic=True, max_pkiss=MAXPK)
    t0 = time.time()
    try:
        r = tool.attempt(target, seed=seed)
        adj = panel(r.seq, target) if (r.solved and r.seq) else None
        return {"rep": rep, "arm": arm, "id": tid, "seed": seed, "solved": bool(r.solved),
                "seq": r.seq, "panel": adj, "n_pkiss": (r.stats or {}).get("n_pkiss", 0),
                "s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"rep": rep, "arm": arm, "id": tid, "seed": seed, "error": repr(e)[:250], "n_pkiss": 0}


def fit_k():
    """Choose k and the trajectory count on ids 0-29 only. Returns (k, S, prediction table)."""
    D = json.load(open(os.path.join(HERE, "pk_ev.json")))
    pk = {r["id"]: r for r in D["rows"]}
    idx = lambda t: int(t.split(":")[-1])
    FIT = sorted([t for t in pk if idx(t) < 30], key=idx)
    dmin = {t: min(pk[t]["draws_norm"][:8]) for t in pk}
    rank = sorted(FIT, key=lambda t: (dmin[t], pk[t]["n"], t))
    # per-trajectory success rate and per-trajectory cost, both from the stored 15-cell sweep
    att = defaultdict(list)
    for a in D["attempts"]:
        if a["arm"] == "panel_auto":
            att[a["id"]].append(a)
    p = {t: sum(a["unanimous"] for a in att[t]) / max(1, len(att[t])) for t in FIT}
    c = {t: float(np.mean([a["n_pkiss"] for a in att[t]])) for t in FIT}
    rows = []
    for k in range(3, 31):
        g = rank[:k]
        cbar = float(np.mean([c[t] for t in g]))
        S = BUDGET / (k * cbar)
        e = sum(1.0 - (1.0 - p[t]) ** S for t in g)
        rows.append((k, cbar, S, e, sum(p[t] for t in g)))
    best = max(rows, key=lambda r: r[3])
    return best[0], best[2], rows, rank, p, c


def main():
    k, S, rows, rank_fit, p_fit, c_fit = fit_k()
    print("=== DECLARED ON ids 0-29 (stored sweep only; no held-out outcome used) ===")
    print(f"  budget/replicate {BUDGET} calls, max_pkiss {MAXPK} (kept high on purpose)")
    print(f"  {'k':>3} {'cost/traj':>10} {'trajectories each':>18} {'E[solves]':>10} "
          f"{'sum p (=1 traj each)':>21}")
    for kk, cbar, Ss, e, sp in rows:
        tag = "   <== chosen" if kk == k else ""
        print(f"  {kk:3d} {cbar:10.1f} {Ss:18.2f} {e:10.2f} {sp:21.2f}{tag}")
    Sint = max(1, int(round(S)))
    print(f"  -> k = {k}, {Sint} trajectories each (real-valued optimum {S:.2f})")

    D = json.load(open(os.path.join(HERE, "pk_ev.json")))
    pkr = {r["id"]: r for r in D["rows"]}
    idx = lambda t: int(t.split(":")[-1])
    HELD = sorted([t for t in pkr if idx(t) >= 30], key=idx)
    dmin = {t: min(pkr[t]["draws_norm"][:8]) for t in pkr}
    rank_h = sorted(HELD, key=lambda t: (dmin[t], pkr[t]["n"], t))
    chosen = rank_h[:k]
    print(f"\n  applied to ids 30-59: fund {[t.split(':')[-1] for t in chosen]}")

    # A dose-response curve, not a single comparison. Seeds are nested (S2 subset S4 subset S8) because
    # trajectory q uses seed base+17j+q, so the three delta arms are paired and the curve is within-target.
    #
    # There is no budget-matched "restarts without screening" arm, and that is the point: at BUDGET calls,
    # 30 targets at 1 trajectory each already exhausts the budget (cost/traj on the full set is 133 vs
    # 42 on the lowest-Delta 11). Extra trajectories can ONLY come from funding fewer, cheaper targets.
    # Screening is not an option here, it is the only source of trajectories.
    arms = {
        "depth1":   [(t, 1) for t in HELD],            # baseline: 30 targets, 1 trajectory each
        "delta_S2": [(t, 2) for t in chosen],
        "delta_S4": [(t, Sint) for t in chosen],       # the declared policy
        "delta_S8": [(t, 2 * Sint) for t in chosen],   # over budget on purpose: does the curve keep rising?
    }
    print("\n=== arms (dose-response; seeds nested across the delta arms) ===")
    for a, spec in arms.items():
        cb = float(np.mean([c_fit[x] for x in rank_fit[:k]])) if a != "depth1" else \
             float(np.mean([c_fit[x] for x in rank_fit]))
        print(f"  {a:9} {len(spec):2d} targets x {spec[0][1]:2d} trajectories = "
              f"{sum(s for _, s in spec):3d} trajectories, ~{sum(s for _, s in spec)*cb:.0f} calls "
              f"({sum(s for _, s in spec)*cb/BUDGET:.2f}x budget)")
    print(f"  rand_S4    11 random targets x {Sint} trajectories  (control: is it Delta, or just "
          f"fewer-targets-more-seeds?)")

    if not os.environ.get("GO"):
        print("\nDeclaration only. Re-run with GO=1 to execute.")
        return

    os.makedirs(OUT, exist_ok=True)
    tgt = {t.task_id: t.target for t in load_pseudobase(limit=60)}
    items = []
    for rep in range(NREP):
        base = 50_000 + rep * 1000            # disjoint seed block per replicate
        rng = random.Random(777 + rep)
        rand_set = rng.sample(HELD, k)
        spec_all = dict(arms)
        spec_all["rand_S4"] = [(t, Sint) for t in rand_set]
        for arm, spec in spec_all.items():
            for j, (t, s) in enumerate(spec):
                for q in range(s):
                    items.append((rep, arm, t, tgt[t], base + 17 * j + q))
    done = {f[:-5] for f in os.listdir(OUT) if f.endswith(".json")}
    key = lambda it: f"r{it[0]}_{it[1]}_{it[2].split(':')[-1]}_s{it[4]}"
    todo = [it for it in items if key(it) not in done]
    print(f"\n{len(items)} trajectories total, {len(todo)} to run "
          f"(~{len(todo)*45*0.22/3600:.1f} core-h at 45 calls each)", flush=True)

    nw = int(os.environ.get("NW", "40"))
    t0 = time.time()
    with mp.get_context("fork").Pool(nw, initializer=_init) as pool:
        for i, rec in enumerate(pool.imap_unordered(work, todo, chunksize=1), 1):
            json.dump(rec, open(os.path.join(OUT, key((rec["rep"], rec["arm"], rec["id"], None,
                                                       rec["seed"]))) + ".json", "w"))
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el/60:.1f} min  eta {el/i*(len(todo)-i)/60:.1f} min",
                      flush=True)
    json.dump({"k": k, "S": Sint, "budget": BUDGET, "maxpk": MAXPK, "nrep": NREP,
               "chosen": chosen, "fit_rows": rows},
              open(os.path.join(HERE, "seeds_plan.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
