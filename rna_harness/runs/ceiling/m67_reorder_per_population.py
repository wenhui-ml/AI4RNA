"""M67 arm A: the leaderboard reorder, run inside each released population separately.

The estimator is imported from the pooled arm, not re-implemented, so a difference
between the pooled number and these can only come from the partition.  Offline, $0.
"""
import collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m61_readout_and_consequence import _compare, _within  # the identical estimator

SEED = 20260902
MIN_METHODS = 9          # fixed in the declaration, before any result
EXCLUDE = {"panel_f1_fifth.json"}   # two methods cannot produce a ranking; declared in advance

F = json.load(open(os.path.join(HERE, "m13_features.json")))
rng = np.random.default_rng(SEED)

def items_of(rows):
    out = []
    for r in rows:
        g = r["F"].get("gc_paired")
        if g is None or r.get("wet") is None:
            continue
        out.append({"m": r["method"], "y": float(r["wet"]), "p": r["puzzle"], "g": g})
    return out

def run(it, tag):
    # quintile edges computed WITHIN the population, as declared
    gq = np.quantile([x["g"] for x in it], [0.2, 0.4, 0.6, 0.8])
    for x in it:
        x["q"] = int(np.searchsorted(gq, x["g"]))
    KM, KY = (lambda x: x["m"]), (lambda x: x["y"])
    # K4: a method with no stratum it shares with another method survives the raw ranking but not
    # the adjusted one. The declaration says such a method is reported unrankable here, never
    # silently pooled, so it is removed from BOTH rankings and named.
    spec = lambda x: (x["p"], x["q"])
    A, _u = _within(it, spec, KM, KY)
    keep = set(A)
    unrankable = sorted({x["m"] for x in it} - keep)
    it2 = [x for x in it if x["m"] in keep]
    return _compare(it2, KM, KY, "wet",
                    [("target_only", lambda x: (x["p"],)),
                     ("target_plus_gc", spec)],
                    rng, cluster=lambda x: x["p"]), unrankable

OUT = {"seed": SEED, "min_methods": MIN_METHODS, "excluded": sorted(EXCLUDE), "populations": {}}
print(f"{'population':24s} {'n':>6s} {'meth':>5s} {'tau(target)':>12s} {'moved':>6s} "
      f"{'tau(+GC)':>9s} {'moved':>6s} {'maxmove':>8s}  bootstrap CI on tau(+GC)")
for fn, d in F.items():
    it = items_of(d["rows"])
    if not it:
        continue
    C, unrankable = run(it, fn)
    role = ("headline" if (C["methods"] >= MIN_METHODS and fn not in EXCLUDE)
            else ("excluded" if fn in EXCLUDE else "under-powered"))
    t0, t1 = C["target_only"], C["target_plus_gc"]
    ci = t1.get("tau_boot_ci")
    OUT["populations"][fn] = dict(role=role, n=C["n"], methods=C["methods"],
                                  tau_target=t0["tau_vs_raw"], moved_target=t0["moved"],
                                  tau_gc=t1["tau_vs_raw"], moved_gc=t1["moved"],
                                  max_move=t1["max_move"], ci=ci,
                                  frac_below_1=t1.get("tau_boot_frac_below_1"),
                                  unrankable_here=unrankable)
    print(f"{fn[:24]:24s} {C['n']:>6d} {C['methods']:>5d} {t0['tau_vs_raw']:>+12.3f} "
          f"{t0['moved']:>6d} {t1['tau_vs_raw']:>+9.3f} {t1['moved']:>6d} {t1['max_move']:>8d}  "
          f"{ci}   [{role}]" + (f"  UNRANKABLE: {unrankable}" if unrankable else ""))

H = {k: v for k, v in OUT["populations"].items() if v["role"] == "headline"}
taus = [v["tau_gc"] for v in H.values()]
OUT["headline"] = dict(
    n_populations=len(H),
    all_move_at_least_one=all(v["moved_gc"] >= 1 for v in H.values()),
    signs_agree=len({np.sign(1.0 - t) for t in taus}) == 1,
    tau_min=float(min(taus)) if taus else None, tau_max=float(max(taus)) if taus else None,
    n_ci_excluding_one=sum(1 for v in H.values() if v["ci"] and v["ci"][1] < 1.0),
    target_only_moves_none=sum(1 for v in H.values() if v["moved_target"] == 0),
)
print("\n--- headline populations:", OUT["headline"]["n_populations"])
print("  every one moves at least one method :", OUT["headline"]["all_move_at_least_one"])
print("  all taus on the same side of 1      :", OUT["headline"]["signs_agree"])
print(f"  tau range                           : {OUT['headline']['tau_min']:+.3f} to {OUT['headline']['tau_max']:+.3f}")
print("  CIs excluding 1                     :", OUT["headline"]["n_ci_excluding_one"], "/", len(H))
print("  target-alone moves NO method in     :", OUT["headline"]["target_only_moves_none"], "/", len(H))

pooled = json.load(open(os.path.join(HERE, "m61_readout_and_consequence.json")))["C"]["benchmark_score"]
pt = pooled["target_plus_gc"]["tau_vs_raw"]
OUT["pooled_tau"] = pt
OUT["pooled_inside_range"] = bool(taus and min(taus) <= pt <= max(taus))
print(f"\n  pooled tau = {pt:+.3f};  inside the per-population range: {OUT['pooled_inside_range']}")
json.dump(OUT, open(os.path.join(HERE, "m67_reorder_per_population.json"), "w"), indent=1)
print("\nwrote m67_reorder_per_population.json")
