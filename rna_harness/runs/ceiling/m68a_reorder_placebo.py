"""M68 arm A: the placebo covariate this paper demands of every mediator, applied at last
to its own leaderboard arm.

Adding a stratification dimension moves method means by itself. The question the published
arm cannot answer is whether the movement is about the covariate or about stratification.
A placebo covariate with the same marginal and the same granularity, but permuted within
target so it cannot know the method, answers it.  Offline, CPU, $0.
"""
import collections, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from m61_readout_and_consequence import _compare

SEED = 20260902
NPLACEBO = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rng = np.random.default_rng(SEED)
F = json.load(open(os.path.join(HERE, "m13_features.json")))

items = []
for fn, d in F.items():
    for r in d["rows"]:
        g = r["F"].get("gc_paired")
        if g is None or r.get("wet") is None:
            continue
        items.append({"m": r["method"], "y": float(r["wet"]), "p": (fn, r["puzzle"]), "g": g})
gq = np.quantile([x["g"] for x in items], [0.2, 0.4, 0.6, 0.8])
for x in items:
    x["q"] = int(np.searchsorted(gq, x["g"]))
print(f"items {len(items)}, methods {len({x['m'] for x in items})}, strata {len({x['p'] for x in items})}")

KM, KY = (lambda z: z["m"]), (lambda z: z["y"])
SPEC = [("target_only", lambda z: (z["p"],)), ("target_plus_gc", lambda z: (z["p"], z["q"]))]

real = _compare(items, KM, KY, "wet", SPEC, rng, nboot=0)
R_tau, R_moved = real["target_plus_gc"]["tau_vs_raw"], real["target_plus_gc"]["moved"]
R_max = real["target_plus_gc"]["max_move"]
print(f"\nreal covariate : tau={R_tau:+.4f}  moved={R_moved}/{real['methods']}  max_move={R_max}")

# --- the placebo: same marginal, same granularity, permuted WITHIN target so it cannot
#     carry any information about which method produced the design.
by_t = collections.defaultdict(list)
for i, x in enumerate(items):
    by_t[x["p"]].append(i)
qs = np.array([x["q"] for x in items])

taus, moveds, maxes = [], [], []
for b in range(NPLACEBO):
    qp = qs.copy()
    for idx in by_t.values():
        a = np.array(idx)
        qp[a] = qs[a][rng.permutation(len(a))]
    for i, x in enumerate(items):
        x["q"] = int(qp[i])
    C = _compare(items, KM, KY, "wet", SPEC, rng, nboot=0)
    t = C["target_plus_gc"]
    taus.append(t["tau_vs_raw"]); moveds.append(t["moved"]); maxes.append(t["max_move"])
    if (b + 1) % 5000 == 0:
        print(f"  placebo {b+1}/{NPLACEBO}  tau median {np.median(taus):+.4f}  "
              f"moved median {np.median(moveds):.1f}")
for i, x in enumerate(items):
    x["q"] = int(qs[i])            # restore

taus, moveds, maxes = np.array(taus), np.array(moveds), np.array(maxes)
p_tau = (float((taus <= R_tau).sum()) + 1) / (NPLACEBO + 1)
p_mov = (float((moveds >= R_moved).sum()) + 1) / (NPLACEBO + 1)
print(f"\nplacebo distribution over {NPLACEBO} draws")
print(f"  tau       median {np.median(taus):+.4f}   2.5th {np.percentile(taus,2.5):+.4f}   "
      f"97.5th {np.percentile(taus,97.5):+.4f}")
print(f"  moved     median {np.median(moveds):.1f}      2.5th {np.percentile(moveds,2.5):.1f}      "
      f"97.5th {np.percentile(moveds,97.5):.1f}")
print(f"  max_move  median {np.median(maxes):.1f}      97.5th {np.percentile(maxes,97.5):.1f}")
print(f"\n  real tau   {R_tau:+.4f}  ->  placebo p = {p_tau:.4g}   "
      f"(S2: real below the 2.5th percentile: {R_tau < np.percentile(taus,2.5)})")
print(f"  real moved {R_moved}       ->  placebo p = {p_mov:.4g}   "
      f"(S1: real above the 97.5th percentile: {R_moved > np.percentile(moveds,97.5)})")
print(f"  S3 stratification alone moves methods (placebo median moved > 0): "
      f"{np.median(moveds) > 0}")
OUT = dict(seed=SEED, n_placebo=NPLACEBO, n_items=len(items), methods=real["methods"],
           real=dict(tau=R_tau, moved=R_moved, max_move=R_max),
           placebo=dict(tau_median=float(np.median(taus)), tau_lo=float(np.percentile(taus,2.5)),
                        tau_hi=float(np.percentile(taus,97.5)),
                        moved_median=float(np.median(moveds)),
                        moved_lo=float(np.percentile(moveds,2.5)),
                        moved_hi=float(np.percentile(moveds,97.5)),
                        maxmove_median=float(np.median(maxes)),
                        maxmove_hi=float(np.percentile(maxes,97.5))),
           p_tau=p_tau, p_moved=p_mov,
           S1=bool(R_moved > np.percentile(moveds,97.5)), S2=bool(R_tau < np.percentile(taus,2.5)),
           S3=bool(np.median(moveds) > 0))
json.dump(OUT, open(os.path.join(HERE, "m68a_reorder_placebo.json"), "w"), indent=1)
print("\nwrote m68a_reorder_placebo.json")
