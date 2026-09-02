"""M67 K2: the pooled tau fell outside the per-population range, so the declaration
requires the cause be found before the pooled number is reported again.

Two candidate causes, tested rather than argued:
  (a) composition -- the pooled arm ranks a method set no single population carries;
  (b) strata      -- the pooled arm's quintile edges are computed across populations.
"""
import collections, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from m61_readout_and_consequence import _compare, _within
SEED = 20260902
F = json.load(open(os.path.join(HERE, "m13_features.json")))
HEAD = ["panel_f1_fresh.json", "panel_f1_designs.json", "panel_f1_third.json"]

def rows(fn):
    return [{"m": r["method"], "y": float(r["wet"]), "p": r["puzzle"], "g": r["F"]["gc_paired"],
             "f": fn}
            for r in F[fn]["rows"]
            if r["F"].get("gc_paired") is not None and r.get("wet") is not None]

per = {fn: rows(fn) for fn in F}
common = set.intersection(*[{x["m"] for x in per[fn]} for fn in HEAD])
print(f"methods common to the three headline populations: {len(common)}")
print("  ", sorted(common))
allm = {x["m"] for fn in F for x in per[fn]}
print(f"methods anywhere in the pooled corpus: {len(allm)};  pooled-only: {sorted(allm - common)}\n")

def tau_of(items, within_pop_bins, restrict=None):
    it = [dict(x) for x in items]
    if restrict: it = [x for x in it if x["m"] in restrict]
    if not it: return None
    if within_pop_bins:
        for fn in {x["f"] for x in it}:
            sub = [x for x in it if x["f"] == fn]
            q = np.quantile([x["g"] for x in sub], [0.2, 0.4, 0.6, 0.8])
            for x in sub: x["q"] = int(np.searchsorted(q, x["g"]))
    else:
        q = np.quantile([x["g"] for x in it], [0.2, 0.4, 0.6, 0.8])
        for x in it: x["q"] = int(np.searchsorted(q, x["g"]))
    KM, KY = (lambda z: z["m"]), (lambda z: z["y"])
    spec = (lambda z: ((z["f"], z["p"]), z["q"]))
    A, _ = _within(it, spec, KM, KY)
    it = [x for x in it if x["m"] in set(A)]
    C = _compare(it, KM, KY, "wet",
                 [("t0", lambda z: ((z["f"], z["p"]),)), ("t1", spec)],
                 np.random.default_rng(SEED), nboot=0)
    return C["t1"]["tau_vs_raw"], C["methods"], C["t1"]["moved"]

pooled_items = [x for fn in F for x in per[fn]]
print(f"{'condition':52s} {'tau':>8s} {'meth':>5s} {'moved':>6s}")
for lab, items, wp, restr in [
    ("pooled, pooled bins, all methods (the published arm)", pooled_items, False, None),
    ("pooled, per-population bins, all methods",             pooled_items, True,  None),
    ("pooled, pooled bins, common methods only",             pooled_items, False, common),
    ("pooled, per-population bins, common methods only",     pooled_items, True,  common),
]:
    r = tau_of(items, wp, restr)
    print(f"{lab:52s} {r[0]:>+8.3f} {r[1]:>5d} {r[2]:>6d}")
print()
res = {}
for fn in HEAD:
    a = tau_of(per[fn], True, None); b = tau_of(per[fn], True, common)
    res[fn] = dict(all_methods=a, common_only=b)
    print(f"{fn[:28]:28s} all methods tau={a[0]:+.3f} ({a[1]} meth) | "
          f"common only tau={b[0]:+.3f} ({b[1]} meth)")
lo = min(res[fn]['common_only'][0] for fn in HEAD); hi = max(res[fn]['common_only'][0] for fn in HEAD)
pc = tau_of(pooled_items, True, common)[0]
print(f"\nOn the SAME method set and the same bin rule:")
print(f"  per-population range = [{lo:+.3f}, {hi:+.3f}];  pooled = {pc:+.3f};  "
      f"inside = {lo <= pc <= hi}")
json.dump({"common_methods": sorted(common), "per_population": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in res.items()},
           "pooled_common_perpop_bins": pc, "range_common": [lo, hi],
           "inside_after_matching_method_set": bool(lo <= pc <= hi), "seed": SEED},
          open(os.path.join(HERE, "m67_k2_diagnostic.json"), "w"), indent=1)
print("\nwrote m67_k2_diagnostic.json")
