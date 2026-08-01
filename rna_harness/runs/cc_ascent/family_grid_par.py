#!/usr/bin/env python3
"""Parallel driver for family_grid.py. Same targets, same arms, same results -- just not on one core.

The serial version spent 29 of every 32 minutes in panel_auto, one target at a time, on a 256-core host.
Nothing about the experiment required that: the arms run under PKX_DETERMINISTIC=1, where counted budgets
replace wall-clock caps and ViennaRNA's generator is seeded per attempt, so a (target, seed) pair fixes the
run bit-for-bit and is independent of what else is executing. Parallelising across targets is therefore a
pure scheduling change, and validate_against() checks that claim rather than assuming it: it re-runs a seed
that was produced serially and asserts the per-target verdicts are identical.
"""
import json, os, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.environ.setdefault("PKX_DETERMINISTIC", "1")
from family_grid import GRID, htype, panel, OUT, LO, HI                      # noqa: E402
from rna_harness.experiments.track_b_pk import build_policies                # noqa: E402
from rna_harness.oracles_pk import rnastructure_available                    # noqa: E402

ARMS = os.environ.get("FAMILY_ARMS", "always_cheap,autonomous,panel_auto").split(",")
NW = int(os.environ.get("NWORKERS", "50"))


def _one(args):
    n, m, seed = args
    T = htype(n, m)
    pols = build_policies(ARMS)                      # per-worker: policies hold no cross-target state
    rnastr = rnastructure_available()
    rec = {"n": n, "m": m, "len": len(T), "target": T, "arms": {}}
    for name, pol in pols.items():
        try:
            r = pol.solve(T, seed=seed)
            s = r.get("seq")
            rec["arms"][name] = {"solved": bool(r.get("solved")), "seq": s,
                                 "panel": panel(s, T, rnastr) if (r.get("solved") and s) else None}
        except Exception as e:
            rec["arms"][name] = {"solved": None, "error": repr(e)[:200]}
    return rec


def run(seed, out_name=None):
    with Pool(NW) as p:
        rows = p.map(_one, [(n, m, seed) for (n, m) in GRID], chunksize=1)
    out = os.path.join(OUT, out_name or f"grid_s{seed}.json")
    json.dump({"seed": seed, "lo": LO, "hi": HI, "rows": rows}, open(out, "w"), indent=1)
    unan = {a: sum(bool((r["arms"].get(a, {}).get("panel") or {}).get("unanimous")) for r in rows)
            for a in ARMS}
    print(f"  seed {seed}: {len(rows)} targets -> {out}   unanimous {unan}", flush=True)
    return rows


def verdicts(rows):
    return [{a: [(v.get("panel") or {}).get(k) for k in ("pkiss", "probknot", "shapeknots")]
             for a, v in r["arms"].items()} for r in rows]


def validate_against(seed):
    """Re-run a serially-produced seed in parallel and require identical per-target verdicts."""
    ref = os.path.join(OUT, f"grid_s{seed}.json")
    old = json.load(open(ref))["rows"]
    new = run(seed, out_name=f"grid_s{seed}_par.json")
    same = verdicts(old) == verdicts(new)
    print(f"  determinism check vs serial seed {seed}: identical = {same}")
    if not same:
        d = [i for i, (a, b) in enumerate(zip(verdicts(old), verdicts(new))) if a != b]
        print(f"    differing targets: {len(d)} of {len(old)}; first few {d[:5]}")
    return same


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        sys.exit(0 if validate_against(int(sys.argv[2])) else 1)
    for s in [int(x) for x in sys.argv[1:]]:
        run(s)
