#!/usr/bin/env python3
"""R10 (reviewer W1/W4c): the one test set that is NOT structurally exhausted, run at scale.

The binding limitation of this paper is that Pseudobase++ holds 251 crossing targets in 60 structural
clusters, of which 24 are disjoint from our development. Every magnitude here is therefore unestimable, and
no reanalysis fixes it. But the paper already defines a target set with no such ceiling: the parameterized
H-type family

    T_{n,m} = '('*n + '.'*L + '['*m + '.'*L + ')'*n + '.'*L + ']'*m ,   L = 4,

the RNA analogue of the count language {a^n b^m c^n d^m}. It is non-context-free for every m >= 1, it is the
family the class-level reading is about, and it is unbounded -- so it is the only place in this study where
power is a budget decision rather than a property of a benchmark. A previous run used 16 targets at one seed
and was reported in an appendix as not relied on. That was the wrong call, and this run replaces it.

Design. Full grid n, m in {3..12} = 100 targets, three arms, five nominal seeds, deterministic:

  always_cheap  REMC under the pseudoknot-free oracle -> the CF floor. Prop. 1 says it cannot realize any of
                these, so any nonzero count here would falsify the negative side outright.
  autonomous    the single-oracle gamer (pKiss only) -> the Delta_gamer baseline, measured on a family that
                shares no structural cluster with anything we developed on.
  panel_auto    the hand-built panel operator (pKiss AND ProbKnot).

The operator's procedure and its size-independent budgets were fixed on n = m in {2,3,4} and are FROZEN; every
(n,m) here is strictly larger than that regime in at least one stem, so nothing below is tuned per size. What
this buys, and does not: a single finite-description operator succeeding across a growing ladder it never saw
is uniform-procedure evidence over an unbounded non-CF family, which a memorized table cannot supply and a CF
generator provably cannot. It is still not a proof of a class ascent -- a length-parameterized template could
do the same -- and the family is synthetic, so it says nothing about natural pseudoknot diversity. Those two
limits are the price of the only unbounded axis available.

Search-only: no LLM, no $ cost.
"""
import json, os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from rna_harness.experiments.track_b_pk import build_policies                      # noqa: E402
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,     # noqa: E402
                                    shapeknots_pairs, rnastructure_available)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "family_grid")
os.makedirs(OUT, exist_ok=True)
LO, HI = 3, 12
GRID = [(n, m) for n in range(LO, HI + 1) for m in range(LO, HI + 1)]


def htype(n, m, L=4):
    return "(" * n + "." * L + "[" * m + "." * L + ")" * n + "." * L + "]" * m


def panel(seq, target, rnastr):
    pk = forms_target(pkiss_pairs(seq), target)
    pb = forms_target(probknot_pairs(seq), target) if rnastr else None
    sk = forms_target(shapeknots_pairs(seq), target) if rnastr else None
    return {"pkiss": bool(pk), "probknot": (None if pb is None else bool(pb)),
            "shapeknots": (None if sk is None else bool(sk)),
            "unanimous": (bool(pk) and bool(pb) and bool(sk)) if rnastr else None}


def main(seed=0):
    os.environ.setdefault("PKX_DETERMINISTIC", "1")
    rnastr = rnastructure_available()
    arms = os.environ.get("FAMILY_ARMS", "always_cheap,autonomous,panel_auto").split(",")
    pols = build_policies(arms)
    out = os.path.join(OUT, f"grid_s{seed}.json")
    print(f"family grid: {len(GRID)} targets (n,m in {LO}..{HI}), seed={seed}, "
          f"arms={list(pols)}, rnastructure={rnastr}", flush=True)
    rows = []
    for (n, m) in GRID:
        T = htype(n, m)
        rec = {"n": n, "m": m, "len": len(T), "target": T, "arms": {}}
        for name, pol in pols.items():
            t0 = time.time()
            try:
                r = pol.solve(T, seed=seed)
                s = r.get("seq")
                rec["arms"][name] = {"solved": bool(r.get("solved")), "seq": s,
                                     "panel": panel(s, T, rnastr) if (r.get("solved") and s) else None,
                                     "s": round(time.time() - t0, 1)}
            except Exception as e:
                rec["arms"][name] = {"solved": None, "error": repr(e)[:200],
                                     "s": round(time.time() - t0, 1)}
        rows.append(rec)
        u = {k: ((v.get("panel") or {}).get("unanimous")) for k, v in rec["arms"].items()}
        print(f"  T_{n},{m} len={len(T):3d} unan={u}", flush=True)
        json.dump({"seed": seed, "lo": LO, "hi": HI, "rows": rows}, open(out, "w"), indent=1)
    print(f"\ndone -> {out}", flush=True)


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
