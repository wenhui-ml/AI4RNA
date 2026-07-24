#!/usr/bin/env python3
"""Phase 3: parameterized non-CF crossing family with OOD-length transfer (chomsky_climb class claim).

The critique: solving a FINITE held-out set cannot prove a Chomsky CLASS ascent (any finite language is
regular). The fix: show a SINGLE finite-description operator realizes members of an UNBOUNDED parameterized
crossing family at parameters larger than any it was validated on -- a uniform procedure over the family,
which a memorized table is not and a CF generator provably cannot supply.

Family (canonical H-type, the count-language analog {a^n b^m c^n d^m}), WITH biophysically necessary loops:
    T_{n,m} = '('*n + '.'*L + '['*m + '.'*L + ')'*n + '.'*L + ']'*m   (round stem n crossed by square stem m)
An RNA stem cannot close with zero unpaired nucleotides, so the idealized loop-free string is undesignable by
construction; we insert fixed unpaired linkers (L=4) between segments so each T_{n,m} is a realizable H-type.

Arms per target (each is autonomous / local; NO $ cost):
  - always_cheap : REMC under the CF (pseudoknot-free) oracle -> the CF FLOOR (must be ~0 for any m>=1).
  - panel_auto   : the invented autonomous operator optimizing pKiss AND ProbKnot agreement (no external solver).
  - always_sota  : DesiRNA -> DESIGNABILITY reference (is T_{n,m} realizable at all under the energy model?).

Every solved sequence is re-folded under the full panel (pKiss / ProbKnot / ShapeKnots); panel-unanimous =
all three confirm the exact crossing pair set. Success at LARGE (n,m) the operator never saw = the uniform
procedure claim. If DesiRNA also fails a target, that target is UNDESIGNABLE and is excluded from the
capability denominator (reported separately) -- so operator failure reads as a capability gap, not undesignability.
"""
import json, os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from rna_harness.experiments.track_b_pk import build_policies
from rna_harness.oracles_pk import (forms_target, pkiss_pairs, probknot_pairs,
                                     shapeknots_pairs, rnastructure_available)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "family_ood")
os.makedirs(OUT, exist_ok=True)


def htype(n, m, L=4):
    # canonical H-type WITH unpaired loops (L nt each): '('*n '.'*L '['*m '.'*L ')'*n '.'*L ']'*m
    return "(" * n + "." * L + "[" * m + "." * L + ")" * n + "." * L + "]" * m


# DECISIVE unbounded-transfer protocol (reviewer concern 2, option B):
#   dev/validation set (NOT tested here): small stems n=m in {2,3,4}. The operator's procedure and its
#   size-INDEPENDENT budgets (PKP_*) were fixed on that regime and are FROZEN -- no per-size tuning below.
#   test set (this GRID): strictly LARGER, unseen (n,m), spanning several length regimes and asymmetric
#   combinations, up to ~3-4x the largest dev stem. A single finite-description operator succeeding across a
#   growing ladder it never saw is uniform-procedure evidence over the unbounded non-CF family {T_{n,m}}; a
#   memorized table cannot (targets are novel), and a CF generator provably cannot (Prop.1). We report the
#   success-vs-size curve AND the compute per size, and use DesiRNA as a designability reference so an
#   operator failure at a size where DesiRNA succeeds reads as a search/budget gap, not undesignability.
GRID = ([(n, n, "diag") for n in (5, 6, 7, 8, 9, 10, 12, 14)]
        + [(4, 10, "asym"), (10, 4, "asym"), (5, 12, "asym"), (12, 5, "asym"),
           (3, 14, "asym"), (14, 3, "asym"), (6, 12, "asym"), (12, 6, "asym")])


def panel(seq, target, rnastr):
    pk = forms_target(pkiss_pairs(seq), target)
    pb = forms_target(probknot_pairs(seq), target) if rnastr else None
    sk = forms_target(shapeknots_pairs(seq), target) if rnastr else None
    unan = (bool(pk) and bool(pb) and bool(sk)) if rnastr else None
    return {"pkiss": bool(pk), "probknot": (None if pb is None else bool(pb)),
            "shapeknots": (None if sk is None else bool(sk)), "unanimous": unan}


def main(seed=0):
    os.environ.setdefault("DESIRNA_CMD",
                          "/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py")
    rnastr = rnastructure_available()
    arms = os.environ.get("FAMILY_ARMS", "always_cheap,panel_auto,always_sota").split(",")
    pols = build_policies(arms)
    print(f"family-OOD: {len(GRID)} targets; arms={list(pols)}; rnastructure={rnastr}", flush=True)
    rows = []
    for i, (n, m, regime) in enumerate(GRID):
        T = htype(n, m)
        rec = {"n": n, "m": m, "regime": regime, "len": len(T), "target": T, "arms": {}}
        for name, pol in pols.items():
            t0 = time.time()
            try:
                r = pol.solve(T, seed=seed)
                seq = r.get("seq")
                adj = panel(seq, T, rnastr) if (r.get("solved") and seq) else None
                rec["arms"][name] = {"solved": bool(r.get("solved")), "seq": seq,
                                     "panel": adj, "s": round(time.time() - t0, 1)}
            except Exception as e:  # never let one arm kill the sweep
                rec["arms"][name] = {"solved": None, "error": repr(e)[:200], "s": round(time.time() - t0, 1)}
            a = rec["arms"][name]
            u = (a.get("panel") or {}).get("unanimous")
            print(f"  T_{n},{m} [{regime}] len={len(T):3d} {name:12} solved={a.get('solved')} "
                  f"panel_unanimous={u} ({a.get('s')}s)", flush=True)
        rows.append(rec)
        json.dump({"seed": seed, "grid": GRID, "rows": rows}, open(os.path.join(OUT, "family_ood.json"), "w"), indent=1)
    print("\ndone -> family_ood/family_ood.json", flush=True)


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
