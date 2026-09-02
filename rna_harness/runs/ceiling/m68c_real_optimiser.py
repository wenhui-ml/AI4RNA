#!/usr/bin/env python3
"""M68 arm C: the control the undesigned-control arm actually needs.

M63's inverse folder stops at the first sequence whose MFE equals the target, so it samples
near-uniformly from correctly-folding sequences and is not a stability optimiser. The claim it
licenses is therefore narrower than the one the manuscript makes. This runs a designer that keeps
optimising after the structure constraint is met -- ViennaRNA's inverse_pf_fold, the library form of
RNAinverse -Fp, which minimises the ensemble free energy difference and has no "constraint met"
stopping rule at all -- on the SAME targets, with the SAME admission rule and the SAME summariser.

DEFECT CAUGHT BEFORE THE FIRST RUN LANDED (recorded in m68_review_gaps_declaration_erratum.json):
the first implementation built its targets by filtering pairs_from_db's multi-family pairing map to
the window and re-emitting it as a single-family dot-bracket. That silently reassigns every crossing
pair by stack order: across the 20 targets only 30.3% of the intended pairs survived, and one target
kept none of them. It is the same flattening the M63 erratum records as F4, made a third time. This
version calls M63.nested_core, exactly as M63's own C' control does, so crossings are DROPPED rather
than reassigned; the designed arm is scored on the same nested-core pairs; and the harness is
verified by reproducing M63's published designed_core figures before the optimiser is run at all.

COST: $0, offline, CPU only.
"""
from __future__ import annotations
import json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import RNA

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m63_undesigned_placement as M63

SEED = 20260902
MINPAIRS = 8          # M63's own nested-core admission rule: len(Q)//2 >= 8


def build_targets(des):
    """M63.core_arms' own target set and designed arm, rebuilt here so the two can be compared."""
    targets, dsg = {}, []
    for (pz, _g, _a, _b, _c, _gc, _np, meth, db, seq, win) in des:
        Q, sub, lo = M63.nested_core(db, win)
        if len(Q) // 2 < MINPAIRS:
            continue
        targets.setdefault(pz, (sub, lo))
        s = seq[lo:lo + len(sub)]
        P = M63.pairs_from_db(sub)
        g = M63.gap_of(s, P)
        if g:
            gc = (s.count("G") + s.count("C")) / max(len(s), 1)
            dsg.append((pz, g[0], g[1], g[2], g[3], gc, len(P) // 2))
    return targets, dsg


def run(targets, fn, label, require_mfe, rng):
    out = []
    n_try = n_hit = 0
    t_all = time.time()
    for pz, (sub, _lo) in targets.items():
        P = M63.pairs_from_db(sub)
        t0 = time.time()
        for _ in range(M63.RESTARTS):
            if time.time() - t0 > M63.BUDGET_S:
                break
            start = "".join("ACGU"[int(x)] for x in rng.integers(0, 4, len(sub)))
            n_try += 1
            try:
                s, _d = fn(start, sub)
            except Exception:
                continue
            s = s.upper().replace("T", "U")
            if require_mfe:
                mfe, _e = RNA.fold(s)
                if mfe != sub:
                    continue
            n_hit += 1
            g = M63.gap_of(s, P)
            if g:
                gc = (s.count("G") + s.count("C")) / max(len(s), 1)
                out.append((pz, g[0], g[1], g[2], g[3], gc, len(P) // 2))
    print(f"  {label}: {n_try} attempts, {n_hit} accepted, {len(out)} admissible molecules, "
          f"{time.time()-t_all:.0f}s", flush=True)
    return out, n_try, n_hit


def main():
    rng = np.random.default_rng(SEED)
    des = M63.designed_rows()
    targets, dsg = build_targets(des)
    print(f"designed rows {len(des)}   nested-core targets {len(targets)}   "
          f"designed molecules on those pairs {len(dsg)}", flush=True)

    # every target string must reparse to the map it was built from, or the flattening bug is back
    for pz, (sub, _lo) in targets.items():
        assert sub.count("(") == sub.count(")"), pz
    D = M63.summarise("designed, nested core", dsg, rng); M63.show(D)

    # the harness control: this must reproduce M63's own published designed_core
    pub = json.load(open(os.path.join(HERE, "m63_undesigned_placement.json"))) \
        .get("Cprime_repaired_control", {}).get("designed_core", {})
    ok = (pub.get("n") == D["n"] and abs((pub.get("gap") or 0) - D["gap"]) < 1e-9)
    print(f"  harness control: M63 published designed_core n={pub.get('n')} gap={pub.get('gap')}"
          f"  ->  reproduced: {ok}", flush=True)

    OUT = {"seed": SEED, "targets": len(targets),
           "designed_core": {k: D[k] for k in ("n", "n_clusters", "gap", "ci", "frac_negative")},
           "reproduces_m63_designed_core": bool(ok),
           "m63_published_designed_core": pub,
           "target_construction": "M63.nested_core (crossings dropped, not reassigned)"}

    for label, fn, req in [
        ("inv_pf_strict", RNA.inverse_pf_fold, True),    # ensemble-optimised AND folds to target
        ("inv_pf_any",    RNA.inverse_pf_fold, False),   # ensemble-optimised, constraint not enforced
    ]:
        rows, n_try, n_hit = run(targets, fn, label, req, rng)
        if len(rows) < 5:
            print(f"  {label}: too few molecules to summarise ({len(rows)})", flush=True)
            OUT[label] = {"n": len(rows), "attempts": n_try, "accepted": n_hit,
                          "note": "under-powered"}
            continue
        S = M63.summarise(label, rows, rng); M63.show(S)
        OUT[label] = {k: S[k] for k in ("n", "n_clusters", "gap", "median", "ci", "frac_negative")
                      if k in S}
        OUT[label].update({"attempts": n_try, "accepted": n_hit, "n_molecules": len(rows)})
        if D.get("gap") and abs(D["gap"]) > 1e-9 and S.get("gap") is not None:
            OUT[label]["share_reproduced"] = round(S["gap"] / D["gap"], 4)
            print(f"  {label} reproduces {100*S['gap']/D['gap']:.1f}% of the designed gap",
                  flush=True)

    st = OUT.get("inv_pf_strict", {})
    OUT["S7"] = (bool(st.get("share_reproduced") is not None
                      and st["share_reproduced"] < 1.0
                      and (st["ci"][0] <= 0 <= st["ci"][1] or abs(st["gap"]) < abs(D["gap"])))
                 if "share_reproduced" in st else None)
    print(f"\nS7 (an optimiser that continues past the structure constraint still does not "
          f"reproduce the designed corpus's placement gap): {OUT['S7']}", flush=True)
    json.dump(OUT, open(os.path.join(HERE, "m68c_real_optimiser.json"), "w"), indent=1)
    print("\nwrote m68c_real_optimiser.json", flush=True)


if __name__ == "__main__":
    main()
