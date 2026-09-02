"""M68 arm C: the control the undesigned-control arm actually needs.

M63's inverse folder stops at the first sequence whose MFE equals the target, so it samples
near-uniformly from correctly-folding sequences and is not a stability optimiser. The claim it
licenses is therefore narrower than the one the manuscript makes. This runs a designer that
keeps optimising after the structure constraint is met -- ViennaRNA's inverse_pf_fold, which
maximises the ensemble probability of the target -- on the SAME targets, with the SAME
admission rule and the SAME summariser.  Offline, CPU, $0.
"""
import json, os, sys, time
import numpy as np
import RNA
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m63_undesigned_placement as M63

SEED = 20260902
rng = np.random.default_rng(SEED)

des = M63.designed_rows()
dv = [(r[0], r[1]) + r[2:7] for r in des]
V = M63.summarise("OK7a designed", dv, rng)
print(f"designed corpus gap: {V['gap']:+.4f}  {V['ci']}  (n={V['n']})")

# the same target windows M63 used
seen = {}
for (pz, _g, _a, _b, _c, _gc, _np, meth, db, seq, win) in des:
    if pz in seen:
        continue
    ss, se = win
    lo, hi = ss - 1, se - 1
    Q = {a: b for a, b in M63.pairs_from_db(db).items() if lo <= a <= hi and lo <= b <= hi}
    if len(Q) < 8:
        continue
    seen[pz] = "".join("(" if (k in Q and Q[k] > k) else (")" if k in Q else ".")
                       for k in range(lo, hi + 1))
print(f"targets: {len(seen)}")

def run(fn, label, require_mfe):
    out = []; n_try = n_hit = 0
    t_all = time.time()
    for pz, sub in seen.items():
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
            P2 = M63.pairs_from_db(sub)
            g = M63.gap_of(s, P2)
            if g:
                gc = (s.count("G") + s.count("C")) / max(len(s), 1)
                out.append((pz, g[0], g[1], g[2], g[3], gc, len(P2) // 2))
    print(f"  {label}: {n_try} attempts, {n_hit} accepted, {len(out)} admissible molecules, "
          f"{time.time()-t_all:.0f}s")
    return out

OUT = {"seed": SEED, "designed": {k: V[k] for k in ("n","n_clusters","gap","ci","frac_negative")},
       "targets": len(seen)}
for label, fn, req in [
    ("inv_pf_strict", RNA.inverse_pf_fold, True),    # ensemble-optimised AND folds to target
    ("inv_pf_any",    RNA.inverse_pf_fold, False),   # ensemble-optimised, constraint not enforced
]:
    rows = run(fn, label, req)
    if len(rows) < 5:
        print(f"  {label}: too few molecules to summarise ({len(rows)})")
        OUT[label] = {"n": len(rows), "note": "under-powered"}
        continue
    S = M63.summarise(label, rows, rng)
    M63.show(S)
    OUT[label] = {k: S[k] for k in ("n","n_clusters","gap","median","ci","frac_negative") if k in S}
    OUT[label]["n_molecules"] = len(rows)

json.dump(OUT, open(os.path.join(HERE, "m68c_real_optimiser.json"), "w"), indent=1)
print("\nwrote m68c_real_optimiser.json")
