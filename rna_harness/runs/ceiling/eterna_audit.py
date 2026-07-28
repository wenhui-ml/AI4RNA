#!/usr/bin/env python3
"""Is the project's headline capability number oracle-inflated, the way 43/60 was?

The regularity that motivates this. The crossing operator drives WHATEVER it optimises to 100%: the
pKiss-only arm gets pKiss 100% / ProbKnot 5% / ShapeKnots 3.4%, and the pKiss+ProbKnot arm gets both at
100% / ShapeKnots 9.1% (runs/ceiling/bottleneck.py). Search power is not the binding constraint on that
benchmark -- the predicate is. And the project already knows what that does to a headline: 43/60 under
pKiss collapsed to 1/60 under a three-oracle panel, a 43x inflation.

Eterna100's 64/100 is a SINGLE-oracle number. rna_harness/cognitive/tools.py offers three strictness
levels on the ViennaRNA path:

    'match'  the target is A minimum-free-energy structure          <- the default, and the loosest
    'umfe'   the target is THE UNIQUE MFE (closes degenerate co-optimals)
    'pfe'    P(target) in the Boltzmann ensemble clears a threshold (a robust fold, not a knife edge)

If the archived runs verified under 'match', then some of the 64 are degenerate co-optimal matches or
knife-edge folds, and the honest capability number is lower. This re-adjudicates the STORED sequences --
no search, no new design work, just the same molecules judged by stricter predicates.

An inflation factor here would say the project's capability lever is the verifier, not the searcher, on
both benchmarks rather than only the crossing one.
"""
import json, os, sys, time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.eterna100 import load_eterna100 as L                 # noqa: E402
from rna_harness.cognitive.tools import verify                                    # noqa: E402


def judge(args):
    tid, target, seq = args
    out = {"task": tid, "n": len(target)}
    for lab in ("match", "umfe", "pfe"):
        t0 = time.time()
        try:
            ok, bp, oracle = verify(seq, target, strictness=lab)
            out[lab] = bool(ok)
            out[lab + "_bp"] = bp
        except Exception as e:
            out[lab] = None
            out[lab + "_err"] = repr(e)[:120]
        out[lab + "_s"] = round(time.time() - t0, 2)
    # extra descriptors that explain a failure
    try:
        import RNA
        fc = RNA.fold_compound(seq)
        fc.pf()
        out["p_target"] = float(fc.pr_structure(target))
        out["mfe_e"] = float(RNA.fold(seq)[1])
        out["tgt_e"] = float(RNA.energy_of_struct(seq, target))
    except Exception:
        pass
    return out


def main():
    T = {f"eterna100:{t['id']}:{t['name']}": t["target"] for t in L()}
    rows = json.load(open(os.path.join(HERE, "..", "..",
                                       "_archive_runs/eterna_paired/memon/rows_partial.json")))
    # take, per target, a sequence the archive recorded as solved
    best = {}
    for r in rows:
        if r.get("solved") and r.get("final_sequence") and r["task_id"] in T:
            best.setdefault(r["task_id"], r["final_sequence"])
    print(f"{len(rows)} archived rows; {len(best)} targets with a stored SOLVED sequence "
          f"(of {len(T)} in the benchmark)")
    args = [(t, T[t], s) for t, s in best.items()]
    nw = int(os.environ.get("NW", "24"))
    with mp.get_context("fork").Pool(nw) as pool:
        res = pool.map(judge, args)
    res.sort(key=lambda r: r["n"])
    json.dump(res, open(os.path.join(HERE, "eterna_audit.json"), "w"), indent=1)

    print(f"\n=== re-adjudicating the STORED solved sequences under stricter predicates ===")
    print(f"  {'n':>4} {'match':>6} {'umfe':>6} {'pfe':>6} {'P(target)':>10} "
          f"{'gap kcal':>9}  task")
    for r in res:
        gap = (r.get("tgt_e", 0) - r.get("mfe_e", 0)) if "tgt_e" in r else float("nan")
        print(f"  {r['n']:4d} {str(r.get('match')):>6} {str(r.get('umfe')):>6} "
              f"{str(r.get('pfe')):>6} {r.get('p_target', float('nan')):10.4f} {gap:9.2f}  "
              f"{r['task'].split(':', 2)[-1][:38]}")

    n = len(res)
    print(f"\n=== the audit ===")
    for lab in ("match", "umfe", "pfe"):
        ok = sum(1 for r in res if r.get(lab) is True)
        bad = sum(1 for r in res if r.get(lab) is False)
        err = sum(1 for r in res if r.get(lab) is None)
        print(f"  {lab:>6}: {ok:3d}/{n} pass, {bad:3d} fail" + (f", {err} error" if err else ""))
    m = sum(1 for r in res if r.get("match") is True)
    for lab in ("umfe", "pfe"):
        ok = sum(1 for r in res if r.get(lab) is True)
        print(f"  inflation, match -> {lab}: {m} -> {ok} = {m/max(ok,1):.2f}x")
    print(f"  (for scale: the crossing benchmark's pKiss -> panel inflation was 43x)")

    print(f"\n=== where the strict predicates bite ===")
    deg = [r for r in res if r.get("match") and r.get("umfe") is False]
    knife = [r for r in res if r.get("match") and r.get("pfe") is False]
    print(f"  degenerate co-optimals (match but not unique MFE): {len(deg)}/{m}")
    if deg:
        print("    " + ", ".join(f"{r['n']}nt" for r in deg[:14]))
    print(f"  knife-edge folds (match but P(target) below the pfe threshold): {len(knife)}/{m}")
    if knife:
        import numpy as np
        ps = [r.get("p_target", float("nan")) for r in knife]
        print(f"    P(target) among them: median {np.nanmedian(ps):.4f}, max {np.nanmax(ps):.4f}")
    import numpy as np
    kept = [r.get("p_target", float("nan")) for r in res if r.get("pfe")]
    if kept:
        print(f"    P(target) among those that pass pfe: median {np.nanmedian(kept):.4f}")

    print(f"\n=== does strictness bite harder on long targets? ===")
    for lo, hi in ((0, 100), (100, 200), (200, 400), (400, 10 ** 9)):
        g = [r for r in res if lo <= r["n"] < hi]
        if not g:
            continue
        mm = sum(1 for r in g if r.get("match") is True)
        print(f"  {lo:3d}-{hi if hi < 10**9 else 'inf'}nt  n={len(g):3d}  match {mm:3d}  "
              f"umfe {sum(1 for r in g if r.get('umfe') is True):3d}  "
              f"pfe {sum(1 for r in g if r.get('pfe') is True):3d}")


if __name__ == "__main__":
    main()
