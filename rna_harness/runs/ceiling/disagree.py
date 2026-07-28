#!/usr/bin/env python3
"""Can ONE model's ensemble uncertainty predict when THREE models will disagree?

Everything else on this branch has been better bookkeeping. This is the one hypothesis that bets on an
unmeasured physical relationship, so it can produce a tool rather than a corrected number -- or die.

The observation. Of 530 sequences this operator drove to pKiss AND ProbKnot agreement, ShapeKnots agrees
on 48. The three oracles are different parameterisations and algorithms over the SAME energy landscape,
so their disagreement is not noise; it is a property of the landscape at that sequence.

The hypothesis. Where the target's basin is deep and dominant, every approximation finds it; where it is
shallow or degenerate, they diverge. "How dominant is this basin" is exactly what a SINGLE model's
ensemble spread measures -- and that is computable from ViennaRNA alone, without running the other two.

    ShapeKnots agreement  <-  predicted by  ->  ViennaRNA ensemble uncertainty on the same sequence

If it holds, an expensive three-oracle conjunction collapses into a cheap single-model objective, and it
explains mechanistically why inverse_pf_fold's 13.59x ensemble robustness matters: robustness would BE
the single-model proxy for cross-model agreement, not a separate quality axis.

Measured on the 530 in-loop solves, 48 positive. Predictors, all from ViennaRNA on the sequence alone:

    positional entropy   mean over positions of the base-pairing-probability entropy -- the direct
                         measure of how undecided the ensemble is
    ensemble diversity   RNA's mean base-pair distance within the ensemble
    P(skeleton)          Boltzmann probability of the target's NESTED skeleton. The full target has
                         crossing pairs that ViennaRNA cannot score, so the skeleton is the scorable part
    skeleton defect      expected number of positions differing from the skeleton, normalised
    MFE gap              E(skeleton) - E(MFE), the single-structure version

The last is the control: it is what the operator already optimises indirectly, so if only the MFE gap
predicts, nothing new has been found.

Kill condition stated first: AUC ~ 0.5 on every ensemble predictor means cross-model disagreement is not
visible from inside one model, the conjunction must be paid for in full, and this idea joins the five
already refuted on this branch.
"""
import json, math, os, sys
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.abspath(os.path.join(HERE, "..", "cc_ascent"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))


def skeleton(db):
    return "".join(c if c in "()." else "." for c in db)


def feats(args):
    """ViennaRNA ensemble descriptors for one sequence, plus the target's nested skeleton."""
    tid, seq, db = args
    import RNA
    sk = skeleton(db)
    try:
        fc = RNA.fold_compound(seq)
        mfe_db, mfe_e = fc.mfe()
        fc.exp_params_rescale(mfe_e)
        pf_db, pf_e = fc.pf()
        bpp = fc.bpp()                      # 1-indexed upper triangle
        n = len(seq)
        # positional entropy: for each i, distribution over partners plus "unpaired"
        H = []
        for i in range(1, n + 1):
            ps = []
            for j in range(1, n + 1):
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                p = bpp[a][b]
                if p > 1e-12:
                    ps.append(p)
            tot = sum(ps)
            ps.append(max(0.0, 1.0 - tot))
            H.append(-sum(p * math.log(p) for p in ps if p > 1e-12))
        ent = sum(H) / n
        div = float(fc.mean_bp_distance())
        p_sk = float(fc.pr_structure(sk))
        e_sk = float(RNA.energy_of_struct(seq, sk))
        # skeleton defect: expected mismatched positions against the skeleton, normalised by n
        pt = RNA.ptable(sk)
        d = 0.0
        for i in range(1, n + 1):
            j = pt[i]
            if j > i:
                a, b = i, j
                d += 2 * (1 - bpp[a][b])
            elif j == 0:
                paired = 0.0
                for k in range(1, n + 1):
                    if k == i:
                        continue
                    a, b = (i, k) if i < k else (k, i)
                    paired += bpp[a][b]
                d += paired
        return {"id": tid, "n": n, "ent": ent, "div": div / n, "p_sk": p_sk,
                "defect": d / n, "gap": e_sk - mfe_e, "gap_n": (e_sk - mfe_e) / n,
                "ens_e": pf_e, "mfe_e": mfe_e, "de": (mfe_e - pf_e)}
    except Exception as e:
        return {"id": tid, "err": repr(e)[:120]}


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum(1 if a > b else .5 if a == b else 0 for a in pos for b in neg) / (len(pos) * len(neg))


def main():
    rows = []
    for e in range(3):
        for s in range(5):
            f = os.path.join(CC, f"gvp_e{e}_s{s}", "sweep.json")
            if not os.path.exists(f):
                continue
            for r in json.load(open(f))["rows"]:
                v = r["arms"].get("panel_auto") or {}
                p = v.get("panel") or {}
                if v.get("solved") and v.get("seq") and p.get("shapeknots") is not None:
                    rows.append({"id": r["id"], "seq": v["seq"], "db": r["target"],
                                 "sk": bool(p["shapeknots"])})
    # one row per distinct sequence: the same sequence found twice carries no extra information
    seen, uniq = set(), []
    for r in rows:
        if r["seq"] in seen:
            continue
        seen.add(r["seq"]); uniq.append(r)
    print(f"{len(rows)} in-loop solves -> {len(uniq)} distinct sequences, "
          f"{sum(1 for r in uniq if r['sk'])} of them ShapeKnots-agreeing")

    nw = int(os.environ.get("NW", "24"))
    with mp.get_context("fork").Pool(nw) as pool:
        F = pool.map(feats, [(r["id"], r["seq"], r["db"]) for r in uniq])
    bad = [f for f in F if "err" in f]
    if bad:
        print(f"  {len(bad)} feature failures, e.g. {bad[0]['err']}")
    ok = [(r, f) for r, f in zip(uniq, F) if "err" not in f]
    P = [f for r, f in ok if r["sk"]]
    N = [f for r, f in ok if not r["sk"]]
    print(f"  scored {len(ok)}: {len(P)} agree, {len(N)} disagree\n")

    import numpy as np
    from scipy.stats import mannwhitneyu
    print("=== does ViennaRNA's own ensemble see the disagreement? ===")
    print("    AUC is oriented so >0.5 means the predictor is HIGHER on ShapeKnots-AGREEING sequences")
    print(f"  {'predictor':>22} {'sign':>5} {'AUC':>6} {'agree med':>11} {'disagree med':>13} "
          f"{'MWU p':>10}")
    cands = [("positional entropy", "ent", -1), ("ensemble diversity / n", "div", -1),
             ("P(skeleton)", "p_sk", +1), ("skeleton defect", "defect", -1),
             ("MFE-ensemble gap (dE)", "de", -1),
             ("--- control ---", None, 0),
             ("skeleton MFE gap", "gap", -1), ("skeleton MFE gap / n", "gap_n", -1),
             ("length n", "n", -1)]
    best = None
    for lab, k, sgn in cands:
        if k is None:
            print(f"  {lab:>22}")
            continue
        a = [sgn * f[k] for f in P]; b = [sgn * f[k] for f in N]
        A = auc(a, b)
        try:
            p = mannwhitneyu(a, b).pvalue
        except Exception:
            p = float("nan")
        print(f"  {lab:>22} {sgn:+5d} {A:6.3f} {np.median([f[k] for f in P]):11.4f} "
              f"{np.median([f[k] for f in N]):13.4f} {p:10.3g}")
        if best is None or abs(A - 0.5) > abs(best[1] - 0.5):
            best = (lab, A, k, sgn)

    print("\n=== is it independent of length, or just length again? ===")
    ln = np.array([f["n"] for f in P + N], float)
    ter = [np.quantile(ln, 1 / 3), np.quantile(ln, 2 / 3)]
    lab, A, k, sgn = best
    print(f"  best predictor: {lab} (AUC {A:.3f}); within length terciles:")
    for tl, sel in ((f"n <= {ter[0]:.0f}", lambda f: f["n"] <= ter[0]),
                    (f"{ter[0]:.0f} < n <= {ter[1]:.0f}", lambda f: ter[0] < f["n"] <= ter[1]),
                    (f"n > {ter[1]:.0f}", lambda f: f["n"] > ter[1])):
        p_ = [sgn * f[k] for f in P if sel(f)]; n_ = [sgn * f[k] for f in N if sel(f)]
        if not p_ or not n_:
            print(f"    {tl:20} degenerate ({len(p_)} agree / {len(n_)} disagree)")
            continue
        print(f"    {tl:20} n={len(p_)+len(n_):4d}  AUC {auc(p_, n_):.3f}  "
              f"({len(p_)} agree / {len(n_)} disagree)")

    print("\n=== verdict ===")
    if abs(A - 0.5) < 0.10:
        print(f"  Best AUC {A:.3f}. Cross-model disagreement is NOT visible from inside one model.")
        print("  The conjunction has to be paid for in full, and this joins the five ideas already")
        print("  refuted on this branch. The ShapeKnots gate stays the only way to get it.")
    else:
        print(f"  {lab} separates agreement from disagreement at AUC {A:.3f}. If that survives the")
        print("  length check above, a single-model ensemble quantity is a usable SURROGATE for the")
        print("  three-oracle conjunction: it can enter the loop at ViennaRNA prices instead of")
        print("  ShapeKnots prices, and it explains why inverse_pf_fold's ensemble objective matters.")
        print("  Next step would be to optimise it directly and check the panel rate, NOT to believe it.")

    json.dump([{**f, "sk": r["sk"]} for r, f in ok],
              open(os.path.join(HERE, "disagree.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
