#!/usr/bin/env python3
"""Are our Eterna designs thermodynamically fragile, or is that just what long RNA looks like?

eterna_audit.py re-judged the 64 archived Eterna solutions under stricter predicates:

    match  64/64      the default, and the loosest: the target is A minimum-free-energy structure
    umfe   55/64      the target is THE UNIQUE MFE -- 9 of our solutions are degenerate co-optimals
    pfe     3/64      P(target) in the Boltzmann ensemble >= 0.90

The umfe number is real and length-robust, so the honest headline is 55/100 rather than 64/100. The pfe
number is NOT usable as stated: the threshold is 0.90, which essentially nothing above ~50 nt can reach,
and the length breakdown confirms it (3/32 below 100 nt, 0/31 above). So "3/64" says something about the
threshold, not about our designs.

What IS informative is the DISTRIBUTION: median P(target) = 0.055 across our solutions. Calling that
fragile requires a reference, and the right reference is not another designer, it is EVOLUTION. If natural
RNAs carry their own native structure at a similar Boltzmann probability, then 0.055 is physics and there
is nothing to fix. If they are an order of magnitude higher, we have a real quality axis that the
MFE-match criterion cannot see.

The reference I first reached for -- native sequences from RNAInvBench -- does not exist: that dataset is
an inverse-folding set, so every sequence is masked to 'N'. The better comparison is available for free and
is more actionable anyway, because ViennaRNA ships an optimiser for each objective:

  ours       the archived Eterna solutions, P(target)
  inv_mfe    RNA.inverse_fold    -- the vanilla MFE-matching designer
  inv_pf     RNA.inverse_pf_fold -- optimises the ENSEMBLE objective directly, i.e. maximises P(target)
  random     random compatible sequences, P(target) -- the no-optimisation floor

That makes this the same experiment as the ShapeKnots gate, one benchmark over. On the crossing set the
operator satisfied whatever predicate it optimised (pKiss 100%, pKiss+ProbKnot 100%) and a predicate it
never consulted rejected 90.9%. Here the predicate we optimise is MFE match and the one we never consult
is ensemble robustness. If inv_pf reaches much higher P(target) while still MFE-matching, the fragility is
an objective-specification deficit and the fix is a different function call, not a better search.

Nested only, because pr_structure cannot score a crossing structure.
"""
import gzip, json, os, pickle, random, sys, warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
SRC = "/data/wenhui/RNAInvBench/data/inverse_rna_folding_train_dotbracket.pkl.gz"
WC = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]


def pairmap(db):
    st, p = [], {}
    for i, c in enumerate(db):
        if c == '(':
            st.append(i)
        elif c == ')':
            j = st.pop(); p[j] = i; p[i] = j
    return p


def score(seq, db):
    import RNA
    fc = RNA.fold_compound(seq)
    fc.pf()
    p = float(fc.pr_structure(db))
    e_t = float(RNA.energy_of_struct(seq, db))
    mfe_db, e_m = RNA.fold(seq)
    return p, e_t - e_m, mfe_db == db


def p_struct(args):
    """P(structure) under the Boltzmann ensemble, the MFE gap, and whether it still MFE-matches.

    For the two designer groups the sequence is produced here rather than passed in, so the design cost is
    measured on the same footing as the score.
    """
    lab, ident, seq, db = args
    import RNA, time
    try:
        t0 = time.time()
        if lab in ("inv_mfe", "inv_pf"):
            RNA.init_rand(abs(hash(ident)) % (2 ** 30))          # reproducible: inverse_* use RNA's own RNG
            start = seq                                           # a random compatible sequence
            seq = (RNA.inverse_fold(start, db) if lab == "inv_mfe"
                   else RNA.inverse_pf_fold(start, db))[0]
        design_s = time.time() - t0
        p, gap, matches = score(seq, db)
        return {"grp": lab, "id": ident, "n": len(db), "p": p, "gap": gap, "match": bool(matches),
                "design_s": round(design_s, 2), "seq": seq}
    except Exception as e:
        return {"grp": lab, "id": ident, "n": len(db), "p": None, "err": repr(e)[:100]}


def nested_db(pairs, n):
    """Dot-bracket from a pair list, or None if any two pairs cross (we need nested only)."""
    ps = sorted({(min(i, j), max(i, j)) for i, j, *_ in pairs if i != j})
    for a in range(len(ps)):
        i, j = ps[a]
        for b in range(a + 1, len(ps)):
            k, l = ps[b]
            if i < k < j < l:
                return None
    db = ["."] * n
    for i, j in ps:
        if i >= n or j >= n:
            return None
        db[i], db[j] = "(", ")"
    return "".join(db)


def main():
    jobs = []

    # --- ours: the archived Eterna solutions
    from rna_harness.benchmarks.eterna100 import load_eterna100 as L
    T = {f"eterna100:{t['id']}:{t['name']}": t["target"] for t in L()}
    rows = json.load(open(os.path.join(HERE, "..", "..",
                                       "_archive_runs/eterna_paired/memon/rows_partial.json")))
    ours = {}
    for r in rows:
        if r.get("solved") and r.get("final_sequence") and r["task_id"] in T:
            ours.setdefault(r["task_id"], r["final_sequence"])
    for t, s in ours.items():
        if "[" not in T[t] and "{" not in T[t]:
            jobs.append(("ours", t, s, T[t]))

    # --- random: the no-optimisation floor, on the same targets
    rng = random.Random(0)
    for t, s in ours.items():
        db = T[t]
        if "[" in db or "{" in db:
            continue
        p = pairmap(db)
        for k in range(3):
            q = ['A'] * len(db)
            for i in range(len(db)):
                if i not in p:
                    q[i] = rng.choice('ACGU')
            for i, j in p.items():
                if i < j:
                    a, b = rng.choice(WC); q[i], q[j] = a, b
            jobs.append(("random", f"{t}#{k}", "".join(q), db))

    # --- the two ViennaRNA designers, on exactly the targets we solved, from a random compatible start
    for t, s in ours.items():
        db = T[t]
        if "[" in db or "{" in db:
            continue
        p = pairmap(db)
        q = ['A'] * len(db)
        for i in range(len(db)):
            if i not in p:
                q[i] = rng.choice('ACGU')
        for i, j in p.items():
            if i < j:
                a, b = rng.choice(WC); q[i], q[j] = a, b
        start = "".join(q)
        jobs.append(("inv_mfe", t, start, db))
        jobs.append(("inv_pf", t, start, db))

    GRPS = ("ours", "inv_mfe", "inv_pf", "random")
    print(f"{len(jobs)} sequences to score: "
          + ", ".join(f"{g}={sum(1 for j in jobs if j[0]==g)}" for g in GRPS))
    nw = int(os.environ.get("NW", "24"))
    with mp.get_context("fork").Pool(nw) as pool:
        res = [r for r in pool.map(p_struct, jobs) if r.get("p") is not None]
    json.dump(res, open(os.path.join(HERE, "robustness.json"), "w"), indent=1)

    import numpy as np
    from scipy.stats import mannwhitneyu, wilcoxon
    print("\n=== P(target) under the Boltzmann ensemble ===")
    print(f"  {'group':>8} {'n':>5} {'median P':>10} {'q25':>8} {'q75':>8} {'>=0.5':>7} {'>=0.9':>7} "
          f"{'still MFE-matches':>18} {'design s':>9}")
    G = {}
    for g in GRPS:
        v = [r["p"] for r in res if r["grp"] == g]
        if not v:
            continue
        G[g] = {r["id"]: r["p"] for r in res if r["grp"] == g}
        m = [r.get("match") for r in res if r["grp"] == g and r.get("match") is not None]
        ds = [r.get("design_s", 0.0) for r in res if r["grp"] == g]
        print(f"  {g:>8} {len(v):5d} {np.median(v):10.4f} {np.percentile(v,25):8.4f} "
              f"{np.percentile(v,75):8.4f} {np.mean([x>=0.5 for x in v]):7.2f} "
              f"{np.mean([x>=0.9 for x in v]):7.2f} "
              f"{(f'{np.mean(m):.2f}' if m else '-'):>18} {np.mean(ds):9.2f}")

    print("\n=== paired on the SAME targets: does the ensemble objective buy robustness? ===")
    print(f"  {'comparison':>22} {'n':>4} {'median P a':>11} {'median P b':>11} {'ratio':>7} "
          f"{'b>a':>6} {'Wilcoxon p':>11}")
    for a, b in (("ours", "inv_mfe"), ("ours", "inv_pf"), ("inv_mfe", "inv_pf")):
        if a not in G or b not in G:
            continue
        keys = sorted(set(G[a]) & set(G[b]))
        va = [G[a][k] for k in keys]; vb = [G[b][k] for k in keys]
        wins = sum(1 for x, y in zip(va, vb) if y > x)
        try:
            p = wilcoxon(va, vb).pvalue
        except Exception:
            p = float("nan")
        print(f"  {b+' vs '+a:>22} {len(keys):4d} {np.median(va):11.4f} {np.median(vb):11.4f} "
              f"{np.median(vb)/max(np.median(va),1e-9):7.2f} {wins:3d}/{len(keys):<3d} {p:11.3g}")

    print("\n=== by length ===")
    print(f"  {'band':>12} {'n':>4} " + " ".join(f"{g:>9}" for g in GRPS))
    for lo, hi in ((20, 60), (60, 100), (100, 150), (150, 250), (250, 401)):
        ids = {r["id"] for r in res if r["grp"] == "ours" and lo <= r["n"] < hi}
        if not ids:
            continue
        cells = []
        for g in GRPS:
            v = [r["p"] for r in res if r["grp"] == g and lo <= r["n"] < hi]
            cells.append(f"{np.median(v):9.4f}" if v else f"{'-':>9}")
        print(f"  {f'{lo}-{hi}nt':>12} {len(ids):4d} " + " ".join(cells))

    if "ours" in G and "random" in G:
        ka = sorted(set(G["ours"]))
        print(f"\n  sanity: ours vs the random floor, MWU p = "
              f"{mannwhitneyu([G['ours'][k] for k in ka], list(G['random'].values())).pvalue:.3g}")

    print("\n=== the verdict this decides ===")
    if "inv_pf" in G and "ours" in G:
        keys = sorted(set(G["ours"]) & set(G["inv_pf"]))
        r = np.median([G["inv_pf"][k] for k in keys]) / max(np.median([G["ours"][k] for k in keys]), 1e-9)
        mt = np.mean([x.get("match") for x in res if x["grp"] == "inv_pf" and x.get("match") is not None])
        if r > 1.5 and mt > 0.5:
            print(f"  inv_pf/ours = {r:.2f} with {mt:.0%} still MFE-matching. The fragility is an")
            print("  OBJECTIVE-SPECIFICATION deficit, not a search deficit: ViennaRNA already ships an")
            print("  optimiser for the criterion we never asked for. Same lesson as the ShapeKnots gate,")
            print("  on the main benchmark, and the fix is a different function call.")
        elif r > 1.5:
            print(f"  inv_pf/ours = {r:.2f} but only {mt:.0%} still MFE-match: the ensemble objective")
            print("  trades away the criterion the benchmark scores. A composite objective is needed,")
            print("  not a substitution.")
        else:
            print(f"  inv_pf/ours = {r:.2f}: optimising the ensemble objective directly does NOT lift")
            print("  P(target) much above what we already get, so 0.059 is close to what is attainable")
            print("  at these lengths and robustness is not an improvement axis.")


if __name__ == "__main__":
    main()
