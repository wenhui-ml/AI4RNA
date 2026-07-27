#!/usr/bin/env python3
"""R7 M1/M2: the metric-neutral statistic -- CONDITIONAL CARRY-OVER.

Why this file exists. The paper's headline comparison used r_mix = pKiss AND ProbKnot AND ShapeKnots.
But r_in = pKiss AND ProbKnot *is the panel operator's own success predicate*, so r_mix scores that arm on
its objective plus one term. Comparing arms on r_mix therefore credits the panel operator for satisfying its
own objective, which Sec. 3 already says cannot evidence generalization. The only genuinely new information
r_mix carries over r_in is the conditional term

    C(arm) = P( held-out predictor confirms | the arm's own in-loop predicate is satisfied ),

which is metric-neutral: every arm is scored on how far its own successes carry to a predictor that was
never in ITS loop. This script computes C for every arm we have data for, with the denominator each arm's
own successes, and with paired/difficulty-matched versions where the same targets are available.

Which predictor is held out depends on the arm, and getting this right is the whole point:

  arm                      in-loop objective          held out from it
  -----------------------  -------------------------  ---------------------------
  DesiRNA                  minimum free energy        pKiss, ProbKnot, ShapeKnots
  gamer (pk_decompose)     pKiss                      ProbKnot, ShapeKnots
  panel operator           pKiss AND ProbKnot         ShapeKnots
  undirected / blind /     pKiss AND ProbKnot         ShapeKnots
    other panel ablations    (same predicate, different guidance)
  agent-written operators  pKiss AND ProbKnot         ShapeKnots
                             (invent_operator.py line 70)

ShapeKnots is the one predictor never in any loop, so carry-over TO ShapeKnots is the comparison that is
uniformly defined across every arm. Carry-over to ProbKnot is only meaningful for the two arms that do not
optimize it (DesiRNA, gamer) -- that pair is Table 5's 26/43 vs 2/43.

Offline over the saved sweeps; no episodes, no oracle calls.
"""
import json, math, os, glob, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
SEEDS = [0, 1, 2, 3, 4]

# panel-family ablations: all six share the pKiss AND ProbKnot success predicate (pk_decompose_panel.py
# line 142: solved = pk_solved and cur_pb == 0); they differ only in how the search is guided.
PANEL_ARMS = {"panel_auto": "panel operator (full)",
              "panel_randwalk": "- objective (undirected walk)",
              "panel_blindrand": "- objective, - bias (oracle-blind)",
              "panel_norepair": "- repair loop entirely",
              "panel_nodecompose": "- CF decomposition (random init)",
              "panel_nocompose": "- complementarity composition"}
PANEL_CELLS = [f"{t}_s{s}" for t in ("pablate", "pabrep1", "pabrep2") for s in (0, 1, 2)]
GVP_CELLS = [f"gvp_e{e}_s{s}" for e in (0, 1, 2) for s in SEEDS]


def clopper(k, n, upper=False):
    """Exact binomial interval endpoint by bisection on the binomial tail."""
    if n == 0:
        return float("nan")
    if not upper:
        if k == 0:
            return 0.0
        lo, hi = 0.0, 1.0
        for _ in range(120):
            m = (lo + hi) / 2
            if sum(math.comb(n, i) * m ** i * (1 - m) ** (n - i) for i in range(k, n + 1)) > 0.025:
                hi = m
            else:
                lo = m
        return (lo + hi) / 2
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(120):
        m = (lo + hi) / 2
        if sum(math.comb(n, i) * m ** i * (1 - m) ** (n - i) for i in range(0, k + 1)) > 0.025:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(2 * sum(math.comb(n, k) * 0.5 ** n for k in range(m + 1)), 1.0)


def fmt(k, n, label):
    if n == 0:
        return f"  {label:38} n=0"
    return (f"  {label:38} {k:4d}/{n:<5d} = {k/n:6.3f}   "
            f"95% CP [{clopper(k, n):.3f},{clopper(k, n, True):.3f}]")


def load_cells(cells, arms, need=60):
    """-> {cell: {arm: {target: {'in':bool,'sk':bool,'pkiss':bool,'pb':bool}}}}"""
    out = {}
    for c in cells:
        p = os.path.join(c, "sweep.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if len(d.get("rows") or []) < need:
            continue
        cell = {}
        for a in arms:
            per = {}
            for r in d["rows"]:
                v = r["arms"].get(a)
                if v is None or v.get("error"):
                    continue
                pn = v.get("panel") or {}
                per[r["id"]] = {"in": bool(v.get("solved")), "sk": bool(pn.get("shapeknots")),
                                "pkiss": bool(pn.get("pkiss")), "pb": bool(pn.get("probknot"))}
            if per:
                cell[a] = per
        if cell:
            out[c] = cell
    return out


def main():
    print("=" * 100)
    print("CONDITIONAL CARRY-OVER  C = P(held-out predictor | the arm's own in-loop success)")
    print("=" * 100)

    # ---------------------------------------------------------------- (1) gamer vs panel operator
    print("\n(1) gamer vs panel operator -- 15 (seed, execution) cells x 60 targets, both arms same worker call")
    g = load_cells(GVP_CELLS, ["autonomous", "panel_auto"])
    print(f"    cells loaded: {len(g)}")
    gin = gsk = pin = psk = 0
    for c, cell in g.items():
        for t, v in cell["autonomous"].items():
            # gamer's in-loop predicate is pKiss alone
            if v["in"]:
                gin += 1
                gsk += v["sk"]
        for t, v in cell["panel_auto"].items():
            if v["in"]:          # already pKiss AND ProbKnot
                pin += 1
                psk += v["sk"]
    print("\n  carry-over to ShapeKnots (held out from BOTH), each on its own successes:")
    print(fmt(gsk, gin, "gamer          (in-loop: pKiss)"))
    print(fmt(psk, pin, "panel operator (in-loop: pKiss+ProbKnot)"))

    # difficulty-matched: restrict to targets where BOTH arms succeeded under their own predicate
    n = mg = mp = b = c_ = 0
    for cl, cell in g.items():
        A, B = cell["autonomous"], cell["panel_auto"]
        for t in set(A) & set(B):
            if A[t]["in"] and B[t]["in"]:
                n += 1
                mg += A[t]["sk"]
                mp += B[t]["sk"]
                if B[t]["sk"] and not A[t]["sk"]:
                    b += 1
                if A[t]["sk"] and not B[t]["sk"]:
                    c_ += 1
    print(f"\n  difficulty-matched (targets where BOTH arms satisfy their own predicate), n={n}:")
    print(fmt(mg, n, "gamer"))
    print(fmt(mp, n, "panel operator"))
    print(f"  paired McNemar b={b} (panel only), c={c_} (gamer only), exact p={mcnemar(b, c_):.4g}; "
          f"risk difference {(b - c_) / n:+.4f}" if n else "")

    # ---------------------------------------------------------------- (2) panel-family ablations
    print("\n(2) panel-family arms -- 9 cells x 60; identical success predicate, different guidance")
    pa = load_cells(PANEL_CELLS, list(PANEL_ARMS))
    print(f"    cells loaded: {len(pa)}")
    print("\n  carry-over to ShapeKnots (the only predictor held out from all six):")
    tot = {}
    for a, lab in PANEL_ARMS.items():
        ni = sk = 0
        for cl, cell in pa.items():
            for t, v in cell.get(a, {}).items():
                if v["in"]:
                    ni += 1
                    sk += v["sk"]
        tot[a] = (sk, ni)
        print(fmt(sk, ni, lab))
    # paired: full operator vs undirected, on targets where both succeed
    for other in ("panel_randwalk", "panel_blindrand"):
        n2 = b2 = c2 = 0
        for cl, cell in pa.items():
            A, B = cell.get("panel_auto", {}), cell.get(other, {})
            for t in set(A) & set(B):
                if A[t]["in"] and B[t]["in"]:
                    n2 += 1
                    if A[t]["sk"] and not B[t]["sk"]:
                        b2 += 1
                    if B[t]["sk"] and not A[t]["sk"]:
                        c2 += 1
        if n2:
            print(f"  paired full vs {PANEL_ARMS[other]}: n={n2}, b={b2} (full only), c={c2} (other only), "
                  f"p={mcnemar(b2, c2):.4g}, diff {(b2 - c2)/n2:+.4f}")

    # ---------------------------------------------------------------- (3) DesiRNA vs gamer
    print("\n(3) DesiRNA vs gamer -- the two arms that do NOT optimize ProbKnot (single run, same 43 targets)")
    files = {"DesiRNA (in-loop: MFE)": "desirna_seqs/panel_always_sota.json",
             "gamer   (in-loop: pKiss)": "generous/panel_autonomous.json"}
    sets = {}
    for lab, p in files.items():
        if not os.path.exists(p):
            print(f"    MISSING {p}")
            continue
        rows = json.load(open(p))["rows"]
        sets[lab] = {r["id"]: r for r in rows}
        pk = [r for r in rows if r.get("pkiss")]
        print(f"\n  {lab}: {len(pk)} pKiss-confirmed designs")
        print(fmt(sum(bool(r.get("probknot")) for r in pk), len(pk), "  -> ProbKnot (held out from both)"))
        print(fmt(sum(bool(r.get("shapeknots")) for r in pk), len(pk), "  -> ShapeKnots (held out from both)"))
    if len(sets) == 2:
        (la, A), (lb, B) = sets.items()
        common = sorted(set(A) & set(B))
        for pred in ("probknot", "shapeknots"):
            b3 = sum(1 for t in common if A[t].get(pred) and not B[t].get(pred))
            c3 = sum(1 for t in common if B[t].get(pred) and not A[t].get(pred))
            print(f"  paired on the same {len(common)} targets, {pred}: b={b3} (DesiRNA only), "
                  f"c={c3} (gamer only), exact p={mcnemar(b3, c3):.4g}, diff {(b3 - c3)/len(common):+.4f}")

    # ---------------------------------------------------------------- (4) agent-written operators
    print("\n(4) agent-written operators -- in-loop predicate is pKiss AND ProbKnot (invent_operator.py:70),")
    print("    so ShapeKnots is their only held-out predictor. Single run each, n=60 pool.")
    agg_k = agg_n = 0
    for p in sorted(glob.glob("invent_operator/*/result.json")) + ["invent_operator/result.json"]:
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rows = d["heldout"]
        err = sum(1 for r in rows if r.get("error"))
        rin = [r for r in rows if not r.get("error") and r.get("pkiss") and r.get("probknot")]
        k = sum(bool(r.get("shapeknots")) for r in rin)
        agg_k += k
        agg_n += len(rin)
        print(fmt(k, len(rin), f"{d['model'][:30]} (err {err}/60)"))
    print(fmt(agg_k, agg_n, "ALL agent operators pooled"))

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 100)
    print("SUMMARY -- carry-over to ShapeKnots, the predictor never in any loop")
    print("=" * 100)
    rows = [("DesiRNA (MFE objective)", None, None),
            ("gamer (pKiss)", gsk, gin),
            ("panel operator (pKiss+ProbKnot)", psk, pin),
            ("undirected walk (same predicate)", *tot.get("panel_randwalk", (0, 0))),
            ("oracle-blind walk (same predicate)", *tot.get("panel_blindrand", (0, 0))),
            ("agent-written (pKiss+ProbKnot)", agg_k, agg_n)]
    for lab, k, nn in rows:
        if k is None:
            continue
        print(fmt(k, nn, lab))
    print("\nRead: making the success predicate a two-oracle conjunction raises the IN-OBJECTIVE rate enormously")
    print("but does not raise the rate at which those successes carry to the un-optimized predictor. That is")
    print("the metric-neutral reading of Delta_gamer, and it is a null.")


if __name__ == "__main__":
    main()
