#!/usr/bin/env python3
"""If every available judge is wrong most of the time, what should "confirmed" mean? Calibrate on ground truth.

THE PROBLEM, measured. The shipped criterion is "all three pseudoknot predictors reproduce the target
exactly". The archived native-sequence control (runs/axis1_native_control, 250 Pseudobase targets) shows what
that criterion does to KNOWN-CORRECT answers -- fold each wild-type sequence and ask whether it recovers its
own published structure:

    exact recovery      pKiss 32.4%   ProbKnot 5.6%   ShapeKnots 7.6%
    crossing recall     pKiss 0.85    ProbKnot 0.62   ShapeKnots 0.63

So unanimous-exact rejects most of nature. A design that fails it has not been shown to be wrong; it has been
shown to be unrecognisable to instruments that mostly cannot recognise biology either. Calling that
population "false" was an overclaim of mine, and tightening or loosening the criterion by taste would just be
a different overclaim.

THE FIX IS NOT A NEW ORACLE, IT IS A CALIBRATION SET. Natural structures are ground truth: the wild-type
sequence really does form the published structure. So a confirmation rule can be CALIBRATED -- choose it so
that it recognises a declared fraction of known-correct structures, then report that fraction alongside every
certificate. "This design passes a criterion that recognises 80% of known natural structures" is a statement
a non-expert can act on. "Three heuristics agreed" is not.

AN ABSOLUTE TOLERANCE IS THE WRONG KNOB, and the relative grid below is why this file reports both. bp_dist
here is the symmetric difference of pair sets, so for a target with P pairs, bp_dist == P is exactly as wrong
as predicting NOTHING and bp_dist == 2P is fully disjoint. Pseudobase targets run 24-100 nt with roughly
8-25 pairs, so a fixed d=8 is a mild tolerance on the largest targets and a vacuous one on the smallest --
the same number means "nearly right" and "completely wrong" depending on which target it lands on. The
relative form r = bp_dist / n_pairs(T) is comparable across targets, and only r < 1 is even meaningful.

CRITERION FAMILY, two interpretable knobs:
    accept_o(seq, T; d)  :=  bp_dist(fold_o(seq), T) <= d          per oracle, tolerance d
    CONFIRMED(k, d)      :=  at least k of the 3 oracles accept
(k=3, d=0) is exactly what ships today, so the current rule is a point in the family rather than a baseline
being argued against.

SENSITIVITY WITHOUT SPECIFICITY IS MEANINGLESS, and loosening d until natives pass is exactly how one would
manufacture a flattering number. So the same grid is evaluated on a NULL: a random ACGU sequence of the same
length as each native target. A tolerance at which random sequences also pass is not a confirmation rule, it
is a rubber stamp. The reported quantity is therefore the pair (sensitivity on natives, false-accept rate on
random), and a rule is only usable where the two separate.

THIS SCRIPT DOES THE CALIBRATION HALF ONLY -- it is free, it runs on the archive, and it decides whether the
approach is viable before any compute is spent re-scanning designs. If no (k, d) reaches useful sensitivity,
the idea dies here at zero cost.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
NATIVE = os.path.join(HERE, "agent", "native_control.json")
ORACLES = ("pkiss", "probknot", "shapeknots")
TARGET_SENSITIVITY = 0.80          # declared before looking: the rule must recognise 80% of natives


def _null_rows(rows, seed=0):
    """Fold ONE random sequence per target under all three oracles -> the false-accept null for each (k, d).
    Cached, because it costs real oracle time and nothing about it changes between runs."""
    cache = os.path.join(HERE, "calibrate_null_v2.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    import random
    from concurrent.futures import ProcessPoolExecutor
    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    tgt = {t.task_id: t.target for t in load_pseudobase()}
    rng = random.Random(seed)
    items = [(r["id"], tgt.get(r["id"]), "".join(rng.choice("ACGU") for _ in range(r["n"])))
             for r in rows]
    items = [it for it in items if it[1] and len(it[1]) == len(it[2])]
    with ProcessPoolExecutor(max_workers=16) as ex:
        out = list(ex.map(_null_one, items))
    out = [o for o in out if o]
    json.dump(out, open(cache, "w"))
    return out


def _null_one(item):
    tid, target, seq = item
    from rna_harness.benchmarks.rnainvbench import pk_bp_distance
    from rna_harness.oracles_pk import pkiss_pairs, probknot_pairs, shapeknots_pairs
    import rna_harness.benchmarks.rnainvbench as rib
    fns = {"pkiss": pkiss_pairs, "probknot": probknot_pairs, "shapeknots": shapeknots_pairs}
    rec = {"id": tid, "n": len(target)}
    tp = rib.pk_pairs(target)
    # the CROSSING pairs are the ones that make this a pseudoknot at all; a predictor can be far off on the
    # nested skeleton and still be the right answer to "did it form the knot", so this is a genuinely
    # different quantity from full-structure distance, not a rescaling of it.
    nested_only = rib.pk_pairs(rib._round_only(target))
    cross = tp - nested_only
    for o, fn in fns.items():
        ps = fn(seq)
        if ps is None:
            rec[o] = {"ok": False, "bp_dist": None, "cross_recall": None}
            continue
        rec[o] = {"ok": True, "bp_dist": len(ps ^ tp),
                  "cross_recall": (len(ps & cross) / len(cross)) if cross else None}
    return rec


def _native_sequences():
    """Parse antaRNA constraint triples (>name / native sequence / dot-bracket) and match each to a
    Pseudobase target by (length, pair set) -- robust to bracket-symbol conventions, the same matching the
    archived native control used."""
    import os as _os
    from rna_harness.benchmarks.rnainvbench import load_pseudobase, pk_pairs
    FILES = ["/data/wenhui/RNAInvBench/rna_design_algorithms/antaRNA/Constraints/Pseudobase++",
             "/data/wenhui/RNAInvBench/rna_design_algorithms/antaRNA/Constraints/Pseudobase++_TEST",
             "/data/wenhui/RNAInvBench/rna_design_algorithms/antaRNA/Constraints/Pseudobase++_TRAIN"]
    key = {}
    for t in load_pseudobase():
        key[(len(t.target), frozenset(pk_pairs(t.target)))] = t.task_id
    out = {}
    for path in FILES:
        if not _os.path.exists(path):
            continue
        lines = [ln.strip() for ln in open(path) if ln.strip()]
        for i in range(0, len(lines) - 2):
            if not lines[i].startswith(">"):
                continue
            seq, db = lines[i + 1].upper().replace("T", "U"), lines[i + 2]
            if set(seq) - set("ACGU") or len(seq) != len(db):
                continue
            tid = key.get((len(db), frozenset(pk_pairs(db))))
            if tid and tid not in out:
                out[tid] = seq
    return out


def main():
    rows = json.load(open(NATIVE))["rows"]
    rows = [r for r in rows if all(r.get(o, {}).get("ok") for o in ORACLES)]
    print(f"calibration set: {len(rows)} natural structures with all three oracles run")
    print(f"  ground truth = the wild-type sequence really does form the published structure\n")

    print(f"  per-oracle native recovery at tolerance d (bp_dist <= d):")
    print(f"  {'d':>4} " + "".join(f"{o:>12}" for o in ORACLES))
    DS = [0, 1, 2, 3, 5, 8, 12, 20]
    for d in DS:
        cells = []
        for o in ORACLES:
            acc = sum(1 for r in rows if r[o].get("bp_dist") is not None and r[o]["bp_dist"] <= d)
            cells.append(f"{acc/len(rows):>11.1%}")
        print(f"  {d:>4} " + "".join(cells))

    print(f"\n=== sensitivity of CONFIRMED(k, d) on ground truth ===")
    print(f"  {'d':>4} " + "".join(f"{'k=' + str(k):>10}" for k in (1, 2, 3)))
    grid = {}
    for d in DS:
        cells = []
        for k in (1, 2, 3):
            n = 0
            for r in rows:
                acc = sum(1 for o in ORACLES
                          if r[o].get("bp_dist") is not None and r[o]["bp_dist"] <= d)
                n += (acc >= k)
            sens = n / len(rows)
            grid[(k, d)] = sens
            cells.append(f"{sens:>9.1%}")
        print(f"  {d:>4} " + "".join(cells))

    print(f"\n=== the rule, chosen by a declared criterion rather than by taste ===")
    print(f"  requirement: sensitivity >= {TARGET_SENSITIVITY:.0%} on known natural structures")
    print(f"  among those, take the STRICTEST (largest k, then smallest d)")
    ok = [(k, d, s) for (k, d), s in grid.items() if s >= TARGET_SENSITIVITY]
    shipped = grid[(3, 0)]
    print(f"  currently shipped rule CONFIRMED(k=3, d=0): sensitivity {shipped:.1%}")
    if not ok:
        best = max(grid.items(), key=lambda kv: kv[1])
        print(f"  NO (k, d) reaches {TARGET_SENSITIVITY:.0%}. Best available is "
              f"k={best[0][0]}, d={best[0][1]} at {best[1]:.1%}.")
        print(f"  -> Confirmation by agreement among THESE THREE predictors cannot be calibrated to a")
        print(f"     useful sensitivity at any threshold. That is a statement about the instruments, and it")
        print(f"     means the honest product move is to report the disagreement profile, not a verdict.")
    else:
        ok.sort(key=lambda t: (-t[0], t[1]))
        k, d, s = ok[0]
        print(f"  -> CONFIRMED(k={k}, d={d}) at sensitivity {s:.1%}")
        print(f"     a certificate can then say: 'passes a criterion that recognises {s:.0%} of known"
              f" natural structures'")

    # ---- the same calibration on a target-NORMALISED tolerance -------------------------------------
    from rna_harness.benchmarks.rnainvbench import load_pseudobase, pk_pairs
    tg = {t.task_id: t.target for t in load_pseudobase()}
    npairs = {r["id"]: len(pk_pairs(tg[r["id"]])) for r in rows if r["id"] in tg}
    nullr = _null_rows(rows)
    RS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

    def grid_rel(recs):
        g = {}
        use = [r for r in recs if r["id"] in npairs and npairs[r["id"]] > 0]
        for rr in RS:
            for k in (1, 2, 3):
                n = 0
                for r in use:
                    lim = rr * npairs[r["id"]]
                    acc = sum(1 for o in ORACLES
                              if r.get(o, {}).get("bp_dist") is not None and r[o]["bp_dist"] <= lim)
                    n += (acc >= k)
                g[(k, rr)] = n / max(len(use), 1)
        return g, len(use)

    gs, n_s = grid_rel(rows)
    gn, n_n = grid_rel(nullr)
    print(f"\n=== RELATIVE tolerance r = bp_dist / n_pairs(T): comparable across target sizes ===")
    print(f"  ({n_s} natives, {n_n} randoms; r=1 is as wrong as predicting nothing, so r>=1 is vacuous)")
    print(f"  {'r':>5} " + "".join(f"{'k=' + str(k):>22}" for k in (1, 2, 3)))
    print(f"  {'':>5} " + "".join(f"{'sens':>10}{'  random':>12}" for _ in (1, 2, 3)))
    usable_rel = []
    for rr in RS:
        cells = []
        for k in (1, 2, 3):
            cells.append(f"{gs[(k, rr)]:>9.1%}{gn[(k, rr)]:>12.1%}")
            if gs[(k, rr)] >= TARGET_SENSITIVITY and gn[(k, rr)] <= 0.05 and rr < 1.0:
                usable_rel.append((k, rr, gs[(k, rr)], gn[(k, rr)]))
        print(f"  {rr:>5.2f} " + "".join(cells))
    print(f"\n  usable under the SAME declared rule, restricted to meaningful r < 1: "
          f"{usable_rel if usable_rel else 'NONE'}")
    if not usable_rel:
        print(f"  -> Once the tolerance is made comparable across target sizes, NO rule in this family")
        print(f"     recognises 80% of known structures while rejecting random sequences. The absolute-d")
        print(f"     winner (k=1, d=8) was an artefact of averaging a mild tolerance on big targets with a")
        print(f"     vacuous one on small targets.")

    # ---- option B: a DIFFERENT quantity, not a rescaling -- crossing-pair recall --------------------
    print(f"\n=== a different judged quantity: CROSSING-PAIR RECALL (the pairs that make it a pseudoknot) ===")
    print(f"  native means: " + ", ".join(
        f"{o} {sum(r[o].get('cross_recall') or 0 for r in rows)/len(rows):.2f}" for o in ORACLES))
    TAUS = [1.0, 0.9, 0.8, 0.67, 0.5]

    def grid_cr(recs):
        g = {}
        use = [r for r in recs if any(r.get(o, {}).get("cross_recall") is not None for o in ORACLES)]
        for tau in TAUS:
            for k in (1, 2, 3):
                n = 0
                for r in use:
                    acc = sum(1 for o in ORACLES
                              if (r.get(o, {}).get("cross_recall") or 0.0) >= tau)
                    n += (acc >= k)
                g[(k, tau)] = n / max(len(use), 1)
        return g, len(use)

    cs, n_cs = grid_cr(rows)
    cn, n_cn = grid_cr(nullr)
    print(f"  ({n_cs} natives, {n_cn} randoms)")
    print(f"  {'tau':>5} " + "".join(f"{'k=' + str(k):>22}" for k in (1, 2, 3)))
    print(f"  {'':>5} " + "".join(f"{'sens':>10}{'  random':>12}" for _ in (1, 2, 3)))
    usable_cr = []
    for tau in TAUS:
        cells = []
        for k in (1, 2, 3):
            cells.append(f"{cs[(k, tau)]:>9.1%}{cn[(k, tau)]:>12.1%}")
            if cs[(k, tau)] >= TARGET_SENSITIVITY and cn[(k, tau)] <= 0.05:
                usable_cr.append((k, tau, cs[(k, tau)], cn[(k, tau)]))
        print(f"  {tau:>5.2f} " + "".join(cells))
    print(f"\n  usable under the SAME declared rule: {usable_cr if usable_cr else 'NONE'}")
    if usable_cr:
        k, tau, se, fa = sorted(usable_cr, key=lambda t: (-t[0], -t[1]))[0]
        print(f"  -> crossing-pair recall SEPARATES where exact match did not: k={k}, tau={tau} "
              f"at {se:.1%} sensitivity / {fa:.1%} random false-accept.")
        print(f"     A certificate could then say 'reproduces the pseudoknot stems under {k} of 3 "
              f"independent predictors, a criterion that recognises {se:.0%} of known structures'.")
    else:
        print(f"  -> crossing-pair recall does NOT separate either. Both quantities fail on these")
        print(f"     instruments, which closes option B and leaves reporting the disagreement profile.")

    # ---- the combined rule: judge each PART with the instrument that can judge it -------------------
    # crossing-pair recall covers only the pairs that make the target a pseudoknot. A design could form the
    # knot and get the nested skeleton wrong, so recall alone is an incomplete structural claim. The nested
    # skeleton, though, has an EXACT and unambiguous judge -- ViennaRNA MFE on the pseudoknot-free reduction
    # -- so the honest criterion judges each part with the instrument that can actually judge it. Whether
    # that composes is a question, not an assumption: a native sequence really forms a pseudoknot, so its
    # ViennaRNA MFE may well NOT equal the nested reduction. Measured, not assumed.
    print(f"\n=== combined rule: nested skeleton by an EXACT judge AND crossing stems by the panel ===")
    import RNA
    from rna_harness.benchmarks.rnainvbench import _round_only
    nat = _native_sequences()
    print(f"  wild-type sequences parsed and matched to targets: {len(nat)}/{len(rows)}")
    if not nat:
        print(f"  SKIPPED: antaRNA constraint files not found; the combined rule stays UNCALIBRATED.")
    else:
        use = [r for r in rows if r["id"] in nat and r["id"] in tg]
        skel = {r["id"]: (RNA.fold(nat[r["id"]])[0] == _round_only(tg[r["id"]])) for r in use}
        n_sk = sum(skel.values())
        print(f"  native skeleton recovered by ViennaRNA MFE alone: {n_sk}/{len(use)} "
              f"({n_sk/max(len(use),1):.1%})")
        print(f"    (a native really does form a pseudoknot, so its MFE need not equal the nested"
              f" reduction -- this is the number that decides whether the two parts compose)")
        # the null: random sequences, same combined rule
        nullm = {r["id"]: r for r in nullr}
        rnd_use = [r for r in use if r["id"] in nullm]
        import random as _rnd
        _g = _rnd.Random(0)
        rnd_seq = {r["id"]: "".join(_g.choice("ACGU") for _ in range(len(tg[r["id"]]))) for r in rnd_use}
        rnd_skel = {i: (RNA.fold(q)[0] == _round_only(tg[i])) for i, q in rnd_seq.items()}
        print(f"\n  {'tau':>5} {'k':>3} {'combined sens':>14} {'combined random':>16}")
        best = None
        for tau in TAUS:
            for k in (1, 2, 3):
                se = sum(1 for r in use
                         if skel[r["id"]]
                         and sum(1 for o in ORACLES
                                 if (r.get(o, {}).get("cross_recall") or 0.0) >= tau) >= k) / max(len(use), 1)
                fa = sum(1 for r in rnd_use
                         if rnd_skel[r["id"]]
                         and sum(1 for o in ORACLES
                                 if (nullm[r["id"]].get(o, {}).get("cross_recall") or 0.0) >= tau) >= k
                         ) / max(len(rnd_use), 1)
                if k == 1:
                    print(f"  {tau:>5.2f} {k:>3} {se:>13.1%} {fa:>15.1%}")
                if se >= TARGET_SENSITIVITY and fa <= 0.05 and (best is None or (k, tau) > best[:2]):
                    best = (k, tau, se, fa)
        print(f"\n  usable combined rules under the SAME declared bar: "
              f"{'yes -> k=%d tau=%.2f at %.1f%% / %.1f%%' % (best[0], best[1], 100*best[2], 100*best[3]) if best else 'NONE'}")
        if not best:
            print(f"  -> the two parts do NOT compose: requiring the nested skeleton to be the ViennaRNA MFE")
            print(f"     costs most of the sensitivity, because a native's MFE is not its nested reduction.")
            print(f"     The defensible criterion therefore stays crossing-stem recall alone, and a")
            print(f"     certificate must say that it speaks about the KNOT, not the whole structure.")

    print(f"\n=== the missing half: false-accept rate on RANDOM sequences (same lengths) ===")
    null = _null_rows(rows)
    print(f"  null set: {len(null)} random ACGU sequences")
    print(f"  {'d':>4} " + "".join(f"{'k=' + str(k):>22}" for k in (1, 2, 3)))
    print(f"  {'':>4} " + "".join(f"{'sens':>10}{'  random':>12}" for _ in (1, 2, 3)))
    usable = []
    for d in DS:
        cells = []
        for k in (1, 2, 3):
            fa = 0
            for r in null:
                acc = sum(1 for o in ORACLES if r[o].get("bp_dist") is not None and r[o]["bp_dist"] <= d)
                fa += (acc >= k)
            fa /= max(len(null), 1)
            cells.append(f"{grid[(k, d)]:>9.1%}{fa:>12.1%}")
            if grid[(k, d)] >= TARGET_SENSITIVITY and fa <= 0.05:
                usable.append((k, d, grid[(k, d)], fa))
        print(f"  {d:>4} " + "".join(cells))
    print(f"\n  usable rules (sensitivity >= {TARGET_SENSITIVITY:.0%} AND random false-accept <= 5%): "
          f"{usable if usable else 'NONE'}")
    if not usable:
        print(f"  -> No tolerance separates known-correct structures from RANDOM SEQUENCES while keeping")
        print(f"     useful sensitivity. Confirmation by agreement among these three predictors cannot be")
        print(f"     calibrated at all. The honest product move is to report the disagreement profile and")
        print(f"     the instruments' measured recovery, not to issue a verdict dressed as ground truth.")

    print(f"\n=== what the shipped rule costs in sensitivity ===")
    for k in (3, 2, 1):
        print(f"  k={k}: " + "  ".join(f"d={d} {grid[(k, d)]:.0%}" for d in DS))

    json.dump({"n_calibration": len(rows), "target_sensitivity": TARGET_SENSITIVITY,
               "grid": {f"k{k}_d{d}": s for (k, d), s in grid.items()},
               "shipped_k3_d0": shipped,
               "viable_by_sensitivity_only": bool(ok),
               "usable_with_specificity": usable,
               "grid_relative_sens": {f"k{k}_r{r}": v for (k, r), v in gs.items()},
               "grid_relative_random": {f"k{k}_r{r}": v for (k, r), v in gn.items()},
               "usable_relative": usable_rel,
               "grid_cross_recall_sens": {f"k{k}_t{t}": v for (k, t), v in cs.items()},
               "grid_cross_recall_random": {f"k{k}_t{t}": v for (k, t), v in cn.items()},
               "usable_cross_recall": usable_cr},
              open(os.path.join(HERE, "calibrate.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/calibrate.json")


if __name__ == "__main__":
    main()
