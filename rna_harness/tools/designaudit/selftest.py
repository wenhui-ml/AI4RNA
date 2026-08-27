"""designaudit.selftest -- reproduce the published headline numbers from the corpus itself.

This is the claim the package has to be able to make: a reader points it at the same public CSV
and gets the same numbers, without our run scripts. Targets are the committed values from the
M26 / M26c artefacts; each is checked against a stated tolerance and every one is printed whether
it passes or not. A tolerance is never widened to make a check pass -- if a number moves, the
selftest says which one and by how much, and the caller decides what that means.

Tier 1 runs anywhere. Tier 2 needs ViennaRNA and is skipped, loudly, when it is absent.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from . import corpora, mining, core, scoring

CORPUS = os.environ.get(
    "DESIGNAUDIT_M2R", "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv")

TARGETS = {
    "n_admitted":        (4317,    30,     "interventions mined"),
    "identity_contrast": (+0.0539, 0.0050, "CG-AU, design x depth x crossed"),
    "weight":            (902,     60,     "pair-weights carrying that contrast"),
}
TIER2 = {
    "sensitivity":       (+0.1206, 0.0150, "within-stratum slope, ViennaRNA P_model"),
    "retention":         (0.963,   0.030,  "identity retained after fixing P_model"),
}


def _corpus_from(interventions):
    recs = [{"design": x.design, "pt": x.pair_type, "rf": x.outcome,
             "depth3": x.meta["depth3"], "crossed": x.meta["crossed"],
             "puzzle": x.meta["puzzle"]} for x in interventions]
    return core.Corpus.from_records(recs, design="design", treatment="pt", outcome="rf",
                                    covariates=("depth3", "crossed"), cluster="puzzle")


def run(path=CORPUS, verbose=True):
    ok, rows = True, []

    def check(name, got, want, tol, what):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        rows.append((name, got, want, tol, good, what))
        if verbose:
            print(f"  [{'PASS' if good else 'FAIL'}] {name:18s} {got:+10.4f}  "
                  f"target {want:+.4f} +/- {tol:.4f}   {what}")
        return good

    if not os.path.exists(path):
        print(f"corpus not found: {path}\n  set DESIGNAUDIT_M2R to the OK7a M2R csv")
        return False, rows

    print(f"designaudit selftest\n  corpus {path}\n")
    constructs = corpora.openknot_m2r(path)
    interventions, report = mining.mine(constructs)
    corpora.annotate_context(interventions)
    print(f"  {report}\n")

    print("tier 1 -- mining and inference, no predictor needed")
    check("n_admitted", len(interventions), *TARGETS["n_admitted"])
    c = _corpus_from(interventions)
    est = core.contrast(c, ("CG", "AU"), strata=("depth3", "crossed"), nperm=2000, nboot=200)
    check("identity_contrast", est.effect, *TARGETS["identity_contrast"])
    check("weight", est.weight, *TARGETS["weight"])
    print(f"       {est}")

    print("\ntier 2 -- scoring a predictor")
    try:
        import RNA
    except ImportError:
        print("  [SKIP] ViennaRNA not importable; tier 2 not run. This is a skip, not a pass.")
        return ok, rows

    cache = {}

    def pmodel(seq, i, j):
        if seq not in cache:
            fc = RNA.fold_compound(seq)
            fc.pf()
            cache[seq] = fc.bpp()
        try:
            return float(cache[seq][min(i, j) + 1][max(i, j) + 1])
        except (IndexError, KeyError):
            return 0.0

    print(f"  scoring {len({x.mut_seq for x in interventions})} mutant ensembles ...")
    for x in interventions:
        x.meta["p_model"] = pmodel(x.mut_seq, x.i, x.j)
    recs = [{"design": x.design, "pt": x.pair_type, "rf": x.outcome,
             "depth3": x.meta["depth3"], "crossed": x.meta["crossed"],
             "puzzle": x.meta["puzzle"], "p_model": x.meta["p_model"]} for x in interventions]
    c2 = core.Corpus.from_records(recs, design="design", treatment="pt", outcome="rf",
                                  covariates=("depth3", "crossed", "p_model"), cluster="puzzle")
    inst = scoring.score(c2, "p_model", levels=("CG", "AU"),
                         strata=("depth3", "crossed"), nperm=500, nplacebo=12)
    check("sensitivity", inst.sensitivity, *TIER2["sensitivity"])
    if inst.residual is None:
        ok = False
        rows.append(("retention", float("nan"), TIER2["retention"][0],
                     TIER2["retention"][1], False, "residual did not compute"))
        print("  [FAIL] retention          could not be computed; a declared check that does not "
              "run is a failure, not a skip")
        for n in inst.notes:
            print(f"         {n}")
    else:
        check("retention", inst.residual.retained, *TIER2["retention"])
    print(f"       {inst}")
    return ok, rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else CORPUS
    ok, _rows = run(path)
    print("\n" + ("selftest PASSED" if ok else "selftest FAILED -- see the FAIL lines above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
