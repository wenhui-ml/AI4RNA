"""designaudit CLI -- mine, score, contrast, selftest.

    python -m designaudit mine     --corpus OK7a_M2R_data.v4.5.1.csv
    python -m designaudit contrast --corpus ... --levels CG,AU
    python -m designaudit score    --corpus ... --predictor vienna --levels CG,AU
    python -m designaudit selftest

`score` ships one predictor, `vienna`, because it is the one we can compute without a network. A
predictor is a callable (sequence, i, j) -> float; pass your own with
--predictor mypkg.mymodule:my_callable and it will be imported and used unchanged.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from . import corpora, mining, core, scoring


def _load_predictor(spec):
    if spec == "vienna":
        import RNA
        cache = {}

        def f(seq, i, j):
            if seq not in cache:
                fc = RNA.fold_compound(seq)
                fc.pf()
                cache[seq] = fc.bpp()
            try:
                return float(cache[seq][min(i, j) + 1][max(i, j) + 1])
            except (IndexError, KeyError):
                return 0.0
        return f
    mod, _, name = spec.partition(":")
    if not name:
        raise SystemExit("predictor must be 'vienna' or 'module:callable'")
    return getattr(importlib.import_module(mod), name)


def _mined(args):
    constructs = corpora.openknot_m2r(args.corpus, outcome=args.outcome)
    iv, report = mining.mine(constructs)
    corpora.annotate_context(iv)
    return iv, report


def _corpus(iv, extra=()):
    recs = [{"design": x.design, "pt": x.pair_type, "rf": x.outcome,
             "depth3": x.meta["depth3"], "crossed": x.meta["crossed"],
             "puzzle": x.meta["puzzle"], **{k: x.meta[k] for k in extra}} for x in iv]
    return core.Corpus.from_records(recs, design="design", treatment="pt", outcome="rf",
                                    covariates=("depth3", "crossed") + tuple(extra),
                                    cluster="puzzle")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="designaudit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--corpus", required=True, help="probing corpus csv")
        p.add_argument("--outcome", default="rescue_factor")
        p.add_argument("--json", help="write the result here as json")

    m = sub.add_parser("mine", help="recover the interventions a corpus contains")
    common(m)
    c = sub.add_parser("contrast", help="effect of a pair-type contrast, design conditioned on")
    common(c)
    c.add_argument("--levels", default="CG,AU")
    c.add_argument("--nperm", type=int, default=2000)
    s = sub.add_parser("score", help="score a predictor on the mined interventions")
    common(s)
    s.add_argument("--predictor", default="vienna", help="'vienna' or 'module:callable'")
    s.add_argument("--levels", default="CG,AU", help="omit to get sensitivity+calibration only")
    s.add_argument("--nbins", type=int, default=5)
    sub.add_parser("selftest", help="reproduce the published headline numbers")

    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        from . import selftest
        return selftest.main([])

    iv, report = _mined(a)
    print(report)
    if not iv:
        print("\nnothing to do: this corpus supplies no interventions.")
        return 1

    out = {"n_admitted": len(iv), "excluded": report.excluded,
           "by_pair_type": report.by_pair_type}

    if a.cmd == "contrast":
        lv = tuple(a.levels.split(","))
        est = core.contrast(_corpus(iv), lv, strata=("depth3", "crossed"), nperm=a.nperm)
        print(f"\n{lv[0]} over {lv[1]}:  {est}")
        out["contrast"] = est.__dict__

    elif a.cmd == "score":
        f = _load_predictor(a.predictor)
        print(f"\nscoring {len({x.mut_seq for x in iv})} mutant sequences with {a.predictor} ...")
        for x in iv:
            x.meta["prediction"] = float(f(x.mut_seq, x.i, x.j))
        lv = tuple(a.levels.split(",")) if a.levels else None
        inst = scoring.score(_corpus(iv, extra=("prediction",)), "prediction", levels=lv,
                             strata=("depth3", "crossed"), nbins=a.nbins)
        print(f"\n{inst}")
        for row in inst.calibration:
            if row["n"]:
                print(f"    bin {row['bin']}  n {row['n']:5d}  prediction {row['prediction']:.4f}"
                      f"  outcome {row['outcome']:+.4f}  centred {row['outcome_centred']:+.4f}")
        out["sensitivity"] = inst.sensitivity
        out["sensitivity_p"] = inst.sensitivity_p
        out["monotone"] = inst.monotone
        out["calibration"] = inst.calibration
        if inst.residual is not None:
            out["retention"] = inst.residual.retained
            out["mediated"] = inst.residual.mediated
            out["mediated_ci"] = list(inst.residual.mediated_ci)
            out["placebo_mean"] = inst.residual.placebo_mean
            out["warnings"] = inst.residual.warnings

    if a.json:
        json.dump(out, open(a.json, "w"), default=str, ensure_ascii=False, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
