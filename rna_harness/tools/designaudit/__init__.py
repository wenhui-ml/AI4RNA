"""designaudit -- score a predictor on interventions a corpus already contains.

Two things live here and they are meant to be used in this order.

`mining` recovers controlled interventions from a probing corpus: wild type and mutant differing
at exactly two positions inside the declared design window, those positions being a declared pair,
the substitution exchanging the two bases so the unordered pair type is preserved. That is a
two-nucleotide experiment with its own control, already executed and already measured.

`scoring` puts a predictor on those interventions and reports three readouts -- sensitivity,
calibration, and what a named sequence property still explains once the prediction is held fixed.
`core` supplies the inference underneath and enforces four rules as code rather than documenting
them: outcome-permutation under a fixed design (R1), same-population denominators (R2), a placebo
mediator that always runs (R3), and a linearity test before any product-of-coefficients route is
reported (R4).

    from designaudit import corpora, mining, scoring, core

    constructs = corpora.openknot_m2r("OK7a_M2R_data.v4.5.1.csv")
    interventions, report = mining.mine(constructs)
    print(report)

`python -m designaudit selftest` reproduces the published headline numbers from the corpus itself.
"""
from . import core, mining, scoring, corpora           # noqa: F401
from .core import Corpus, contrast, placement, mediate, ladder, audit   # noqa: F401
from .mining import mine, Intervention, MiningReport   # noqa: F401
from .scoring import score, sensitivity, calibration, Instrument        # noqa: F401

__version__ = "0.1.0"
__all__ = ["core", "mining", "scoring", "corpora", "Corpus", "contrast", "placement",
           "mediate", "ladder", "audit", "mine", "Intervention", "MiningReport",
           "score", "sensitivity", "calibration", "Instrument"]
