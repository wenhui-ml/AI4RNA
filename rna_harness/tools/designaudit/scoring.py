"""designaudit.scoring -- score a predictor on interventions it did not choose.

A conventional structure benchmark asks whether a predicted structure agrees with a reference.
That question saturates: on a corpus where the reference is recovered for ~90% of the pairs being
compared, agreement has almost no dynamic range left, and the answer moves several-fold with the
choice of which pairs enter the comparison. An intervention benchmark asks a different question --
when the molecule was actually perturbed, did the model's prediction move the way the measurement
moved -- and that question has no pair-set freedom, because the interventions are the ones that
were performed.

A predictor enters as a callable that returns, for a sequence and a position pair, a number it
believes about that pair forming. It need not be a probability and need not come from a
thermodynamic model; it must only be comparable across the sequences of one corpus. Three readouts
come back, and they are meant to be read in this order:

  SENSITIVITY   does the predictor's number move with the measurement at all, inside the design?
                Reported as a within-stratum slope with a permutation p. If this is null, nothing
                below it can be read as "beyond the model" -- the model predicted nothing to be
                beyond. `Instrument.readable` records that.

  CALIBRATION   binned mean outcome against binned prediction, and whether it is monotone. This is
                descriptive and it is what licenses treating the predictor's scale as meaningful
                rather than merely rank-ordered.

  RESIDUAL      what a named sequence property still explains once the prediction is held fixed,
                via `core.mediate`, which enforces the same-population denominator and refuses to
                report a mediated fraction without a placebo mediator.

The residual is the point of the instrument. A predictor that scores well on the first two
readouts and still leaves a residual on the third has been shown to be missing something specific
-- which is a diagnosis, not a score.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from . import core


@dataclass
class Instrument:
    sensitivity: float
    sensitivity_p: float
    n_scored: int
    calibration: list
    monotone: bool
    residual: object            # core.Mediation, or None when the corpus cannot support it
    readable: bool
    notes: list

    def __repr__(self):
        head = (f"sensitivity {self.sensitivity:+.4f} (p {self.sensitivity_p:.4f}, "
                f"n {self.n_scored})   calibration {'monotone' if self.monotone else 'NOT monotone'}")
        if not self.readable:
            head += "\n  ! the predictor has no measurable skill here; the residual below is not " \
                    "evidence of anything beyond the model"
        if self.residual is not None:
            head += "\n  residual: " + repr(self.residual)
        return head + ("" if not self.notes else "\n  " + "\n  ".join(self.notes))


def _slope_within(sid, x, y):
    """pooled within-stratum slope of y on x -- the design is never a regressor, it is removed."""
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)
    sxy = sxx = 0.0
    n = 0
    for _s, ix in by.items():
        if len(ix) < 2:
            continue
        xa = np.asarray(x)[ix].astype(float)
        ya = np.asarray(y)[ix].astype(float)
        xa = xa - xa.mean()
        ya = ya - ya.mean()
        sxy += float((xa * ya).sum())
        sxx += float((xa * xa).sum())
        n += len(ix)
    return (sxy / sxx if sxx else None), n


def sensitivity(c: core.Corpus, prediction, strata=None, nperm=500, seed=0):
    """Within-stratum slope of the outcome on the prediction, with the outcome permuted.

    R1 again: the design is held fixed and the OUTCOME is permuted inside it. Permuting the
    prediction would rebuild the stratification it defines."""
    x = np.asarray(c.covariates[prediction], float)
    y = np.asarray(c.outcome, float)
    sid = c.strata(strata)
    obs, n = _slope_within(sid, x, y)
    if obs is None:
        return None, 1.0, 0
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)
    ge = 0
    for _ in range(nperm):
        yp = y.copy()
        for _s, ix in by.items():
            yp[ix] = y[np.asarray(ix)[rng.permutation(len(ix))]]
        s, _n = _slope_within(sid, x, yp)
        ge += (s is not None and abs(s) >= abs(obs))
    return float(obs), (ge + 1) / (nperm + 1), int(n)


def calibration(c: core.Corpus, prediction, strata=None, nbins=5):
    """Binned prediction against the outcome, centred inside the design so the bins are not
    reporting which designs happen to be easy."""
    x = np.asarray(c.covariates[prediction], float)
    y = np.asarray(c.outcome, float)
    sid = c.strata(strata)
    edges = [float(np.quantile(x, q)) for q in np.linspace(0, 1, nbins + 1)[1:-1]]
    b = np.searchsorted(edges, x, side="right")
    cen = np.empty(len(y))
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)
    for _s, ix in by.items():
        cen[ix] = y[ix] - y[ix].mean()
    rows = []
    for k in range(nbins):
        m = b == k
        if not m.any():
            rows.append({"bin": k, "n": 0})
            continue
        rows.append({"bin": k, "n": int(m.sum()), "prediction": float(x[m].mean()),
                     "outcome": float(y[m].mean()), "outcome_centred": float(cen[m].mean())})
    seen = [r for r in rows if r["n"]]
    mono = all(seen[i]["outcome_centred"] <= seen[i + 1]["outcome_centred"]
               for i in range(len(seen) - 1))
    return rows, bool(mono), edges


def score(c: core.Corpus, prediction, levels=None, strata=None, nbins=5, nperm=500,
          nplacebo=12, seed=0):
    """The three readouts. `levels` names the property whose residual is asked for; omit it to get
    sensitivity and calibration only."""
    notes = []
    s, p, n = sensitivity(c, prediction, strata=strata, nperm=nperm, seed=seed)
    if s is None:
        return Instrument(0.0, 1.0, 0, [], False, None, False,
                          ["no stratum holds two scorable records"])
    cal, mono, _edges = calibration(c, prediction, strata=strata, nbins=nbins)
    readable = (p <= 0.05 and s > 0)
    if not readable:
        notes.append("sensitivity is null or negative: report it and stop, or fix the index "
                     "convention before reading anything else")
    resid = None
    if levels is not None:
        try:
            resid = core.mediate(c, levels, prediction, strata=strata, nbins=nbins,
                                 nplacebo=nplacebo, seed=seed)
        except ValueError as e:                                                # noqa: BLE001
            notes.append(f"residual not computable: {e}")
    return Instrument(float(s), float(p), int(n), cal, mono, resid, bool(readable), notes)
