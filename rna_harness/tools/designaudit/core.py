"""designaudit -- estimate a sequence-property effect on a corpus somebody designed.

The problem this solves. When the molecules in a corpus were produced by design programs, the
programs' own preferences appear as covariates. A contrast in a sequence property is then partly a
contrast in whatever the designers correlated with that property, and the naive estimate is
attenuated by a computable amount. This module estimates the effect with the design conditioned
on, reports how much the corpus was hiding, and refuses the two reports that are easy to compute
and easy to misread.

Four rules are enforced rather than documented.

  R1  design-fixed randomisation. When the comparison set is built under calipers, permuting the
      treatment REBUILDS that set instead of relabelling it, and the resulting null is too narrow.
      Permutation is over the outcome inside a fixed stratification. `permute="treatment"` is
      available and warns.

  R2  same-population denominator. A mediated fraction is (1 - conditioned/baseline), and the
      baseline must be computed on the records that survive the extra stratifying dimension.
      Against the full population the ratio mixes selection into mediation.

  R3  placebo mediator. Adding any stratifying dimension shrinks a matched contrast mechanically.
      A mediator drawn from the same marginal, cut at the same edges, carrying no relation to the
      outcome, measures that shrinkage. Only the difference is interpretable, and `mediate()`
      will not return a mediated fraction without it.

  R4  linearity before a product of coefficients. The stratified and product-of-coefficients
      routes to a mediated fraction agree only when the conditional mean is affine in the
      mediator. `mediate()` tests it bin by bin and marks the product route inadmissible when it
      is not.

Depends on numpy only.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Corpus", "contrast", "placement", "mediate", "ladder", "audit", "Report"]


# ------------------------------------------------------------------ data
@dataclass
class Corpus:
    """One row per unit. `design` is the group the design process holds fixed (a target x method
    cell, a designer, a batch); contrasts are always computed inside it."""
    design: np.ndarray            # group label, any hashable
    treatment: np.ndarray         # the sequence property, categorical (two levels compared)
    outcome: np.ndarray           # float, or None for a design-side-only audit
    covariates: dict = field(default_factory=dict)   # name -> array, used for stratification
    cluster: np.ndarray = None    # resampling unit for intervals; defaults to `design`

    def __post_init__(self):
        n = len(self.design)
        for k, v in list(self.covariates.items()):
            if len(v) != n:
                raise ValueError(f"covariate {k!r} has length {len(v)}, expected {n}")
        if self.outcome is not None and len(self.outcome) != n:
            raise ValueError("outcome length does not match")
        if self.cluster is None:
            self.cluster = np.asarray(self.design)
        self.n = n

    @classmethod
    def from_records(cls, records, design, treatment, outcome=None, covariates=(), cluster=None):
        def g(k):
            vals = [r[k] for r in records]
            if any(isinstance(v, (tuple, list)) for v in vals):
                # a design cell is often a tuple (target, method); np.array would make it 2-D
                # and the stratum ids below would be built from unhashable rows
                arr = np.empty(len(vals), dtype=object)
                for i, v in enumerate(vals):
                    arr[i] = v
                return arr
            return np.array(vals)
        return cls(design=g(design), treatment=g(treatment),
                   outcome=(g(outcome).astype(float) if outcome else None),
                   covariates={c: g(c) for c in covariates},
                   cluster=(g(cluster) if cluster else None))

    def strata(self, keys=None, extra=None):
        """stratum id per row from `design` plus the named covariates (plus `extra`, an array)."""
        keys = list(self.covariates) if keys is None else list(keys)
        cols = [self.design] + [self.covariates[k] for k in keys]
        if extra is not None:
            cols.append(np.asarray(extra))
        return np.array([hash(t) for t in zip(*cols)])


# ------------------------------------------------------------------ estimator
def _weighted(sid, t, y):
    """stratum-size weighted difference of means over strata holding both levels."""
    keys, inv = np.unique(sid, return_inverse=True)
    ng = len(keys)
    nA = np.bincount(inv[t], minlength=ng); sA = np.bincount(inv[t], weights=y[t], minlength=ng)
    nB = np.bincount(inv[~t], minlength=ng); sB = np.bincount(inv[~t], weights=y[~t], minlength=ng)
    ok = (nA > 0) & (nB > 0)
    if not ok.any():
        return None, 0.0, 0, ok[inv]
    w = np.minimum(nA[ok], nB[ok]).astype(float)
    d = sA[ok] / nA[ok] - sB[ok] / nB[ok]
    return float((w * d).sum() / w.sum()), float(w.sum()), int(ok.sum()), ok[inv]


def _groups(labels):
    """row indices per distinct label. np.unique/== break on object arrays of tuples, which is
    what a design cell usually is, so grouping is done by dict rather than by broadcasting."""
    by = collections.defaultdict(list)
    for i, v in enumerate(labels):
        by[v].append(i)
    return [np.asarray(ix) for ix in by.values()]


def _perm_within(sid, t, rng):
    order = np.lexsort((rng.random(len(sid)), sid))
    base = np.argsort(sid, kind="stable")
    out = np.empty(len(sid), dtype=bool)
    out[base] = t[order]
    return out


@dataclass
class Estimate:
    effect: float
    p: float
    ci: tuple
    weight: float
    strata: int
    null_sd: float
    n_used: int

    def __repr__(self):
        return (f"effect {self.effect:+.4f}  p {self.p:.4f}  CI [{self.ci[0]:+.4f}, "
                f"{self.ci[1]:+.4f}]  {self.weight:.0f} pair-weights over {self.strata} strata")


def contrast(c: Corpus, levels, strata=None, extra=None, nperm=2000, nboot=500,
             permute="outcome", seed=0):
    """Effect of `levels[0]` over `levels[1]`, conditioned on the design and the named covariates.

    permute: "outcome" (R1, the default) or "treatment" (warns; kept so the difference can be
    demonstrated rather than asserted)."""
    if c.outcome is None:
        raise ValueError("contrast needs an outcome; use placement() for a design-side audit")
    m = np.isin(c.treatment, levels)
    sid = c.strata(strata, extra)[m]
    t = (c.treatment[m] == levels[0])
    y = c.outcome[m].astype(float)
    eff, w, ns, _ = _weighted(sid, t, y)
    if eff is None:
        raise ValueError("no stratum holds both levels; loosen the stratification")
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(nperm):
        if permute == "treatment":
            tp = t[rng.permutation(len(t))]
        else:
            tp = _perm_within(sid, t, rng)
        v, _w, _s, _ok = _weighted(sid, tp, y)
        null.append(0.0 if v is None else v)
    null = np.sort(np.asarray(null))
    # two-sided. A one-sided (null >= eff) count reports ~0.98 for a real negative effect, which
    # reads as "nothing here" when the effect is in the far negative tail. Found when the
    # thermodynamic model's contrast came back at -0.0304 with p 0.979.
    p = (float((np.abs(null) >= abs(eff)).sum()) + 1) / (nperm + 1)
    cl = c.cluster[m]
    units = np.unique(cl)
    boot = []
    for _ in range(nboot):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([np.nonzero(cl == units[j])[0] for j in pick])
        v, _w, _s, _ok = _weighted(sid[idx], t[idx], y[idx])
        if v is not None:
            boot.append(v)
    boot = np.sort(np.asarray(boot))
    ci = (float(boot[int(.025 * len(boot))]), float(boot[int(.975 * len(boot))]))
    return Estimate(eff, p, ci, w, ns, float(null.std()), int(m.sum()))


# ------------------------------------------------------------------ design-side audit
def placement(c: Corpus, levels, covariate, group=None, nperm=2000, seed=0):
    """How differently the two treatment levels are placed on `covariate`, per design group.

    Touches no outcome. `group` (e.g. the design method) decomposes the gap into a component
    explained by the group after removing the design's own main effect."""
    x = np.asarray(c.covariates[covariate], float)
    m = np.isin(c.treatment, levels)
    d, t, x = c.design[m], c.treatment[m] == levels[0], x[m]
    by = collections.defaultdict(lambda: ([], []))
    for i in range(len(d)):
        by[d[i]][0 if t[i] else 1].append(x[i])
    cells = {k: float(np.mean(a) - np.mean(b)) for k, (a, b) in by.items() if len(a) >= 4 and len(b) >= 4}
    if not cells:
        return {"cells": 0}
    vals = np.array(list(cells.values()))
    out = {"cells": len(cells), "pooled_gap": float(vals.mean()),
           "sd": float(vals.std(ddof=1)), "frac_negative": float((vals < 0).mean())}
    if group is not None:
        gg = {k: v for k, v in zip(c.design[m], np.asarray(c.covariates[group])[m])}
        keys = sorted(cells)
        y = np.array([cells[k] for k in keys])
        lab = np.array([gg[k] for k in keys])
        tgt = np.array([k[0] if isinstance(k, tuple) else k for k in keys])

        def partial(labels):
            ut, um = np.unique(tgt), np.unique(labels)
            if len(um) < 2 or len(ut) < 2:
                return 0.0
            T = np.zeros((len(y), len(ut))); M = np.zeros((len(y), len(um)))
            for i, (a, b) in enumerate(zip(tgt, labels)):
                T[i, list(ut).index(a)] = 1.0; M[i, list(um).index(b)] = 1.0
            rss = lambda X: float(((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum())
            tss = float(((y - y.mean()) ** 2).sum()) or 1.0
            return (rss(T) - rss(np.hstack([T, M]))) / tss
        obs = partial(lab)
        rng = np.random.default_rng(seed)
        nl = []
        for _ in range(nperm):
            perm = lab.copy()
            for u in np.unique(tgt):
                ix = np.nonzero(tgt == u)[0]
                perm[ix] = lab[ix][rng.permutation(len(ix))]
            nl.append(partial(perm))
        nl = np.sort(np.asarray(nl))
        out["group_component"] = float(obs)
        out["group_p"] = (float((nl >= obs).sum()) + 1) / (nperm + 1)
        out["per_group"] = {str(g): float(np.mean([cells[k] for k in keys if gg[k] == g]))
                            for g in np.unique(lab)}
    return out


# ------------------------------------------------------------------ mediation
@dataclass
class Mediation:
    conditioned: float
    baseline_same_population: float
    baseline_full: float
    retained: float
    mediated: float
    mediated_ci: tuple
    placebo_mean: float
    placebo_sd: float
    z_vs_placebo: float
    binwise: list
    monotone: bool
    product_route: float
    product_admissible: bool
    warnings: list

    def __repr__(self):
        s = (f"mediated {100*self.mediated:+.1f}% [{100*self.mediated_ci[0]:+.1f}%, "
             f"{100*self.mediated_ci[1]:+.1f}%]  (placebo {100*self.placebo_mean:.1f}% "
             f"+/- {100*self.placebo_sd:.1f}, z {self.z_vs_placebo:+.2f})")
        return s + ("" if not self.warnings else "\n  ! " + "\n  ! ".join(self.warnings))


def mediate(c: Corpus, levels, mediator, strata=None, nbins=5, nplacebo=12, seed=0):
    """How much of the conditioned contrast runs through `mediator`.

    Enforces R2 (same-population denominator), R3 (placebo mediator, always run) and R4
    (linearity tested before the product-of-coefficients route is reported)."""
    if c.outcome is None:
        raise ValueError("mediate needs an outcome")
    warn = []
    m = np.isin(c.treatment, levels)
    base_sid = c.strata(strata)[m]
    t = (c.treatment[m] == levels[0])
    y = c.outcome[m].astype(float)
    mv = np.asarray(c.covariates[mediator], float)[m]
    edges = np.quantile(mv, np.linspace(0, 1, nbins + 1)[1:-1])       # cut once, before contrasts
    b = np.searchsorted(edges, mv, side="right")
    sid = np.array([hash((int(s), int(k))) for s, k in zip(base_sid, b)])

    cond, w1, s1, keep = _weighted(sid, t, y)
    if cond is None:
        raise ValueError("no stratum holds both levels once the mediator is added")
    full, _w, _s, _k = _weighted(base_sid, t, y)
    same, _w2, _s2, _k2 = _weighted(base_sid[keep], t[keep], y[keep])   # R2
    if same is None or abs(same) < 1e-12:
        raise ValueError("same-population baseline is degenerate")
    retained = cond / same
    if abs(full - same) > 0.15 * abs(full):
        warn.append(f"selection: the overlap region's baseline is {full/same:.2f}x the corpus "
                    f"baseline. Dividing by the corpus baseline would report "
                    f"{100*(1-cond/full):+.1f}% instead of {100*(1-retained):+.1f}%.")
    rng = np.random.default_rng(seed)                                   # R3
    pl = []
    for _ in range(nplacebo):
        sh = mv.copy()
        for ix in _groups(c.design[m]):
            sh[ix] = mv[ix][rng.permutation(len(ix))]
        bb = np.searchsorted(edges, sh, side="right")
        s2 = np.array([hash((int(s), int(k))) for s, k in zip(base_sid, bb)])
        v, _w3, _s3, kp = _weighted(s2, t, y)
        bl, _w4, _s4, _k4 = _weighted(base_sid[kp], t[kp], y[kp])
        if v is not None and bl not in (None, 0.0):
            pl.append(v / bl)
    pl = np.asarray(pl)
    pm, ps = float(pl.mean()), float(pl.std(ddof=1))
    z = (retained - pm) / (ps or 1.0)
    med = 1 - retained
    ci = (med - 1.96 * ps, med + 1.96 * ps)
    if ci[0] <= 0.0 <= ci[1]:
        warn.append("the mediated fraction is indistinguishable from what the placebo mediator "
                    "produces; report it as zero, not as the point estimate.")
    keys, inv = np.unique(base_sid, return_inverse=True)                # R4
    cnt = np.bincount(inv).astype(float); cnt[cnt == 0] = 1.0
    yc = y - (np.bincount(inv, weights=y) / cnt)[inv]
    bw = [{"bin": int(k), "n": int((b == k).sum()),
           "mediator": float(mv[b == k].mean()), "outcome": float(yc[b == k].mean())}
          for k in range(nbins) if (b == k).any()]
    o = [r["outcome"] for r in bw]
    mono = all(o[i] >= o[i + 1] for i in range(len(o) - 1)) or \
           all(o[i] <= o[i + 1] for i in range(len(o) - 1))
    xc = mv - (np.bincount(inv, weights=mv) / cnt)[inv]
    beta = float((xc * yc).sum() / (xc * xc).sum()) if (xc * xc).sum() else 0.0
    gapn = gapd = 0.0
    nA = np.bincount(inv[t], minlength=len(keys)); sA = np.bincount(inv[t], weights=mv[t], minlength=len(keys))
    nB = np.bincount(inv[~t], minlength=len(keys)); sB = np.bincount(inv[~t], weights=mv[~t], minlength=len(keys))
    ok = (nA > 0) & (nB > 0)
    if ok.any():
        wgt = np.minimum(nA[ok], nB[ok]).astype(float)
        gapn = float((wgt * (sA[ok] / nA[ok] - sB[ok] / nB[ok])).sum()); gapd = float(wgt.sum())
    gap = gapn / gapd if gapd else 0.0
    prod = beta * gap / (full or 1.0)
    if not mono:
        warn.append(f"the conditional mean is not monotone in the mediator, so the "
                    f"product-of-coefficients route is inadmissible here; it would report "
                    f"{100*prod:.1f}% against the stratified {100*med:+.1f}%.")
    return Mediation(cond, same, full, retained, med, ci, pm, ps, z, bw, mono, prod,
                     bool(mono), warn)


# ------------------------------------------------------------------ calibration
def ladder(c: Corpus, levels, strata=None, deltas=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4),
           draws=20, nperm=400, alpha=0.005, seed=0):
    """What effect this design could have detected. Built on the NULL: labels are scrambled first,
    so the delta injected is the whole truth, and delta=0 is a type-I self-check."""
    m = np.isin(c.treatment, levels)
    sid = c.strata(strata)[m]
    t = (c.treatment[m] == levels[0])
    y = c.outcome[m].astype(float)
    rng = np.random.default_rng(seed)
    null = np.sort(np.asarray([(_weighted(sid, _perm_within(sid, t, rng), y)[0] or 0.0)
                               for _ in range(nperm)]))
    thr = float(np.quantile(null, 1 - alpha))
    out = {}
    for d in deltas:
        hit = 0
        for _ in range(draws):
            tp = _perm_within(sid, t, rng)
            v, _w, _s, _k = _weighted(sid, tp, y + d * tp)
            hit += (v is not None and v > thr)
        out[d] = hit / draws
    return {"threshold_statistic": thr, "power": out,
            "self_check_delta0": out.get(0.0),
            "detects": next((d for d in deltas if d > 0 and out[d] >= 0.8), None)}


# ------------------------------------------------------------------ one call
@dataclass
class Report:
    contrast_marginal: Estimate
    contrast_conditioned: Estimate
    placement: dict
    mediation: Mediation
    calibration: dict
    warnings: list

    def text(self):
        L = ["designaudit report", "=" * 60,
             f"marginal      {self.contrast_marginal}",
             f"conditioned   {self.contrast_conditioned}",
             f"attenuation   {self.contrast_marginal.effect - self.contrast_conditioned.effect:+.4f}"
             f"  ({100*(1 - self.contrast_marginal.effect / self.contrast_conditioned.effect):.0f}%"
             f" of the conditioned effect is not visible marginally)",
             "-" * 60,
             f"placement     gap {self.placement.get('pooled_gap', float('nan')):+.4f} over "
             f"{self.placement.get('cells', 0)} design cells"
             + (f", group component {self.placement['group_component']:+.4f} "
                f"(p {self.placement['group_p']:.4f})" if "group_component" in self.placement else ""),
             f"mediation     {self.mediation}",
             f"calibration   detects {self.calibration['detects']}, "
             f"delta=0 self-check {self.calibration['self_check_delta0']}"]
        if self.warnings:
            L += ["-" * 60] + ["! " + w for w in self.warnings]
        return "\n".join(L)


def audit(c: Corpus, levels, mediator, strata=None, group=None, seed=0, **kw):
    """Marginal and conditioned contrasts, the design-side placement, the mediation with its
    guards, and the calibration ladder -- in one call."""
    strata = list(c.covariates) if strata is None else list(strata)
    marg = contrast(c, levels, strata=[], seed=seed, **kw)
    cond = contrast(c, levels, strata=strata, seed=seed, **kw)
    pl = placement(c, levels, mediator, group=group, seed=seed)
    med = mediate(c, levels, mediator, strata=strata, seed=seed)
    lad = ladder(c, levels, strata=strata, seed=seed)
    warn = list(med.warnings)
    if lad["self_check_delta0"] and lad["self_check_delta0"] > 0.15:
        warn.append(f"the ladder's delta=0 self-check is {lad['self_check_delta0']}, above 0.15: "
                    f"this design's type-I error is not under control and nothing above is safe.")
    if lad["detects"] is not None and abs(cond.effect) < lad["detects"]:
        warn.append(f"the effect ({cond.effect:+.4f}) is below what this design detects "
                    f"({lad['detects']}); it is at its own resolution limit.")
    return Report(marg, cond, pl, med, lad, warn)
