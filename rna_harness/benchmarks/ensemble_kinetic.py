"""The co-transcriptional criterion as a distribution rather than a point, ready for the factory process.

WHAT IS BEING REPLACED. The shipping rule is pointwise and binary:

    trap_fraction = #{k : MFE(seq[:k]) contains ANY pair the target lacks} / #{k},  accept if <= 0.10

MFE is one structure out of an ensemble and "any" is a 0/1 verdict on a continuous quantity. Measured
consequences of that: it fires on 91.7% of natural transcription intermediates, and the ensemble version
predicts an independent simulator markedly better (rho -0.617 against -0.470, difference +0.147 with a
target-clustered bootstrap 95% CI of [+0.081, +0.216]) while being STRONGER on the adversarial null where
composition is identical by construction (+20.2% against +10.7%).

THE QUANTITY, and why it is a SHARE rather than a mass. The first attempt thresholded raw off-target
probability mass and every pre-registered threshold fired on everything, because the grid was declared at
0.1-2.0 while the quantity's mean on the relevant population is 18.3. Raw mass scales with length and pair
count -- 18-24 against targets carrying 86-92 pairs, and nothing like that on a 30 nt design -- so no fixed
threshold on it can be right. The share is bounded:

    off_share(k) = off_mass(k) / (off_mass(k) + on_mass(k))

measured at 47.4%-64.0% on natural prefixes, median 55.8%. A threshold on a bounded quantity is at least
comparable across molecules.

THRESHOLD SELECTION IS SELECTIVITY-MATCHED, and without this the whole exercise is self-confirming. A looser
criterion accepts more candidates and therefore delivers more; that is not an improvement, it is loosening. So
the operating point is chosen so the new rule's ACCEPTANCE RATE on the design pool equals the shipping rule's,
and only then are the two compared. Any comparison at unequal selectivity is refused.

WHAT THE FACTORY PROCESS REQUIRES, corrected during the pre-run review. An earlier version of this checklist
listed the register-shift (0/386) and nested (0/183) decoys -- those are nulls for the PSEUDOKNOT criterion
knot_confirmed and are untouched by a change here. This criterion has exactly two nulls of its own:

    isomer          within a family the base multiset is identical BY CONSTRUCTION, so a composition proxy
                    cannot separate at all. The shipping rule scores +10.7 [+7.1, +14.3]; the replacement may
                    not be weaker.
    grow invariance  the accepted-minus-rejected gap must keep its sign across a 16x span of transcription
                    speed, as the shipping rule does (+16.8 / +14.4 / +17.7).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..instrument import Declared, InstrumentError

MIN_PREFIX = 12                  # same as kinetic_suite, so both rules see identical prefixes
SHARE_GRID = (0.2, 0.35, 0.5, 0.65, 0.8)     # a grid on a BOUNDED quantity; still checked against the scale


def off_share_profile(seq: str, target: str) -> List[Dict]:
    """Per-prefix off-target share of pairing probability, plus the shipping rule's flag from the same fold.

    Both rules are derived from ONE pass of folding so a comparison between them cannot be confounded with
    solver noise or with a different set of prefixes.
    """
    import RNA
    from .kinetic_suite import pair_set
    tgt = pair_set(target)
    out = []
    for k in range(MIN_PREFIX, len(seq) + 1):
        sub = seq[:k]
        fc = RNA.fold_compound(sub)
        fc.pf()
        b = fc.bpp()
        on = off = 0.0
        for i in range(1, k + 1):
            row = b[i]
            for j in range(i + 1, k + 1):
                if (i - 1, j - 1) in tgt:
                    on += row[j]
                else:
                    off += row[j]
        tot = on + off
        out.append({"k": k, "on_mass": on, "off_mass": off,
                    "off_share": (off / tot) if tot > 0 else 0.0,
                    "mfe_trap": bool(pair_set(RNA.fold(sub)[0]) - tgt)})
    return out


def summarise(profile: Sequence[Dict]) -> Dict[str, float]:
    """The aggregates, including the shipping rule computed from the same profile."""
    n = max(1, len(profile))
    shares = [p["off_share"] for p in profile]
    return {"trap_fraction": sum(1 for p in profile if p["mfe_trap"]) / n,   # today's rule
            "share_mean": sum(shares) / n,
            "share_max": max(shares, default=0.0),
            "shares": shares}


def calibrate_theta(pool_summaries: Sequence[Dict], statistic: str = "share_mean",
                    shipping_max: float = 0.10) -> Dict[str, object]:
    """Pick the operating point so the new rule is EXACTLY AS SELECTIVE as the shipping rule.

    Without this the comparison is rigged: a threshold that accepts more will deliver more, and that is
    loosening rather than improvement. The shipping rule's acceptance rate on this pool is measured first,
    then the new statistic's threshold is set to the matching quantile.

    The grid is additionally checked against the statistic's measured scale, because declaring a threshold
    without measuring the scale is the defect that killed the previous attempt -- and instrument.Declared
    refuses to hand over a grid that misses it.
    """
    if not pool_summaries:
        raise InstrumentError("calibrate_theta: empty pool")
    acc_ship = sum(1 for s in pool_summaries if s["trap_fraction"] <= shipping_max) / len(pool_summaries)
    vals = sorted(s[statistic] for s in pool_summaries)
    grid = Declared(f"{statistic}_grid", SHARE_GRID).against(vals)
    # .against() only MEASURES; it is .value that enforces. The first version of this line took grid.scale and
    # never read grid.value, so the straddle check never fired -- the third instance today of writing a gate
    # and then not going through it. Reading .value is the enforcement.
    _ = grid.value
    # the matching quantile: accept the same FRACTION of the pool, from the low (better) end
    idx = max(0, min(len(vals) - 1, int(round(acc_ship * len(vals))) - 1))
    theta = vals[idx] if acc_ship > 0 else vals[0]
    acc_new = sum(1 for v in vals if v <= theta) / len(vals)
    return {"statistic": statistic, "shipping_acceptance": acc_ship, "theta": float(theta),
            "matched_acceptance": acc_new, "scale": grid.scale,
            "selectivity_matched": abs(acc_new - acc_ship) <= 1.0 / len(vals) + 1e-9}


def held(summary: Dict[str, float], theta: float, statistic: str = "share_mean") -> bool:
    """Does this candidate pass the ensemble criterion at the calibrated operating point?"""
    return summary[statistic] <= theta


def compare_at_equal_selectivity(pool_summaries: Sequence[Dict], judge: Sequence[float],
                                 statistic: str = "share_mean",
                                 shipping_max: float = 0.10) -> Dict[str, object]:
    """Both rules at the SAME acceptance rate: which one's accepted set scores better on the judge?

    Refuses to report anything if the selectivities do not match, because a difference at unequal selectivity
    says nothing about which rule is better.
    """
    import statistics as st
    cal = calibrate_theta(pool_summaries, statistic, shipping_max)
    if not cal["selectivity_matched"]:
        return {"refused": "selectivity could not be matched on this pool", **cal}
    ship = [j for s, j in zip(pool_summaries, judge) if s["trap_fraction"] <= shipping_max]
    new = [j for s, j in zip(pool_summaries, judge) if held(s, cal["theta"], statistic)]
    if len(ship) < 5 or len(new) < 5:
        return {"refused": f"too few accepted (shipping {len(ship)}, new {len(new)})", **cal}
    return {"theta": cal["theta"], "shipping_acceptance": cal["shipping_acceptance"],
            "matched_acceptance": cal["matched_acceptance"],
            "n_shipping": len(ship), "n_new": len(new),
            "judge_shipping": st.mean(ship), "judge_new": st.mean(new),
            "delta": st.mean(new) - st.mean(ship), "scale": cal["scale"]}
