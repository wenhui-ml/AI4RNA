"""The factory gates for the ensemble co-transcriptional criterion, as tests rather than as intentions."""
import pytest

from rna_harness.benchmarks.ensemble_kinetic import (SHARE_GRID, calibrate_theta,
                                                     compare_at_equal_selectivity, held, off_share_profile,
                                                     summarise)
from rna_harness.instrument import InstrumentError


def test_off_share_is_bounded_and_the_two_rules_share_one_folding():
    """The previous attempt thresholded raw mass, whose mean on the relevant population is 18.3 while the

    declared grid ran 0.1-2.0. A share is bounded, so a threshold on it is at least comparable across
    molecules. Both rules must also come out of the SAME folding pass, or their comparison is confounded with
    solver noise.
    """
    seq = "GGGGAAAACCCCUUUUGGGGAAAACCCC"
    import RNA
    prof = off_share_profile(seq, RNA.fold(seq)[0])
    assert prof and all(0.0 <= p["off_share"] <= 1.0 for p in prof)
    assert all("mfe_trap" in p for p in prof), "the shipping rule must come from the same profile"
    s = summarise(prof)
    assert 0.0 <= s["share_mean"] <= 1.0 and 0.0 <= s["trap_fraction"] <= 1.0


def _pool(n=40):
    # a synthetic pool: trap_fraction and share_mean correlated but not identical
    return [{"trap_fraction": i / n, "share_mean": 0.2 + 0.6 * (i / n)} for i in range(n)]


def test_threshold_is_selectivity_matched_not_chosen():
    """A looser rule accepts more and therefore delivers more; that is loosening, not improvement.

    So the operating point is set to the quantile that reproduces the shipping rule's acceptance rate, and the
    match is asserted rather than assumed.
    """
    cal = calibrate_theta(_pool(), "share_mean", shipping_max=0.10)
    assert cal["selectivity_matched"]
    assert abs(cal["matched_acceptance"] - cal["shipping_acceptance"]) <= 1 / 40 + 1e-9
    # and the threshold actually admits that fraction
    p = _pool()
    n_new = sum(1 for s in p if held(s, cal["theta"], "share_mean"))
    assert abs(n_new / len(p) - cal["shipping_acceptance"]) <= 1 / 40 + 1e-9


def test_a_grid_that_misses_the_statistic_scale_is_refused():
    """instrument.Declared refuses the grid rather than letting a mis-scaled threshold through, which is what

    happened when theta in {0.1..2.0} was declared for a quantity centred on 18.3.
    """
    off_scale = [{"trap_fraction": 0.0, "share_mean": 0.001 + 0.0001 * i} for i in range(30)]
    with pytest.raises(InstrumentError, match="misses the observed scale"):
        calibrate_theta(off_scale, "share_mean")
    assert min(SHARE_GRID) > 0.0 and max(SHARE_GRID) < 1.0


def test_comparison_is_refused_when_selectivity_cannot_be_matched():
    """A difference measured at unequal selectivity says nothing about which rule is better."""
    p = _pool(8)
    out = compare_at_equal_selectivity(p, [0.5] * 8, "share_mean")
    assert "refused" in out, "with 8 candidates and a 10% shipping rate there is nothing to compare"
