"""Every test here reproduces a defect that actually happened, on the day it happened.

The point is not coverage. A check that cannot be traced to something that went wrong is a guess about the
future; a check that reproduces a real defect is a guarantee it cannot come back silently. Three of the ten
defects below would have shipped a false positive, and those three are marked.
"""
import pytest

from rna_harness.instrument import (Clustered, InstrumentError, Reference, check_duplex_antiparallel,
                                    check_pair_set, constancy, has_dynamic_range, run_controls,
                                    scale_then_threshold, stable_seed, tally)


# --- 1. structures that cannot exist ---------------------------------------------------------------------
def test_parallel_duplex_is_rejected():
    """WOULD HAVE FLIPPED A VERDICT. Matching last-opener-to-last-closer gave (15,0),(16,1),(17,2)...

    That is a parallel duplex. It shares zero pairs with the correct antiparallel answer, so two tools that
    agree exactly looked like they disagreed -- and the third domain's whole entry verdict (P2: do the
    predictors disagree?) turned on that comparison.
    """
    wrong = [(15, 0), (16, 1), (17, 2), (18, 3)]
    right = [(15, 11), (16, 10), (17, 9), (18, 8)]
    with pytest.raises(InstrumentError, match="parallel"):
        check_duplex_antiparallel(wrong)
    check_duplex_antiparallel(right)                      # must not raise


def test_pair_set_rejects_reused_positions_and_bad_order():
    check_pair_set([(0, 9), (1, 8)], 10)
    with pytest.raises(InstrumentError, match="matching"):
        check_pair_set([(0, 9), (0, 8)], 10)              # position 0 paired twice
    with pytest.raises(InstrumentError, match="i<j"):
        check_pair_set([(9, 0)], 10)
    with pytest.raises(InstrumentError, match="outside"):
        check_pair_set([(0, 12)], 10)
    # intermolecular: i and j index different strands, so i >= j is legal
    check_pair_set([(15, 11), (16, 10)], 45, 12)


# --- 2. tools that return a degenerate answer ------------------------------------------------------------
def test_a_tool_that_returns_nothing_fails_its_positive_control():
    """WOULD HAVE FLIPPED A VERDICT. bifold's multi-structure ct was read as one block, so every call

    returned an empty pair set -- and an empty set scores Cohen's kappa of exactly 0.000 against every other
    tool, which reads as total disagreement rather than as a broken parser.
    """
    broken = lambda *_: set()                                                  # noqa: E731
    r = run_controls("bifold", broken, ("GGGGAAAACCCC", "GGGGUUUUCCCC"), ("AAAA", "AAAA"),
                     positive_ok=lambda v: len(v) > 0, negative_ok=lambda v: len(v) == 0)
    assert not r.usable and not r.positive_ok
    working = lambda a, b: ({(0, 3), (1, 2)} if "G" in a and "C" in b else set())   # noqa: E731
    assert run_controls("ok", working, ("GGGG", "CCCC"), ("AAAA", "AAAA"),
                        lambda v: len(v) > 0, lambda v: len(v) == 0).usable


def test_a_tool_that_says_yes_to_everything_fails_the_negative_control():
    """The failure mode that reads as success: RNAPKplex in a dirty working directory, in reverse."""
    always = lambda *_: {(0, 3)}                                               # noqa: E731
    r = run_controls("always", always, ("GGGG", "CCCC"), ("AAAA", "AAAA"),
                     lambda v: len(v) > 0, lambda v: len(v) == 0)
    assert r.positive_ok and not r.negative_ok and not r.usable


def test_constancy_says_which_side_was_constant():
    """The finding inverts depending on the answer, and scipy's warning does not give it.

    17 of 22 constructs came back with Kinfold reach identically zero. Reading that as "not enough molecules"
    instead of "the judge has no dynamic range" is the difference between a sample-size problem and a
    measured ceiling on the instrument.
    """
    judge_dead = constancy([0.0] * 17)
    assert judge_dead["constant"] and judge_dead["where"] == "all-zero"
    criterion_varies = constancy([0.74, 0.83, 0.91])
    assert not criterion_varies["constant"]


def test_one_nonzero_construct_is_not_dynamic_range():
    """WOULD HAVE FLIPPED A VERDICT. "not all zero" let 1/17 print "dynamic range recovered" at sd 0.059."""
    assert not has_dynamic_range([0.0] * 16 + [0.25])
    assert has_dynamic_range([0.0, 1.0, 0.3, 0.5, 0.6, 0.2, 0.9, 0.4, 0.1])
    # SECOND VERSION, and its absence almost shipped a wrong claim. The barriers refolding judge returned
    # 0.9787..0.9979 -- every value strictly between 0 and 1, so the first version passed it -- and a
    # correlation against it came out rho=+0.169, p=0.39, which reads as "the criterion does not predict
    # refolding". The truth is a 1.9-point range, so nothing is detectable either way.
    saturated = [0.9787, 0.98, 0.985, 0.99, 0.993, 0.995, 0.997, 0.9979] * 4
    assert not has_dynamic_range(saturated), "a judge spanning 2 points has no usable range"


# --- 3. statistics at the wrong level of replication -----------------------------------------------------
def test_clustered_reports_the_honest_n():
    """20 trajectories of one sequence are not 20 observations; 12 candidates of one target are not 12."""
    c = Clustered(values=[0.5] * 240, clusters=[i // 20 for i in range(240)])
    assert c.n_obs == 240 and c.n_clusters == 12
    assert c.honest_n() == 12
    assert "report n=12" in c.warn_if_inflated()
    assert Clustered(values=[1, 2], clusters=["a", "b"]).warn_if_inflated() is None
    with pytest.raises(InstrumentError):
        Clustered(values=[1, 2, 3], clusters=["a"])


# --- 4. thresholds declared before the scale is known ----------------------------------------------------
def test_threshold_grid_that_misses_the_scale_is_refused():
    """theta in {0.1 ... 2.0} was pre-registered for a quantity whose mean on the population is 18.3.

    Every threshold then fired on everything, and the pre-registered condition failed for a reason that had
    nothing to do with the hypothesis. Raising theta afterwards would have been selective reporting, so the
    grid has to be checked against the scale BEFORE it is declared.
    """
    off_mass = [12.0, 15.0, 18.0, 21.0, 25.0]
    bad = scale_then_threshold(off_mass, [0.1, 0.25, 0.5, 1.0, 2.0])
    assert not bad["straddles_scale"] and "misses the observed scale" in bad["verdict"]
    good = scale_then_threshold(off_mass, [5.0, 15.0, 30.0])
    assert good["straddles_scale"] and good["verdict"] == "ok"


# --- 5. a reference fit to the data it scores ------------------------------------------------------------
def test_reference_derived_from_the_same_measurement_is_circular():
    """Its advantage grew from +0.007 early to -0.033 near full length -- exactly where the prefix IS the

    molecule -- because there the comparison degenerates into SHAPE-directed fold versus MFE on SHAPE data,
    which the SHAPE-directed fold wins by construction.
    """
    r = Reference("((((....))))", derived_from="FLUORSW_full_length_reactivity", quality=0.72)
    assert not r.independent_of("FLUORSW_full_length_reactivity")
    with pytest.raises(InstrumentError, match="circular"):
        r.assert_independent_of("FLUORSW_full_length_reactivity")
    r.assert_independent_of("some_other_experiment")      # must not raise
    assert r.passes_quality(0.65)
    assert not Reference("((..))", quality=0.507).passes_quality(0.65)


# --- 6. seeds that are not seeds ---------------------------------------------------------------------------
def test_stable_seed_does_not_depend_on_process_hash_randomisation():
    """random.Random(hash(str)) looked seeded and drew a different sample every run, which made the sharpest

    null for the co-transcriptional criterion non-reproducible. The value below is pinned so a refactor back
    to hash() fails here rather than silently.
    """
    assert stable_seed("16581", 0) == stable_seed("16581", 0)
    assert stable_seed("16581", 0) != stable_seed("16581", 1)
    assert stable_seed("16581", 0) == 2650798945          # pinned across processes and machines


# --- 7. a score that disagrees with its own table ---------------------------------------------------------
def test_tally_counts_the_table_instead_of_the_author():
    """Goal 4's fraction disagreed with its own table in two consecutive weekly reports, low by one each."""
    rows = ["a ✅", "b ✅", "c ❌", "d ✅"]
    t = tally(rows)
    assert t == {"yes": 3, "no": 1, "total": 4, "fraction": 0.75}
    with pytest.raises(InstrumentError, match="exactly one verdict"):
        tally(["a ✅", "b (未测)"])


# --- 8. a binary gate on a continuous statistic ----------------------------------------------------------
def test_a_gate_decided_within_noise_is_reported_as_a_knife_edge():
    """Found by rerunning the same entry check at two sample sizes rather than by reasoning about it.

    P2's kill rule is "every pairwise kappa >= 0.80". At 60 pairs IntaRNA~bifold was 0.799 and P2 read as
    PASSED; at 120 pairs the same quantity was 0.804 and P2 read as FAILED. Nothing about the domain changed.
    The overall entry verdict survived only because P3 failed either way -- luck, not design.
    """
    from rna_harness.instrument import gate_margin
    edge = gate_margin([0.80, 0.79, 0.81, 0.80, 0.79, 0.82, 0.78, 0.81], 0.80)
    assert not edge["decisive"] and "knife edge" in edge["verdict"]
    clear = gate_margin([0.95, 0.96, 0.94, 0.97, 0.95, 0.96], 0.80)
    assert clear["decisive"] and clear["verdict"] == "clear" and clear["margin"] > 0.1


# --- 9. a rate matrix used as a master-equation generator -------------------------------------------------
def test_generator_checks_orientation_and_irreducibility():
    """Two defects found ten minutes apart while trying to build a second kinetic judge out of `barriers`.

    Orientation: rates.out is [source][target] with a nonzero self-rate on the diagonal; read the other way
    the populations look plausible and are wrong. Detailed balance decides it -- median log deviation 0.019
    one way, 8.792 the other.

    Reducibility: at --max 12 the matrix came back in two disconnected components with the open chain in one
    and the ground state in the other, so the long-time limit is Boltzmann WITHIN the start's component.
    Comparing against global Boltzmann made a correct solver look broken.
    """
    import numpy as np
    from rna_harness.instrument import check_generator
    RT = 0.0019872 * 310.15
    G = [-10.0, -8.0, -6.0]
    R = np.zeros((3, 3))
    R[0, 1] = 1.0                                          # 0 -> 1
    R[1, 0] = R[0, 1] * np.exp(-(G[0] - G[1]) / RT)        # detailed balance
    R[1, 2] = 1.0
    R[2, 1] = R[1, 2] * np.exp(-(G[1] - G[2]) / RT)
    ok = check_generator(R, G, start=2)
    assert ok["irreducible"] and ok["orientation_ok"] and ok["detailed_balance_err"] < 1e-6

    with pytest.raises(InstrumentError, match="wrong direction"):
        check_generator(R.T, G, start=2)

    split = np.zeros((4, 4))
    split[0, 1] = 1.0
    split[1, 0] = split[0, 1] * np.exp(-(G[0] - G[1]) / RT)
    split[2, 3] = split[3, 2] = 1.0                        # a second, disconnected component
    d = check_generator(split, G + [-6.0], start=2)
    assert not d["irreducible"] and d["n_components"] == 2 and "by construction" in d["warning"]


# --- 10. the checks that were written and then not used --------------------------------------------------
def test_a_control_that_cannot_fail_is_rejected():
    """Both negative controls I wrote today passed for reasons unrelated to discrimination.

    poly-A made the pipeline return None and the predicate accepted None as a pass -- a completely broken
    install would have passed identically. The second attempt used a "foreign" target that turned out to be
    that sequence's own MFE and scored 0.88.
    """
    from rna_harness.instrument import assert_control_can_fail
    too_permissive = lambda v: True                                            # noqa: E731
    with pytest.raises(InstrumentError, match="cannot fail"):
        assert_control_can_fail(too_permissive, [None, {}, {"judge": 1.0}])
    # the real defect: accepting None, which means "the pipeline did not run"
    accepts_none = lambda v: (not v) or v.get("judge", 1) < 0.05               # noqa: E731
    with pytest.raises(InstrumentError, match="cannot fail"):
        assert_control_can_fail(accepts_none, [None])
    strict = lambda v: bool(v) and v.get("judge", 1) < 0.05                    # noqa: E731
    assert_control_can_fail(strict, [None, {}, {"judge": 1.0}])                # must not raise


def test_a_threshold_cannot_be_used_before_its_scale_is_measured():
    """SUBOPT_E = 8.0 and theta in {0.1..2.0} were both declared without calling scale_then_threshold,

    which already existed. Writing a check and not using it is indistinguishable from not having it.
    """
    from rna_harness.instrument import Declared
    d = Declared("theta", [0.1, 0.25, 0.5, 1.0, 2.0])
    with pytest.raises(InstrumentError, match="before its scale"):
        _ = d.value
    with pytest.raises(InstrumentError, match="misses the observed scale"):
        _ = d.against([12.0, 18.0, 24.0]).value
    ok = Declared("share", [0.2, 0.5, 0.8]).against([0.47, 0.56, 0.64])
    assert list(ok.value) == [0.2, 0.5, 0.8] and ok.scale["straddles_scale"]


def test_entry_population_must_be_the_deployment_population():
    """P2 said "measure whether they disagree" and never said on which population, and the population decided

    the answer: on perfect-complement triggers all six kappas cleared 0.80 and strict already delivered 45.8%.
    """
    from rna_harness.instrument import assert_population_is_deployment
    assert_population_is_deployment("natural sRNA-mRNA pairs", "natural sRNA-mRNA pairs")
    with pytest.raises(InstrumentError, match="declare the deployment population"):
        assert_population_is_deployment("perfect-complement triggers", "natural sRNA-mRNA pairs")
