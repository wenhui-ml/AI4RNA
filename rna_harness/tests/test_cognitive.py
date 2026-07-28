"""Foundation tests for the domain-agnostic cognitive core: cost accounting, mechanical features,
correct-oracle verification, and the uniform tool interface (cost-tagged, verified)."""
import RNA

from rna_harness.cognitive.cost import solver_cost_usd, llm_cost_usd
from rna_harness.cognitive.features import features, FEATURE_NAMES
from rna_harness.cognitive.tools import RemcTool, verify

HAIRPIN = "((((....))))"
PK = "((((....[[[[))))....]]]]"   # a simple pseudoknot (crossing [] over ())


def test_cost_monotonic_and_positive():
    assert solver_cost_usd(10, cores=8) > solver_cost_usd(10, cores=1) > 0
    assert llm_cost_usd(1000, 1000, "z-ai/glm-5.2") > 0
    assert solver_cost_usd(0, 8) == 0.0


def test_features_mechanical_hairpin():
    f = features(HAIRPIN)
    assert f["len"] == 12 and f["n_pairs"] == 4 and f["has_pseudoknot"] == 0.0
    assert f["n_hairpins"] == 1 and f["nesting_depth"] == 4
    assert set(f.keys()) == set(FEATURE_NAMES)   # stable vector


def test_features_pseudoknot_robust():
    f = features(PK)                              # must not crash on extended dot-bracket
    assert f["has_pseudoknot"] == 1.0 and f["n_pk_brackets"] > 0
    assert f["len"] == len(PK)


def test_verify_mfe_gate():
    # a canonical GC hairpin folds to the target
    seq = "GGGGAAAACCCC"
    st, _ = RNA.fold(seq)
    solved, bp, oracle = verify(seq, HAIRPIN)
    assert oracle == "mfe"
    assert solved == (st == HAIRPIN)
    # a nonsense seq is rejected
    s2, b2, _ = verify("A" * 12, HAIRPIN)
    assert s2 is False and b2 > 0


def test_verify_strictness_axes():
    """Axis 2: the same sequence gets different truth values under stricter declared predicates."""
    T = "(((((....)))))"                       # 5bp-stem tetraloop hairpin
    robust = "GCGCGAAAACGCGC"                  # folds to T with high ensemble probability
    knife = "GGGGGGAAUUUCUC"                   # folds to T but P(T)~0.19 -- a fragile 1-of-many
    # robust: passes all three strictness levels
    assert verify(robust, T, "match")[0] is True
    assert verify(robust, T, "umfe")[2] == "mfe:umfe" and verify(robust, T, "umfe")[0] is True
    assert verify(robust, T, "pfe")[2] == "mfe:pfe" and verify(robust, T, "pfe")[0] is True
    # knife-edge: exact-match says solved, but the strict ensemble predicate rejects it
    assert verify(knife, T, "match")[0] is True
    assert verify(knife, T, "pfe")[0] is False   # P(T) below _PFE_MIN -> not a robust solution
    # unknown strictness is a hard error, not a silent default
    import pytest
    with pytest.raises(ValueError):
        verify(robust, T, "bogus")


def test_remc_tool_pseudoknot_fails_gracefully():
    """Regression: REMC (MFE-only, nested warm-start) must NOT crash on a pseudoknot target -- it designs
    the nested skeleton and pKiss judges the full target (a graceful failure, not a ValueError on '[')."""
    r = RemcTool(wall_s=2).attempt(PK, seed=0)     # PK has crossing [] over ()
    assert r.tool.startswith("remc@")              # returned a ToolResult, did not raise
    assert r.solved is False                       # REMC cannot form the crossing stem
    assert r.oracle in ("pkiss", "none")           # judged by the pseudoknot-aware oracle, never 'mfe'


def test_remc_tool_solves_small_and_costs():
    r = RemcTool(wall_s=10, n_replicas=6).attempt(HAIRPIN, seed=0)
    assert r.tool.startswith("remc@")
    assert r.cores == 6 and r.wall_s >= 0 and r.cost_usd > 0   # cost>0 is the real cost-tracking check
    assert r.solved is True and r.bp == 0 and r.oracle == "mfe"
    assert r.seq is not None and RNA.fold(r.seq)[0] == HAIRPIN


# ---- router + core ----------------------------------------------------------
from rna_harness.cognitive.tools import default_toolbox, DesirnaTool, RemcTool as _R
from rna_harness.cognitive.router import Router
from rna_harness.cognitive.core import CognitiveCore, FixedPolicy


def test_router_cold_start_cheapest_first():
    tb = default_toolbox()
    order = Router(tb).plan(features(HAIRPIN))
    costs = [t.nominal_cost_usd() for t in order]
    assert costs == sorted(costs)                 # cold-start = ascending nominal cost
    assert order[0].nominal_cost_usd() == min(costs)


def test_router_orders_by_cost_per_unit_success():
    """The router's learned order is by $ / P(solve), not by P(solve). At EQUAL cost the reliable rung must
    come first; a much cheaper rung legitimately stays ahead even at low P, because in a cascade the price of
    trying it and failing is exactly its cost. (An earlier version of this test asserted "prefer the reliable
    one" using a 2000x cost ratio, which asserted the wrong thing -- the router was right.)"""
    tb = default_toolbox()
    rt = Router(tb)
    f = features(HAIRPIN)
    a, b = tb[0].name, tb[1].name
    for _ in range(6):                              # SAME cost, different outcomes
        rt.record(f, a, solved=False, cost=1e-3)
        rt.record(f, b, solved=True, cost=1e-3)
    assert rt.plan(f)[0].name == b                  # at equal cost, the one that works goes first
    rt2 = Router(tb)
    for _ in range(6):                              # now make the loser 1000x cheaper
        rt2.record(f, a, solved=False, cost=1e-6)
        rt2.record(f, b, solved=True, cost=1e-3)
    assert rt2.plan(f)[0].name == a                 # trying it costs almost nothing, so it is tried first


def test_cognitive_core_stops_at_first_verified_solve():
    """The cascade MECHANISM, exercised on the built-in solver so the unit test needs no external binary."""
    core = CognitiveCore([_R(wall_s=5), _R(wall_s=30), DesirnaTool(timelimit_s=180)], learn=True)
    out = core.solve(HAIRPIN, seed=0)
    assert out["solved"] is True and out["cost_usd"] > 0
    assert out["trace"][0]["tool"] == "remc@5s"                   # tried cheapest first
    assert len(out["trace"]) == 1                                 # and stopped there
    assert all("desirna" not in s["tool"] for s in out["trace"])  # never needed SOTA on an easy one


def test_default_toolbox_is_the_measured_ladder_or_the_fallback():
    """default_toolbox() must be the DesiRNA budget ladder when the solver is installed, and the REMC ladder
    when it is not -- never empty. The measured basis is in the function's own docstring; the point here is
    that a missing external solver degrades instead of leaving the router with nothing to fire."""
    from rna_harness.generators.desirna import desirna_available
    tb = default_toolbox()
    assert tb, "default_toolbox must never be empty"
    names = [t.name for t in tb]
    if desirna_available():
        assert all(n.startswith("desirna@") for n in names), names
        assert [t.budget_s for t in tb] == [5.0, 15.0, 60.0, 180.0], names
    else:
        assert all(n.startswith("remc@") for n in names), names
    assert [t.nominal_cost_usd() for t in tb] == sorted(t.nominal_cost_usd() for t in tb)


def test_panel_abstains_when_an_oracle_cannot_run(monkeypatch):
    """A missing panel oracle must ABSTAIN, not reject. Before this, forms_target(None, ...) -> None was read
    as 'rejected', so with RNAstructure absent every candidate failed and the tool reported 0 solves with no
    diagnostic."""
    from rna_harness import oracles_pk
    monkeypatch.setattr(oracles_pk, "probknot_pairs", lambda seq, timeout_s=60.0: None)
    monkeypatch.setattr(oracles_pk, "shapeknots_pairs", lambda seq, timeout_s=120.0: None)
    t = DesirnaTool(timelimit_s=5, panel_screen=True)
    r = t.attempt(PK, seed=0)
    if r.available is False:
        pytest.skip("DesiRNA not installed")
    # It may or may not solve PK in 5 s; what must NOT happen is a silent panel verdict with no flag.
    assert r.oracle != "panel:pkiss+probknot+shapeknots" or r.solved is False
    assert r.stats.get("panel_unavailable") is True or r.stats.get("n_unanimous", 0) == 0
