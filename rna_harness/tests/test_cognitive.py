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


def test_router_learns_to_prefer_cheap_winner():
    tb = default_toolbox()
    rt = Router(tb)
    f = features(HAIRPIN)
    for _ in range(6):                            # cheap tool reliably solves this feature region
        rt.record(f, "remc@5s", solved=True, cost=1e-6)
    for _ in range(6):                            # an expensive tool also solves but dear
        rt.record(f, "desirna@180s", solved=True, cost=2e-3)
    order = rt.plan(f)
    assert order[0].name == "remc@5s"             # learned: put the cheap reliable winner first


def test_cognitive_core_solves_cheap_and_stops_early():
    core = CognitiveCore(default_toolbox(), learn=True)
    out = core.solve(HAIRPIN, seed=0)
    assert out["solved"] is True and out["cost_usd"] > 0
    assert out["trace"][0]["tool"] == "remc@5s"   # tried cheapest first
    assert all("desirna" not in s["tool"] for s in out["trace"])  # never needed SOTA on an easy one


def test_fixed_policy_always_cheap_baseline():
    out = FixedPolicy([_R(wall_s=10, n_replicas=6)], name="always-cheap").solve(HAIRPIN)
    assert out["solved"] is True and out["cost_usd"] > 0
