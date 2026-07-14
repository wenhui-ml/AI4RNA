"""Unit tests for the structural-retrieval library. Fast (tiny targets, tiny search budgets).
Bug-rejection focus: oracle gate, leakage guard, exact-key round-trip, cross-target unit transfer,
determinism, canonical-prior validity, and persistence."""
import RNA

from rna_harness.experiments.retrieval import (
    RetrievalLibrary, element_bag, bag_distance, _canonical_unit_seq)
from rna_harness.mcp.finisher import _fold, parallel_inverse_fold
from rna_harness.mcp.motif_kb import decompose_structure

HAIRPIN = "((((....))))"          # 12 nt, one leaf unit spanning the whole target


def _solve_small(target, n_restarts=3, tries=25):
    r = parallel_inverse_fold(target, n_restarts=n_restarts, tries=tries)
    s = r.get("seq")
    assert s and _fold(s) == target, f"test setup could not solve {target!r}"
    return s


def test_oracle_gate_rejects_nonsolve():
    lib = RetrievalLibrary()
    res = lib.add_solution(HAIRPIN, "A" * len(HAIRPIN))     # all-A cannot fold to a hairpin
    assert res["added"] is False and res["reason"] == "not-a-verified-solve"
    assert lib.stats()["n_unit_seqs"] == 0


def test_length_mismatch_guard():
    lib = RetrievalLibrary()
    assert lib.add_solution(HAIRPIN, "ACGU")["added"] is False


def test_leakage_guard():
    lib = RetrievalLibrary()
    lib.forbid([HAIRPIN])
    res = lib.add_solution(HAIRPIN, _solve_small(HAIRPIN))
    assert res["added"] is False and res["reason"] == "eval-leakage-guard"
    assert lib.stats()["n_unit_seqs"] == 0


def test_unit_roundtrip_and_exact_hit():
    lib = RetrievalLibrary()
    s = _solve_small(HAIRPIN)
    r = lib.add_solution(HAIRPIN, s)
    assert r["added"] and r["units"] >= 1
    seed, cov, meta = lib.warm_start(HAIRPIN)
    assert cov == 1.0                       # the single unit shape is now known
    assert meta["assembled_bp"] == 0        # retrieved seed folds exactly to target
    assert _fold(seed) == HAIRPIN


def test_determinism():
    lib = RetrievalLibrary()
    lib.add_solution(HAIRPIN, _solve_small(HAIRPIN))
    assert lib.warm_start(HAIRPIN)[0] == lib.warm_start(HAIRPIN)[0]


def test_shared_unit_transfers_across_targets():
    lib = RetrievalLibrary()
    t1 = "((((....))))"
    t2 = "...((((....))))..."     # same hairpin unit, different whole target
    lib.add_solution(t1, _solve_small(t1))
    units = [u["sub_target"] for u in decompose_structure(t2)["units"]]
    assert "((((....))))" in units          # t2's leaf unit shares t1's shape
    seed, cov, meta = lib.warm_start(t2)
    assert cov == 1.0                        # ...so retrieval hits it, no fresh search


def test_bag_distance_properties():
    b = element_bag(HAIRPIN)
    assert bag_distance(b, b) == 0.0
    assert bag_distance(b, {"nonexistent": 1}) == 1.0


def test_canonical_unit_seq_is_designable():
    s = _canonical_unit_seq(HAIRPIN)
    assert len(s) == len(HAIRPIN)
    assert _fold(s) == HAIRPIN               # GC stem + GAAA cap folds to the hairpin


def test_clamp_boundary_stiffens_hit_units():
    lib = RetrievalLibrary()
    s = _solve_small(HAIRPIN)
    lib.add_solution(HAIRPIN, s)
    seed, cov, _ = lib.warm_start(HAIRPIN, clamp_hits=True)
    assert cov == 1.0
    assert seed[0] == "G" and seed[-1] == "C"     # terminal pair GC-clamped
    assert _fold(seed) == HAIRPIN                  # still designable after clamp


def test_save_load_roundtrip(tmp_path):
    lib = RetrievalLibrary()
    lib.add_solution(HAIRPIN, _solve_small(HAIRPIN))
    p = str(tmp_path / "lib.json")
    lib.save(p)
    lib2 = RetrievalLibrary.load(p)
    assert lib2.stats()["n_unit_seqs"] == lib.stats()["n_unit_seqs"]
    assert lib2.warm_start(HAIRPIN)[1] == 1.0
