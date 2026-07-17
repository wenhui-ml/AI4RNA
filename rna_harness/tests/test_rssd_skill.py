"""Lock the DecomposePKTool skill's contract (docs/RSSD_design.md):
 * decomposition is a correct, disjoint partition (nested skeleton + crossing pairs);
 * the mutation neighbourhood widens the raw mismatch by a halo (so a single missing terminal pair is
   fixable -- the bug the first prototype hit);
 * the tool ABSTAINS cheaply on a non-pseudoknot target (nothing to add there);
 * a full attempt on a small pseudoknot is oracle-honest: solved => pKiss bp == 0 and a sequence is set.
The end-to-end test is skipped when pKiss is unavailable (keeps the suite green off the pk env)."""
import os

import pytest

from rna_harness.benchmarks.rnainvbench import pk_pairs, pkiss_available
from rna_harness.cognitive.skills.pk_decompose import (DecomposePKTool, _partner_map,
                                                       _wrong_positions)

PK = "((.((((.....[[[[))))..))......]]]]"   # a real pseudoknot (crossing [] over ())


def test_decomposition_is_disjoint_partition():
    tgt = PK
    tp = pk_pairs(tgt)
    skel = "".join(c if c in "()." else "." for c in tgt)
    sp = pk_pairs(skel)
    cross = tp - sp
    # skeleton pairs are exactly the () pairs; crossing pairs are exactly the []{} pairs; disjoint, union=all
    assert sp and cross and (sp & cross) == set()
    assert sp | cross == tp
    # every position is in at most one pair (extended dot-bracket invariant the compose step relies on)
    partner = _partner_map(tp)
    assert len(partner) == 2 * len(tp)


def test_wrong_positions_halo_widens_single_missing_pair():
    # target has pair (0,9); folded is MISSING it (no spurious pair) -> raw mismatch = {0,9}, but the
    # halo must add neighbours so the walk can strengthen the local context, not just poke 0 and 9.
    target_pairs = {(0, 9), (1, 8)}
    folded_pairs = {(1, 8)}
    raw = {0, 9}
    got = set(_wrong_positions(folded_pairs, target_pairs, n=10, halo=2))
    assert raw <= got                       # includes the mismatch itself
    assert got - raw                        # AND strictly more (the halo)
    assert all(0 <= p < 10 for p in got)    # clamped to sequence bounds


def test_abstains_on_nested_target_cheaply():
    tool = DecomposePKTool(repair_wall_s=1)
    r = tool.attempt("((((....))))", seed=0)   # no pseudoknot
    assert r.available is True and r.solved is False
    assert r.cost_usd == 0.0 and r.oracle == "none"   # abstained without touching pKiss


@pytest.mark.skipif(not pkiss_available(), reason="pKiss not installed (conda env 'pk')")
def test_end_to_end_is_oracle_honest():
    tool = DecomposePKTool(skel_wall_s=3, repair_wall_s=8, max_repair_iters=20, skel_restarts=2)
    r = tool.attempt(PK, seed=0)
    assert r.oracle == "pkiss"
    assert r.bp is not None and r.bp >= 0
    if r.solved:                                # a claimed solve must be a REAL pKiss bp==0 with a sequence
        assert r.bp == 0 and r.seq and len(r.seq) == len(PK)
    else:
        assert r.seq is None                    # never emit a sequence for an unsolved target
