"""designaudit.mining -- recover the controlled interventions a probing corpus already contains.

Chemical-probing corpora that carry mutant libraries were not built for causal inference, but some
of the mutants they contain are controlled interventions: a wild type and a mutant that differ at
exactly two positions, those two positions being a base pair, and the substitution exchanging the
two bases so that the unordered pair type is preserved. Nothing about the rest of the molecule
changes. That is a two-nucleotide experiment with its own control, executed and measured, sitting
in data that has already been published.

This module states when a construct pair qualifies, and refuses everything else by a named reason
that is counted and reported. The criterion is the measurement protocol; it is not a filter that
can be loosened to increase yield. A corpus that yields nothing yields nothing, and `MiningReport`
says so rather than degrading quietly.

  INCLUSION      wt and mut are the same length; they differ at exactly two positions inside the
                 declared window; those two positions form a pair in the declared target
                 structure; the wild-type pair is canonical (A-U, C-G, G-U); and the substitution
                 is an orientation flip -- mut[i] == wt[j] and mut[j] == wt[i].

  EXCLUSION      every other case, under one of the reasons in `EXCLUSIONS`. Counts are reported.

  AMBIGUITY      a mutant that admits more than one wild type, or whose two changed positions are
                 explained by more than one pair in the target structure, is excluded as
                 `ambiguous` rather than resolved by a rule. Resolving it would let the corpus
                 choose which intervention it supplies.

  CONFIDENCE     each admitted intervention carries the corpus's own quality field when one is
                 present (signal-to-noise, read depth), normalised to [0, 1], and 1.0 when the
                 corpus declares none. Confidence is reported and may be used to weight, but the
                 criterion above is not a function of it -- quality never decides admission.

An orientation flip preserves the unordered pair type while changing the nearest-neighbour
stacking context. It is therefore narrow: it tests whether a model's prediction responds correctly
to a substitution, not what a base pair's identity does. That limit belongs next to every number
this module makes possible.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

CANONICAL = ("AU", "CG", "GU")

EXCLUSIONS = (
    "no_wild_type",             # no wild type declared for this construct's design cell
    "length_mismatch",          # wt and mut are not the same length
    "not_two_positions",        # differ at zero, one, or more than two positions in the window
    "positions_not_a_pair",     # the two changed positions are not paired in the target
    "pair_not_canonical",       # the wild-type pair is not A-U, C-G or G-U
    "not_orientation_flip",     # the substitution changes the unordered pair type
    "ambiguous",                # more than one wild type or more than one explaining pair
    "no_outcome",               # the corpus records no measured outcome for this construct
)


@dataclass
class Intervention:
    """One executed two-nucleotide experiment, with its own control."""
    design: tuple          # the cell the design process holds fixed (target x method, batch, ...)
    i: int                 # 0-based position, i < j
    j: int
    pair_type: str         # "AU", "CG" or "GU" -- unchanged by the flip, by construction
    wt_seq: str
    mut_seq: str
    outcome: float         # the corpus's measured response to this intervention
    confidence: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def key(self):
        return (self.design, self.i, self.j)


@dataclass
class MiningReport:
    n_input: int
    n_admitted: int
    excluded: dict
    by_pair_type: dict
    ambiguous_examples: list
    confidence_summary: dict

    @property
    def yield_rate(self):
        return self.n_admitted / self.n_input if self.n_input else 0.0

    def __repr__(self):
        head = (f"mined {self.n_admitted} interventions from {self.n_input} constructs "
                f"({100*self.yield_rate:.1f}%)  {self.by_pair_type}")
        if not self.n_admitted:
            head += "\n  ! this corpus contains no minable intervention; it cannot be scored"
        rows = "\n".join(f"    {k:24s} {v}" for k, v in sorted(self.excluded.items(),
                                                               key=lambda kv: -kv[1]) if v)
        return head + ("\n  excluded:\n" + rows if rows else "")


def _diff_positions(wt, mut, lo, hi):
    return tuple(k for k in range(lo, hi) if wt[k] != mut[k])


def mine(constructs, *, canonical=CANONICAL):
    """Recover interventions from an iterable of construct dicts.

    Each construct must carry: design, sequence, window (lo, hi) half-open 0-based, pairs (a dict
    or mapping i -> j of the declared target structure), is_wild_type, and outcome. Wild types are
    matched to mutants by their design cell.

    Returns (list[Intervention], MiningReport). The criterion is fixed; there is no threshold to
    tune and no argument that increases the yield.
    """
    constructs = list(constructs)
    wt_by_design = collections.defaultdict(list)
    for c in constructs:
        if c.get("is_wild_type"):
            wt_by_design[c["design"]].append(c)

    excluded = {k: 0 for k in EXCLUSIONS}
    admitted, ambiguous_examples, confs = [], [], []

    for c in constructs:
        if c.get("is_wild_type"):
            continue
        cands = wt_by_design.get(c["design"], [])
        if not cands:
            excluded["no_wild_type"] += 1
            continue
        if len({w["sequence"] for w in cands}) > 1:
            excluded["ambiguous"] += 1
            if len(ambiguous_examples) < 5:
                ambiguous_examples.append({"design": c["design"], "why": "multiple wild types"})
            continue
        wt = cands[0]["sequence"]
        mut = c["sequence"]
        if len(wt) != len(mut):
            excluded["length_mismatch"] += 1
            continue
        lo, hi = c.get("window", (0, len(mut)))
        d = _diff_positions(wt, mut, lo, hi)
        if len(d) != 2:
            excluded["not_two_positions"] += 1
            continue
        i, j = d
        pairs = c["pairs"]
        explaining = [(a, b) for a, b in ((i, j), (j, i)) if pairs.get(a) == b]
        if not explaining:
            excluded["positions_not_a_pair"] += 1
            continue
        if len({tuple(sorted(p)) for p in explaining}) > 1:
            excluded["ambiguous"] += 1
            if len(ambiguous_examples) < 5:
                ambiguous_examples.append({"design": c["design"], "why": "multiple explaining pairs"})
            continue
        pt = "".join(sorted({wt[i], wt[j]}))
        if pt not in canonical:
            excluded["pair_not_canonical"] += 1
            continue
        if not (mut[i] == wt[j] and mut[j] == wt[i]):
            excluded["not_orientation_flip"] += 1
            continue
        if c.get("outcome") is None:
            excluded["no_outcome"] += 1
            continue
        conf = float(c.get("confidence", 1.0))
        confs.append(conf)
        admitted.append(Intervention(design=c["design"], i=i, j=j, pair_type=pt,
                                     wt_seq=wt, mut_seq=mut, outcome=float(c["outcome"]),
                                     confidence=conf, meta=c.get("meta", {})))

    by_pt = dict(collections.Counter(x.pair_type for x in admitted))
    cs = ({"min": min(confs), "mean": sum(confs) / len(confs), "max": max(confs)}
          if confs else {})
    return admitted, MiningReport(n_input=len(constructs), n_admitted=len(admitted),
                                  excluded=excluded, by_pair_type=by_pt,
                                  ambiguous_examples=ambiguous_examples,
                                  confidence_summary=cs)
