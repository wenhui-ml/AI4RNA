"""Instrument discipline, as code rather than as habit.

WHY THIS FILE EXISTS. In one day this project found ten defects that were NOT wrong hypotheses -- they were
wrong measurements -- and three of them would have shipped a false positive:

    ct2dot writes header + sequence + structure; concatenating non-header lines doubled the length and every
      construct was reported as "the tool did not run", a tool failure that never happened
    no reference-quality gate; five references could not explain the data they were fit to (worst 0.507) and
      the headline moved p=0.049 -> p=0.109 once they were excluded
    the alignment offset was chosen by maximising the AUROC of the model under test -- letting the accused
      pick the jury
    bracket matching last-opener-to-last-closer produced a PARALLEL duplex, physically impossible, and it
      fabricated a disagreement between two tools that agree exactly           <- would have flipped P2
    bifold's multi-structure ct read as one block, so every pair came back empty, and an empty set scores
      kappa exactly 0.000 against everyone, which reads as total disagreement  <- would have flipped P2
    Kinfold's X marker undercounts by 6.5 points; every Kinfold number in the project was the marker
    "usable neighbourhood" required only "not all zero", so 1 nonzero construct in 17 printed
      "dynamic range recovered"                                                <- would have flipped a verdict
    a pre-registered threshold grid declared without knowing the scale, low by an order of magnitude
    the goal-4 fraction disagreed with its own table, in two consecutive weekly reports
    random.Random(hash(str)) -- Python randomises str hashing per process, so a "seeded" construction drew a
      different sample every run

The hypothesis-level failure rate here is far lower than the instrument-level failure rate. That means the
binding constraint is measurement integrity, and every future direction inherits the same defect rate unless
the checks stop being habits and become assertions.

Each function below corresponds to a class of defect above, and each has a regression test that reproduces
the original defect. Nothing here is generic hygiene: if a check cannot be traced to something that actually
went wrong, it does not belong in this file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


class InstrumentError(AssertionError):
    """Raised when a measurement is not merely wrong but impossible. Never caught to 'keep going'."""


# ---------------------------------------------------------------------------------------------------------
# 1. structures that cannot exist
# ---------------------------------------------------------------------------------------------------------
def check_pair_set(pairs: Iterable[Tuple[int, int]], n_left: int, n_right: Optional[int] = None,
                   name: str = "pairs") -> List[Tuple[int, int]]:
    """A parsed pair set must be a matching on distinct in-range positions. Returns it sorted.

    Intramolecular: pass n_left only; indices live in [0, n_left) and i < j.
    Intermolecular: pass both; i indexes the left strand, j the right, and there is no ordering between them.

    The bug this catches: a parser that pops a stack in the wrong order still produces plausible-looking
    tuples, so nothing downstream complains -- it just quietly answers a different question.
    """
    P = sorted((int(i), int(j)) for i, j in pairs)
    seen_l, seen_r = set(), set()
    for i, j in P:
        if not (0 <= i < n_left):
            raise InstrumentError(f"{name}: left index {i} outside [0,{n_left})")
        hi = n_left if n_right is None else n_right
        if not (0 <= j < hi):
            raise InstrumentError(f"{name}: right index {j} outside [0,{hi})")
        if n_right is None and i >= j:
            raise InstrumentError(f"{name}: intramolecular pair ({i},{j}) is not i<j")
        if i in seen_l or j in seen_r:
            raise InstrumentError(f"{name}: position reused -- ({i},{j}) is not a matching")
        seen_l.add(i)
        seen_r.add(j)
    return P


def check_duplex_antiparallel(pairs: Sequence[Tuple[int, int]], name: str = "duplex") -> None:
    """Two strands hybridise antiparallel: as i increases along the left strand, j must decrease.

    THE DEFECT THIS EXISTS FOR. Matching the last opener to the last closer instead of LIFO yields
    (15,0),(16,1),(17,2)... -- a parallel duplex. RNA does not do that. It shared zero pairs with the
    correctly parsed answer, which made two tools that agree exactly look like they disagreed, and the entire
    third-domain entry verdict turned on exactly that comparison.
    """
    P = sorted(pairs)
    for (i1, j1), (i2, j2) in zip(P, P[1:]):
        if i2 > i1 and j2 > j1:
            raise InstrumentError(
                f"{name}: ({i1},{j1}) then ({i2},{j2}) runs parallel; a duplex must be antiparallel")


# ---------------------------------------------------------------------------------------------------------
# 2. tools that return a degenerate answer, read as a finding
# ---------------------------------------------------------------------------------------------------------
@dataclass
class ControlResult:
    tool: str
    positive_ok: bool
    negative_ok: bool
    positive_value: object = None
    negative_value: object = None
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.positive_ok and self.negative_ok


def run_controls(tool: str, fn: Callable, positive: tuple, negative: tuple,
                 positive_ok: Callable[[object], bool],
                 negative_ok: Callable[[object], bool]) -> ControlResult:
    """Run a positive and a negative control EVERY time the tool is used, not once during development.

    Two separate defects motivate this. RNAPKplex silently returns nested-only structures when its working
    directory holds leftovers -- 0/3 on natural pseudoknots instead of 12/14 -- which would have been written
    up as "the external judge rejects us entirely". bifold returned an empty pair set on every input because
    of a parse bug, and an empty set scores kappa exactly 0.000 against every other tool, i.e. it looks like
    perfect disagreement.

    A tool that fails BOTH controls is broken. A tool that passes the positive and fails the negative is
    answering yes to everything, which is the failure mode that reads as success.
    """
    try:
        pv = fn(*positive)
    except Exception as e:                                                     # noqa: BLE001
        return ControlResult(tool, False, False, note=f"positive raised {e!r}"[:120])
    try:
        nv = fn(*negative)
    except Exception as e:                                                     # noqa: BLE001
        return ControlResult(tool, bool(positive_ok(pv)), False, pv, note=f"negative raised {e!r}"[:120])
    return ControlResult(tool, bool(positive_ok(pv)), bool(negative_ok(nv)), pv, nv)


def assert_control_can_fail(predicate: Callable[[object], bool], degenerate: Sequence[object],
                            name: str = "control") -> None:
    """A control that cannot fail is not a control. Prove the predicate rejects something before trusting it.

    BOTH OF MY NEGATIVE CONTROLS TODAY PASSED FOR REASONS UNRELATED TO DISCRIMINATION, and that is what this
    exists for -- not a hypothetical.

      poly-A gave RNAsubopt one structure, barriers found zero saddles and wrote no rates.out, so the judge
      returned None -- and the predicate accepted None as a pass. A completely broken install would have
      "passed" identically. None means the pipeline did not run; it is not evidence about folding.

      The second attempt used a "foreign" sequence and target of the same length. It scored 0.88, because the
      target I picked happened to be that sequence's own MFE.

    Pass the degenerate values the predicate MUST reject -- None, an empty result, a saturated result. If it
    accepts all of them, the control is decorative and the run stops here rather than producing a number.
    """
    accepted = []
    for d in degenerate:
        try:
            if predicate(d):
                accepted.append(d)
        except Exception:                                                      # noqa: BLE001
            pass                                    # raising on garbage is a legitimate rejection
    if len(accepted) == len(degenerate):
        raise InstrumentError(
            f"{name}: predicate accepted every degenerate input {accepted!r} -- a control that cannot fail "
            f"is not a control")


@dataclass
class Declared:
    """A threshold that refuses to be used unless it was built against the quantity's measured scale.

    scale_then_threshold already existed and I declared SUBOPT_E = 8.0 and theta in {0.1..2.0} without calling
    it -- twice, on the same defect class. Writing a check and not using it is indistinguishable from not
    having it, so the value is now unreachable until the scale has been measured.
    """
    name: str
    grid: Sequence[float]
    _scale: Optional[Dict[str, object]] = None

    def against(self, sample: Sequence[float]) -> "Declared":
        self._scale = scale_then_threshold(sample, self.grid, name=self.name)
        return self

    @property
    def value(self) -> Sequence[float]:
        if self._scale is None:
            raise InstrumentError(
                f"{self.name}: threshold used before its scale was measured -- call .against(sample) first")
        if not self._scale["straddles_scale"]:
            raise InstrumentError(f"{self.name}: {self._scale['verdict']}")
        return self.grid

    @property
    def scale(self) -> Dict[str, object]:
        if self._scale is None:
            raise InstrumentError(f"{self.name}: scale not measured")
        return self._scale


def assert_population_is_deployment(entry: str, deployment: str, name: str = "population") -> None:
    """The population an entry check runs on must be the population the criterion will sign for.

    The protocol's P2 said "measure whether the predictors disagree" and never said ON WHICH POPULATION, and
    the answer turned out to be decided by it: on perfect-complement triggers -- the easiest case in the
    domain -- all six pairwise kappas cleared 0.80 and the strict rule already delivered 45.8%. That is in the
    protocol as prose; this puts it in code.
    """
    if entry != deployment:
        raise InstrumentError(
            f"{name}: entry check ran on '{entry}' but the criterion will sign for '{deployment}' -- "
            f"declare the deployment population and test on it, or state the mismatch as a scope limit")


def constancy(values: Sequence[float]) -> Dict[str, object]:
    """Which side is constant, and at which end -- because a warning that omits this inverts the finding.

    scipy's ConstantInputWarning does not say which array was constant. When the CRITERION is constant it
    cannot rank the population; when the OUTCOME is constant the JUDGE cannot. Those are opposite
    conclusions, and reading one as the other is how "not enough molecules" got written down when the truth
    was "Kinfold reaches nothing at this size".
    """
    v = [float(x) for x in values]
    if not v:
        return {"constant": True, "where": "empty", "value": None, "n": 0}
    lo, hi = min(v), max(v)
    if lo == hi:
        where = "all-zero" if lo == 0 else ("all-one" if lo == 1 else "constant")
        return {"constant": True, "where": where, "value": lo, "n": len(v)}
    return {"constant": False, "where": "varies", "value": None, "n": len(v),
            "n_strictly_between": sum(1 for x in v if 0 < x < 1)}


def has_dynamic_range(values: Sequence[float], min_fraction_between: float = 1 / 3,
                      min_count: int = 3, min_spread: float = 0.10) -> bool:
    """A judge is usable only when a real share of the population sits strictly between the extremes AND the
    spread is large enough to detect anything.

    "Not all zero" is far too weak: one nonzero construct out of seventeen gives sd 0.059 and was reported as
    dynamic range recovered. Both degenerate directions are equally useless -- a judge that says 1.0 for
    everything tells you nothing that one saying 0.0 does not.

    THE SPREAD REQUIREMENT IS THE SECOND VERSION, AND ITS ABSENCE ALMOST SHIPPED A WRONG CLAIM. The barriers
    refolding judge returned populations spanning 0.9787 to 0.9979 -- 28 of 28 "strictly between 0 and 1", so
    the first version passed it -- and a correlation against it came out rho=+0.169, p=0.39. Reported as
    written that would have read "the criterion does not predict refolding". The truth is that the judge has
    a 1.9-percentage-point range on this population, so NO correlation is detectable either way. Compare
    Kinfold on the same designs: 10th percentile 0.27, i.e. a usable spread.
    """
    v = [float(x) for x in values]
    if not v:
        return False
    between = sum(1 for x in v if 0 < x < 1)
    if between < max(min_count, math.ceil(min_fraction_between * len(v))):
        return False
    q = sorted(v)
    lo, hi = q[max(0, int(0.10 * len(q)) - 1)], q[min(len(q) - 1, int(0.90 * len(q)))]
    return (hi - lo) >= min_spread


# ---------------------------------------------------------------------------------------------------------
# 3. statistics computed at the wrong level of replication
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Clustered:
    """Values that carry their cluster labels, so nobody can compute an interval at the wrong level.

    Twenty Kinfold trajectories of one sequence are not twenty independent observations, and twelve
    candidates of one target are not twelve independent sequences. Trajectory-pooled Clopper-Pearson
    intervals are precisely why the first co-transcriptional result looked so cleanly significant.
    """
    values: List[float] = field(default_factory=list)
    clusters: List[object] = field(default_factory=list)

    def __post_init__(self):
        if len(self.values) != len(self.clusters):
            raise InstrumentError("Clustered: values and clusters differ in length")

    @property
    def n_obs(self) -> int:
        return len(self.values)

    @property
    def n_clusters(self) -> int:
        return len(set(self.clusters))

    def honest_n(self) -> int:
        """The n a reader should be shown. Never n_obs when clusters are far fewer."""
        return self.n_clusters

    def inflation(self) -> float:
        return self.n_obs / max(1, self.n_clusters)

    def warn_if_inflated(self, factor: float = 2.0) -> Optional[str]:
        if self.inflation() >= factor:
            return (f"{self.n_obs} observations from only {self.n_clusters} clusters "
                    f"({self.inflation():.1f}x); report n={self.n_clusters}, not n={self.n_obs}")
        return None


# ---------------------------------------------------------------------------------------------------------
# 4. thresholds declared before the scale is known
# ---------------------------------------------------------------------------------------------------------
def scale_then_threshold(sample: Sequence[float], grid: Sequence[float],
                         name: str = "threshold") -> Dict[str, object]:
    """Refuse a threshold grid that does not straddle the quantity's actual scale.

    The defect: theta in {0.1 ... 2.0} was pre-registered for off-target probability mass whose mean on the
    relevant population is 18.3. Every threshold fired on everything, the pre-registered condition failed for
    a reason that had nothing to do with the hypothesis, and raising theta afterwards would have been
    selective reporting. Measuring the scale first costs one pass over a sample and makes the declaration
    meaningful instead of arbitrary.
    """
    v = sorted(float(x) for x in sample)
    if not v:
        raise InstrumentError(f"{name}: cannot declare a threshold against an empty sample")
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]                          # noqa: E731
    lo, hi = q(0.10), q(0.90)
    g = sorted(float(x) for x in grid)
    straddles = g[0] <= hi and g[-1] >= lo
    return {"p10": lo, "median": q(0.50), "p90": hi, "grid": g, "straddles_scale": bool(straddles),
            "verdict": ("ok" if straddles else
                        f"grid {g[0]}..{g[-1]} misses the observed scale {lo:.3g}..{hi:.3g}")}


# ---------------------------------------------------------------------------------------------------------
# 5. a reference fit to the data it is scored against
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Reference:
    """A reference structure plus where it came from, so circular scoring cannot be silent.

    The defect: the reference was produced by SHAPE-directed folding on the FULL-LENGTH reactivity and then
    scored against per-length reactivity from the same experiment. Its advantage over the model under test
    grew from +0.007 early in transcription to -0.033 near full length -- exactly where a prefix IS almost
    the whole molecule -- so the comparison degenerated into "SHAPE-directed fold beats MFE on SHAPE data",
    which it wins by construction. The artefact appeared precisely where the claimed effect did.
    """
    structure: str
    derived_from: Optional[str] = None      # id of the measurement used to build it, if any
    quality: Optional[float] = None         # how well it explains the data it was fit to

    def independent_of(self, measurement_id: str) -> bool:
        return self.derived_from is None or self.derived_from != measurement_id

    def assert_independent_of(self, measurement_id: str, what: str = "scoring") -> None:
        if not self.independent_of(measurement_id):
            raise InstrumentError(
                f"reference was derived from {self.derived_from} and is now used for {what} against the "
                f"same measurement -- the comparison is circular")

    def passes_quality(self, floor: float = 0.65) -> bool:
        return self.quality is not None and self.quality >= floor


# ---------------------------------------------------------------------------------------------------------
# 6. seeds that are not seeds
# ---------------------------------------------------------------------------------------------------------
def stable_seed(*parts: object) -> int:
    """A reproducible seed from arbitrary parts. NEVER hash() -- Python randomises str hashing per process.

    The defect: random.Random(hash((task_id, seed))) looked seeded and drew a different set of isomers on
    every run, so the sharpest null this project has for the co-transcriptional criterion was not
    reproducible. The certificate this project ships distinguishes deterministic re-verification from
    non-reproducible search; the bug sat exactly on that line.
    """
    import zlib
    return zlib.crc32("|".join(str(p) for p in parts).encode())


# ---------------------------------------------------------------------------------------------------------
# 8. a rate matrix used as a master-equation generator
# ---------------------------------------------------------------------------------------------------------
def check_generator(rates, energies: Sequence[float], start: int, rt: float = 0.0019872 * 310.15,
                    name: str = "generator") -> Dict[str, object]:
    """A rate matrix is only a usable generator if it is irreducible AND obeys detailed balance.

    TWO DEFECTS THIS EXISTS FOR, both found within ten minutes of each other while trying to build a second
    kinetic judge out of `barriers`.

    ORIENTATION. rates.out from barriers is [source][target] and carries a nonzero self-rate on the diagonal.
    Reading it the other way round, or exponentiating it with the diagonal left in, produces populations that
    look plausible and are wrong. Detailed balance decides the orientation empirically -- the median log
    deviation was 0.019 one way and 8.792 the other, which is not a judgement call.

    REDUCIBILITY. With --max 12 the matrix came back in TWO disconnected components, the open chain in one
    and the ground state in the other. Long-time populations then converge to the Boltzmann distribution
    WITHIN the start's component, not the global one -- so a target sitting in the other component scores
    population zero forever, as an artefact of truncation rather than as kinetics. Comparing against global
    Boltzmann made a correct solver look broken; the restricted comparison agreed to 0.0045.

    Returns the diagnosis; raises only when the matrix cannot be a generator at all.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    R = np.array(rates, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise InstrumentError(f"{name}: rate matrix is not square")
    n = R.shape[0]
    diag_nonzero = bool(np.any(np.diag(R) != 0))
    np.fill_diagonal(R, 0.0)
    G = np.asarray(energies, dtype=float)[:n]
    err = []
    for i in range(n):
        for j in range(n):
            if i != j and R[i, j] > 0 and R[j, i] > 0:
                err.append(abs(math.log(R[i, j] / R[j, i]) + (G[j] - G[i]) / rt))
    db = float(np.median(err)) if err else float("nan")
    flipped = []
    for i in range(n):
        for j in range(n):
            if i != j and R[i, j] > 0 and R[j, i] > 0:
                flipped.append(abs(math.log(R[j, i] / R[i, j]) + (G[j] - G[i]) / rt))
    db_flipped = float(np.median(flipped)) if flipped else float("nan")
    lab = connected_components(csr_matrix(((R > 0) | (R.T > 0)).astype(int)), directed=False)[1]
    comp = [int(i) for i in np.where(lab == lab[start])[0]]
    out = {"n": n, "diagonal_was_nonzero": diag_nonzero,
           "detailed_balance_err": db, "detailed_balance_err_if_transposed": db_flipped,
           # `db == db` is True for every non-NaN, so the original short-circuited to True always and the
           # orientation check never fired. Its own test caught it on the first run, which is the point.
           "orientation_ok": bool(math.isnan(db) or db <= db_flipped),
           "n_components": int(lab.max() + 1), "irreducible": bool(lab.max() == 0),
           "start_component": comp, "start_component_size": len(comp)}
    if not out["orientation_ok"]:
        raise InstrumentError(
            f"{name}: detailed balance fits the TRANSPOSED matrix ({db_flipped:.3f} vs {db:.3f}) -- "
            f"the rate matrix is being read in the wrong direction")
    if not out["irreducible"]:
        out["warning"] = (f"reducible: {out['n_components']} components; long-time populations converge to "
                          f"Boltzmann WITHIN component {comp}, and anything outside it is zero by "
                          f"construction, not by kinetics")
    return out


# ---------------------------------------------------------------------------------------------------------
# 9. a binary kill-gate sitting on top of a continuous statistic
# ---------------------------------------------------------------------------------------------------------
def gate_margin(values: Sequence[float], threshold: float, n_boot: int = 2000,
                seed: int = 0) -> Dict[str, object]:
    """A pass/fail gate must clear its threshold by more than sampling error, or it is not a verdict.

    THE DEFECT THIS EXISTS FOR, found by rerunning the same entry check at two sample sizes. P2's kill rule
    is "every pairwise kappa >= 0.80". At 60 pairs IntaRNA~bifold came out at 0.799 and P2 read as PASSED;
    at 120 pairs the same quantity was 0.804 and P2 read as FAILED. Nothing about the domain changed. The
    overall verdict happened to be unchanged because P3 failed either way, which is luck, not design.

    Returns the margin and a bootstrap interval for the statistic closest to the line, so a gate decided
    within noise is reported as ON THE KNIFE EDGE rather than as a decision.
    """
    import random as _r
    v = [float(x) for x in values]
    if not v:
        raise InstrumentError("gate_margin: empty sample")
    rng = _r.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(v) for _ in v) / len(v))
    means.sort()
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]
    obs = sum(v) / len(v)
    decisive = (lo > threshold) or (hi < threshold)
    return {"observed": obs, "threshold": threshold, "margin": obs - threshold,
            "ci95": (lo, hi), "decisive": bool(decisive),
            "verdict": ("clear" if decisive else
                        f"knife edge: {lo:.3f}..{hi:.3f} straddles {threshold}")}


# ---------------------------------------------------------------------------------------------------------
# 9. a score that disagrees with its own table
# ---------------------------------------------------------------------------------------------------------
def tally(rows: Sequence[str], yes: str = "✅", no: str = "❌") -> Dict[str, int]:
    """Count a scorecard from the rows themselves rather than by hand.

    The defect: goal 4's fraction was hand-counted and disagreed with its own table in two consecutive
    weekly reports, undercounting by one both times. The ratio barely moved, but a score that disagrees with
    the table it summarises gives a reader no reason to trust anything outside the table.
    """
    y = sum(1 for r in rows if yes in r)
    n = sum(1 for r in rows if no in r and yes not in r)
    if y + n != len(rows):
        raise InstrumentError(
            f"tally: {len(rows)} rows but {y} yes and {n} no -- every row must carry exactly one verdict")
    return {"yes": y, "no": n, "total": len(rows), "fraction": y / max(1, len(rows))}
