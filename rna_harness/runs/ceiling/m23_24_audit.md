# M23 / M24 audit — the null condition, and three held-out rounds

Declarations `m23_exact_null_declaration.json` and `m24_cross_round_declaration.json`, frozen and
pushed at `e8d4907f` **before** either runner existed. Offline, $0.

Both answer objections from an external review of the manuscript.

---

## M23 — "the null condition is stated too weakly"

The review is right. Proposition 1 said *"under the null $y\perp t$ within cell"*, but the
statistic depends on a matched set built jointly from $(t,\gamma,\pi)$, and $\gamma$ (global GC)
and $\pi$ (the panel prognostic score) predict $y$. Permuting $y$ within cell therefore breaks the
$y$–$\gamma$ relation as well, and the test is not exact under that condition.

### Arm A — four reference distributions under an exchangeable null (stands)

DGP1 permutes $y$ within cell, so $y$ is independent of everything. Six populations, 40 draws,
400 permutations, nominal $0.005$:

| null | rejections | rate | mean null SD |
|---|---|---|---|
| N1 permute the treatment | 9/240 | **0.037** | 0.4340 |
| N2 permute $y$ within cell | 1/240 | **0.004** | 0.6745 |
| N3 Freedman–Lane | 0/240 | 0.000 | 0.6394 |
| N4 blocked on $(\gamma,\pi)$ calipers | 0/240 | 0.000 | 0.2956 |

The treatment-side null over-rejects sevenfold and every design-fixed null is at or below nominal.
K4 fired as the declaration anticipated: N4's blocks have median size $1.0$, so it has almost no
permutation freedom and is recorded as `blocked_underpowered`.

### Arm B — VOID. There is no zero-effect dataset that keeps the real $y$–$\gamma$ relation

DGP2 was meant to be the regime the review is about: $y$ keeps its real relation to $\gamma$ and
$\pi$, only the $t$–$y$ relation is cut. It permuted $t$ inside fixed-grid $\gamma$ blocks of width
`CAL_GC`. Under it, N2 rejected 45/240 and N3 16/240, which would have been a catastrophic finding
about the paper's inference. It is not one, because **DGP2 is not a zero-effect process.**

- the estimator's caliper is a *moving window* of width `CAL_GC` while the blocks are a fixed
  grid of the same width, so **44.4%** of matched pairs straddle a block boundary and keep their
  original treatment--outcome association;
- measured leak, population by population: **4.7%, 16.4%, 10.9%, 22.2%, 33.7%, 31.4%** of the real
  effect, with **88--100%** of draws positive;
- the leak's predicted rejection rate correlates **+0.857** with the observed one and its pooled
  prediction (0.254) *exceeds* what was observed (0.188).

The obvious repair is worse. DGP3 estimates $\mathbb E[y\mid\gamma,\pi]$ non-parametrically
(8-nearest-neighbour within cell), permutes the residual and adds the fit back; it retains
**47.5%** of the real effect. The reason is structural and worth stating, because it is not a
failure of simulation craft: **$\gamma$ and $t$ are dependent, so part of the real $y$--$\gamma$
relation *is* the effect.** Any construction that preserves that relation preserves some of the
effect. On this design the question the review raises cannot be settled by simulation.

Arm B's outputs are kept as `m23_exact_null_DGP2_VOID_not_a_null_process.{json,log}`.
**Staked prediction (3) --- "N2 is conservative under DGP2" --- is recorded as wrong**, and so is
the reasoning behind it, because the object it was about turned out not to exist.

### Arm C — the same concern, bounded analytically instead

The failure mode the review identifies is residual within-caliper imbalance: a matched pair
contributes $h(\gamma_i,\pi_i)-h(\gamma_j,\pi_j)$, which vanishes only if the caliper closes
exactly. That is measurable with quantities the estimator already carries:

$$\text{bias} \;=\; \beta_\gamma\,\overline{\Delta\gamma} \;+\; \beta_\pi\,\overline{\Delta\pi}$$

| arm | bias as a share of the effect, six populations |
|---|---|
| panel-free | $+22.8\%$ to $+37.7\%$ |
| **panel-matched (the paper's main arm)** | $+0.95\%$ to $+3.16\%$ |

Almost all of it runs through $\overline{\Delta\pi}$, which the panel caliper cuts from $1.50$–$2.76$
to $-0.02$–$0.01$. **The panel caliper is what makes the estimator robust to the failure the review
names, and this is the first time its role has been quantified.** It also says plainly that the
panel-free arm is an upper bound rather than a cleaner measurement.

### What Proposition 1 should say

Not *"$y\perp t$ within cell"*. The exact condition is **within-cell exchangeability of $y$ given
the design**, i.e. $y\perp(t,\gamma,\pi)$; under the weaker $y\perp t$ alone the test is not exact,
the failure channel is residual within-caliper imbalance, and its size is the arm-C table. The
robust content --- which arm A measures and which does not depend on the condition --- is that
**inference must preserve the joint covariate structure used to construct the matched set.**

---

## M24 — "one corpus is one ecological environment"

OpenKnotBench v4.5.1 carries four rounds. Everything in this paper used Round 3. Rounds 1, 2 and 4
have **disjoint puzzle sets** (verified: empty intersection with Round 3 in all three) and
different method mixes, and had never been read.

**P2, the reproduction probe, caught a bug --- in a different experiment.** The declaration said
the code path must reproduce "M20's nine numbers". There are two such sets: the pre-run figures in
M20's `BLIND_STATUS` and the committed results in `m20_designer_confounding.json`. Against the
committed results the agreement is exact (max cell deviation $0$, per-method deviation $0$).
Against the declaration's pre-run set it is not, because those predate the `MIN_CELL` filter and
differ by up to $0.145$. **M22 had hardcoded the pre-run set**; it has been changed to read the
artifact, and its correlation moves from Spearman $-0.40$ to $-0.267$ with no change to any
conclusion. Never transcribe a number that already lives in an artifact.

### The placement structure, four rounds

| round | targets | methods | cells | pooled $g$ | cells $<0$ | methods $<0$ | Eterna rank | method component | $p$ |
|---|---|---|---|---|---|---|---|---|---|
| 1 (PK50) | 17 | 9 | 141 | $-0.0877$ | 56.7% | 5/9 | 2 | $+0.3185$ | 0.0005 |
| 2 | 17 | 7 | 104 | $-0.3404$ | 81.7% | 7/7 | 0 | $+0.0796$ | 0.0275 |
| 3 | 20 | 10 | 200 | $-0.2713$ | 72.0% | 9/10 | 0 | $+0.1778$ | 0.0005 |
| 4 | 20 | 10 | 194 | $-0.3508$ | 79.4% | 10/10 | 2 | $+0.0875$ | 0.0120 |

**All four pooled gaps negative. 31 of 36 (round, method) cells negative, sign test
$p = 6.5\times10^{-6}$.** Eterna is in the most negative third of every round.

The method component is positive in all four and clears $0.005$ in two. On the benchmark's full
pair set this is stronger than the $p=0.0120$ the manuscript reports, and the two are **not
interchangeable**: the manuscript's figure is on the compensatory-arm pair set, which is smaller.
The cross-round numbers are a replication on held-out rounds, not a restatement.

**Round 1 is the shape.** Four of its nine methods are positive --- `3DRNA`, `MPNN-RFdiff`,
`Rosetta`, `Rosetta-LoRes` --- and it is the round whose targets come from PK50. It is the weakest
pooled gap by a factor of three. Reported, not smoothed.

### Verdict versus staked prediction

The declaration's K2 asks for all four rounds negative and a **majority** of methods negative in
each; that is met (5/9, 7/7, 9/10, 10/10), so the verdict is
`placement_replicates_across_rounds`. **Staked prediction (2) asked for $\ge 2/3$ per round and
fails on Round 1's 5/9.** The two thresholds are different and both are reported, so that a failed
prediction is not later described as a failed verdict, or the reverse.

### What this cannot do

All four rounds are **designed** corpora, one laboratory, one platform, one player community. This
replicates targets and method mix. It says nothing about natural RNA, and no natural corpus is
present in this data release.

## Files

- `m23_exact_null_declaration.json`, `m23_exact_null.py`, arm-A/B output archived as
  `m23_exact_null_DGP2_VOID_not_a_null_process.{json,log}`
- `m23v_dgp_check.{py,json,log}`, `m23v2_leak_per_pop.{py,json,log}` — the void's diagnosis
- `m23c_imbalance_bound.{py,json,log}` — arm C
- `m24_cross_round_declaration.json`, `m24_cross_round.{py,json,log}`
