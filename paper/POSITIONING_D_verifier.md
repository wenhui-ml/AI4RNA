# Paper D (verifier economics) — positioning & anti-collision spec  [REV 2, 2026-07-14]

Written before drafting so this paper never retrofits de-collision. Sixth paper in the InceptLabs
family; second RNA-under-ViennaRNA member (after Paper C). Read this AND `POSITIONING.md` (Paper C).

**REV 2 note**: this revision folds in an external adversarial review (2026-07-14). Three headline claims
from REV 1 were RETRACTED as unsupportable — see "Retractions" — and the contribution recentred on the
one defensible core (C3: deterministic cost × optimization-pressure × strong-acceptance measurement).

**Working title**: *Solved Relative to Which Verifier? Deterministic Cost–Robustness Frontiers in RNA
Inverse Design.* (provisional). Alt: *A Deterministic Testbed for Costed Verifier Misspecification.*
Cite key: `verifiereconomics`.

**One-line thesis (REV 2)**: under a fixed thermodynamic model, RNA inverse design admits a *provably
nested* family of decidable acceptance predicates (returned-MFE match ⊇ unique-MFE ⊇ P(target)≥θ, for
θ>½) plus a *pseudoknot branch* where pseudoknot-blind verifiers are out-of-domain. Because these
predicates are deterministic (no evaluator variance), we can measure — exactly and model-relative — how
the proxy↔strong acceptance gap trades against verification COST and OPTIMIZATION PRESSURE, separating
ordinary criterion mismatch from optimization-induced (Goodhart) gaps, and when a verifier should abstain.

## Retractions from REV 1 (do NOT ship these as worded)
- ❌ "the first deterministic verifier ladder / measure reward-hacking exactly / prior work could only
  estimate" — the broad *deterministic reward-hacking measurement* claim is taken by
  **Hack-Verifiable Environments** (`arXiv:2605.20744`, TextArena; "deterministic and automated
  measurement", "verifiable by design"). We claim only the *narrow* new instance: a model-internal,
  cost-priced, provably-nested predicate family on a natural-science design problem.
- ❌ "we first show the gap is Goodhart not solver-weakness" — **LLMs Gaming Verifiers**
  (`arXiv:2604.15149`) already shows extensional verifiers induce shortcuts, isomorphic ones remove
  them, and prevalence grows with task complexity and inference-time compute. C2 is **replication-plus-
  extension in a deterministic domain**, not a first.
- ❌ "MFE→uMFE→P(T)→pKiss strictly ordered ladder" — pKiss is NOT a top rung; it is a **branch** (see
  Formal spec). Only MFE⊇uMFE⊇P(T)≥θ (θ>½) is a provable nest.
- ❌ "truth value is computed" → use "each predicate is evaluated deterministically **relative to a
  fully specified computational folding model**". Folding oracles are model outputs, not biophysical truth.

## The family (six papers, all Chen et al.)
A2 Determinism Manifold (`determinismmanifold`, owns 4-coord Dsym/Rdet/Ttrace/A + substrate taxonomy) ·
A1 Actuation (`actuation`, owns A + Diagnosis–Actuation Gap; RNA A≈0.86) · A3 Oracle Residue
(`oracleresidue`, owns αf/Rdet-ceiling; RNA=αf≈1 clean pole) · B Compounding/SAT (`compoundingprofile`,
owns compounding-vs-amplification) · C RNA self-improvement/green-zone (`rnaselfimprovement`, owns
self-improvement MODE taxonomy + green-zone NULL + A_θ-headroom) · **D this paper** (verifier economics).

## Paper D's DISJOINT core (owned by none of A1/A2/A3/B/C, and surviving external review)
1. **A provably-nested, model-internal, decidable predicate family with a formal ordering** (MFE ⊇ uMFE
   ⊇ P(T)≥θ for θ>½), on a natural-science design problem — distinct from the LLM-judge/rubric/test-suite
   verifiers of the RLVR-verifier literature (noisy, not analytically nested, not priced) and from
   TextArena's embedded-hack environments (`2605.20744`).
2. **A joint COST × OPTIMIZATION-PRESSURE × STRONG-ACCEPTANCE surface** measured with zero evaluator
   variance. Not "verifiers disagree" (known) but *how the disagreement moves* as you (a) spend more on
   optimizing a cheap proxy and (b) pay more to verify — the deterministic instrument the LLM-judge
   world cannot build cleanly. **This is the core (was C3).**
3. **A domain-of-validity / abstention treatment of oracle misspecification**: pseudoknot-blind verifiers
   are total-but-invalid on pseudoknotted targets; the correct action is abstain, not accept a projected
   target. A verifier *graph* (branch), not a linear ladder.

## Formal spec (must be pinned in the draft — REV 1 was informal here)
Let acceptance sets over sequences x for target T, under one fixed energy model + params + tie-break:
- `A_match` = { x : RNAfold's returned structure == T } — MUST state: returned-MFE (deterministic
  tie-break) vs member-of-MFE-set; float-energy tolerance; this is the CHEAP proxy.
- `A_uMFE` = { x : T is the unique MFE } — state the test (RNAsubopt -e 0 enumeration / energy-gap>0).
- `A_pfe(θ)` = { x : P(T | x) ≥ θ }, P from the SAME pseudoknot-free ensemble.
Provable nesting (verified empirically: 0 violations on 56 targets):
  `A_pfe(θ) ⊆ A_uMFE ⊆ A_match`  **for θ > ½**  (if P(T)>½, no other structure can have energy ≤ E(T),
  so T is the strict unique MFE). For θ ≤ ½ this is only a *threshold family*, NOT a stronger rung — the
  main nested claim MUST fix θ>½; sub-½ sweeps are reported as a threshold family, not "stronger".
- **pKiss is a BRANCH, not a rung**: on pseudoknotted T, `A_match/uMFE/pfe` are *inapplicable* (ViennaRNA
  cannot emit crossing pairs); on nested T, pKiss uses a different (larger) space + heuristic + possibly
  different params, so `A_pKiss ⊄ A_Vienna` in general (model disagreement, not nesting). Structure it as
  a verifier graph: [nested branch: match→uMFE→pfe→ensemble-defect] and [pk branch: projected-backbone →
  ABSTAIN / pKiss-heuristic → pKiss-restricted-class → cross-oracle].

## Claims — honest novelty + required reframe (post-review)
- **C1** (exact-match over-accepts vs ensemble robustness). Phenomenon NOT novel (SAMFEO, sampling-based
  continuous optimization, "Fitness Functions for RNA Design" all show MFE/uMFE solutions can have low
  P(T)). Reframe as a **quantified conditional distribution** `Pr[P(T)<θ | x∈A_match]` across θ, model,
  solver, length, difficulty — a benchmark statistic, NOT a discovery. **A single 93% is a metric
  mismatch, not reward hacking** until C2 supplies controlled optimization.
- **C2** (Goodhart, not solver-weakness). Collides with `2604.15149`; frame as replication-plus-extension
  in a deterministic domain. REQUIRES: fix the proposer, vary only the reward predicate; and because
  exact-match is BINARY and SATURATES, use a **continuous cheap proxy** (MFE structural distance / energy
  margin) or a generative policy whose accept-rate rises, else "more optimization pressure" is undefined.
- **C3** (verification cost vs acceptance reliability) — **the defensible core**. Do honest end-to-end
  costing: MFE and partition-function are BOTH ~O(n³) (no clean order-of-magnitude ladder — drop "six
  orders of magnitude" unless measured); pKiss heuristic O(n⁴). Report cold/warm latency, CPU+wall,
  memory, empirical scaling in n, timeouts, and DISTRIBUTIONS (uMFE enumeration is heavy-tailed — itself
  a result). Unified rule: cost = zero→accept/reject end-to-end, same for every rung.
- **C4** (wrong oracle → false positives). RNAInvBench + antaRNA already split RNAfold/pKiss on 251
  PseudoBase++. Reframe as a **domain-of-validity / category-error case study + abstention**, not
  "pKiss is the strongest rung / 100% false positives" (that framing is near-tautological by construction).
- **C5** (adaptive verifier selection + abstention). Cascade/selective-verification/abstention literature
  is mature (`Cost-Saving LLM Cascades with Early Abstention`, `Anytime Verified Agents`). Reframe as a
  **system/mitigation** result with a risk-constrained formalization (min E[cost] s.t.
  Pr[V*≠V_π]≤ε, abstain allowed); route by structure-class + cost, NOT LLM confidence. Novelty only if a
  finite-sample risk or oracle-regret bound is given.

## Revised experiment list (E1'–E6')
- **E1'** Nested frontier: `Pr[P(T)<θ | A_match]` over θ∈(½,0.99], with uMFE stratification; verify set
  nesting; per-target aggregation, multi-seed, CIs; STATE the solver + selection (REMC-solvable subset;
  add DesiRNA/SAMFEO solutions for the hard tail + as a "strong-objective solver is less fragile" control).
- **E2'** Controlled Goodhart (**core, exceed-bar**): continuous cheap proxy; fixed proposer; cheap/strong/
  mixed reward; hold-out; draw G(budget, θ) surface (cheap improves, strong stalls/declines, strong-reward
  arm shows no split). Cite `2604.15149`; position as deterministic-domain extension.
- **E3'** Oracle-misspecification case study (full 251 PseudoBase++): domain-of-validity + abstain, not
  "strongest rung".
- **E4'** End-to-end costing (**core**): honest per-rung cost distributions vs n; O(n³)/O(n⁴) reality.
- **E5'** Risk-constrained verifier routing (mitigation): formalize + measure cost at matched reliability;
  report domain-dependence honestly (nested-MFE wins; pk single-pass loses).
- **E6'** Second deterministic domain (generality; REQUIRED to beat "RNA metric repackaging"): SAT/SMT
  (sat-assignment ⊇ certified-unsat / MaxSAT-margin) or program-test strength. **B-collision guard**: B
  owns SAT *compounding*; E6' is *verifier-economics on SAT* — a different axis; cite B, do not import
  compounding framing.

## vs family — collision rules (unchanged in spirit; A3 still sharpest)
- **A3**: our nested-frontier is verifier SELECTION/STRENGTH, not αf oracle-residue; RNA-on-proper-domain
  stays A3's αf≈1 clean pole. Cite, don't re-derive. (C4's pk branch is model DISAGREEMENT, still not αf.)
- **A1**: cost-router is "routing/selection policy", never "A_θ"/actuation; cite A≈0.86.
- **C green-zone**: any learning/cost result is COST headroom not CAPABILITY; solves fixed; router≠A_θ.
- **B**: E6' SAT is verifier-economics, not compounding; keep the learning-vs-cost result subordinate.
- **A2**: adopt 4-coord; D's axis (verifier selection×strength×cost) ≠ substrate taxonomy.

## Consistency ledger (own numbers, preliminary)
- **E1' Frontier** (61 REMC-solvable Eterna100-V2 targets, single-seed pilot): survival match 100% / uMFE
  90% / P(T)≥0.9 7%; **set-nesting violations = 0** (empirically confirms A_pfe⊆A_uMFE⊆A_match at θ=0.9);
  median P(T)=0.25–0.30. fp-rate (exact-match but not pfe) by length: <50 →75%, 50–99 →100%, 100–199
  →100%, ≥200 →92% (fragility JUMPS at n≥50). MARK preliminary: single-seed, REMC-solvable subset
  (selection bias), needs multi-seed + CIs + hard-tail via DesiRNA/SAMFEO.
- **E2' controlled Goodhart** (15 targets n≤90, same Metropolis proposer, only reward differs; budgets
  500/4000/20000): CHEAP-reward arm strong-accept(P≥0.9) DECLINES 26.7→20→13.3% while gap to proxy WIDENS
  20→40→53pp (median P(T) 0.40); STRONG-reward arm strong-accept RISES 20→26.7→40% (median P(T) 0.59).
  → optimization-induced separation (not mere criterion mismatch), RL-free, controlled. MARK: small n,
  single-seed, ±6.7pp quantization; needs more targets/seeds/CIs; frame as extension of `2604.15149`.
- **E4' verifier cost** (lengths 30–300, 15 reps): match/uMFE/pfe scale ~n^2.78/2.79/2.84 (all ~O(n³));
  at n=300 medians 98/301/384 ms — a **~4× constant-factor ladder, NOT orders of magnitude**; pKiss is a
  startup-dominated BRANCH (~121 ms flat over n=30–150). uMFE tail modest in the clean regime (p95≈1.1×
  median). RSS flat ~28 MB. → CONFIRMS: drop "six orders of magnitude"; the big cost jump is the oracle
  BRANCH, not a strictness rung.
- **E6' SAT second domain** (`experiments/sat_verifier.py`; ladder V0 satisfies-φ ⊇ ρ≥θ single-flip
  robustness; hardness = clause density α):
  * frontier (n=60, budget 60000, k=20, ρ conditioned on SAT): V0-accept 85–100%; **median ρ|SAT drops
    MONOTONICALLY 0.233→0.167→0.133→0.117→0.117→0.100 as α 3.0→4.267** — clean cross-domain mirror of E1'.
  * cost: V0 O(m) 0.1–0.27 ms → ρ O(nm) 1.7–14 ms → r=2 O(n²m) 20–118 ms — within-domain cost-robustness
    ladder; span (~200–1000×) STEEPER than RNA's (~4×) — an honest cross-domain contrast.
  * Goodhart (witness-start, same kernel): WEAK / domain-dependent (α=4.2 strong>cheap by 0.03; α=3.6 no
    gap) → report as "Goodhart severity is domain-dependent: strong in RNA (E2'), mild in SAT". RNA carries
    the Goodhart leg. (Did NOT launch a big SAT-Goodhart run — weak signal.)
  * → cross-domain claim rests on frontier+cost mirroring in BOTH domains ("not RNA repackaging").
- **E5' verifier routing** (`experiments/verifier_routing.py`; nested ladder, gold=pfe):
  * THEOREM (sound cascade, ε=0): nesting ⇒ cheap-rung REJECT is a certain gold REJECT ⇒ cheapest-first
    cascade = gold verdict EXACTLY (correctness 100%, **unsound_cases=0** on both workloads — verified).
  * Sound-cascade cost saving is WORKLOAD-DEPENDENT (honest): reject-heavy (135 cands, 7% match-accept)
    E[cost] **0.444 vs 1.133 ms = +60.8% saving** (cheap rungs filter non-solutions); accept-heavy (90
    cands, 61% match-accept) **−25.3%** (nothing is cheaply rejected → cheap rungs are pure overhead atop
    gold). → the cascade is a REJECT-filter, valuable on failure-heavy (early-RLVR-like) workloads, not a
    universal free lunch.
  * Risk-cost frontier (accept-heavy): cheap-rung stop-and-accept FP = **98%** (match 54/55, uMFE 47/48);
    Clopper-Pearson **min-ε = 0.999** for both → the frontier is FLAT at gold through ε=0.99. Quantified
    cost-of-robustness: **you cannot trade cost for correctness unless tolerating ~99.9% false positives →
    robustness certification is near-incompressible; you must pay the strong rung.** (finite-sample bound).
- fp-demo pilot: 4/4 MFE false positives on pseudoknots (pKiss bp 4/6/5/7) — reframe as category error.
- cogeval: cognitive 18/18 $0.269 vs always_sota 17/18 $0.464 (+1 = noise). Track B: cognitive 6/10 =
  always_sota 6/10 but pricier + 2× pKiss; REMC 0/10 (domain-dependence, report honestly).
- learning-vs-cold: 73% cost saving, solves 18/18 unchanged — MAGNITUDE inflated by soft-wall bug; report
  only after hard-wall fix; frame as cost-not-capability, subordinate.
- Established methods to CITE not claim: SAMFEO / EnsembleDesign / RiboPO (ensemble objectives), uMFE,
  ensemble defect, P(T); RNAInvBench + antaRNA (RNAfold/pKiss split, 251 PseudoBase++); DesiRNA 97@24h.

## Safer abstract (draft to iterate)
> We introduce a deterministic, model-relative testbed for measuring how proxy↔target acceptance gaps vary
> with optimization pressure and verification cost. In pseudoknot-free RNA inverse design we formalize,
> under one thermodynamic model, a provably nested family of decidable predicates — returned-MFE match ⊇
> unique-MFE ⊇ target Boltzmann probability ≥ θ (θ>½). Because these predicates have deterministic
> outcomes, disagreement rates are measured without evaluator variance. We separate ordinary criterion
> mismatch from optimization-induced (Goodhart) gaps by holding the proposer fixed and varying only search
> pressure and the reward predicate. We show pseudoknotted targets do not form a stronger rung but expose a
> verifier-domain mismatch, for which pseudoknot-blind verifiers should abstain rather than accept a
> projected target. Across RNA benchmarks we characterize the joint frontier among verification cost, proxy
> acceptance, and strong-predicate acceptance, and evaluate risk-constrained adaptive verification. We do
> not claim biophysical ground truth; all predicates are relative to a specified computational folding
> model, and we position the study against the RLVR-verifier and RNA-design literatures it bridges.

## Do NOT (updated)
- No "first deterministic reward-hacking measurement", no "measure exactly", no "truth is computed", no
  "MFE→pKiss strict ladder", no "six orders of magnitude" (unless measured), no first-Goodhart claim.
- Do not call a single 93% "reward hacking" — that needs controlled C2.
- Do not claim 4-coord/A/oracle-residue/compounding — cite owners. Do not claim uMFE/ensemble/pKiss/
  cascade/abstention as novel — cite.
- No RLVR-training result (we run none); frame as testbed + measurement + honest agent. No in-vivo claims.
