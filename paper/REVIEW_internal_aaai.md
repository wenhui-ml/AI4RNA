# Internal Peer Review — Paper C (Empirical Skeleton), AAAI-2027 caliber

Reviewer stance: simulate a tough but fair AAAI/NeurIPS-caliber reviewer, informed by a fresh
(2026-07-10) literature sweep. Reviewing `EMPIRICAL_SKELETON.md` + `POSITIONING.md`.

## Recommendation

**Reject (borderline) for main track as currently scoped; viable only after repositioning + rigor.**
Confidence: high on novelty concerns, medium on empirics (skeleton, not full paper).
Overall merit ≈ **3 / 6**. The honesty and the closed chain are real assets; the novelty is largely
pre-empted by stronger contemporaneous work, and the empirics are under-powered for a top venue.

## Summary of contributions (as I read them)

C1: memory/skills self-improvement measures null because the harness pre-encodes the procedure
(scaffold redundancy); a reversible de-scaffolding ablation exposes the agent's floor. C2: given the
ideal environment the agent's selection-A_θ is exercised but only competitive-not-superior to a
hand-coded policy. C3: RNA-under-exact-oracle as a residue-free positive control. Bridge thesis:
harness-confound and self-improvement-null are one phenomenon; prescription = report floor-and-ceiling.

## Strengths

- **S1. Intellectual honesty is exemplary.** The paper retracts its own headline ("73% is scaffolding")
  after multi-round replication (S3), and reports a null and a competitive-not-superior result without
  spin. This is rare and valuable.
- **S2. The closed, mechanism-grounded chain** (null → redundancy → de-scaffold → attribution →
  self-revise → agent-gate → wall) is coherent and every number is artifact-traceable.
- **S3. Residue-free exact-oracle positive control** is a genuinely good methodological choice: it
  licenses "the null indicts the mechanism, not the domain" in a way SWE/terminal-bench work cannot.
- **S4. The de-scaffolding *instrument* (reversible flag, byte-identical off)** is a clean, reusable
  measurement tool.

## Weaknesses

### W1 (MAJOR) — Novelty is largely pre-empted by stronger 2026 work.
The scaffold-vs-model attribution problem — the paper's C1/C2 core — is now a crowded area, and by
studies that *outclass this one on rigor*:
- **Agent psychometrics** (arXiv:2604.00594) already decomposes agent ability into LLM-ability +
  scaffold-ability and tests additivity, across multiple coding benchmarks and models.
- **ClawsBench** (2604.05172) decomposes scaffolding into independent levers and varies them across
  **6 models × 4 harnesses** — exactly the multi-model/multi-harness design this paper lacks.
- **A Unified Framework for LLM Agentic Capabilities** (2605.27898) explicitly "disentangles intrinsic
  LLM capabilities from framework- and environment-induced artifacts."
- **Harness-Bench** (2605.27922), **"Stop Comparing… without Disclosing the Harness"** (2605.23950),
  **CORE-Bench** (2606.26158): the attribution/"don't credit the model alone" thesis.
- **SIA** (2605.27276): the observation that scaffold self-improvement is "SE hygiene (parsing/retries),
  rarely domain reasoning the base model couldn't produce" — i.e. the paper's S3 finding, already stated.
- **MOST DANGEROUS — "AI scientists produce results without reasoning scientifically"** (2604.18805):
  a *scientific-agent* paper arguing scaffold-dominance with de-scaffolding ablations, and (per a
  search-level read that the authors MUST verify directly) possibly stating self-improvement/memory
  show null gains because the scaffold already contains the structure. **If that paper says what the
  search suggests, the bridge thesis + C1 + C3 are a near-collision.** Authors: read 2604.18805 in full
  before submission; the whole positioning may need to pivot around it.
→ What plausibly survives as novel: (a) the *specific unification* that the harness-confound and the
  self-improvement-loop-null are the same phenomenon **viewed from two sides**, (b) the residue-free
  exact-oracle instantiation, (c) the discovery-criterion accept-gate (from the MIT categorical paper),
  (d) the honest replication overturning a single-point attribution. This is a *measurement cautionary
  tale + clean-oracle instrument*, not a new attribution method. The paper must be re-centered on that
  and cite the above as prior art, or it reads as re-deriving a known result on weaker evidence.

### W2 (MAJOR) — Under-powered for a top venue: single model, tiny n, single domain.
S1 n=1/arm; S3 n=3 rounds/arm; S6 n=6 targets × 2 seeds; **all on one model (GLM-5.2)**. The
contemporaneous work (ClawsBench: 6×4; Agent psychometrics: multi-model) sets a bar this misses by an
order of magnitude. No headline claim here would survive a reviewer's "single seed / single model /
n<10" objection. The paper's own family (Paper B: K-seed, Wilcoxon, pre-registration) is far more
rigorous — a shared reviewer will notice.

### W3 (MAJOR) — S3 has a confound that may undercut the redundancy thesis.
S3 (floor ≈ ceiling on the hard tail) is the strongest-powered result, but on the **hard tail** both
arms may fail *because the targets are solver-hard* (S7's thermodynamic wall), not because the scaffold
is redundant. "Floor ≈ ceiling where nothing works" does not establish "scaffold buys no capability."
The redundancy claim actually lives in the **solvable** regime (where the scaffold's procedure carries
the solve and memory has nothing to add — that is S4's attribution). As written, S3 and the redundancy
thesis are in tension. Either measure floor-vs-ceiling on the *solvable* set (where the effect should
appear), or reframe S3 as "the residual is a hard wall, indistinguishable across arms" and rest
redundancy on S4.

### W4 (MODERATE) — Single domain; generality asserted, not shown.
One domain (RNA). The positive-control framing helps but does not substitute for a second domain. Paper
B was dinged for two domains; one is more exposed. State the limit plainly and lean on the
positive-control argument, but expect the "n=1 domain" objection.

### W5 (MODERATE) — S5/self-revise reads as solver engineering, not a scientific contribution.
The close_pair operator + greedy policy are hand-tuned heuristics; a reviewer will ask "what is the
scientific claim beyond 'we wrote a better RNA local-search endgame'?" The two mechanism findings
(diagnosis-timing > blind-selection; self-revision > blind) are **n=1 existence proofs** (pid 66, 49).
Keep S5 only as scaffolding for S6; do not present it as a capability contribution.

### W6 (MODERATE) — S6's conclusion is contradicted by stronger contemporaneous work.
**ARC / "Learning to Configure Agentic AI Systems"** (2602.11574) reports a *learned* policy beating
strong hand-designed baselines by up to +25% across benchmarks. This paper's S6 (agent-driven ≈/<
hand-coded greedy, n=6, one model) is the same comparison with the *opposite, weaker* result. A reviewer
will read S6 as "under-powered evidence against a conclusion the field already has better evidence for."
Frame S6 narrowly (operator-selection, one weak model, existence not rate) and cite ARC.

## Questions to the authors

1. Does 2604.18805 already claim scaffold-redundancy → self-improvement null in scientific agents? If so,
   what is left that is yours?
2. Why is S3 on the hard tail and not the solvable set? Can you show floor < ceiling where the scaffold's
   procedure actually carries solves?
3. What changes with a second model (DeepSeek/Sonnet are already run at base — extend them through the
   arms)? Without it, why should we trust GLM-5.2-only conclusions?
4. Is the "bridge" a testable claim or a reframing? What prediction does it make that the attribution
   literature does not?

## Path to viability (constructive)

1. **Re-center on the disjoint core**: the residue-free exact-oracle *instrument* + the discovery-criterion
   accept-gate + the honest replication cautionary tale. Demote the attribution decomposition to "we
   confirm, in a clean scientific domain, what [ClawsBench/psychometrics/2604.18805] show in SE/other
   domains, and add X." Make X explicit and defensible.
2. **Power it**: run S1 and S6 through ≥2 models (base runs exist) and more seeds; keep S3 (strongest) but
   add the solvable-set version (W3).
3. **Engage the prior art head-on** (a full related-work section citing the ~10 papers above); the current
   POSITIONING de-collides only against the *internal* family, not the external field.
4. Consider **workshop** (measurement/eval-of-agents workshop) as the honest venue for the current
   evidence, main-track only after (2).

## Minor

- SOTA framing (76 vs DesiRNA ~85 @minute) is now correct and honest — keep.
- Define "budget" precisely everywhere (wall vs search-compute; the p1 25-min cap is LLM-dominated).
- The retraction of "73%" should be a stated *finding* (single-point estimates mislead), not buried.
