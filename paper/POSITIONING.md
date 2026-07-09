# RNA paper (Paper C) — positioning & anti-collision spec

Written 2026-07-08, BEFORE drafting, so this paper never has to retrofit de-collision the way the
SAT paper did. This is the fifth paper in the InceptLabs self-improvement family. Read this before
touching `rna-selfimprovement.tex`.

## The family (five papers, all Chen et al., all AAAI-targeted)

- **[A2] Determinism Manifold** (`determinismmanifold`) — OWNS the four-coordinate framework
  (Dsym, Rdet, Ttrace, A), the manifold, τ, the substrate taxonomy (weights/context/harness) and the
  Law of Substrate Selection. Catalysis data only.
- **[A1] Actuation Fidelity** (`actuation`) — the A coordinate + Diagnosis–Actuation Gap. **Already
  reports RNA's A≈0.86 and a memory-severing ablation** (RNA is one of its cross-domain rows), and
  explicitly reserves "RNA RL-data generation" for THIS paper.
- **[A3] Oracle Residue** (`oracleresidue`) — Rdet-ceiling for physical oracles; uses the
  clean/self-checkable oracle (αf≈1) as its POSITIVE pole. RNA-under-ViennaRNA IS that pole.
- **[B] Compounding Profile / SAT** (`compoundingprofile`) — adjudicates compounding-vs-amplification
  (loop vs matched best-of-N), two-domain reversal, graded actuation ablation, preregistered
  falsification. The other clean-oracle four-coordinate paper.
- **[C] this paper** — RNA.

## Paper C's DISJOINT core (owned by none of A1/A2/A3/B)

1. An **honest mode-taxonomy of agent self-improvement** in a residue-free domain: memory
   distillation (NULL), autonomous de-scaffolding (~73% of capability is scaffolding), solve-rate
   attribution (LLM orchestrates classical search), tool acquisition (feasible), algorithm
   self-optimization (FunSearch-style; needs a capable subject). No companion enumerates the MODES
   and measures each mode's real efficacy boundary.
2. The **capability-attribution methodology** (autonomous floor vs scaffolded; designability split of
   agent-limited vs oracle-limited) — RNA-specific measurement contribution.
3. If run with a capable subject: the **weight-substrate RL / algorithm-evolution** result A1
   reserved — the family's only end-to-end context→weights realization.

## Collision surface & the rule for each companion

### vs [A1] actuation — HIGHEST consistency risk (the memory result)
A1 reports, on RNA, that memory HELPS: a learned playbook cuts oracle-calls-to-solve 6.8→2.1, and a
controlled severing of a FIXED correct memory on a classical executor raises calls 15.2→3.0 (CI
excludes 0); natural A≈0.86. THIS paper's ① reports memory as NULL (three-arm solve-rate 76/75/76;
skills 3/35 invoked, 0 solved). These are RECONCILABLE but a shared reviewer will call it a
contradiction unless stated head-on:
- A1's result = **oracle-calls-to-solve on already-solved tasks, on a classical local-search
  executor with a motif warm-start** (an adoption-mechanism / efficiency measurement).
- This paper's ① = **solve-COUNT on the LLM agent** (does memory make more puzzles solvable?).
- The honest synthesis: memory improves search EFFICIENCY on solvable tasks (A1) but does not expand
  the SOLVABLE SET for a tool-rich LLM agent (this paper) — the binding limit is classical-search
  convergence, not strategy recall. State this explicitly, cite A1, and use A1's A numbers verbatim
  (DeepSeek 0.86 / GLM 0.86 / Sonnet 0.85). Do NOT re-headline the A measurement or the severing
  ablation — those are A1's; cite them.

### vs [A3] oracle-residue — route ③ is a POSITIVE INSTANCE, cite don't claim
This paper's ③ ("solve-rate = LLM orchestrating classical search; the search/solver is the residual
computation the LLM does not internalize") is A3's judgment–computation asymmetry, in a NEW domain.
Frame ③ as an independent instance CONFIRMING A3 (judgment internalizes; the classical
search/oracle computation stays external), and frame RNA-under-ViennaRNA as A3's αf≈1 clean-oracle
pole. Do not re-derive the oracle-residue thesis. Note: RNA's oracle is residue-FREE (search is
amortizable), which is the CONTRAST to A3's catalysis (residue-bound) — that contrast is the point.

### vs [B] SAT compounding-profile — route ⑥ is the ADJACENT one, must differentiate
Route ⑥ (FunSearch-style: LLM evolves the RNA-design algorithm A_θ under the verifier V) is
structurally close to SAT's heuristic discovery (LLM edits solver hooks under a gaming-proof
verifier). Keep them distinct:
- SAT [B] = **adjudication**: does verified self-improvement compound vs amplify? (matched best-of-N,
  profile-not-verdict, two-domain reversal). A=1.0 obedience.
- RNA ⑥ = **discovery/capability**: can an LLM evolve a cheaper / higher-crack-rate RNA-design
  algorithm from the verifier? A natural A<1 (0.86). Different question.
- RULE: do NOT import SAT's best-of-N / profile-not-verdict / compounding-adjudication framing into
  the RNA paper. Cite [B] as the adjudication counterpart. RNA's "compounding" claims must be framed
  as **applied capability / mode-efficacy**, never as re-adjudicating amplification-vs-compounding.

### vs [A2] determinism-manifold — adopt the framework, distinguish the taxonomy axis
- Adopt (Dsym,Rdet,Ttrace,A) from [A2]; cite, do not claim. RNA's realized coordinates: Rdet≈1,
  A≈0.86 (cite A1 for A).
- [A2] has a SUBSTRATE taxonomy (weights/context/harness). This paper has a MODE taxonomy
  (memory / scaffolding / tool-acquisition / algorithm-invention) — a DIFFERENT axis (mechanisms of
  self-improvement, not storage locations). State the distinction; cite [A2].
- The two-stage memory→weights vision is [A2]'s composition law instantiated; cite it as such.

## Consistency ledger (numbers that must not contradict the family)
- A (RNA): DeepSeek 0.86, GLM 0.86, Sonnet 0.85 — must equal [A1].
- Rdet (RNA) ≈ 1.0 (clean ViennaRNA oracle) — consistent with [A1] Table-2 RNA row.
- Memory-free Eterna100-V2: GLM 56 / Sonnet 51 / DeepSeek 47 (this paper's own numbers; no family
  conflict). SOTA anchors: DesiRNA 97@24h (V2), Montparnasse 100 (V1) — from docs/eterna100_sota.md.
- The "76 / 56-of-65 / 3-of-35" figures from the experiment scan are marked [verify-vs-artifacts]
  in the draft until confirmed against rna_harness/runs; do not ship unverified counts.

## Do NOT
- Do not claim the four-coordinate system, A coordinate, oracle-residue thesis, or the compounding
  adjudication. Cite the owning companion.
- Do not re-report A1's RNA A-measurement or severing ablation as this paper's contribution.
- Do not frame ⑥ as a compounding-vs-amplification adjudication (that is [B]).
- Do not ship the off-mission "push Eterna to SOTA" story as the contribution; the honest thesis is
  the mode taxonomy + efficacy boundaries (score-boosting is classical-solver+compute, off-mission).
