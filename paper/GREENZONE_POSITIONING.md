# Paper C — Positioning v2: The Green-Zone Null (the algorithmic-core headroom coordinate)

Supersedes the mode-taxonomy framing of `POSITIONING.md` (which the 2026 harness-effect cluster
pre-empted — see `REVIEW_internal_aaai.md`). This is the reframe that threads the crack between the
InceptLabs family (A1/A2/A3/B) and the crowded external field. Written 2026-07-11.

Working title: **"Everything Green, Still Null: An Algorithmic-Core Headroom Coordinate for
Self-Improving Agents, and a Clean-Oracle Scientific Domain Where the Field's Diagnostic Apparatus
Predicts Compounding but Measures None."**

Domain: RNA secondary-structure inverse design under ViennaRNA 2.7.2 (Turner2004). `solved` =
design's MFE fold equals target; no in-vivo claim. Red line: distilled procedure only, never answers.

---

## 1. The boundary, and the crack

The most complete diagnostic system in the field for **where self-improvement is possible** is the
InceptLabs family's coordinate set:

- **A2 (Determinism Manifold)** — a domain is learnable when its determinisms (D_sym, R_det, T_trace)
  clear τ; below τ is the *pseudo-learning regime* (a null caused by LOW determinism).
- **A1 (Actuation Fidelity)** — the loop must be closed: A = P(a correct correction controls the next
  execution), A = A_harness × A_model; a low A is a null caused by an *unclosed loop*.
- **A3 (Oracle Residue)** — the reward oracle must be amortizable: α_f, the amortizable fraction; a
  physical oracle caps R_det, a null caused by *oracle residue*. Its POSITIVE pole is the
  closed/exact/self-checkable oracle (games, code, formal math) at α_f≈1, "where search amortizes
  cleanly into weights."

By **every one of these axes**, RNA-under-ViennaRNA is in the **green zone** — and we take each
number from the owning companion, verbatim, not re-derived:

- learnable: D_sym = R_det = T_trace ≈ 1 (A2 places RNA here).
- loop-closed: **A ≈ 0.85** (A1's own RNA measurement: DeepSeek-V4-pro 0.86 / GLM-5.2 0.86 / Sonnet-5 0.85).
- oracle-amortizable: **α_f ≈ 1** (A3's clean-oracle pole; RNA-under-ViennaRNA is free, exact,
  self-checkable — exactly A3's "search amortizes cleanly" case).

**All three companions therefore predict RNA self-improvement should compound. We measure NULL.**

And this is the family's own **named missing testbed**: A2 future work — *"a naturally low-improvement
LLM self-improvement loop on a learnable family… remains the missing testbed"*; A1 — the RNA causal
claim runs on a *classical* executor, *"not a single end-to-end causal claim on an LLM self-improving
agent."* We supply exactly that: an end-to-end LLM self-improvement loop, on a learnable,
loop-closed, clean-oracle domain, that is null for a reason that is **neither low determinism (A2),
nor unclosed loop (A1), nor oracle residue (A3).**

## 2. The fifth coordinate

> Self-improvement additionally requires **algorithmic-core headroom**: the agent's transferable
> algorithmic core A_θ (reasoning · search · planning, per the Cognitive Core K_θ/A_θ/C_t/V) must have
> something the loop can *compound*. In the green zone this headroom is ≈ 0, and it is zero for **two
> composable, separable reasons**:
>
> - **(H1) Scaffold-encoded procedure (a measurement artifact).** The harness already inlines the
>   solving procedure, so distilled memory/skills are redundant — the loop measures null because there
>   is nothing to learn that the scaffold does not already provide. Reversibly removable by a
>   de-scaffolding ablation.
> - **(H2) An A_θ ceiling (a capability fact).** Even de-scaffolded and given the ideal environment
>   (rich diagnosis + operator library + exact verifier), the agent *exercises* A_θ but does not
>   *exceed* a trivial hand-coded policy; there is no residual capability for memory to compound.

The two are separated by the de-scaffolding flag; together they make the green-zone headroom zero.
This is a coordinate **orthogonal** to D/R/T (a domain property), to A (a delivery property), to α_f
(an oracle property), and to the substrate taxonomy (a *storage* property, A2). It is a property of
the **agent's core**, and it binds in the exact region all four say is clear.

**Operationalizing headroom-consumption (borrowed instrument, cited not owned):** we use the
retrieval / search / **discovery** trichotomy of Wang & Buehler (arXiv:2606.01444) — *discovery =
verified regime transition = residual content beyond left-Kan functorial transport*. Self-improvement
consumes headroom only if it produces **discovery** (residual beyond transport); if it is **search**
(transportable within a fixed schema) the headroom was already zero. Our measured self-improvement is
search: the accumulated memory/skill is functorial transport, residual = 0.

## 3. De-collision vs the family (precise)

| | Owns | Our fifth-coordinate claim (orthogonal / extends; does not collide) |
|---|---|---|
| **A1** | actuation A (adoption of a delivered correction); A=A_h×A_m | We measure A_θ *generation/orchestration* headroom, not correction adoption — orthogonal. We FILL A1's explicitly-named missing testbed (end-to-end LLM loop). RNA A≈0.85 cited verbatim; the severing/efficiency ablation is A1's, we do not restate it. |
| **A2** | D/R/T, manifold, τ, substrate taxonomy, pseudo-learning regime (a LOW-determinism null) | Our null is in a HIGH-determinism domain — a *different null cause*. The fifth coordinate is orthogonal to D/R/T and to "which substrate stores the improvement." We adopt (D,R,T) and cite; we add the axis A2's system omits. |
| **A3** | α_f, R_det ceiling for PHYSICAL oracles, judgment–computation asymmetry, "capability eats knowledge not computation" | We sit at A3's α_f≈1 CLEAN-oracle positive pole and show self-improvement can still be null — sourced not in the oracle but in the agent's A_θ. Boundary case that COMPLETES A3: capability can fail to eat even already-clean knowledge when the algorithmic core that would digest it is at ceiling. α_f cited, not re-derived. The residual hard tail (solver-hard) IS A3's residue — cited as A3, not claimed. |
| **B** | compounding-vs-amplification adjudication (loop vs matched best-of-N), SAT/A* | Different domain, different axis. We do not adjudicate amplification-vs-compounding; we locate a headroom coordinate. Cite as the adjudication counterpart. |

## 4. De-collision vs the external field (the crack's real novelty risk)

The 2026 field already has pieces near this; we must cite them and state the delta precisely.

- **"Verifiable reward is necessary but not sufficient / learnable-information-gain"** — *Self-Play Only
  Evolves When… Ensures Learnable Information Gain* (arXiv:2603.02218): task metrics rise while
  learnable information is unchanged. Plus the RLVR base-model-ceiling line (Yue et al.; Wu et al.,
  "Invisible Leash"), which A3 already cites at the RL-vs-base boundary. → **"Self-improvement can be
  null" is NOT our claim.** Our delta: (a) we give the null a *mechanism* decomposed into H1+H2, (b)
  we give it a *home* in the field's most complete where-can-self-improve coordinate system and show
  that system has a blind axis, (c) we isolate it in a *clean-oracle scientific domain* so it cannot
  be blamed on oracle/determinism/actuation, (d) we make headroom-consumption *measurable* via
  discovery-vs-search.
- **Diagnostic-framework space** — TIDE (2602.02196, AUV/LR/MI test-time metrics); "Truly
  Self-Improving Agents Require Intrinsic Metacognitive Learning" (2506.05109). → We are not another
  test-time-improvement metric suite; we add a *coordinate* to the where-is-improvement-possible map
  and pin it with a null in the all-green zone.
- **Scaffold-vs-model attribution** — Harness-Bench (2605.27922), ClawsBench (2604.05172), Agent
  psychometrics (2604.00594), "AI scientists produce results without reasoning scientifically"
  (2604.18805). → These do generic score decomposition in SWE/general/other-science domains; **must
  verify 2604.18805 directly** (it is the nearest scientific-agent scaffold-dominance paper). Our
  delta: coordinate-system blind spot + clean-exact-oracle isolation + discovery-vs-search, not a
  score-attribution method.
- **Discovery-vs-search measurable** — Wang & Buehler (2606.01444) OWN the residual-beyond-functorial-
  transport criterion. We BORROW it as an instrument and cite; we do not claim it.

## 5. Grounded evidence chain (green-zone framing; every number artifact-traced)

Green-zone coordinates (cited): D=R=T≈1 [A2]; A≈0.85 [A1]; α_f≈1 [A3]. All clear.

| # | Claim in the green-zone frame | Value | Source |
|---|---|---|---|
| S1 | Memory/skill self-improvement NULL in the green zone (frozen / K_θ-content / A_θ-skills) | **76 / 75 / 76** /100; skills 3/35 invoked, 0 decisive | `runs/meta_ladder_{frozen,accumulate,askill}/epoch_progress.json` |
| S2 | Mechanism H1 (scaffold redundancy): de-scaffold floor ≈ crutched ceiling, n=3 rounds | ceiling {6,4,3} μ4.33 · floor {8,3,5} μ5.33 (indistinguishable) | `runs/p1_logs/*.log` |
| S3 | Self-improvement here is SEARCH not DISCOVERY (residual beyond transport = 0) | 56/65 solves = LLM orchestrating classical search | provenance scan (documented) |
| S4 | Mechanism H2 (A_θ ceiling): agent exercises A_θ but ≈/< trivial policy, ideal environment | greedy 3/6 vs agent 2/6, complementary, **0/52 invalid decisions** | `runs/self_revise_agent/agent_compare.json` |
| S5 | The self-revising harness lifts modestly (diagnosis-timing > blind-selection) — bounds H2 | blind 1 / roundrobin 2 / greedy 3 of 12 short-hard | `runs/self_revise_v2/phase2_controllers.json` |
| S6 | Residual hard tail = α_f≈1 solver/thermodynamic wall (A3's residue, cited) | 258nt→bp0 only via 24-min brute SA; 108nt 13.5min fail; SOTA DesiRNA 85@1min·97@24h | probes + `docs/eterna100_sota.md` |
| S0 | Base capability spread (context) | DeepSeek 47 / GLM 58 / Sonnet 51 /100 | `runs/*_eterna100/summary.json` |

## 6. Contributions (what this paper owns)

- **C1 (primary):** the **algorithmic-core headroom coordinate** — a necessary condition for
  self-improvement that the D/R/T/A/α_f apparatus omits — and the **green-zone null** that exhibits
  it: a learnable, loop-closed, clean-oracle domain where every companion predicts compounding and the
  loop measures none.
- **C2:** the **H1/H2 decomposition** (scaffold-redundancy artifact vs A_θ-ceiling capability), made
  separable by a reversible de-scaffolding instrument.
- **C3:** an **existence + mechanism** result filling the family's named missing testbed (end-to-end
  LLM loop, clean-oracle, natural — not injected — null), operationalized via discovery-vs-search.

## 7. Honest limits / path to a top venue

- **Power (the binding weakness):** S1 n=1/arm; S2 n=3 rounds; S4 n=6×2; all one model (GLM-5.2).
  Current status = **existence + case study**, defensible at the boundary but not a rate. To land
  main-track: run S1 and S4 through ≥2 models (base runs for DeepSeek/Sonnet already exist) and more
  seeds; ideally add a **second clean-oracle domain** (formal-math-lite / SAT-design) to lift the
  fifth coordinate above n=1 domain and echo A3's positive pole.
- **Must engage prior art head-on** (§4): 2603.02218, TIDE, 2604.18805, and cite Wang & Buehler for
  the borrowed instrument. Verify 2604.18805 in full before submission.
- Do **not** claim: the discovery-vs-search criterion (Wang & Buehler), the A coordinate / α_f / D-R-T
  (companions), or a rate for H2 at current n.

## 8. Figure plan

1. **fig_greenzone** — the D/R/T/A/α_f radar for RNA, all axes green, with a fifth (headroom) axis
   pinned at 0; caption: "every companion axis clear, self-improvement null."
2. **fig_null** — S1 three-arm bars (76/75/76) + skills 3/35 invoked / 0 decisive.
3. **fig_H1H2** — de-scaffold floor≈ceiling (H1) beside agent-vs-greedy grid (H2).
4. **fig_search_not_discovery** — S3 attribution (56 orchestrated / 9 hand) annotated "residual beyond
   transport = 0."
5. **fig_gap** — budget-aligned SOTA (76 vs DesiRNA 85@1min/97@24h), residual = A3 wall (cited).

---

*One line: the field can now say where self-improvement is possible on four axes; we exhibit a domain
green on all four where it still fails, name the missing fifth axis (algorithmic-core headroom),
decompose why it is zero, and measure it as search-not-discovery — filling the family's own named
missing testbed without colliding with any of its owned claims.*
