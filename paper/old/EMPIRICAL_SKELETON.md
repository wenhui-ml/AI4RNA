# Paper C — Empirical Skeleton (grounded evidence chain)

Assembled 2026-07-10 from `openclaude_rna/rna_harness/runs/`. Every number below is verified against a
named run artifact (the POSITIONING gate: no unverified counts ship). This is the honest backbone the
paper is written from. Domain: RNA secondary-structure inverse design under ViennaRNA 2.7.2
(Turner2004, 37°C, dangles=2). `solved` = design's MFE fold string-equals the target; no in-vivo claim.

Cross-refs: `POSITIONING.md` (de-collision vs family), memory `project-selfrevise-capability`,
`project-paperC-lit-calibration`, `reference-mit-categorical-discovery`.

---

## The closed evidence chain (the spine)

> Memory/skills self-improvement measures **NULL** (S1) — but the null is a **scaffold-redundancy
> artifact**: the harness pre-encodes the solving procedure A_θ (S2). Stripping it (de-scaffolding)
> shows the scaffold buys **no measurable capability** on the hard tail (S3). The realized capability
> is **LLM-orchestrated classical search**, not design-from-scratch (S4). A stronger *self-revising*
> harness helps only **modestly** and yields two clean mechanism findings (S5). Handing the wheel to
> the **agent's own A_θ** shows it is *exercised and competitive-but-not-superior* to a trivial
> hand-coded policy (S6). The residual is a **thermodynamic/search wall**, not a diagnosis or
> tool-availability gap (S7).

The unifying claim (the **bridge**, the externally-novel core per calibration): the *harness-effect
confound* (attribution literature), the *self-evolving-agent* question, and the *self-improving-harness*
programme are **one phenomenon** — a harness rich enough to score well has pre-encoded the procedure,
so the self-improvement loop has nothing to learn and measures null. Prescription: to test whether a
self-improving agent is *learning*, measure the **de-scaffolded floor** and report floor-and-ceiling,
not one harness-confounded number; count accumulation as learning only when it yields **residual
capability beyond functorial transport, verified by V** (discovery criterion, [MIT 2606.01444]).

---

## Grounded numbers (claim → value → artifact)

| # | Claim | Value | Source artifact |
|---|---|---|---|
| S0 | Base capability, no memory, full Eterna100-V2 (model spread) | DeepSeek-V4-pro **47** / GLM-5.2 **58** / Sonnet-5 **51** /100 | `runs/{deepseek_v4pro,glm5.2,sonnet5}_eterna100/summary.json` |
| S0′ | Memory-on, full 100 (GLM-5.2) | **65**/100 | `runs/glm52_eterna_memON/summary.json` |
| **S1** | **3-arm meta-learning NULL** (frozen / K_θ-content / A_θ-skills), 4 epochs, GLM-5.2 | **76 / 75 / 76** /100 | `runs/meta_ladder_{frozen,accumulate,askill}/epoch_progress.json` |
| S1′ | Distilled skills essentially never load-bearing | **3/35 invoked, 0 decisive** | `runs/meta_ladder_askill/` traces (documented) |
| **S3** | De-scaffolding floor vs crutched ceiling, tail-35, **n=3 rounds each**, GLM-5.2 (raw solve, same denom) | ceiling **{6,4,3}** μ=4.33 · floor **{8,3,5}** μ=5.33 → **indistinguishable (floor ≥ ceiling)** | `runs/p1_logs/p1_{ceiling,floor}_tail35_r{1,2,3}.log` |
| S3′ | Single-epoch autonomous floor (baseline-add denom) | **68/100 = 3/35 cracked** vs crutched 76 | `runs/meta_ladder_auto_frozen/epoch_progress.json` |
| **S4** | 76 baseline solves = LLM-orchestrated classical search | **56/65 solver-carried** (37 MCP + 16 Bash inverse_fold + 3 self-SA); 9 hand-design | provenance scan (documented, memory `project-m2-compounding-finding`) |
| **S5** | Self-revising harness on 12 short-hard (blind / roundrobin / greedy, same budget) | cracked **1 / 2 / 3** | `runs/self_revise_v2/phase2_controllers.json` |
| S5a | Mechanism: **diagnosis-timing > blind-selection** | pid 66 cracked by greedy ONLY (roundrobin has close_pair, blind-selects, fails) | same |
| S5b | Mechanism: **self-revision > blind** | pid 49 cracked by greedy+roundrobin, not blind | same |
| **S6** | **Agent-driven A_θ gate**: LLM picks operator from diagnosis vs hand-coded greedy, 6 discriminating targets ×2 seeds | greedy **3/6** · agent **2/6** · **0/52 invalid decisions** | `runs/self_revise_agent/agent_compare.json` |
| S6a | Not a subset — **complementary**: agent uniquely cracks 62 (greedy-fail); greedy uniquely 57,66; union 4/6; {65,61} = wall | — | same |
| S6b | Agent over-picks close_pair 26/52 (miscalibrated) yet fully exercises selection | op profile {close_pair 26, monolithic 12, gc 10, hier 4} | same |
| **S7** | Residual hard tail is solver-hard (thermodynamic wall) | 258nt→bp0 only via 24-min brute SA; 108nt short-hard 13.5min fail | probes (documented) |
| SOTA | Public anchors (deep-research verified) | DesiRNA **97@24h** (85@<1min, 95@<1h) · NEMO ~94 · LLM-method ~80 | `docs/eterna100_sota.md`, NAR 2025 gkae1306 |

---

## Mapping to POSITIONING contributions (+ calibration reframe)

The calibration (`project-paperC-lit-calibration`) demoted two of POSITIONING's original pillars:
capability-attribution is now shared with a 2026 harness-effect cluster (Harness-Bench 2605.27922,
"Stop Comparing…" 2605.23950, CORE-Bench 2606.26158); M4/FunSearch is a crowded paradigm. **Surviving
novel core → make it the thesis:**

- **C1 (primary) — the bridge + de-scaffolding measurement.** S1→S3 is the confound; the prescription
  (report floor-and-ceiling; measure the de-scaffolded floor) is the contribution the harness-effect
  papers lack (they swap engineered harnesses, never strip to a floor, never touch the self-improvement
  loop, no scientific domain). **Owns: the connection nobody made + the reversible de-scaffolding instrument.**
- **C2 — the A_θ-exercise gate (S6).** Given the *ideal* environment (rich diagnosis + operator
  library, agent decides), selection-A_θ is exercised (0/52 invalid) but competitive-not-superior to a
  trivial policy (2 vs 3, complementary) → the binding ceiling is search/operators + the thermodynamic
  wall, **not** tool/diagnosis availability nor A_θ absence. This sharpens the harness-effect story.
- **C3 — clean-oracle positive control (S0,S7,SOTA).** RNA-under-ViennaRNA = A3's α_f≈1 pole; residue-
  free, so any null indicts the *mechanism*, not the domain. Attribute A/coordinate to A1/A2 (cite,
  don't claim); SOTA framed budget-aligned (76 vs DesiRNA ~85 @minute, not 97@24h).
- **DEMOTE M4** to "one mode we probed; needs a capable subject" — cite AlphaEvolve/ShinkaEvolve, don't
  claim as method.

De-collision (unchanged, from POSITIONING): don't restate A1's RNA A-measurement / severing ablation
(cite); don't re-derive oracle-residue (cite A3); don't import B's compounding-vs-amplification framing.

---

## Figure plan

1. **fig_null** — S1 three-arm bar (76/75/76) + skills-invoked annotation (3/35, 0 decisive). "Memory null."
2. **fig_floor** — S3 floor-vs-ceiling, 3 rounds each, dot-plot with means; band shows overlap. "Scaffold buys nothing on the hard tail; single-point estimates mislead."
3. **fig_attrib** — S4 stacked bar (56 solver-carried / 9 hand-design). "76 = orchestrated search."
4. **fig_selfrevise** — S5 controller cracks (1/2/3) + the two labelled mechanism wins (66,49).
5. **fig_agent** — S6 2×6 grid greedy vs agent (green/blank), complementarity highlighted (62 agent-only, 57/66 greedy-only, 65/61 wall).
6. **fig_gap** — SOTA budget-aligned ladder (76 vs DesiRNA 85@1min/97@24h vs LLM ~80), "gap is budget/search, not methodology."

`make_figs.py`-style: each measured fig reads its run artifact; annotate n and caveats in-caption.

---

## Honest caveats (must appear; do not overclaim)

- **S1/S3/S6 are small-n**: S1 n=1/arm; S3 n=3 rounds/arm; S6 n=6×2. These are *existence/indistinguishability*
  observations, not certified rates. The **strongest, most-powered** claim is S3 (floor ≈ ceiling, n=3,
  clear overlap) — lead with it; frame S6 as a single-domain existence result.
- The old "**~73% of capability is scaffolding**" (single-point 3/35 vs 11/35) is **RETRACTED** — it does
  not survive multi-round replication (S3). Report the honest version: *scaffold contribution is within
  round-to-round noise on the hard tail.*
- S6 tests operator-**selection** A_θ only (not operator invention).
- Single domain (RNA). Generality is the family's job, not this paper's; the positive-control framing
  carries it partway but state the limit.
- To reach AAAI rigor: S1 and S6 want multi-seed × ≥2 models before the headline; S3 is closest to bar.

---

## What NOT to do next (per the plan)

Consolidate/write from this skeleton. Do **not** build more operators, bigger self-revise runs, or a
FunSearch engine — that is solver engineering (本末倒置), off the A_θ mission and off this paper's thesis.
