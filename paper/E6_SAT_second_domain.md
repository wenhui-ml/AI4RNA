# E6' — SAT as Paper D's second deterministic domain, and the anti-collision boundary vs family B

Purpose: reviewers of a single-domain verifier-economics paper will ask "does this only hold for RNA, or
is it a repackaging of RNA metrics?" E6' answers by instantiating the SAME structure — a provably-nested,
decidable, dial-able, cost-priced verifier ladder + a controlled Goodhart measurement — in a second,
unrelated deterministic domain: Boolean satisfiability. This file specifies the design AND pins the
collision boundary against family paper **B (Compounding Is a Profile, Not a Verdict; openclaude_sat)**,
which also uses SAT.

## 1. The SAT verifier ladder (mirror of the RNA ladder)
Candidate = a truth assignment x∈{0,1}ⁿ to a CNF formula φ (n vars, m clauses, density α=m/n).
- `V0(φ,x)` = x satisfies φ. CHEAP, O(m); the proxy a plain SAT search optimizes.
- `V_r(φ,x)` = the whole Hamming ball of radius r around x satisfies φ (r-flip-robust). Decidable,
  O(Σ_{i≤r} C(n,i)·m); r is the STRENGTH KNOB. Provable nest: `V_r ⊆ V_{r-1} ⊆ … ⊆ V0`.
- (strong end) cluster/backbone-consistency: x lies in a LARGE solution cluster / respects the backbone —
  the SAT analog of "high Boltzmann probability". Expensive (sampling/#SAT-flavored); an approximate rung.
- HARDNESS KNOB = clause density α. Near the 3-SAT threshold (α≈4.267) solutions fragment into small,
  isolated clusters protected by frozen variables → satisfying assignments become FRAGILE (low r-robust).

Structural correspondence (why this is the same paper, not two):
| RNA (nested branch) | SAT |
|---|---|
| exact-match `fold==T` (cheap proxy) | `V0` satisfies φ |
| unique-MFE | (near-)isolated/forced assignment |
| P(target)≥θ (ensemble robustness) | r-flip-robust / large-cluster membership |
| length n (hardness) | clause density α (phase transition) |
| pKiss = out-of-domain BRANCH | (SAT analog optional: UNSAT certification is a different task) |

## 2. Controlled Goodhart on SAT (mirror of E2')
Fixed proposer (one local-search kernel, e.g. WalkSAT-style flips), vary ONLY the reward:
- cheap arm: reward = clauses-satisfied (→ stops at V0, on the constraint boundary).
- strong arm: reward = r-robustness / neighbor-satisfaction count (→ cluster interior).
Sweep optimization budget (flips). Prediction: cheap arm's r-robust-accept DECLINES / stays low as budget
rises while V0-accept →100% (gap widens), strongest near the threshold α; strong arm's r-robust-accept
rises. Same signature as RNA E2' → the effect is domain-general, not an RNA artifact. Cost per rung
(V0 vs V_r) is measured end-to-end as in E4' (V_r grows with r; a within-domain cost–robustness ladder).

## 3. Anti-collision vs family B — the load-bearing part
**What B owns** (read from `openclaude_sat/paper/compounding-profile.tex`): B studies self-improving LLM
agents EDITING solver heuristics (SAT/A*/TSP) under a **correctness-pinned, gaming-proof, SPEED-ONLY**
verifier; its results are two identification limits (actuation under discard-gating; oversight-burden
migration) and depth-vs-breadth **compounding** as a configuration-level fact. B's design *deliberately
removes* reward hacking by pinning correctness by construction ("give an LLM a verifier it cannot fool").

**Why D/E6' does not collide (they are opposite sides of the same axis):**
- B PINS the verifier ungameable to isolate *compounding*; D/E6' DIALS verifier strength to measure the
  *gaming↔cost* tradeoff B pins away. Different question, opposite move on the verifier axis.
- B's SAT task = LLM *edits a solver heuristic*, optimizing SPEED at fixed correctness. E6' SAT task =
  *find an assignment* under a robustness ladder. Different task on a shared domain (same relation as
  RNA C-vs-D: shared domain, different question).
- **The sharpening D adds (and must foreground):** B shows a correctness-pinned EXACT verifier removes
  *correctness* gaming. D shows a cheap EXACT verifier can still be a *weak proxy for a ROBUSTNESS
  objective* — an orthogonal gaming axis exactness does NOT close. So "decidable/exact ≠ ungameable":
  V0 is exact and unfoolable on correctness yet a weak proxy for r-robustness. This CONTRASTS with, and
  does not contradict, B.

**RULES (mirror POSITIONING_D's family rules):**
- Cite B as the SAT-compounding / gaming-proof companion; frame E6' as verifier-economics on SAT.
- Do NOT import: compounding-vs-amplification, depth-vs-breadth, best-of-N matching, actuation-under-
  discard, "gaming-proof by construction" as OUR claim (it is B's construction; we cite it as the contrast).
- Do NOT study solver-heuristic editing or LLM self-improvement here — E6' is about the verifier, not the
  agent's improvement loop. No A_θ / actuation language (that is A1/B).
- Keep it correctness-orthogonal: our "gaming" is robustness-proxy gaming, NOT correctness violation
  (all V0-accepted x are genuinely satisfying) — this is precisely what distinguishes it from the field's
  correctness-gaming and from B's correctness-pinning.

## 4. External positioning (cite, differentiate)
- SAT solution robustness / clustering / frozen variables / backbone / phase transition: ESTABLISHED
  statistical-physics of SAT (Zhou et al. frozen-variables/entropy; clustering transition) — CITE, do NOT
  claim the robustness notion.
- Reward hacking with correlated proxies / RLVR reward-hacking testbeds (`Robust Optimization … Correlated
  Proxies` arXiv:2604.12086; `Countdown-Code` arXiv:2603.07084): the general "cheap proxy gamed under
  pressure" is covered; our differentiator = a DECIDABLE, DIAL-ABLE, PRICED ladder in TWO natural
  deterministic domains (RNA + SAT), measured with zero evaluator variance, separating correctness-gaming
  (closed by exactness) from robustness-proxy-gaming (NOT closed by exactness).

## 5. Minimal implementation sketch (for when we build it)
- Instances: random 3-SAT, n∈{50,100,150}, α swept across {3.0, 3.8, 4.0, 4.2, 4.267, 4.4} (through the
  threshold). Use a standard generator; keep only satisfiable instances (verify via a complete solver).
- Verifiers: V0 (clause eval); V_r for r∈{1,2,3} (Hamming-ball enumeration; cap n·r for cost); optional
  cluster-size estimate via short random-walk sampling from x (approximate strong rung, clearly labelled).
- E6'-frontier: for assignments found by a cheap WalkSAT-style search, report V_r-survival vs r and vs α
  (mirror E1'); E6'-Goodhart: cheap vs robust reward, same kernel, budget sweep (mirror E2'); E6'-cost:
  end-to-end V0/V_r wall vs (n,r) (mirror E4').
- Deliverable = one cross-domain figure: the SAME cost–robustness / Goodhart shape appears in RNA and SAT
  → the phenomenon is a property of cheap-proxy verification, not of RNA.

## 5b. BUILT + RESULTS (2026-07-15, `experiments/sat_verifier.py`)
- **frontier** (n=60, budget 60000, k=20, ρ|SAT): V0-accept 85–100%; **median ρ|SAT drops monotonically
  0.233→0.167→0.133→0.117→0.117→0.100 as α 3.0→4.267** — clean mirror of RNA E1' (fragility grows with
  hardness). NOTE: early runs at n=80/budget 20000 were contaminated by solve-failures at high α (ρ set
  to 0 for non-SAT); fixed by conditioning ρ on SAT + raising the cheap-search budget (V0-accept reported
  separately as solver reach).
- **cost**: V0 O(m) 0.1–0.27 ms → ρ O(nm) 1.7–14 ms → r=2 O(n²m) 20–118 ms — within-domain cost-robustness
  ladder; span ~200–1000× (STEEPER than RNA's ~4×) — honest cross-domain contrast.
- **Goodhart** (witness-start, same kernel, cheap=sat-count vs strong=sat+slack): WEAK / domain-dependent
  (α=4.2 strong ρ 0.167 > cheap 0.133; α=3.6 no gap). → report "Goodhart severity is domain-dependent:
  strong in RNA (E2'), mild in SAT". RNA carries the Goodhart leg; did NOT launch a big SAT-Goodhart run.
- **Verdict**: the cross-domain claim rests on **frontier + cost mirroring in BOTH domains** (established
  → "not RNA repackaging"); Goodhart is RNA-strong / SAT-mild (honest boundary, not a failure).

## 6. Honest risks
- r-flip-robustness is a SPECIFIC, possibly narrow robustness notion; cluster/backbone is the richer but
  costlier one. State which we use and why; do not overclaim it is "the" SAT robustness.
- Building E6' is net-new work (a SAT harness); it is the generality insurance, not a quick add. Scope it
  as one clean mirror (frontier + Goodhart + cost), not a full SAT study.
