# HANDOFF — Verifier-Economics → Cost-Repair RLVR (read this first)

**Purpose.** You are resuming, in a fresh session on server `dibm01` (38.9.57.135), a research program
led by Wenhui Chen (InceptLabs). This doc is the complete context to continue **without** the prior chat.
Archived raw chat log: `archive/session.jsonl` (same dir). Paper repo: `/data/wenhui/openclaude_rna`.
RLVR pipeline (new): `/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr`.

---

## 1. Where we are (one paragraph)

The paper `openclaude_rna/paper/verifier_economics/verifier_economics.tex` ("**'Solved' Is a Choice of
Verifier, Not a Fact**") went through **4 external review rounds**; it now compiles clean (~38pp, 0 undefined
refs) and every fix is in. Then we did an **AAAI-level self+independent review** and a **deep-research
frontier map**, which returned a hard verdict: as a *measurement* paper the topic is **borderline** for a top
venue, because its two would-be headline claims are **already taken** (see §3). We decided (user-approved) to
**reposition** around the one genuinely open, non-colliding contribution — **the verifier as a priced,
dial-able, routable object, demonstrated as a COST-REPAIR result in a real RLVR training loop** — and to run
that training on dibm01's 8×B300.

---

## 2. THE DECISION (what to build now)

**Reframe from "verifiers over-accept" (taken) to "you can buy soundness cheaply by dialing/routing the
verifier" (open).** The load-bearing novel positive result to produce:

> A **sound cheapest-first cascade / risk-calibrated dial** achieves the **strong-verifier's held-out RLVR
> training quality** (staying on Youden's **J = TPR − FPR > 0**) at **N× lower verifier compute** than
> always-strong; the cheap verifier alone games (quality plateaus/collapses, J→0).

This is COST-repair, not degrade-repair (degrade-repair via a stronger verifier is already published — §3).

---

## 3. Frontier map — what is TAKEN (avoid) vs OPEN (ours). From deep-research, adversarially verified.

**TAKEN (do NOT claim as novel):**
- "Deterministic/executable verifier over-accepts by representational scope" → **Helff et al. 2026,
  arXiv:2604.15149** ("LLMs Gaming Verifiers"; already cited in the paper as `\citep{gamingverifiers}`).
- "**Degrade-then-repair via a STRONGER verifier** in real RLVR training" → **Helff 2026** (extensional
  induces shortcuts; isomorphic/invariance verifier eliminates them). Also **Cai et al. 2025,
  arXiv:2510.00915** (repair via reward-gradient noise-correction PGFC/PGBC).
- Verifier fuzzing to surface over-acceptance (buggy vs strict, FPR 0.83/0.87/0.56) → **Ray 2026,
  arXiv:2606.01066** (diagnostic, NOT end-to-end trained RLVR — it lists that as a limitation).
- Generation-verification gap + weak-verifier ensembling for SELECTION → **Weaver, arXiv:2506.18203**.
- RLVR phase transition governed by **Youden's J = TPR−FPR** ("rate, not fate": while J>0, over-acceptance
  rescales convergence TIME, not final quality; degradation only once J≤0) → **"Rate or Fate?"
  arXiv:2601.04411**. USE this as principled grounding for the dial (stay J>0 at min cost).
- RLVR reliability/measurement position paper → **arXiv:2509.21882** (Leskovec/Choi/Saberi/Yao).

**OPEN / UNCONTESTED (ours):** the verified Helff finding literally states their repair "is NOT a
cost-priced, dial-able-strength, or abstaining verifier and NOT a routing/cascade — leaving the 'verifier as a
priced, dial-able, abstaining object' framing uncontested." No source in the sweep prices the verifier, builds
a continuous strength dial, or a **sound cheapest-first cascade in the RLVR reward path** with a
cost×robustness frontier. That is our lane.

---

## 4. Family non-collision boundaries (all same author; MUST NOT collide)

- **compounding-profile** (`/data/wenhui/openclaude_sat/paper/compounding-profile.tex`): PINS correctness
  *ungameable* (sealed reward) to study **compounding of self-improvement**; SAT/A*/TSP; frozen-weights
  editing an external solver. Explicitly disclaims "reward hacking / verifier noise" (its line 284).
  → verifier_economics owns the OPPOSITE (verifier is mis-specified/gameable; we DIAL strength).
- **VerifGaps / "Verification Gaps or Interface Gaps?"** (`/data/wenhui/openclaude_numerics`): verifier
  **FALSE NEGATIVES** (under-acceptance) + **interface contract** + a **verifier-validity audit protocol**;
  numerical/PDE domain. Frames itself as the *dual* of reward-hacking (cedes the false-POSITIVE/over-accept
  side). → do NOT lean on "audit the verifier's validity" or false-negatives; stay on over-acceptance +
  PRICING/ROUTING.
- **CycleMorph / "Order Matters"** (`/data/wenhui/cycle-morph`): **certified null hypothesis /
  per-instance machine-checked measurement validity**; order/commutator; transform-validity audit
  (25.4%→0%); pre-registered cross-model **null**; Z3/FOL. → do NOT lean on "certified null / we validated
  our measurement instrument" as novelty (theirs).
- **Net:** verifier_economics' ONLY safe unique lane = **pricing / dial / routing / cost-repair of the
  verifier** (no sibling touches it).

---

## 5. Experiment design (cost-repair RLVR) — build this on dibm01

- **Policy:** Qwen2.5-3B-Instruct (pilot) → Qwen2.5-7B-Instruct (main) → optional 14B (capability sweep).
  All cached in `~/.cache/huggingface/hub`.
- **Domain:** math. Train prompts from NuminaMath-CoT / MetaMathQA (cached); held-out eval MATH500-style.
  (Optional 2nd domain: code, reusing `openclaude_rna` HumanEval apparatus.)
- **Trainer:** ms-swift GRPO (`/data/wenhui/ms-swift`, has grpo_trainer) + vLLM rollouts. Envs with full
  stack (torch+vllm+trl+swift+deepspeed+flash_attn): **`wh-llm`, `gemma`, `cua`**.
- **Three verifier arms (built + CPU-validated, see §6):** cheap (loose: any number ≈ gold value;
  over-accepts) · strong (boxed answer value/symbolic-equal to gold) · cascade (cheap gate → escalate to
  strong on cheap-accepts; sound ε=0 because cheap ⊇ strong; pays strong only on survivors).
- **Metrics per arm (train + held-out):** ① held-out **true quality** (strict-verified acc); ② **verifier
  cost** in the reward path (wall/calls); ③ **Youden's J = TPR−FPR** vs strict-gold; ④ **gaming rate**
  (cheap-accept ∧ strong-reject).
- **Arms × seeds:** cheap/strong/cascade × 2 seeds (pilot 1 seed). Optional 4th arm: risk-calibrated dial.
- **Headline figure:** held-out quality vs cumulative verifier-cost, per arm — cascade tracks strong at a
  fraction of cost; cheap games. Plus J-over-training.

**GPU budget (8×B300):** pilot (3B, ~250 steps, 3 arms, 1 seed) ≈ ~0.5–1 day; main (7B, ~800 steps, 3–4 arms
× 2 seeds) ≈ ~3–5 days; +14B/+code ≈ +2–4 days. Memory is a non-issue (275GB/card); throughput dominates.

---

## 6. What is ALREADY built (in `/data/wenhui/openclaude_rna/paper/verifier_economics_rlvr`)

- `verifiers/arms.py` — the three arms + `reward_and_metrics(arm, completion, gold)` returning
  reward + cheap/strong/gamed/cost fields. **CPU unit test `test_arms.py` PASSES**: cheap over-accepts,
  cheap ⊇ strong (0 violations → cascade sound), cheap arm rewards a gamed sample (reward 1.0) while cascade
  rejects it (reward 0.0). Run: `cd /data/wenhui/openclaude_rna/paper/verifier_economics_rlvr && ~/miniconda3/envs/wh-llm/bin/python test_arms.py`.
- Dirs: `verifiers/ configs/ runs/ data/`.

## 7. Next steps (in order)

1. **(GPU-free, do now)** Wire `reward_and_metrics` as an ms-swift GRPO **custom reward plugin** (one per
   arm). Prepare NuminaMath train prompts (extract question + gold `\boxed{}` answer) + a held-out eval set.
   Write J/cost aggregation over training logs. Write pilot GRPO config (3B).
2. **(when GPUs free — a monitor was running on the OLD host; on dibm01 just check `nvidia-smi`)** Launch
   **3B pilot** (3 arms, ~250 steps) → validate full loop + get preliminary cost-repair curve.
3. **Main 7B runs** (3–4 arms × 2 seeds) → the headline result. Then fold into a repositioned paper draft
   (LLM/RLVR-core; RNA/SAT demoted to the clean controlled study; lead with the priced/dial/route thesis).

## 8. Server / logistics

- **dibm01 = 38.9.57.135**, user's own 8×B300, path `/data/wenhui`. SSH key `~/incept_sh_rsa`.
- Envs: `wh-llm`/`gemma`/`cua` (RLVR stack). Models & math datasets cached. 256 cores, 3TB RAM.
- The org Claude-API monthly spend limit was hit → heavy multi-agent workflows are capped, but **local GPU
  training is unaffected** (no API cost).
- Paper build: `cd openclaude_rna/paper/verifier_economics && pdflatex ... ; bibtex ; pdflatex×2`. Figures via
  `make_figs.py` (uses STIX serif font; run in an env with matplotlib + the run JSONs).
- The verifier_economics paper's own RLVR-adjacent evidence + all 4 review-round fixes are DONE; the new
  RLVR training is a SEPARATE, additive workstream to lift the paper from borderline to competitive.

## 9. Key constraints/preferences (carry forward)

- Chinese-preferring user (jiacheng / Wenhui Chen), targets AAAI-2027, values honesty over hype (no
  fabrication, no hiding contradicting data; hedge honestly).
- Be a polite shared-GPU user (cap memory, don't disrupt others' running jobs).
- Never echo secrets (OpenRouter key lives in a 0600 .orkey on the OLD host; not needed for local training).
