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
- RLVR phase transition governed by **Youden's J = TPR−FPR** ("rate, not fate": while J>0 noise
  rescales convergence TIME, not final quality; degradation only once J≤0). USE as principled grounding
  for the dial (stay J>0 at min cost). **CORRECT CITATION (verified online 2026-07-21 — an EARLIER
  version of this doc listed a WRONG/Anonymous title, which propagated a fabricated bib entry; now
  fixed):** Rad, Filom, Keivan, Mohajerin Esfahani, Kamalinejad. "Rate or Fate? RLV$^\epsilon$R:
  Reinforcement Learning with Verifiable Noisy Rewards." arXiv:2601.04411 (2026). NB the paper is about
  general verifier NOISE (FP+FN); over-acceptance is the false-positive (FPR) direction — cite it that
  way, do not imply the paper is specifically about over-acceptance.
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
  rejects it (reward 0.0).
- `verifiers/plugin.py` — ms-swift GRPO external reward plugin; registers `ve_cheap/ve_strong/ve_cascade`
  ORMs; logs per-sample {step, cheap, strong, gamed, cost} to `$VE_RUN_DIR/samples.rank*.jsonl`.
  **CPU test `test_plugin.py` PASSES** (run both tests with `~/miniconda3/envs/cua/bin/python`).
- `data/prepare_data.py` (DONE, ran 2026-07-19): `train.jsonl` 50k NuminaMath prompts w/ canonical numeric
  gold (`solution` column), `eval.jsonl` 500 held-out from source=='math' (MATH500-style, held out of train
  entirely), `train_smoke.jsonl` 256, `metadata.json` provenance.
- `scripts/aggregate.py` — samples.jsonl → per-step curve.csv (reward, strong_acc, gaming rate, J=TPR−FPR,
  cumulative cost & strong-calls). `scripts/eval_checkpoints.py` — vLLM greedy eval of checkpoints on
  eval.jsonl → RUN/heldout.jsonl (idempotent, skips done ckpts). **GPU-validated end-to-end 2026-07-19**:
  base Qwen2.5-3B-Instruct held-out strict acc = **0.586** (n=500, greedy) — the step-0 baseline for the
  headline figure; checkpoint saving + vLLM loading of saved ckpts also verified (smoke checkpoint-2: 0.576).
- `configs/pilot_3b.sh` — parameterized GRPO launcher (arm, gpus, seed, steps, tag; env-overridable
  VLLM_UTIL/DATASET/...). `scripts/smoke_test.sh` — 2-step full-loop smoke. `scripts/gpu_watch_launch.sh` —
  waits for stably-idle GPUs then tmux-launches all 3 arms, each auto-chaining train → held-out eval →
  aggregate (sessions ve_cheap/ve_strong/ve_cascade; monitor session ve_monitor; log runs/monitor.log).

**PILOT RESULTS (complete, 2026-07-20 02:00; 5 runs × 250 steps, all clean after port fix):**
- Port-collision lesson: concurrent same-host torchrun jobs MUST have distinct MASTER_PORT
  (pilot_3b.sh now maps cheap/strong/cascade → 29611/12/13; swift cli reads the env).
- **Held-out is FLAT in all 3B runs** (58.8→±2 noise; lr 1e-6 and 3e-6 alike). 7B lr 2e-6 drifts up
  mildly (66.6→68.4). But **on-policy train accuracy rises everywhere** (3B +4-6pts, 7B +11pts) —
  learning happens, transfer needs longer horizons + denser signal.
- Root cause of thin signal: 60–90% of GRPO groups had zero reward variance → zero gradient.
  **Fix = DAPO-style difficulty filtering** (keep prompts with base-7B pass-rate 1..7 of 8).
- **cheap arm does NOT game within 250 steps** (gaming 0.15→0.13, held-out flat, J≈0.58 with ~13pt
  free-reward premium). HEADLINE REFRAME (honest): don't promise collapse; sell the Rate-or-Fate-
  consistent story — cheap (J≈0.6) converges SLOWER; strong (J=1) fast but pricey; **cascade = strong's
  decisions at 60% of strong-calls (9,605 vs 16,000, measured)**. Cost axis is already solid.
- 7B-on-2-GPUs feasibility confirmed: 218GB peak, 16.4s/step, 800 steps ≈ 3.6h/arm.

**MAIN 7B RUN (launched 2026-07-20 02:06, tmux `ve_main7b` orchestrator):**
- `scripts/main7b_pipeline.sh`: phase 1 = 8-GPU difficulty filter of 24k prompts →
  `data/train_7b_filtered.jsonl`; phase 2 = 3 arms × 800 steps × lr 2e-6 × SAVE_STEPS=100 on GPUs
  0-5 (tmux ve_m7_cheap/strong/cascade), each auto-chaining eval+aggregate. ETA ~08:00 same day.
- Analysis when done: heldout.jsonl (quality) × curve.csv (cum strong-calls/J/gaming) per arm;
  then seed 1 (same script, SEED=1) and the headline figure.
- Rolling evals on GPUs 6/7 (`scripts/rolling_eval.sh`, tmux ve_eval6/7) eval checkpoints as they save
  (--min-age-s 180 guards against half-written saves) → all 8 GPUs busy, ETA pulled to ~06:30.

**EXPERIMENT QUEUE (user-approved 2026-07-20 02:37, tmux `ve_queue` → `scripts/queue_after_main.sh`):**
1. main7b seed 0 completes → auto-launch **main7b seed 1** (same filtered data, 3 arms × 800 steps).
2. seed 1 completes → auto-launch **Llama-3.1-8B-Instruct seed 0** (cross-family proof; weights at
   `/data/wenhui/LowRankClone_improve/models/Llama-3.1-8B-Instruct`; runs its OWN difficulty filter —
   pass rates are policy-dependent — into `data/train_llama8b_filtered.jsonl`, tag `llama8b`).
3. Code domain: DROPPED for AAAI-2027 timeline (camera-ready extra at best).
- `main7b_pipeline.sh` is now parameterized (MODEL/TAG/FILTERED/FILTER_DIR/SEED/STEPS) and serves any
  family. Filter yield 7B: kept 7113/24000 (32% all-wrong, 38% all-right → 70% zero-gradient explains
  the flat 3B pilot). Track progress: `runs/queue.log`, `runs/<tag>_pipeline.log`, `tmux ls`.

**RISK-CALIBRATED DIAL ARM (added 2026-07-20 ~05:40, user-approved "highest single-item value"):**
- 4th arm `dial` in `verifiers/arms.py`: escalate a cheap-accept to strong with prob `dial_p` via a
  DETERMINISTIC content coin (sha256 of completion+gold — reproducible across dist workers, unlike
  per-process-salted `hash()`). dial_p=0 ≡ cheap, dial_p=1 ≡ cascade; interior p traces the CONTINUOUS
  cost×robustness frontier (makes the title's "dial-able" literal, not 3 discrete points).
- Plumbed: plugin `ve_dial` reads env `VE_DIAL_P`; `pilot_3b.sh` (arm=dial, port 29614, VE_DIAL_P
  passthrough — both ADDITIVE, backward-neutral); `aggregate.py` counts strong-calls via `escalated`
  flag with legacy fallback (verified: old cascade logs still report 9605, unchanged).
- CPU-tested offline BEFORE queueing (`test_dial.py`, all PASS): p=0≡cheap, p=1≡cascade, escalation
  frac≈p (0.257/0.519/0.759), cost monotone↑ & rewarded-gaming monotone↓ in p, deterministic, bad-p
  raises. test_arms/test_plugin regressions still PASS; live trainings unaffected by edits.
- QUEUED (NOT touching the running ve_queue): tmux `ve_queue_dial` → `queue_dial.sh` waits for the 4
  llama8b artifacts, then `run_dial_sweep.sh` runs p∈{0.25,0.5,0.75} on 7B Qwen+filtered (6 GPUs);
  with existing cheap(p0)/cascade(p1) main7b runs that's a 5-point frontier. Zero GPU contention.

**7B SEED-0 RESULT (COMPLETE 2026-07-20 06:25) + LONG-HORIZON PROBE (queued):**
- On-policy: strong 55.1→62.4, cascade 55.0→61.2 (learn, gaming↓); cheap 52.2→53.3 (NO gain),
  gaming↑ 23.4→26.1. Held-out FLAT for all (~66→68 ±2). Cost: cascade 39,503 vs strong 51,200 = 23%
  fewer strong-calls (3B was 40%). Paper §"Cost-Repair..." updated, compiles clean, 0 PENDING;
  makes NO held-out quality claim (honest) — sells cost + on-policy reward-hacking dissociation.
- Because even STRONG doesn't move held-out at 800 steps / lr 2e-6, a longer-horizon probe is queued
  (user-approved): tmux `ve_queue_long` → `queue_longprobe.sh` waits for the dial sweep, then runs
  cheap/strong/cascade × **2000 steps × lr 5e-6** (tag long7b, GPUs 0-5, ports 2964x, SAVE_STEPS=200).
  Goal: get strong's held-out to move and show cheap falls behind at that horizon. max_steps overrides
  epochs (verified: config has no --num_train_epochs), dataloader wraps — 2000 steps will run fully.
- FULL CHAIN ORDER (all auto, tmux): ve_queue (seed1 → llama8b) → ve_queue_dial (dial sweep) →
  ve_queue_long (long probe). Each waits on the prior's curve.csv artifacts. Check `runs/queue.log`.
- 8-GPU PACKING (user directive 2026-07-20 09:56 "maximize GPU util"): dial + long phases were 6-GPU
  (6,7 idle ~15h total); now fill 6,7 with a useful 2nd seed. dial sweep = p{0.25,0.5,0.75} s0 +
  p0.5 s1 (run_dial_sweep.sh rewritten to explicit `p:seed:g1:g2:port` specs). long probe = cheap/
  strong/cascade s0 + strong s1. Ports 2963x (dial) / 2964x (long), all-distinct (verified). Both
  waiters restarted from the parked (not-yet-fired) state = zero risk. long-probe prereq now also
  waits main7bd050_dial_s1 so 6,7 is free before it reuses that pair.

**ENV GOTCHAS (learned the hard way, 2026-07-19):**
- `wh-llm` and `gemma` have a BROKEN flash_attn (no `flash_attn.ops`) → vLLM crashes at model load.
  **Use `~/miniconda3/envs/cua`** (flash_attn 2.8.3 + vllm 0.23 + swift 4.3.2 site-packages, API-compatible
  incl. `swift.rewards.orms`). CPU-only work (data prep, aggregate) is fine in wh-llm.
- HF hub cache entries for Qwen2.5-3B/7B-Instruct are EMPTY husks (refs only, no weights). Real local
  weights: `/data/wenhui/LowRankClone_improve/models/Qwen2.5-{3B,7B}-Instruct` (pass as --model path;
  pilot_3b.sh already does).

## 7. Next steps (in order)

1. ~~GPU-free wiring~~ **DONE 2026-07-19** (plugin, data, aggregation, launcher — see §6).
2. **Smoke test** (2-step full GRPO loop on shared GPUs) then **hang `gpu_watch_launch.sh` in tmux** —
   it launches the 3B pilot (3 arms × 250 steps, 2 GPUs/arm) automatically when 6 GPUs are stably idle,
   and each arm auto-runs held-out eval + aggregation on completion.
3. Pilot analysis: headline curve = heldout.jsonl (quality) × curve.csv (cum cost/J/gaming) per arm.
4. **Main 7B runs** (3–4 arms × 2 seeds; model path in §6) → the headline result. Then fold into a
   repositioned paper draft (LLM/RLVR-core; RNA/SAT demoted to the clean controlled study; lead with the
   priced/dial/route thesis).

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
