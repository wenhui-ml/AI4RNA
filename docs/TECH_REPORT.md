# openclaude_rna: An RNA Inverse-Design Meta-Learning Harness — Technical Report

**2026-07-03 · InceptLabs · Judge: ViennaRNA 2.7.2 (Turner2004, 37 °C, dangles=2)**
*Language: **English** · [中文](TECH_REPORT_zh.md)*

> **Red line**: throughout this report, "success / solved" means **design-correct under the folding oracle only** (the designed sequence's MFE structure == the target dot-bracket). It does **not** imply any in-vivo function or expression.

## Project Philosophy & Goal

**End goal**: a general RNA inverse-design agent. What Claude Code is for people who cannot code, this is for RNA inverse design — hand it a target structure, get back a correct sequence, no expertise required.

**Path — two stages of self-improvement:**

- **Stage 1 · Push meta-learning to its limit.** The agent explores and solves each Eterna100-V2 puzzle using openclaude's **full Claude-Code-class capabilities** (tools, code execution, memory, MCP) in a self-loop that distills every trace into transferable strategy. We drive *inference-time* meta-learning to the ceiling a **fixed** model can reach. The **verifier is the deterministic folding oracle**; mature inverse-design algorithms (e.g. DesiRNA) define the target/ceiling to match or beat. We then **generalize from the fixed 100 puzzles to any open RNA target** (mature algorithms certify designability and generate open tasks). Closing this loop yields an agent adapted to arbitrary RNA inverse-design problems — useful to non-experts, the way Claude Code is.
- **Stage 2 · Raise the base model.** Every trace is collected as agentic RLVR data. Once meta-learning **saturates against the model's ceiling**, we use that data for on-policy RL (GRPO) to train a **stronger base model** — the same ceiling-lifting progression as opus 4.1 → 4.6/4.7/4.8.

**Why RNA makes this tractable** (see Part 3): the folding oracle is free, exact, and self-checkable → R_det ≈ 1 (no oracle residue) and A is high (loop closed), so self-improvement can actually **compound** — the precondition catalysis lacked. Success is measured by **generalization to open problems and usefulness to non-experts**, not by an Eterna100 score alone.

## Abstract

We specialize the general coding agent openclaude into an **RNA inverse-design solver** and layer on a **cross-task self-improvement (meta-learning) layer**. The goal is not to pick the strongest static model, but to **drive open frontier models, via a self-evolution loop (memory distillation + RL), toward closed-model / SOTA parity**. Three **memory-free** base-model runs on Eterna100-V2: **GLM-5.2 56 · Sonnet-5 51 · DeepSeek-v4-pro 47** (/100). **90–98 % of failures are 1500 s wall-clock timeouts**, rooted in classical search failing to converge on hard long structures — **not model reasoning, not tool selection**. Theoretically, the framework absorbs two InceptLabs papers and argues that **RNA-under-ViennaRNA is a positive control that simultaneously clears both of their obstacles**: R_det ≈ 1 (no oracle residue), measured **A ≈ 0.86** (loop closed), and its residual bottleneck (search) is of the **amortizable** kind.

## 0. Overview

**Self-evolution loop (the mission)** — a deterministic judge supplies verifiable reward, driving open models to bootstrap toward SOTA:

```mermaid
flowchart LR
    BASE["Open frontier base models<br/>DeepSeek-v4-pro · GLM-5.2 · Kimi"]
    HARNESS["openclaude_rna<br/>Meta-Learning Harness"]
    EVO{"Self-evolution loop<br/>memory distillation + RL"}
    ORACLE["Eterna100-V2<br/>folding-oracle judge"]
    GOAL["Approach closed / SOTA<br/>Sonnet-5 · DesiRNA 97/100"]
    BASE --> HARNESS
    HARNESS -->|"design sequence"| ORACLE
    ORACLE -->|"deterministic verifiable reward"| EVO
    HARNESS --> EVO
    EVO -->|"transferable knowledge + stronger weights"| HARNESS
    EVO --> GOAL
    classDef goal fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef judge fill:#e3f2fd,stroke:#1976d2,color:#0d47a1;
    class GOAL goal;
    class ORACLE judge;
```

**Per-task inner loop (diagnosis-driven repair) + cross-task outer loop (distillation / RL):**

```mermaid
flowchart TB
    AG["openclaude Agent<br/>autonomous · not a fixed workflow"]
    T["target structure (dot-bracket)"] --> AG
    K["knowledge: rna_expert + motif_kb"] --> AG
    M[("memory (memdir)")] --> AG
    AG <-->|"call"| TOOL["tools<br/>RnaVerify · parallel_search 160 cores · Bash"]
    AG --> SEQ["candidate sequence"] --> V["verifier<br/>ViennaRNA folding"]
    V -->|"diagnosis: missing/spurious/mismatched pairs"| AG
    V --> Q{"MFE == target?"}
    Q -->|"solved / unsolved"| REC["RLVR sample<br/>trajectory + verifiable reward"]
    REC --> DIST["distill_strategies"] --> M
    REC --> RL["weight-level RL / GRPO"] -.->|"update weights"| AG
    classDef store fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef judge fill:#e3f2fd,stroke:#1976d2,color:#0d47a1;
    class M,DIST store;
    class V,Q judge;
```

---

# 1 · Framework

**Problem**: given a target secondary structure + constraints (GC window / fixed motif / forbidden subsequence), design a sequence over {A,C,G,U} whose ViennaRNA MFE == the target. `solved` = MFE string equals target; `uMFE` = target is the unique MFE (stricter).

**Layers (open-ended and autonomous, not a fixed workflow):**

| Layer | File | Role |
|---|---|---|
| Judge | `verifier.py` | Deterministic folding oracle + multi-energy-model agreement; emits **per-position diagnosis**; cannot be gamed |
| Agent | `agent_runner.py` | Drives openclaude over OpenRouter; **swap models by changing only `OPENAI_MODEL`**; auto-enables reasoning for OpenRouter; each task = one agentic RLVR sample |
| Tools (MCP) | `mcp/` | Knowledge: `motif_kb` (tetraloop tables / by-loop-length recommendation / structure analysis); compute: `parallel_search` (multi-restart SA, fan-out ≤ **160 cores**), `batch_fold` (clean subprocess to avoid deadlock); native `RnaVerify/Fold/Diagnose/Constraints` |
| Memory | `meta/` | Native memdir; distills **checkable** strategies from solved tasks (`hairpin_motif` / `gc_closing`) → supports A_memory; externalized text rules, **transferable across models** (stored in `memory_library/`) |
| Dashboard | `dashboard/` | Self-contained SVG; **live scoreboard** for in-progress runs (scan traces → oracle scores on the fly, 20 s auto-refresh) |

**Metrics = a four-coordinate diagnostic system** (adopted from the two papers; the first three are domain preconditions, the fourth is a system property):

| Coordinate | Meaning | RNA value |
|---|---|---|
| **D_sym** data purity | Is what we learn the thing itself? | ≈ 1 (dot-bracket is a reproducible symbol) |
| **R_det** reward determinism | Is the criterion unambiguous and non-drifting? | ≈ 1 (MFE == target: deterministic, exact) |
| **T_trace** trace fidelity | Is failure attributable to a step? | ≈ 1 (per-position diagnosis) |
| **A** actuation fidelity | Does a correct diagnosis actually control the next step? | **measured 0.85–0.86** |

A = A_harness × A_model; A_memory is the cross-task version of A. All four high → the **"learning compounds (goal cell)"** of Paper 1 (see §3).

---

# 2 · Three-base-model comparison (Eterna100-V2)

**Setup**: identical config for all three, `--n-tasks 100 --max-len 400 --timeout-s 1500 --workers 3`, **pure base model · no memory · no budget cap**, reasoning on.

| Model | solved | capability | mean_reward | agent_A | mean RNA calls |
|---|---|---|---|---|---|
| **GLM-5.2** | **56** | 58 | 1.21 | 0.86 | 2.61 |
| **Sonnet-5** | 51 | 51 | 1.08 | 0.85 | 4.44 |
| **DeepSeek-v4-pro** | 47 | 47 | 0.93 | 0.86 | 4.36 |

**Failure analysis (300 attempts):**
- **~90–98 % are 1500 s timeouts** (DeepSeek 46/53 · GLM 41/44 · Sonnet 48/49); genuine wrong answers only 1–6; actuation loses almost nothing (delivered ≈ capability).
- **Frontier is 400 nt for all three** → length is not a death sentence; **structural hardness** is. **33 puzzles are unsolved by all three** (nearly all long + all timeouts).
- **Root cause**: on hard long structures classical SA returns `n_solved=0` within 1500 s (e.g. Cesspool 378 nt: 9 `parallel_search` calls, all 0); the model re-invokes it into the wall — **not reasoning, not tool selection**.

**Public SOTA (the target is to beat this, not local self-scores)**: DesiRNA **97** (24 h/puzzle) · NEMO 90 · LLM-SOTA ~79–81 · this framework, bare base models 47–56.

---

# 3 · Theoretical grounding: clearing catalysis's two obstacles

**The two obstacles (the self-improvement failure modes the two InceptLabs papers diagnose):**
- **Obstacle 1 · A→0 (loop unclosed)**: a correct diagnosis is dropped before it controls the next step. Catalysis A_before ≈ **0.095**, SAC **0/1781** (a Stage-4 silent bypass overrode the correct directive).
- **Obstacle 2 · oracle residue**: a physical oracle is expensive / approximate / not self-checkable → R_det has a structural ceiling `α_f + q_f(B)(1−α_f) < 1`. Reaction barriers are a low-α_f, oracle-bound family, flat at fixed budget. *"Capability eats knowledge, not computation."*

| Dimension | Catalysis (failed) | RNA-under-ViennaRNA (this work) |
|---|---|---|
| Oracle nature | DFT/experiment: expensive · approximate · **not self-checkable** | ViennaRNA: near-free · exact · **self-checkable** |
| **R_det** | < 1, has oracle residue | **≈ 1, no residue** (closed/symbolic end) |
| **α_f** | low (oracle-bound) | high (same end as games/code/math) |
| **A_harness** | → 0 (silent override) | **≈ 1** (diagnosis goes straight to the next turn, no silent override) |
| **A (measured)** | 0.095 / SAC 0.00 | **0.85–0.86** |
| Can learning compound? | No (both obstacles hit) | **Yes (both cleared)** |

```mermaid
flowchart TD
    F["Learning curve flat?"] --> MA["Measure A first<br/>(only log-countable coord)"]
    MA -->|"A→0"| SEAM{"Which seam?"}
    SEAM -->|"A_harness→0 silent override"| H["repair control path<br/>[catalysis SAC 0/1781]"]
    SEAM -->|"A_model→0 prior overrides context"| MM["swap / retrain model"]
    MA -->|"A high"| DOM{"min D,R,T low?"}
    DOM -->|"low: oracle residue"| DE["determinism engineering<br/>[catalysis barrier: must buy oracle]"]
    DOM -->|"high: domain learnable"| GOAL["learning compounds (goal cell)<br/>[RNA: A≈0.86 · R_det≈1]"]
    classDef good fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef bad fill:#ffebee,stroke:#c62828,color:#b71c1c;
    class GOAL good;
    class H,MM,DE bad;
```

**Reconciling the 90–98 % timeouts**: these are neither A→0 (A = 0.86) nor oracle residue (R_det ≈ 1); they are a **search-compute** convergence limit on hard long structures. And search is the **amortizable** kind of computation (search → policy, the AlphaZero class) — **exactly what the self-evolution loop is built to eat** — the opposite of catalysis's non-amortizable residue. ⚠️ **Attribution guard** (echoing Oracle Residue §7): the LLM must **internalize better search / decomposition** (genuine learning), *not* merely be given a larger `parallel_search` step count (spending compute + the attribution trap). Clean test: **hold the tool budget fixed, add only experience, and see whether the hard family's solve-rate rises**.

---

# 4 · Progress & advantages: have we cleared the fundamental obstacles?

**Progress**: three memory-free base-model baselines complete; dashboard made live; memory library preserved (playbook support 20/10); framework validated on both determinism and actuation.
**Advantages**: ① model-agnostic (one-line model swap) ② deterministic, self-checkable oracle (R_det ≈ 1) ③ **A is a first-class, measured metric (0.86)** ④ externalized memory, transferable across models ⑤ positive-control positioning — it **disentangles** catalysis's confounded triple ("domain unlearnable / loop unclosed / oracle residue").

**Have we cleared them — an honest reading:**

| Fundamental obstacle (the two papers) | Status | Evidence / framing |
|---|---|---|
| Obstacle 2: oracle residue (R_det < 1) | **Structurally sidestepped** | Chose a free·exact·self-checkable folding oracle → R_det ≈ 1; not "solved the residue" but **chosen a domain where it does not bind** |
| Obstacle 1 · harness seam (A_harness→0) | **Cleared (measured)** | No silent override; **A ≈ 0.86**, two orders of magnitude above catalysis; A is a first-class metric |
| Obstacle 1 · model seam (A_model < 1) | **Partial** | A ≈ 0.86 → ~14 % of correct diagnoses still not adopted |
| Does the self-improvement curve actually rise (compound)? | **Not yet demonstrated** | All three runs are no-memory baselines; A_memory and multi-epoch lift remain to be run |

**In one line**: the two **structural preconditions** are cleared (R_det ≈ 1 by domain choice; A ≈ 0.86 by loop closure + instrumentation), placing us in the **only cell where learning can compound** — but that it **does compound** is not yet shown. **We have crossed "able to learn"; we have not yet proven "is learning"** — which is exactly what the next experiment targets.

> **Red line, restated**: all of the above holds only under the folding oracle. The moment in-vivo function is the target, the physical oracle and its residue return, and Obstacle 2's ceiling reapplies.

---

# Next steps

1. **Distill experience (offline)**: run `distill_strategies` over all solved tasks from the three runs → refresh `memory_library`.
2. **Attack the hard-core 33**: strengthen search + add a decomposition strategy, **strictly separating "the tool got stronger" from "the LLM's orchestration got stronger."**
3. **Close the self-evolution loop**: ① memory-on, multi-epoch (measure A_memory + lift); ② weight-level RL / GRPO.
4. **Transfer**: GLM-5.2 no-memory baseline → with DeepSeek-distilled memory; measure the transfer gain.

# Appendix · Reproduce

```bash
python -m rna_harness.experiments.agent_runner \
  --model z-ai/glm-5.2 --n-tasks 100 --max-len 400 --timeout-s 1500 --workers 3 \
  --out rna_harness/runs/<name>
# env: CLAUDE_CODE_USE_OPENAI=1 · OPENAI_BASE_URL=…/openrouter/api/v1 · OPENAI_API_KEY=<key> · RNA_MAX_CORES=160
# view: http://<host>:8811/report/<name>
```

**Key paths**: `agent_runner.py` (driver) · `verifier.py` (judge) · `mcp/` (tools) · `meta/` (memory) · `dashboard/` (visualization) · `memory_library/` (learned knowledge).
**References**: Chen, W. *Actuation Fidelity* (InceptLabs, 2026); Chen, W. & Li, Z. *The Oracle Residue Thesis* (InceptLabs, 2026).
