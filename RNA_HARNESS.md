# openclaude_rna — an RNA inverse-design specialization on openclaude

`openclaude_rna` turns the general **openclaude** code agent into an **RNA inverse-design /
hybrid-constraint-design expert agent**, and uses it as a **Meta-Learning Harness** that
collects clean, verifiable RL data ("data, reward, trace") for later RL training of an open
policy. It is an additive, feature-flagged specialization — none of openclaude's native
capabilities (tool calling, MCP, web search, memory, code execution, sub-agents) are
removed.

> **Red line:** everything here verifies *design success under a computational folding
> oracle* (ViennaRNA). No claim of in-vivo function is made or supported.

## Why RNA (the design compass)
Two InceptLabs papers select the terrain:
- **Oracle Residue Thesis** — a self-improving scientific agent is bounded by its reward
  oracle. Pick a domain whose oracle is *cheap, exact, self-checkable* (high α_f). RNA
  secondary-structure folding (ViennaRNA dynamic programming) is exactly that — the
  antithesis of DFT/MLIP. There is **no error-bearing surrogate in the reward path**.
- **Actuation Fidelity** — a self-improving loop only learns if the correct diagnosis
  actually controls the next action (A > 0). We *instrument A directly*.

RNA is therefore the **positive control** the catalysis harness lacked: a domain where the
four coordinates are all high, so a closed loop demonstrably learns.

The harness measures four coordinates:
| coordinate | meaning | RNA |
|---|---|---|
| **D_sym** | observation is an exact reproducible symbol | 1.0 (dot-bracket) |
| **R_det** | reward is exact / self-checkable, no surrogate | 1.0 (ViennaRNA) |
| **T_trace** | failure localizes to specific positions/steps | 1.0 (folding mismatch) |
| **A** | correct diagnosis controls the next action | *instrumented* |

## Architecture
```
openclaude agent (full autonomy: bash, code, web, memory, MCP, sub-agents)
   driver: OpenRouter (anthropic/claude-opus-4.8 | deepseek/deepseek-v4-pro | z-ai/glm-5.2 | local)
      │  RNA specialization tools (feature-flagged OPENCLAUDE_RNA=1)
      │    RnaVerify (canonical reward) · RnaFold · RnaDiagnose · RnaConstraints · RnaBenchmark
      ▼  JSON-over-stdio bridge (src/tools/rna/bridge.ts)
rna_harness/ (Python, conda env `rna`) — the exact science/verifier/RL-data substrate
   oracles/ (ViennaRNA multi-model panel) · verifier.py (canonical reward) · constraints.py
   metrics.py · actuation/ (diagnosis + A) · determinism.py (D_sym/R_det/T_trace)
   generators/ (classical floor) · llm/ (Opus baseline arm) · benchmarks/ (Eterna100 + hybrid)
   rl/ (RLVR schema, RFT/GRPO export, online reward_fn) · rl_train/ (RFT→GRPO reference)
   experiments/ (run.py zero-credit; agent_runner.py full-agent RL environment)
```
The TS tools are thin; all deterministic ground truth is Python. The agent may also just
write ViennaRNA code with Bash — the specialization plays to its code-agent strength.

## Quickstart

### 1. Environment (already provisioned as conda env `rna`)
```bash
conda create -y -n rna python=3.10 && conda activate rna
pip install "ViennaRNA>=2.6" numpy pandas matplotlib scipy requests tqdm pyyaml rich pytest
```

### 2. Zero-credit experiments (classical generators; no LLM needed)
```bash
conda run -n rna python -m rna_harness.experiments.run --experiment all \
    --max-len 100 --seeds 3 --max-steps 70 --k 8 --workers 48 --out rna_harness/runs/zero_credit_main
conda run -n rna python -m rna_harness.report.figures --summary rna_harness/runs/zero_credit_main/summary.json
```
Produces: the actuation-fidelity ablation, classical baselines, four-coordinate report, and
an RLVR dataset (`rft.jsonl` / `grpo.jsonl` / `trajectories.jsonl`).

### 3. The full agent (RL environment, needs OpenRouter creds)
```bash
export OPENAI_API_KEY=sk-or-...          # OpenRouter key
conda run -n rna python -m rna_harness.experiments.agent_runner \
    --model anthropic/claude-opus-4.8 --max-len 30 --n-tasks 5 --max-budget-usd 1 \
    --out rna_harness/runs/agent_opus
```
Drives the full autonomous openclaude agent per task and writes `agentic_rlvr.jsonl` —
(prompt, final sequence, verifiable reward, **full stream-json trace**) for each task.
Swap `--model deepseek/deepseek-v4-pro` or `z-ai/glm-5.2` — one flag, no code change.

**Actuation fidelity on the live agent.** `rna_harness/actuation/trace.py` reconstructs A
from the real openclaude tool-call chain: for consecutive RnaVerify calls it checks whether
the agent's next sequence actually changed the positions the previous call diagnosed as
wrong (`selected`) and whether that improved the fold (`affected`). `agent_runner` reports
per-task and aggregate `agent_A` / `agent_A_model`. This is the meta-learning-harness loop
closed and measured on the *actual* agent, not a simulation (measured A_model ≈ 1.0 for
Opus-4.8 on solved repair chains — the agent does act on the diagnosis).

### 4. Interactive agent (openclaude runs via bun; requires Node≥22 or bun)
```bash
export CLAUDE_CODE_USE_OPENAI=1 OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
       OPENAI_API_KEY=sk-or-... OPENAI_MODEL=anthropic/claude-opus-4.8 \
       OPENCLAUDE_RNA=1 RNA_HARNESS_DIR=$PWD RNA_HARNESS_PYTHON=$(which python)
bun dist/cli.mjs --allowedTools RnaVerify RnaFold RnaDiagnose Bash
# then:  /rna-design ((((....))))
```

### 5. Meta-learning harness (cross-case self-improvement over openclaude memdir)
```bash
# zero-credit mechanism validation (classical executor, 3 ablation arms):
conda run -n rna python -m rna_harness.meta.meta_runner --executor classical \
    --source hairpin_family --n-tasks 12 --out rna_harness/runs/meta
# live (openclaude agent recalls the learned playbook via CLAUDE_COWORK_MEMORY_PATH_OVERRIDE):
conda run -n rna python -m rna_harness.meta.meta_runner --executor agent \
    --model anthropic/claude-opus-4.8 --n-tasks 4 --arms with_memory no_memory
```
The harness distills **checkable** strategies from solved cases (`rna_harness/meta/`), writes
them to an openclaude-format memdir, and the live agent recalls them through its NATIVE memory
path (verified: a seeded memdir is echoed verbatim by the headless agent). `A_memory` measures
whether a recalled strategy applicable to a new case is actually adopted.

Zero-credit result (hairpin family, mean oracle calls to solve):
| arm | solve | mean evals-to-solve | A_memory |
|---|---|---|---|
| with_memory (learned) | 1.0 | **3.2** | 1.0 |
| no_memory (control) | 1.0 | 5.8 | — |
| permuted_memory (wrong) | 1.0 | 8.5 | 1.0 |

Learned memory cuts oracle cost (5.8→3.2 — capability compounds with experience); a *wrong*
playbook is still adopted (A_memory=1.0) yet *hurts* (8.5) — a direct demonstration that
adoption ≠ efficacy, and that the gain comes from the right learned strategies, not from
"having memory." This is the cross-case analog of the single-case actuation ablation and a
direct positive test of the Actuation-Fidelity thesis on a real closed loop.

### 6. RL training (deferred; data-format-ready) — see `rna_harness/rl_train/README.md`
RFT (rejection-sampling SFT) then GRPO with the ViennaRNA verifier as the online reward
(`rna_harness.rl.reward_fn.rna_reward`). Target open policy: Gemma4-31b (any HF causal LM;
or plug the data into an existing Qwen2.5 SFT/RL stack).

## Headline result (zero-credit, Eterna100 ≤100 nt, 40 tasks × 3 seeds)
The actuation-fidelity ablation on a hard, cheap oracle:

| feedback mode | solve rate | A | A_model | A_harness |
|---|---|---|---|---|
| **diagnosis** (loop closed) | **0.233** | **0.812** | 0.812 | 1.0 |
| random (model ignores directive) | 0.158 | 0.336 | 0.336 | 1.0 |
| silent_fallback (harness drops directive) | 0.158 | **0.0** | 0.336 | **0.0** |

Closing the loop lifts solve-rate (~+47% relative) and A; breaking it drops both, and A
cleanly attributes *where* it broke (model vs harness). The random-feedback ablation
confirms the loop genuinely uses the signal (not a "feedback-insensitive" null). Classical
baseline floor: rnainverse 0.60, local_search 0.45. Four coordinates D_sym=R_det=T_trace=1.0.

## Model portability
The correctness floor is the verifier (model-independent); a weaker driving model costs more
oracle calls, not correctness, and the classical generator is always available as a floor.
Running Opus-4.8 vs deepseek-v4-pro vs glm-5.2 on the same fixed verifier is a built-in
ablation (`agent_runner --model ...`).

## Remote dashboard (live web visualization)
A read-only FastAPI dashboard serves run summaries, figures, the actuation / four-coordinate
/ meta results, agent traces, and a **live log tail** (auto-refresh) for monitoring in-flight
runs.
```bash
conda run -n rna pip install fastapi "uvicorn[standard]"   # one-time
conda run -n rna python -m rna_harness.dashboard.app --host 0.0.0.0 --port 8811
# open http://<server-ip>:8811   (health at /health, JSON at /api/runs)
```
Tabs: **Overview** (headline metrics), **Runs** (artifacts/datasets), **Figures**, **Logs
(live)**. Set `RNA_DASHBOARD_LOG_DIRS=dir1:dir2` to expose extra log directories to the tail.
⚠️ Binding to `0.0.0.0` with no auth exposes it to anyone who can reach the port.

## Tests
```bash
conda run -n rna python -m pytest rna_harness/tests -q   # 31 passed
```
