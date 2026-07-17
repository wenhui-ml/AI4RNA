# RSSD — Residual-targeted Self-Revising Skill Distillation

*Design of record. 2026-07-16. Grounds `rna_harness/cognitive/skills/pk_decompose.py` +
`rna_harness/experiments/rssd_compounding.py`.*

## The one open question this attacks

The cognitive core (`rna_harness/cognitive/`) cleared only the **low bar** (cheap-first cascade ≪
always-SOTA on cost). The **real bar — does accumulated PROCEDURE actually compound?** — is unproven.
Two prior results bracket it:

- **Content accumulation → NULL** (retrieval warm-start; memorising Eterna answers). Memorising doesn't help.
- **`learning_vs_cold` can only REORDER a fixed toolbox** → it can trim cost but can *never* crack a
  residual. If no tool in the box solves a puzzle, reordering solves nothing.

RSSD adds the missing degree of freedom: a **new distilled procedure** that cracks residuals no existing
tool solves, plus the router **learning when to fire it** — and proves the gain is genuine by measuring it
on **held-out** targets the skill and router never saw (MIT discovery-vs-retrieval criterion).

## The residual we target (measured, not assumed)

Full-251 Pseudobase++ baseline (DesiRNA -t 300, R=10, pKiss-verified): **180 solved, 71 residual.**
Of the 71 residual, DesiRNA's best candidate is:

| pKiss bp from target | count | note |
|---|---|---|
| **1** | **23** | one crossing base-pair off — prime repair targets |
| 2 | 12 | |
| 3–4 | 21 | |
| ≥5 | 15 | genuinely hard / long (n up to 137) |

35/71 are bp ≤ 2 near-misses. **Key fact:** DesiRNA and REMC both optimise *MFE energy* and never call
pKiss during search, so nothing in the current toolbox ever repairs against the true pseudoknot oracle.

## The distilled skill: `DecomposePKTool` (pKiss-guided decompose–compose–repair)

A new `Tool` (same uniform interface, cost-tagged, oracle-verified) that encodes a transferable PROCEDURE,
not memorised answers:

1. **Decompose** target `T` → nested skeleton `round_only(T)` (the `()` part) + crossing pairs
   `pk_pairs(T) \ pk_pairs(skeleton)` (the `[] {}` layers). Positions are disjoint by construction.
2. **Sub-solve the skeleton cheaply** — REMC on the nested skeleton (bounded wall).
3. **Compose / enforce crossing complementarity** — set each crossing pair to a Watson–Crick pair
   (seed-varied G-C/C-G). This is the constraint MFE can't optimise but the pseudoknot needs.
4. **Verify with the CORRECT oracle** — pKiss on the full `T`. Solved ⟺ bp = 0.
5. **Targeted repair (the novel bit)** — pKiss-in-the-loop hill-climb: localise the mismatch (symmetric
   difference of folded vs target pair sets), mutate only wrong positions (paired positions mutated as a
   complementary pair), accept on bp decrease, bounded by a wall budget. **No existing tool does this.**

Cost = solver CPU-seconds + pKiss oracle-seconds via `cost.solver_cost_usd`, so it slots directly into the
cost-to-solve metric alongside REMC/DesiRNA.

## Gaming-proof held-out compounding test

Deterministic split of the 251 targets into **TRAIN** and **HELD-OUT** (index parity). Simulation-based to
stay affordable (DesiRNA column reused from the cached pk251 run; REMC + `DecomposePKTool` columns computed):

- **Baseline arm:** toolbox = {REMC, DesiRNA}. No skill.
- **RSSD arm:** toolbox = {REMC, DesiRNA, DecomposePKTool}. Router LEARNS on TRAIN, is **FROZEN**, then
  evaluated on HELD-OUT.

**Two separately-reported claims on HELD-OUT:**

- **Capability compounding (Δsolve):** RSSD.solved − Baseline.solved > 0 ⟺ the distilled skill cracked
  held-out residual the toolbox couldn't. (The thing `learning_vs_cold` structurally cannot show.)
- **Cost compounding (Δcost):** at equal solve set, learned routing reaches it cheaper than a cold cascade
  (routes to the skill / DesiRNA directly on predicted-hard, skips doomed cheap tries). This is the "真门槛".

**Honesty gates:** the skill and router never touch HELD-OUT targets → any gain is transfer, not memorisation.
A NULL is reported as a NULL and localises where procedure-learning helps vs not (e.g. cracks bp=1 but not
bp≥5). Costs are a loaded-box upper bound; the arms share one outcome table so the *delta* is robust.

## Findings (2026-07-16, measured)

**Capability (Δsolve) = NULL — and we explain it mechanistically.** pkdecomp did NOT crack any bp≤2
near-miss residual (0/35; every one stayed at *exactly* bp=1). Diagnosis: on pseudobase:160 the crossing
pair (9,18) is missing in **11/12** random Watson–Crick assignments, on :185 the pair (10,17) in 8/12 — a
specific **fragile terminal crossing pair** pKiss will not make the ground state for almost any sequence.
That **DesiRNA (an MFE optimiser) and pkdecomp (a pKiss optimiser) independently plateau at the same bp=1
pair** is strong evidence the residual is **designability-limited, not search-limited**. A skill cannot
crack what has no solution. (Parallels the Eterna designability diagnosis: the wall is the problem, not the
solver.)

**Cost (the "真门槛") = large win.** On the *solvable* PK targets pkdecomp solves at **~$3×10⁻⁶**, and most
solve at **zero repair** — the decompose+compose sequence already folds correctly under pKiss (16/16 of the
DesiRNA-solvable short targets tested). So the skill's value is not a bigger solve set (it solves a SUBSET
of DesiRNA's 180) — it is reaching that set **far cheaper**. The held-out compounding study
(`rssd_compounding.py`) measures how much, with a router that learns the routing rule on TRAIN and is
evaluated frozen on HELD-OUT.

**Honest scope:** this reframes RSSD from "crack the residual" (blocked by a designability wall) to "reach
the solvable set at lower cost, and prove the routing transfers" — the exact cost claim the cognitive-core
thesis rests on.

## Roadmap

- **v1 (this):** skill is hand-authored; the scientific contribution is the *cost-compounding mechanism +
  held-out proof* (capability null is reported as a mechanistically-explained designability wall).
- **v2:** hand the authoring wheel to an LLM agent (`self_revise` line) — the agent reflects on TRAIN
  residual and *writes* the skill; RSSD then tests whether an agent-distilled procedure compounds held-out.
