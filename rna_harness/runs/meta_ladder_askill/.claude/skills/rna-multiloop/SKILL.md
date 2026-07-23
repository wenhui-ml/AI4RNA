---
name: rna-multiloop
description: Inverse-design method for multiloop RNA targets (≤120nt) with dominant long-range pairs
when_to_use: Target is a multiloop, length 61–120nt band, has 1+ dominant long-range pair spanning >50% of length, tandem_repeat=False
allowed-tools: mcp__rna_motifs__analyze_target, mcp__rna_motifs__inverse_fold_search, mcp__rna_motifs__finish, RnaVerify, Bash
---

# RNA Multiloop Inverse-Design Skill

## Phase 1 — Decompose & Plan
1. Call `mcp__rna_motifs__analyze_target` on the full target to get the pair map, multiloop junctions, and decomposition into sub-units.
2. Identify: (a) which pairs are long-range (span > half the length), (b) any 1-bp or lonely-pair helices adjacent to loops, (c) closing pairs of each stem.

## Phase 2 — Initial Search & Escalation Ladder
3. Call `mcp__rna_motifs__inverse_fold_search(n_restarts=0, tries=40)` on the WHOLE target. Submit the exact-length output VERBATIM to `RnaVerify`.
4. If solved (bp_distance=0, uMFE confirmed): done. Emit `FINAL_SEQUENCE:`.
5. If not solved, escalate restart depth progressively: `n_restarts=8, tries=40` → repeat → `n_restarts=16, tries=30`. Each round warm-starts from the prior best candidate implicitly. Re-verify after each.
6. If still not solved after the escalation ladder, call `analyze_target` again to re-examine decomposition; then run `inverse_fold_search` on sub-units and hand-assemble (A-rich linkers, G-C closing pairs), verifying the full assembly.

## Phase 3 — Near-Miss Hill-Climb (bp_distance 1–3)
7. If RnaVerify shows bp_distance 1–3: call `mcp__rna_motifs__finish(iters=30000, n_restarts=16)` on that candidate. The diagnosis-guided SA handles the fragile pair.
8. Re-verify. If still near-miss, loop: verify→`finish` until solved.
9. **NEVER hand-edit bases on a near-miss.** Manual rational G-C edits regress — the solver's SA explores neighborhoods you cannot.

## Phase 4 — Lonely-Pair / Long-Range Frustration (search plateaus at same bp)
10. If the same pair fails across multiple rounds, it is a lonely-pair or long-range frustration problem.
11. Warm-start: call `mcp__rna_motifs__inverse_fold_search(n_restarts=16, tries=30)` with `warm=<best_candidate>`. The expanded search window lets the solver reinforce the fragile pair's flanking context.
12. Re-verify. If bp_distance ≤3, loop back to Phase 3 (`finish`).

## Decision Rules
- **Over-pairing** (neighbour stem zips through a bulge, spurious pair forms): `finish` handles automatically by breaking the competing helix; do not hand-insert mismatches.
- **Never-forms** (endpoints stay unpaired): needs heterogeneous flanking bases + higher `n_restarts`; warm-start from best.
- **Degeneracy** (solved by bp_distance but not unique MFE): add G-C support to the fragile pair via `finish` warm-start.
- **Escalation order**: n_restarts 0→8→8→16, then switch to `finish` with n_restarts=16. Do not skip to high restarts prematurely — low-restart rounds sometimes solve outright.
- **Length discipline**: always submit tool-returned sequences verbatim (already exact length). If hand-assembling, count before emitting.
- **Negative design**: never hand-build all-G-C (collapses). Winners are A-U-heavy with A-rich unpaired linkers as anti-zippers.

## Anti-Patterns (forbidden)
- Do NOT run `inverse_fold_search` in background or poll logfiles — synchronous only.
- Do NOT hand-write ViennaRNA SA loops in Bash.
- Do NOT retype dot-bracket from memory — copy verbatim or mis-counting corrupts the target.
- Do NOT tile identical repeat units (cross-pairing collapse) — use distinct sequences per unit.
