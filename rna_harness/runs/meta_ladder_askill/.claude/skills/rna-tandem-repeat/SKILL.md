---
name: rna-tandem-repeat
description: Inverse-design method for tandem-repeat RNA targets across length bands (M 61-120 through XL >250), including all-unpaired (n_pairs=0) and general multiloop/long-range paired cases. Uses whole-target search, unit decomposition, distinct-repeat stitching with anti-cross-pair stem design, escalating-restart warm-starts, diagnosis-guided finish, and Bash-ViennaRNA fallback for XL timeouts.
when_to_use: Target tandem_repeat=True, any length band. Sub-classes: (a) n_pairs=0 / all-unpaired — homopolymer shortcut; (b) n_pairs>0, length ≤ ~120 — whole-target search with escalating restarts; (c) n_pairs>0, length > ~120 — unit decomposition pipeline; (d) length > ~250 — XL escalation with parallel search and Bash fallback.
allowed-tools:
  - mcp__rna_motifs__analyze_target
  - mcp__rna_motifs__inverse_fold_search
  - mcp__rna_motifs__finish
  - mcp__rna_motifs__parallel_search
  - mcp__rna_motifs__batch_fold
  - RnaVerify
  - RnaDiagnose
  - Bash
---

# Tandem-Repeat Inverse-Design Skill

## Phase 0 — Classify & verify length
1. Inspect n_pairs, length band, multiloop, max_pair_span. If n_pairs=0 → **Phase 0A**. If n_pairs>0 and length ≤ ~120 → **Phase 1M**. If n_pairs>0 and length > ~120 → **Phase 1L**. If length > ~250 → **Phase 1XL**.
2. **Length audit (critical):** confirm exact target length from the prompt before any search. Mis-counting wastes the entire budget.

## Phase 0A — All-unpaired shortcut
3. Emit all-A at exact target length. Count before emitting. Verify; if spurious pairs appear, call `finish` to hill-climb them away.

## Phase 1M — Medium tandem-repeat (≤ ~120 nt, n_pairs>0)
4. Run `inverse_fold_search` on the WHOLE target first (`n_restarts=0, tries=40`). It often solves outright — submit the exact-length return verbatim.
5. If no solution: call `analyze_target` for the decomposition plan. Run `inverse_fold_search` on the shortest unit `sub_target` (`n_restarts=0, tries=40`) to get unit sequences. Re-run to collect DISTINCT solutions — one per tandem copy.
6. **Anti-cross-pair design (critical):** alternate stem orientations across copies (G-C in one, C-G in the next) so no two repeated stems are reverse-complementary. Use stable tetraloop caps (GAAA / UUCG). Fill linkers A-rich. Set multiloop closing pairs to G-C.
7. Assemble distinct units at mapped positions; warm-start `inverse_fold_search` with `warm=<assembly>`.
8. **Escalating restarts:** re-run with n_restarts=4 then 8, warm-starting from best candidate each time. Vary `tries` (30–80). If bp_distance drops to 1–5: call `finish`. For bp_distance > ~5, prefer warm-started `inverse_fold_search`.
9. **Cross-pairing diagnosis:** if assembly over-pairs, call `RnaDiagnose` to identify cross-pairing stems. Break competing helices with A-rich bulges or stem orientation flips. Then `finish(iters=40000)` on corrected warm candidate.
10. **Full-target re-search:** even if unit decomposition stalls, re-run `inverse_fold_search` on the FULL target (no warm-start, `tries=60–80`). Global SA can find solutions that stitched assemblies miss.

## Phase 1L — Long tandem-repeat (> ~120 nt, n_pairs>0)
11. Call `analyze_target`; extract shortest repeating unit as `sub_target`.
12. Run `inverse_fold_search` on one unit (`n_restarts=32, tries=60`) → unit_0. Run again for additional distinct solutions — need as many distinct unit sequences as tandem copies. **Never tile one identical unit** — identical repeats cross-pair into a giant degenerate helix.
13. Assemble with anti-cross-pair design (step 6 rules). Call `batch_fold`; compare MFE to target. Over-pairing → linker not A-rich enough; under-pairing → strengthen innermost closing pair to G-C.
14. If bp_distance ≤ ~5 after assembly: call `finish(iters=40000)` on the warm candidate. If still stuck: `RnaDiagnose` to locate cross-pairs, fix flanking bases, re-`finish`.
15. **Dominant long-range pairs (max_pair_span > ~150):** these are the hardest to stabilize. If a long-range pair never forms, warm-start `inverse_fold_search` from best candidate — ViennaRNA's own inverse_fold is smarter than random mutation. Give the fragile pair strong G-C support AND break competing local helices by inserting deliberate mismatches in the isolating bulge. Expand the edit window to the far end of the adjacent helix.
16. **Iterated warm-start ladder (proven on L-band with dominant long-range):** run `inverse_fold_search` with `n_restarts=20, tries=40` repeatedly (8+ rounds), each time warm-starting
