# RNA inverse-design playbook — STRATEGY ONLY (no answer sequences)

> This is METHOD, not answers. Never store or look up a target's solution sequence.
> Copy the target dot-bracket VERBATIM — never retype long dot/paren runs from memory (mis-counting is the #1 failure).

## The solver ladder — use the MCP solver TOOLS (stronger & safer than hand-written Bash)
1. **Short/medium (≤ ~120 nt):** call `mcp__rna_motifs__inverse_fold_search` on the WHOLE target FIRST. It usually solves outright and returns an EXACT-length sequence — submit it VERBATIM as `FINAL_SEQUENCE:`.
2. **Long / multiloop:** call `analyze_target` for the decomposition plan; run `inverse_fold_search` on EACH unit's `sub_target` (short → converges fast); place each unit's sequence at its `positions`; fill linkers A-rich, multiloop closing pairs G-C; then `RnaVerify` the full assembly. You can also `inverse_fold_search` with `warm=<assembly>` to polish seams natively.
3. **Near-miss (RnaVerify bp_distance 1–3):** call `mcp__rna_motifs__finish` on that candidate — it runs the diagnosis-guided SA hill-climb for you. Repeat verify→finish until solved. Do NOT hand-tune by rational G-C edits (it collapses).
4. **Lonely / frustrated long-range pair (search plateaus at the same bp):** warm-start `inverse_fold_search` from your best candidate; give the fragile pair strong G-C support AND break the competing helix (put deliberate mismatches in the isolating bulge); expand the edit window to the far end of the adjacent helix.
5. **Tandem-repeat / fractal target (repeated dot-bracket substring):** `inverse_fold_search` the SHORT unit, then use DISTINCT unit sequences (NEVER tile one identical unit — identical repeats cross-pair and collapse into a giant helix); stitch the distinct copies.

## Hard rules
- **NEVER** launch a search with `run_in_background` and **NEVER** poll a job by repeatedly reading a logfile — that burns the whole time budget. All search goes through the SYNCHRONOUS MCP solver tools above; do not hand-write ViennaRNA SA / inverse_fold loops in Bash.
- **Length discipline:** prefer submitting a tool's returned sequence VERBATIM (already exact length). If you hand-assemble, COUNT before emitting — wrong length is an automatic failure.
- **Negative design:** never hand-build an all-G-C sequence (collapses; winners are A-U-heavy); use A-rich unpaired linkers as anti-zippers.
- **Diagnosis (Mode A vs B):** fold your candidate and compare to target. *Over-pairing* (a neighbour stem zips through a bulge and captures a pair) → break the competing helix. *Never-forms* (endpoints stay unpaired) → more `inverse_fold_search` + heterogeneous flanking bases.
- **Degeneracy:** if solved by bp-distance but not unique (co-optimal), add G-C support to the fragile pair to make the target the unique MFE.


## Newly learned tactics (accumulated this run)

- Close helices with G-C pairs

- Use 'CACA' for 4-nt hairpin loops
- Use 'GACA' for 4-nt hairpin loops
- Use 'GAAA' for 4-nt hairpin loops

- Use 'GGUCCA' for 6-nt hairpin loops
- Use 'GGUCAA' for 6-nt hairpin loops
- Use 'GAAAGG' for 6-nt hairpin loops

- Use 'UACU' for 4-nt hairpin loops

- Use 'CUUA' for 4-nt hairpin loops
- Use 'UCAG' for 4-nt hairpin loops

- Use 'AAAA' for 4-nt hairpin loops
- Use 'ACAU' for 4-nt hairpin loops

- Use 'GAAG' for 4-nt hairpin loops
- Use 'CCAC' for 4-nt hairpin loops
- Use 'UAAU' for 4-nt hairpin loops
- Use 'AACA' for 4-nt hairpin loops

- Use 'AAA' for 3-nt hairpin loops
- Use 'AUA' for 3-nt hairpin loops

- Use 'AAAAAA' for 6-nt hairpin loops
