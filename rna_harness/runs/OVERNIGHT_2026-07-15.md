# Overnight runs launched 2026-07-15 (evening) — results for the morning

Three background runs to harden Paper D (verifier economics) toward submission-grade. All are pure
ViennaRNA/SAT CPU — **no LLM/OpenRouter cost**. Prerequisite bug (anneal `coordinated_endgame` ignored
`wall_s` on large-n targets → hang) was fixed + regression-tested before launch (11/11 tests pass).

| # | tmux | experiment | outdir | done-marker (in log) |
|---|------|-----------|--------|----------------------|
| 1 | `msf`     | **Full-scale multi-seed frontier** — all 100 Eterna targets × 5 seeds, wall 30s. Upgrades §3 axis-2 from single-seed/61-target to full-scale + **solver-seed-variance CI**. | `runs/verifier_frontier_multiseed/` | `=== FRONTIER over` in `verifier_frontier_multiseed.log` |
| 2 | `satdeep` | **SAT deep sweep** — frontier (8 alphas, k=40/alpha), goodhart (4 budgets), cost (7 sizes, 25 reps). Tightens the cross-domain mirror CIs. | `runs/sat_verifier_deep/` | `SATDEEP_ALL_DONE` in `sat_deep.log` |
| 3 | `ghpms`   | **Goodhart multi-seed** — 20 targets × 5 seeds, budgets [200,1k,5k,20k], θ=0.9. Bootstrap CI on the carrying (RNA) Goodhart leg. | `runs/goodhart_multiseed/` | `GHMS_ALL_DONE` in `ghms.log` |

**What each buys the paper**
- Run 1 → §3: replace "bootstrap CI over targets (single seed)" with full-scale + seed-variance CI; the
  100→90→7% survival collapse gets a real error bar and the fragility-vs-length curve gets more points.
- Run 3 → §4: the 26.7→13.3% (cheap) / 20→40% (strong) Goodhart trend gets a bootstrap CI, so it is no
  longer a single-run anecdote — directly answers the "one seed" reviewer objection.
- Run 2 → §3/§5: SAT frontier ρ-collapse and the cost ladder get k=40 CIs, strengthening "not RNA-specific."

**To read results in the morning**: `tail` each log above, or the combined dump the waiter writes.
Next step after results land: fold the CIs into `paper/verifier_economics/verifier_economics.tex`
(§3 survival + §4 Goodhart), regenerate figures (`make_figs.py`), recompile.

Known residual (not blocking): the main replica-exchange loop still overshoots `wall_s` by ~1 epoch on
very large n (e.g. n=258 → ~50s at wall_s=30) — bounded, not a hang; fine for these runs.
