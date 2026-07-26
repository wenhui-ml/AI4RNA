#!/usr/bin/env bash
# R6 Major Issue 1: EXECUTION-LEVEL REPLICATION OF THE HEADLINE COMPARISON.
#
# Why. The R5 revision measured execution-level nondeterminism (~2/60 at a fixed nominal seed) only for the
# panel-operator ABLATION (Table 10). The headline empirical component Delta_gamer = p_panel - p_gamer was
# still resampled over targets and nominal seeds only, so its interval did not propagate the third variance
# component the paper itself had discovered. This script runs the two arms of the headline comparison
#
#     autonomous  (the pKiss-only "gamer")   vs   panel_auto  (the two-oracle panel operator)
#
# on the SAME 60 targets, at the SAME 3 nominal seeds, with 3 independent executions per nominal seed, and
# -- crucially -- BOTH ARMS INSIDE THE SAME PROCESS AND THE SAME WORKER CALL, so the pairing holds at the
# (target, seed, execution) level rather than only at the target level.
#
# Both arms now emit realized-budget stats (pKiss calls, iterations, candidates, oracle seconds), so the
# head-to-head can report realized compute per arm and not just the matched nominal cap.
#
# Search-only: no LLM, no $ cost.
set -uo pipefail
cd "$(dirname "$0")"
ARMS=autonomous,panel_auto
export NWORKERS=${NWORKERS:-60}
for rep in 0 1 2; do
  for s in 0 1 2; do
    echo "=== gamer-vs-panel execution $rep, nominal seed $s ==="
    /home/ubuntu/miniconda3/envs/rna/bin/python sweep.py "gvp_e${rep}_s$s" 0 60 "$s" "$ARMS"
  done
done
echo GAMER_VS_PANEL_REPLICATE_DONE
