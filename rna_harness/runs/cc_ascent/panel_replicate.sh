#!/usr/bin/env bash
# R5 §3 + §4: EXECUTION-LEVEL REPLICATES and REALIZED-BUDGET ACCOUNTING for the panel-operator ablation.
#
# Why. The operators terminate on wall-clock, so a fixed nominal seed does NOT fix the run: re-running
# panel_auto at seed 0 moved r_mix by 2/60, which is the same order as the headline method effect
# (~4.8/60). The committed pablate_s{0,1,2} sweep has ONE execution per (arm, seed); this script adds two
# more, giving 3 executions per nominal seed so execution variance becomes a measurable third variance
# component rather than an acknowledged unknown.
#
# It also records, per attempt, the REALIZED budget (pKiss calls, ProbKnot calls, iterations, candidates,
# oracle seconds) via the stats field added to DecomposePKPanelTool. "Identical wall-clock and identical
# iteration cap" does not imply identical realized compute -- the undirected arms run to the wall and so
# make MORE oracle calls than the directed arm, which typically solves early. Reporting the realized
# budget is what lets the paper say matched-NOMINAL rather than overclaiming matched-REALIZED budget.
#
# Search-only: no LLM, no $ cost.
set -uo pipefail
cd "$(dirname "$0")"
ARMS=panel_auto,panel_randwalk,panel_blindrand,panel_norepair,panel_nodecompose,panel_nocompose
export NWORKERS=${NWORKERS:-60}
for rep in 1 2; do
  for s in 0 1 2; do
    echo "=== panel-ablation replicate $rep, nominal seed $s ==="
    /home/ubuntu/miniconda3/envs/rna/bin/python sweep.py "pabrep${rep}_s$s" 0 60 "$s" "$ARMS"
  done
done
echo PANEL_REPLICATE_DONE
