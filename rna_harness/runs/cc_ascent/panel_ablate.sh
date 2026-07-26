#!/usr/bin/env bash
# R4: per-component ablation of the PANEL operator + matched-budget undirected-search controls, all scored
# under the SAME three-predictor panel on the primary held-out pool (Pseudobase idx 0-59), 3 seeds.
#
# Every arm is DecomposePKPanelTool at IDENTICAL budgets (skeleton 12s, repair 90s, <=800 iters, ProbKnot
# gate 3, <=120 ProbKnot calls/target), identical oracle access (pKiss + ProbKnot) and identical
# initialization; only the named component is removed:
#   panel_auto        full operator (reference; re-run here so the comparison is paired on seed+target)
#   panel_randwalk    lexicographic acceptance removed  -> matched-budget UNDIRECTED walk, oracle-informed proposals
#   panel_blindrand   acceptance AND proposal-bias removed -> fully oracle-blind matched-budget random search
#   panel_norepair    repair walk removed entirely (skeleton + complementarity composition only)
#   panel_nodecompose random init instead of the CF-skeleton inverse fold
#   panel_nocompose   crossing positions not composed by complementarity
# Search-only: no LLM, no $ cost.
set -uo pipefail
cd "$(dirname "$0")"
ARMS=panel_auto,panel_randwalk,panel_blindrand,panel_norepair,panel_nodecompose,panel_nocompose
export NWORKERS=${NWORKERS:-60}
for s in 0 1 2; do
  echo "=== panel-ablation seed $s ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python sweep.py "pablate_s$s" 0 60 "$s" "$ARMS"
done
echo PANEL_ABLATE_DONE
