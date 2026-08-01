#!/usr/bin/env bash
# R8: re-run the ablation AND the headline gamer-vs-panel comparison in the DETERMINISTIC,
# COMPUTE-MATCHED regime (PKX_DETERMINISTIC=1).
#
# Why this run exists. Every measurement in the paper up to R7 was made under the historical wall-clock
# stopping rules, which have two defects the reviews kept hitting:
#   (a) execution-level nondeterminism at a fixed seed, of the same order as the method effect, so the
#       component attribution under r_mix was "unresolvable at n=60" and had to be averaged over 3
#       executions per seed rather than removed;
#   (b) the arms shared a matched NOMINAL cap but not matched REALIZED compute -- the undirected arms ran
#       to the wall and consumed 33-63% MORE pKiss calls than the full operator.
# Deterministic mode fixes both at the root: every stopping rule is counted (iterations + a pKiss-call
# allowance identical across arms), and ViennaRNA's own global RNG is seeded per attempt, so a nominal
# seed fixes the run bit-for-bit. Verified here before use: two independent runs of 3 arms x 4 targets
# gave 12/12 bit-identical sequences.
#
# Consequences for the design. No execution replicates are needed (that is the point), so the same compute
# buys SEEDS instead: 5 nominal seeds, which matches the paper's headline seed count. PKD_ITERS is raised to
# the panel operator's 800 so the gamer and the panel operator differ ONLY in the objective, not in the
# iteration allowance -- with max_pkiss shared this is a genuinely compute-matched comparison rather than a
# cap-matched one.
#
# Search-only: no LLM, no $ cost.
set -uo pipefail
cd "$(dirname "$0")"
ARMS=autonomous,panel_auto,panel_randwalk,panel_blindrand,panel_norepair,panel_nodecompose,panel_nocompose
export NWORKERS=${NWORKERS:-60}
export PKX_DETERMINISTIC=1
export PKX_MAXPKISS=${PKX_MAXPKISS:-150}
export PKD_ITERS=800          # match the panel operator's iteration allowance (default 400)
for s in 0 1 2 3 4; do
  echo "=== deterministic sweep, nominal seed $s ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python sweep.py "det_s$s" 0 60 "$s" "$ARMS"
done
echo PANEL_DETERMINISTIC_DONE
