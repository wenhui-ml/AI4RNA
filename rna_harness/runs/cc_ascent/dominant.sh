#!/usr/bin/env bash
# R11, D3 seventh test, STAGE 2 -- the test that decides.
#
# Stage 1 (d3_distributional.py) caliper-matched the two arms on GC (negative control p=0.98, matched mean
# 0.4643 vs 0.4640 over 399 pairs) and found the agent operators' accepted designs differ at MATCHED
# composition in one quantity above all others: the ensemble gap (MFE - EFE)/n, 0.005 for them against our
# 0.012 (KS D=0.466). Their designs concentrate the Boltzmann ensemble on a single fold; a dominant fold is
# one any model confirms, which is mechanistically why it should raise carry-over.
#
# pv_dominant minimises that gap at composition and breaks acceptance ties on it. It uses ViennaRNA's nested
# partition function, which is in no arm's objective and outside the counted pKiss/ProbKnot allowance, so no
# compute confound is possible. Judged by the rule every transplant here has faced, fixed before the run:
# carry-over must RISE while r_in does NOT collapse. An observational difference is not a mechanism; `decoy`
# cleared exactly this bar and then reversed.
set -uo pipefail
cd "$(dirname "$0")"
ARMS=panel_auto,pv_dominant
export NWORKERS=${NWORKERS:-60}
export PKX_DETERMINISTIC=1
export PKX_MAXPKISS=${PKX_MAXPKISS:-150}
export PKD_ITERS=800
for s in 0 1 2 3 4; do
  echo "=== dominant sweep, nominal seed $s ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python sweep.py "dom_s$s" 0 60 "$s" "$ARMS"
done
echo DOMINANT_DONE
