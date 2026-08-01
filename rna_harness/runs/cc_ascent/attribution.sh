#!/usr/bin/env bash
# R8, owed experiment (4): WHY do the frozen agent-written operators carry over better than ours?
#
# The gap to explain. On carry-over -- P(de-novo ShapeKnots confirms | the arm's OWN pKiss AND ProbKnot
# predicate confirmed), a statistic no arm's objective can flatter -- the agent operators reach 0.302
# [0.275,0.330] while every hand-built variant sits at 0.09-0.16 (panel_auto 0.108 deterministic, 0.091
# wall-clock). App. F could only say "we cannot yet say why". This run tries to say why.
#
# Method: transplant ONE agent mechanism at a time into the hand-built operator and re-measure carry-over.
# Reading the two sources side by side, the agent operators differ from ours in four ways:
#   ungated  ProbKnot is consulted on EVERY candidate, not only once pKiss is already within `gate`. Since
#            ProbKnot is the least-correlated panel member (kappa 0.27 / 0.12, panel_agreement.py), this
#            transplants "let the independent voice steer the whole search" -- the mechanism we expect most.
#   scalar1  a SUMMED cost over both predictors replaces the lexicographic (pk_bp, pb_bp) rule.
#   scalar2  that sum weights SPURIOUS pairs 2x, as the agents write it. Against scalar1 this isolates the
#            false-positive asymmetry, which is the natural way to make a different predictor agree.
#   anneal   Metropolis acceptance on the summed cost, replacing pure non-worsening.
# Each differs from the previous by exactly one thing, so the chain attributes the gap stepwise.
#
# The control that decides whether this is a principle or just compute:
#   bigpk    3x the pKiss allowance and NOTHING else. The agents also ran 15 restarts x 3000 iterations
#            against our 6 x 800. If only this arm moves carry-over, the agents bought their advantage with
#            compute, and the honest write-up is "search", not "a transferable design principle".
#
# Run deterministically so each cell is one noise-free observation and no execution replicates are needed.
# Search-only: no LLM, no $ cost.
set -uo pipefail
cd "$(dirname "$0")"
ARMS=panel_auto,pv_ungated,pv_scalar1,pv_scalar2,pv_anneal,pv_bigpk
export NWORKERS=${NWORKERS:-60}
export PKX_DETERMINISTIC=1
export PKX_MAXPKISS=${PKX_MAXPKISS:-150}
export PKD_ITERS=800
for s in 0 1 2 3 4; do
  echo "=== attribution sweep, nominal seed $s ==="
  /home/ubuntu/miniconda3/envs/rna/bin/python sweep.py "attr_s$s" 0 60 "$s" "$ARMS"
done
echo ATTRIBUTION_DONE
