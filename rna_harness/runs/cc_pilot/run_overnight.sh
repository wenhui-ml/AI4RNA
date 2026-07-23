#!/usr/bin/env bash
# chomsky_climb overnight queue (E0/E1 CF-floor + the +/-DesiRNA autonomy ablation).
# Detached in tmux; survives disconnect. All classical solvers -> $0 LLM cost.
# Results write INCREMENTALLY (run.json dumped per task), so partial output is always usable.
# NOTE (honest): the current "agent" is a tool ROUTER, not an operator inventor. This run therefore
# measures the BORROWING baseline: always_cheap(REMC, no DesiRNA) is expected ~0 on pseudoknots
# (CF-confined, cannot form crossing stems = empirical Prop.1 floor); always_sota(DesiRNA) is the
# borrowed MCF ceiling. The gap = "capability lives in the DesiRNA tool, not the agent". A positive
# Chomsky ASCENT needs the operator-invention build (to-build #2) which does not exist yet.
set -uo pipefail
cd /data/wenhui/openclaude_rna
export DESIRNA_CMD="/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py"
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
OUT=rna_harness/runs/cc_pilot
mkdir -p "$OUT/fp" "$OUT/ablation" "$OUT/cognitive"

echo "[cc-overnight] START $(date -u)  DESIRNA_CMD set; pKiss oracle."

echo "[cc-overnight] STEP 1/3  E0/E1 CF-confinement floor: MFE false-positive control"
echo "               (REMC designs the nested skeleton; pKiss judges the full crossing target)"
$PY -m rna_harness.experiments.track_b_pk --fp-demo 80 --outdir "$OUT/fp" 2>&1
echo "[cc-overnight] STEP 1 done $(date -u)"

echo "[cc-overnight] STEP 2/3  +/-DesiRNA ablation (PRIORITY): always_cheap(REMC,no-DesiRNA) vs always_sota(DesiRNA)"
$PY -m rna_harness.experiments.track_b_pk --run 60 --policies always_cheap always_sota --outdir "$OUT/ablation" 2>&1
echo "[cc-overnight] STEP 2 done $(date -u)"

echo "[cc-overnight] STEP 3/3  router baseline (bonus): cognitive core over default toolbox"
$PY -m rna_harness.experiments.track_b_pk --run 60 --policies cognitive --outdir "$OUT/cognitive" 2>&1
echo "[cc-overnight] STEP 3 done $(date -u)"

echo "[cc-overnight] ALL DONE $(date -u)"
