#!/bin/bash
# PARALLEL queue for chomsky_climb T2/T3 (256-core box). Each sweep parallelizes across targets
# (NWORKERS); steps run sequentially so no oversubscription. E0 already done. No LLM/$ cost.
set +e
cd /data/wenhui/openclaude_rna/rna_harness/runs/cc_ascent
export DESIRNA_CMD="/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py"
export PYTHONPATH=/data/wenhui/openclaude_rna
export NWORKERS=60
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
LOG=night_queue.master.log
echo "=== PARALLEL QUEUE start ===" > $LOG

echo "[1/4] C: second disjoint held-out pool Pseudobase[60:110]" | tee -a $LOG
export PKP_SKEL_S=20 PKP_REPAIR_S=120 PKP_ITERS=1000 PKP_GATE=3 PKP_MAXPB=120
$PY sweep.py pool2 60 50 0 always_cheap,panel_auto > pool2.log 2>&1 ; echo "  pool2 rc=$?" | tee -a $LOG

echo "[2/4] T3: panel_auto seed 3 (first 60)" | tee -a $LOG
export PKP_REPAIR_S=120 PKP_ITERS=1000
$PY sweep.py seed3 0 60 3 panel_auto > seed3.log 2>&1 ; echo "  seed3 rc=$?" | tee -a $LOG

echo "[3/4] T3: panel_auto seed 4 (first 60)" | tee -a $LOG
$PY sweep.py seed4 0 60 4 panel_auto > seed4.log 2>&1 ; echo "  seed4 rc=$?" | tee -a $LOG

echo "[4/4] T3: budget-scaling on first 20 (panel_auto @ 60/180/300 s)" | tee -a $LOG
for B in 60 180 300; do
  export PKP_REPAIR_S=$B PKP_ITERS=$((B*8))
  $PY sweep.py budget$B 0 20 0 panel_auto > budget$B.log 2>&1 ; echo "  budget$B rc=$?" | tee -a $LOG
done

echo "=== PARALLEL QUEUE complete ===" | tee -a $LOG
