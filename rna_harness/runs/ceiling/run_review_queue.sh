#!/bin/bash
# Sequential review queue. Each job is offline and $0. Single-process numpy work MUST cap BLAS threads
# (memory: a SIGKILL at low RSS on this host is a core-count symptom, not a memory one).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling
PY=/home/ubuntu/miniconda3/envs/rna/bin/python

run () {  # run <tag> <args...>
  tag=$1; shift
  { echo "=== $tag start $(date -Is) ==="
    $PY -u "$@"
    echo "=== $tag done $(date -Is) rc=$? ==="; } 2>&1 | tee "rev_${tag}.log"
}

run B revB_feature_ceiling.py --nperm 120 --nfold 5 --rounds 150
run D revD_target_information.py
run C revC_typing.py --thr 0.25 --min-designs 20
run A revA_lll_proxy.py --targets 40 --draws 300 --workers 24
echo "=== QUEUE COMPLETE $(date -Is) ==="
