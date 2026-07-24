#!/bin/bash
# Cross-source robust-ascent test (chomsky_climb, the "does the certificate certify a real capability
# or Pseudobase-topology overfit?" lever). On the 60 bpRNA/RNAInvBench cross-source crossing targets:
#   - panel_auto with a BOOSTED search budget (does more search realize crossings cross-source?)
#   - always_sota = DesiRNA (the achievability CEILING under the panel, cross-source)
# The CF floor is 0 by Prop 1 (oracle image subset of nested, pool-independent) + E0; measured separately by
# xsrc_floor.py (top-level REMC, since replica_exchange cannot spawn its pool inside a sweep worker).
# DESIRNA_CMD / OPENROUTER_API_KEY are read from the environment, never stored here.
set +e
cd /data/wenhui/openclaude_rna/rna_harness/runs/cc_ascent
export PYTHONPATH=/data/wenhui/openclaude_rna
export DESIRNA_CMD="${DESIRNA_CMD:-/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py}"
export TARGETS_JSON=bprna_pk/targets.json
export NWORKERS=40
# boosted panel_auto budget (default was SKEL_S=12/REPAIR_S=90/ITERS=800)
export PKP_SKEL_S=20 PKP_REPAIR_S=240 PKP_ITERS=2500 PKP_GATE=3 PKP_MAXPB=120
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
echo "=== XSRC start uptime=$(cut -d' ' -f1 /proc/uptime)s: panel_auto(boosted)+DesiRNA x60 bpRNA ==="
$PY sweep.py xsrc 0 60 0 panel_auto,always_sota
echo "=== XSRC done rc=$? -> xsrc/sweep.json ==="
