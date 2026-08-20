#!/bin/bash
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling || exit 1
log(){ echo "=== $1  $(date -u +%H:%M:%SZ) ===" >> run_p2.log; }
log START
log "STAGE1 第六批面板 begin"
/home/ubuntu/miniconda3/envs/vpk/bin/python -u panel_f1_designs.py --sixth 10 --workers 40 \
    --out $PWD/panel_f1_sixth.json > stage1_sixth_panel.log 2>&1
log "STAGE1 rc=$?"
log "STAGE2 P2 begin"
/home/ubuntu/miniconda3/envs/rna/bin/python -u p2_kladder_blind.py > p2_kladder_blind.log 2>&1
log "STAGE2 rc=$?"
log DONE
