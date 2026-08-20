#!/bin/bash
# 8-20 夜间串行队列。串行是故意的:pKiss 是 Perl 包装器,每 worker 派生 ~2.9 个进程
# (见 feedback-cpu-160-core-cap),两个折叠任务并行会把机器推到 366 的负载。
cd /data/wenhui/openclaude_rna/rna_harness/runs/ceiling || exit 1
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
VPK=/home/ubuntu/miniconda3/envs/vpk/bin/python
TORCH=/home/ubuntu/miniconda3/envs/wh-llm/bin/python   # torch 只在这个环境里
log(){ echo "=== $1  $(date -u +%H:%M:%SZ) ===" >> run_night_0820.log; }

log "START"

# ---- 阶段 1:第五批人群 + 面板三判官(P1 的输入)
log "STAGE1 第五批面板 begin"
$VPK -u panel_f1_designs.py --fifth 24 --workers 40 \
     --out $PWD/panel_f1_fifth.json > stage1_fifth_panel.log 2>&1
log "STAGE1 rc=$?"

# ---- 阶段 2:P1 —— k 阶梯上半的功效版(主判官 = RibonanzaNet,主人群 = 合池)
log "STAGE2 P1 begin"
$PY -u p1_kladder_power.py > p1_kladder_power.log 2>&1
log "STAGE2 rc=$?"

# ---- 阶段 3:D4S —— 第四个领域到底有没有环外校准物
log "STAGE3 D4S begin"
$TORCH -u d4s_linker_smoke.py > d4s_linker_smoke.log 2>&1
log "STAGE3 rc=$?"

log "ALL DONE"
