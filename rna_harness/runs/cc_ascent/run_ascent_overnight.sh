#!/usr/bin/env bash
# ============================================================================================
# chomsky_climb ASCENT experiment (the REAL capability line, tightly coupled to the topic).
#
# QUESTION: does the INVENTED MCF operator (DecomposePKTool: decompose T -> CF skeleton -> compose the
# crossing pairs -> pKiss-guided repair) achieve an AUTONOMOUS CF->MCF ascent -- i.e. solve held-out
# crossing (pseudoknot) targets WITHOUT ever calling DesiRNA or any external pseudoknot solver?
#
# The `autonomous` policy is FixedPolicy([DecomposePKTool]) -- it never constructs a DesiRNA call, so a
# nonzero held-out solve rate is a certified ascent by the agent's OWN procedure, not a borrowed one.
#   * always_cheap (REMC, CF-confined) is the 0-floor control (empirical Prop.1).
#   * always_sota  (DesiRNA) already measured = 43/60 on the SAME 60 targets / seed (pilot, runs/cc_pilot).
#
# Detached in tmux -> survives disconnect / laptop close. Results write INCREMENTALLY (run.json per task).
# ============================================================================================
set -uo pipefail
cd /data/wenhui/openclaude_rna
export DESIRNA_CMD="/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py"
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
ROOT=rna_harness/runs/cc_ascent
mkdir -p "$ROOT/generous" "$ROOT/small"

echo "[ascent] START $(date -u)   (autonomous = invented MCF operator, NO DesiRNA in call path)"

# --- STEP 1 (PRIORITY): the definitive ascent run @ GENEROUS budget = the operator's fair ceiling.
#     always_cheap floor re-confirmed in the same file. Full 60 held-out crossing targets, seed 0.
echo "[ascent] STEP 1/4  autonomous(generous skel15/repair90/iters600) + always_cheap floor, n=60"
PKD_SKEL_S=15 PKD_REPAIR_S=90 PKD_ITERS=600 \
  $PY -m rna_harness.experiments.track_b_pk --run 60 \
      --policies always_cheap autonomous --outdir "$ROOT/generous" 2>&1
echo "[ascent] STEP 1 done $(date -u)"

# --- STEP 2: cross-oracle PANEL adjudication of the generous autonomous solves (pKiss/ProbKnot/ShapeKnots).
#     Turns the pKiss-only positive rate into the honest panel-unanimous rate + envelope.
echo "[ascent] STEP 2/4  panel re-adjudication of generous autonomous solves"
$PY rna_harness/runs/cc_ascent/panel_adjudicate.py "$ROOT/generous/run.json" 2>&1 || echo "[ascent] panel step skipped (error above, non-fatal)"
echo "[ascent] STEP 2 done $(date -u)"

# --- STEP 3: budget-CALIBRATION point @ SMALL budget -> brackets the rate-vs-budget curve.
echo "[ascent] STEP 3/4  autonomous(small skel8/repair30/iters250) calibration, n=60"
PKD_SKEL_S=8 PKD_REPAIR_S=30 PKD_ITERS=250 \
  $PY -m rna_harness.experiments.track_b_pk --run 60 \
      --policies autonomous --outdir "$ROOT/small" 2>&1
echo "[ascent] STEP 3 done $(date -u)"

# --- STEP 4: panel adjudication of the small-budget solves (for the calibration table).
echo "[ascent] STEP 4/4  panel re-adjudication of small-budget autonomous solves"
$PY rna_harness/runs/cc_ascent/panel_adjudicate.py "$ROOT/small/run.json" 2>&1 || echo "[ascent] panel step skipped (non-fatal)"
echo "[ascent] STEP 4 done $(date -u)"

echo "[ascent] ================ SUMMARY ================"
$PY - <<'SUMMARY' 2>&1 || true
import json, os
ROOT="rna_harness/runs/cc_ascent"
for tag in ("generous","small"):
    rj=os.path.join(ROOT,tag,"run.json")
    if not os.path.exists(rj):
        print(f"  [{tag}] no run.json"); continue
    d=json.load(open(rj)); agg=d.get("agg",{}); n=len(d.get("rows",[]))
    for name,a in agg.items():
        print(f"  [{tag}] {name:12} solved={a['solved']}/{n}  $={a['cost_usd']:.3e}  pKiss={a['oracle_s']:.0f}s")
    pj=os.path.join(ROOT,tag,"panel_autonomous.json")
    if os.path.exists(pj):
        p=json.load(open(pj))
        print(f"  [{tag}] PANEL: pKiss-reconfirmed={p['pkiss_reconfirmed']}/{p['n_solved']}  "
              f"unanimous={p['panel_unanimous']}  (ProbKnot/ShapeKnots avail={p['rnastructure_available']})")
print("  NOTE: always_sota (DesiRNA) = 43/60 on the SAME pool/seed (pilot runs/cc_pilot/ablation).")
SUMMARY
echo "[ascent] ALL DONE $(date -u)"
