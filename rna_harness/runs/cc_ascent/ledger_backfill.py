#!/usr/bin/env python3
"""Backfill the TransferLedger 'learned + decisive' quantification of capability ACQUISITION (the ② axis).

An operator counts as a LEARNED capability only if it is REUSED across held-out episodes AND its presence
LIFTS the solve rate (vs. not using it), and is DECISIVE if removing it collapses the gain. We build the
ledger from the logged ascent runs: "with-operator" episodes (autonomous / panel_auto, one per target) vs.
"without-operator" episodes (always_cheap = the CF-confined repertoire, 0/60). Crucially we recompute the
verdict under DIFFERENT oracles -- because, as the ascent experiment showed, 'learned' is itself
oracle-relative: an operator that games pKiss is 'learned' under pKiss but not under an independent panel.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase           # noqa: E402
from rna_harness.oracles_pk import forms_target, probknot_pairs          # noqa: E402
from rna_harness.meta.transfer_gate import TransferLedger                # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__))
id2t = {t.task_id: t.target for t in load_pseudobase(limit=60)}


def ledger(op_name, solved_with, solved_without):
    led = TransferLedger()
    for tid, s in solved_with.items():
        led.record_episode(f"op:{tid}", items_used=[op_name], solved=s)
    for tid, s in solved_without.items():
        led.record_episode(f"cheap:{tid}", items_used=[], solved=s)
    r = led.report(min_reuse=3, min_lift=0.0)["items"][op_name]
    without_rate = sum(solved_without.values()) / max(1, len(solved_without))
    with_rate = sum(solved_with.values()) / max(1, len(solved_with))
    r["with_rate"] = round(with_rate, 3)
    r["without_rate"] = round(without_rate, 3)
    r["decisive"] = (without_rate == 0.0 and with_rate > 0.0)   # ablation collapses the gain to zero
    return r


gen = json.load(open(os.path.join(ROOT, "generous", "run.json")))["rows"]
cheap = {r["id"]: bool(r["by_policy"]["always_cheap"]["solved"]) for r in gen}
auto = {r["id"]: r["by_policy"]["autonomous"] for r in gen}

rows = []
# pkdecomp under pKiss (the oracle it optimized)
rows.append(("pkdecomp", "pKiss (optimized)",
             ledger("pkdecomp", {t: bool(v["solved"]) for t, v in auto.items()}, cheap)))
# pkdecomp under pKiss AND ProbKnot (independent robustness)
print("recomputing ProbKnot for pkdecomp solves ...", flush=True)
auto_pk_pb = {}
for t, v in auto.items():
    auto_pk_pb[t] = bool(v["solved"]) and v.get("seq") is not None and \
        bool(forms_target(probknot_pairs(v["seq"]), id2t[t]))
rows.append(("pkdecomp", "pKiss AND ProbKnot (robust)", ledger("pkdecomp", auto_pk_pb, cheap)))

# pkpanel under pKiss AND ProbKnot (its native solve criterion)
pa_path = os.path.join(ROOT, "panel_auto", "run.json")
if os.path.exists(pa_path):
    pa = json.load(open(pa_path))["rows"]
    pa_solved = {r["id"]: bool(r["by_policy"]["panel_auto"]["solved"]) for r in pa}
    cheap_pa = {k: cheap[k] for k in pa_solved if k in cheap}
    rows.append(("pkpanel", f"pKiss AND ProbKnot (optimized pair, n={len(pa_solved)})",
                 ledger("pkpanel", pa_solved, cheap_pa)))

print(f"\n{'operator':<10}{'oracle':<34}{'reuse':>6}{'lift':>8}{'learned':>9}{'decisive':>10}")
for op, oracle, r in rows:
    print(f"{op:<10}{oracle:<34}{r['reuse']:>6}{r['lift']:>+8.3f}{str(r['learned']):>9}{str(r['decisive']):>10}"
          f"   (with={r['with_rate']} without={r['without_rate']})")
json.dump({"rows": [{"operator": op, "oracle": oc, **r} for op, oc, r in rows]},
          open(os.path.join(ROOT, "ledger.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(ROOT, 'ledger.json')}")
print("NOTE: full-panel (incl. held-out ShapeKnots) 'learned' verdict pending the panel adjudication step.")
