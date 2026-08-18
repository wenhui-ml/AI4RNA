#!/usr/bin/env python3
"""Substitute the `screen_first` rule into the full run and score it against the published field.

RUNS IN THE RibonanzaNet venv. No solver, no new generation: every sequence here was already produced,
already adjudicated and already paid for in the CERTIFY_POOL_DUMP run. Only the choice changes.

WHY THIS STEP EXISTS. The replay says `screen_first` delivers a median of 0.917 against the shipped rule's
0.800 (24 better / 8 worse of 51, Wilcoxon p = 1.5e-4, under a Bonferroni threshold of 0.025/7). That is a
statement about OUR OWN pools under OUR OWN judge. The question a user cares about is different: does it
move where we sit against the 13 published methods? Substituting and re-scoring answers that, and it can
come out worse -- a rule can raise the median while losing the targets that were carrying our wins.

    WHAT CHANGES.   knot_confirmed / knot_screened targets take the first candidate whose mean crossing
                    recall clears 0.1, which is the solver's own top-ranked survivor (order carries signal:
                    within-target Spearman between pool index and RNet F1 is -0.184, negative on 48 of 51,
                    p = 3.8e-10; the first candidate beats its pool median by +0.162, 42 of 51, p = 1.5e-5).
    WHAT DOES NOT.  `certified` (panel-unanimous, ladder returned early) and `cannot_certify` keep their
                    shipped answer. The verdict LABEL is left alone here; this measures the selection change
                    in isolation, and what a `knot_confirmed` target should be called once it stops using
                    the confirm rung to choose is a separate product decision.

COST: $0.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from selector_replay import load_dump, pick, KNOT_VERDICTS

R = json.load(open(os.path.join(HERE, "leaderboard_replay.json")))
T = {p: v["target"] for p, v in R["targets"].items()}
P = load_dump(os.path.join(HERE, "pool_dump"), T)
from rna_harness.cognitive.certify import calibrated_rule_applies
G = {p: not calibrated_rule_applies(t) for p, t in T.items()}

n_ch = 0
for r in R["rows"]:
    p = r["puzzle"]
    if r["verdict"] not in KNOT_VERDICTS or p not in P:
        continue
    c = pick(P[p], "screen_first", gated=G[p])
    if c and c["seq"] != r["sequence"]:
        r["sequence"] = c["seq"]
        r["reason"] = "SELECTOR REPLAY (screen_first): first candidate clearing the screen. " + r["reason"]
        n_ch += 1
out = os.path.join(HERE, "leaderboard_screenfirst.json")
json.dump(R, open(out, "w"), ensure_ascii=False)
print(f"改动了 {n_ch}/{len(R['rows'])} 个靶标的交付序列;其余原样保留")
print(f"  wrote {out}")
