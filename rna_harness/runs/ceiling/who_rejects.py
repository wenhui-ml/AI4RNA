#!/usr/bin/env python3
"""Which oracle actually rejects? The operator optimises two of three; the metric requires all three.

why_stop.py found that a successful attempt reaches NO resource limit (pKiss 37 of 150, ProbKnot 33 of
120, iters 36 of 800) -- it stops because it hit its own target -- and that panel-unanimous and
solved-but-panel-NO attempts are statistically indistinguishable in every resource (Mann-Whitney p = 0.88
on ProbKnot calls, 0.83 on pKiss calls). So the panel verdict is decided at the moment the operator stops,
and no amount of extra compute is even attempted.

The operator (DecomposePKPanelTool) optimises pKiss AND ProbKnot. The scored metric adjudicates pKiss AND
ProbKnot AND ShapeKnots. ShapeKnots is therefore OUT OF LOOP. If it is what rejects, then the whole
allocation programme was aimed at the wrong thing: the bottleneck is an oracle the operator never
consults, not a scheduling decision.

This counts the panel verdicts directly. Stored data only.
"""
import json, os
from collections import Counter
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
CC = os.path.abspath(os.path.join(H, "..", "cc_ascent"))

pat = Counter()
solved_n = 0
rows = []
for e in range(3):
    for s in range(5):
        f = os.path.join(CC, f"gvp_e{e}_s{s}", "sweep.json")
        if not os.path.exists(f):
            continue
        for r in json.load(open(f))["rows"]:
            for arm in ("panel_auto", "autonomous"):
                v = r["arms"].get(arm) or {}
                if not v.get("solved"):
                    continue
                p = v.get("panel") or {}
                rows.append({"arm": arm, "id": r["id"], "pk": p.get("pkiss"),
                             "pb": p.get("probknot"), "sk": p.get("shapeknots"),
                             "un": p.get("unanimous")})
print(f"{len(rows)} in-loop-SOLVED attempts with a panel adjudication\n")

for arm in ("panel_auto", "autonomous"):
    g = [r for r in rows if r["arm"] == arm]
    if not g:
        continue
    print(f"=== {arm}: n = {len(g)} in-loop solves ===")
    c = Counter((r["pk"], r["pb"], r["sk"]) for r in g)
    print(f"  {'pKiss':>6} {'ProbKnot':>9} {'ShapeKnots':>11} {'n':>5} {'share':>7}")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {str(k[0]):>6} {str(k[1]):>9} {str(k[2]):>11} {v:5d} {v/len(g):7.1%}")
    for lab, key in (("pKiss", "pk"), ("ProbKnot", "pb"), ("ShapeKnots", "sk")):
        ok = sum(1 for r in g if r[key])
        print(f"  {lab:>11} agrees on {ok:4d}/{len(g)} = {ok/len(g):6.1%}"
              + ("   <- IN LOOP" if key in ("pk", "pb") and arm == "panel_auto" else
                 "   <- IN LOOP" if key == "pk" and arm == "autonomous" else "   <- OUT OF LOOP"))
    # the marginal rejector: among attempts that pass everything else, how often does X fail?
    print("  marginal rejection (passes the other two, fails this one):")
    for lab, key, others in (("pKiss", "pk", ("pb", "sk")), ("ProbKnot", "pb", ("pk", "sk")),
                             ("ShapeKnots", "sk", ("pk", "pb"))):
        base = [r for r in g if all(r[o] for o in others)]
        if not base:
            print(f"    {lab:>11}  no attempts pass the other two")
            continue
        bad = sum(1 for r in base if not r[key])
        print(f"    {lab:>11}  {bad:4d}/{len(base):4d} = {bad/len(base):6.1%} of otherwise-passing "
              f"solutions are killed by {lab}")
    print()

print("=== what the ceiling would be if each oracle were satisfied ===")
g = [r for r in rows if r["arm"] == "panel_auto"]
tot = len(g)
un = sum(1 for r in g if r["un"])
print(f"  panel_auto: {tot} in-loop solves -> {un} unanimous = {un/tot:.1%} yield")
for lab, key in (("ShapeKnots", "sk"), ("ProbKnot", "pb")):
    would = sum(1 for r in g if all(r[k] for k in ("pk", "pb", "sk") if k != key))
    print(f"  if {lab} were also driven to agreement in-loop, the ceiling on this set is "
          f"{would}/{tot} = {would/tot:.1%}  ({would/max(un,1):.1f}x the current yield)")

print("\n=== distinct targets, which is what the benchmark counts ===")
byt = {}
for r in g:
    t = byt.setdefault(r["id"], {"un": 0, "pkpb": 0, "n": 0})
    t["n"] += 1
    t["un"] += int(bool(r["un"]))
    t["pkpb"] += int(bool(r["pk"]) and bool(r["pb"]))
print(f"  targets with >=1 in-loop solve:              {len(byt)}")
print(f"  targets with >=1 pKiss+ProbKnot agreement:   {sum(1 for v in byt.values() if v['pkpb'] > 0)}")
print(f"  targets with >=1 unanimous (all three):      {sum(1 for v in byt.values() if v['un'] > 0)}")
