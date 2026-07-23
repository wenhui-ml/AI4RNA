#!/usr/bin/env python3
"""Grammar-graded capability frontier (chomsky_climb, no new compute).

Bins the n=60 held-out crossing targets by their intrinsic crossing complexity -- the crossing-pair count
n_pk_pairs, an exact O(|P|^2) grade fixed before any design -- and reports, per grade, the solve rate of each
arm. Question: does formal crossing complexity predict where the autonomous operator's panel-robust
capability breaks down? Joins existing run.json + panel adjudications; adds no live episodes.
"""
import json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return json.load(open(os.path.join(HERE, p)))


# grade + panel_auto solved (pKiss AND ProbKnot) from its run
pa = load("panel_auto/run.json")
grade = {r["id"]: r["n_pk_pairs"] for r in pa["rows"]}
pa_solved = {r["id"]: bool(r["by_policy"]["panel_auto"]["solved"]) for r in pa["rows"]}
# panel_auto panel-unanimous (all 3 oracles)
pa_uni = {r["id"]: bool(r.get("unanimous")) for r in load("panel_auto/panel_panel_auto.json")["rows"]}
# gamer (autonomous, pKiss) + its panel-unanimous
gv = load("generous/run.json")
gm_pk = {r["id"]: bool(r["by_policy"]["autonomous"]["solved"]) for r in gv["rows"]}
gm_uni = {r["id"]: bool(r.get("unanimous")) for r in load("generous/panel_autonomous.json")["rows"]}

by = defaultdict(list)
for tid, g in grade.items():
    by[g].append(tid)

rows = []
print(f"{'grade':>5} {'n':>3} | {'gamer pKiss':>11} {'gamer unan':>10} | {'pnl pKiss&PB':>12} {'pnl unanim':>10}")
for g in sorted(by):
    ids = by[g]; n = len(ids)
    gpk = sum(gm_pk.get(i, False) for i in ids)
    gun = sum(gm_uni.get(i, False) for i in ids)
    ppb = sum(pa_solved.get(i, False) for i in ids)
    pun = sum(pa_uni.get(i, False) for i in ids)
    rows.append({"grade": g, "n": n, "gamer_pkiss": gpk, "gamer_unanimous": gun,
                 "panel_pk_pb": ppb, "panel_unanimous": pun})
    print(f"{g:>5} {n:>3} | {gpk:>7}/{n:<3} {gun:>6}/{n:<3} | {ppb:>8}/{n:<3} {pun:>6}/{n:<3}")

tot = {k: sum(r[k] for r in rows) for k in ("n", "gamer_pkiss", "gamer_unanimous", "panel_pk_pb", "panel_unanimous")}
print(f"{'TOT':>5} {tot['n']:>3} | {tot['gamer_pkiss']:>7}/{tot['n']:<3} {tot['gamer_unanimous']:>6}/{tot['n']:<3} "
      f"| {tot['panel_pk_pb']:>8}/{tot['n']:<3} {tot['panel_unanimous']:>6}/{tot['n']:<3}")
json.dump({"by_grade": rows, "totals": tot}, open(os.path.join(HERE, "grade_frontier.json"), "w"), indent=1)
print("-> grade_frontier.json")
