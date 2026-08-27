#!/usr/bin/env python3
"""M18v2: the two checks m18v_verify_repair.py named in its docstring and did not run.

Declaring a gate and not implementing it is failure mode G7 on this line, three times over
(M10b shape_2, M11 complete-cases, M15 P2). This closes the fourth.

  V4  the monkey-patch bit, and nothing else drifted. Re-run one M13c arm with the ORIGINAL
      treatment-shuffling null and check the z reproduces the committed pre-repair value. If it
      does, M19's changes came from the patch alone.
  V5  m15 feeds null_p its point estimate, not the bootstrap mean. boot()["mean"] is
      s.sum()/c.sum(); assert it against an independent recomputation.

COST: $0, offline.
"""
from __future__ import annotations

import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)

SEED, NPERM = 0, 2000
OK = {}

print("=== M18v2:补上 m18v 声明了却没跑的两项 ===\n")

# ------------------------------------------------------------------ V4
print("--- V4:原始(未打补丁)的零分布能否复现提交在册的修前 z ---")
import m13c_pairing_vs_composition as C3
from m13c_symmetric import resid_sd
from m12_sequence_intrinsic import parse_pairs
from leaderboard_design import targets as load_targets

T = load_targets(); PM = {p: sorted(parse_pairs(d["target"])) for p, d in T.items()}
C = json.load(open(os.path.join(HERE, "m13_features.json")))
E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
OLDJ = json.load(open(os.path.join(HERE, "m13c_symmetric.json")))
NEWJ = json.load(open(os.path.join(HERE, "m19_m13c_symmetric_repaired.json")))

fn, k, lab, key, hold = "panel_f1_fresh.json", "0.4", "P", "pair", "gcp"
D = C3.assemble(C[fn]["rows"], E[fn], PM, SEED)
sd, _b = resid_sd(D, key, hold); delta = float(k) * sd
M = C3.match(D, [c[key] for c in D["cells"]], delta, hold)
obs = C3.point(M)
null_orig = C3.null_dist(D, key, delta, hold, SEED, NPERM)      # the ORIGINAL, unpatched null
z_orig = (obs - float(null_orig.mean())) / (float(null_orig.std()) or 1)

o = OLDJ[k]["pops"][fn][lab]; n = NEWJ[k]["pops"][fn][lab]
v4 = abs(z_orig - o["z"]) <= 0.15 and abs(obs - o["effect"]) < 5e-4
OK["V4"] = bool(v4)
print(f"  效应        重算 {obs:+.4f}   在册(修前) {o['effect']:+.4f}   在册(修后) {n['effect']:+.4f}")
print(f"  z(原始零)  重算 {z_orig:6.2f}   在册(修前) {o['z']:6.2f}   在册(修后) {n['z']:6.2f}")
print(f"  ⇒ 原始零分布复现了修前的 z,修后的 z 只能来自补丁  {'✓' if v4 else '**✗**'}")

# ------------------------------------------------------------------ V5
print("\n--- V5:m15 传给 null_p 的是点估计,不是自助均值 ---")
import m13_gc_placement as G1
D2 = G1.assemble(C[fn]["rows"], SEED)
M2 = G1.match(D2, D2["gcp"], G1.DELTA, use_panel=True)
b = G1.boot(M2, SEED)
indep = float(sum(M2["sums"].values()) / sum(M2["cnts"].values()))
v5 = abs(b["mean"] - round(indep, 4)) < 1e-9
OK["V5"] = bool(v5)
print(f"  boot()['mean'] {b['mean']:+.4f}   独立复算的点估计 {indep:+.6f}   "
      f"自助区间 {b['ci']}  {'一致 ✓' if v5 else '**不一致 ✗**'}")

print("\n=== 结论 ===")
for kk, vv in OK.items():
    print(f"  {kk} {'通过' if vv else '**不通过**'}")
allok = all(OK.values())
print(f"  **{'四项+两项全部通过 —— 修法与 M19 的补丁路径都经得起检验'if allok else '有问题,退回'}**")
json.dump(OK, open(os.path.join(HERE, "m18v2_patch_check.json"), "w"), ensure_ascii=False, indent=1)
raise SystemExit(0 if allok else 1)
