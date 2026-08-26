#!/usr/bin/env python3
"""M15 addendum: the part of P2 the main script left out.

The frozen declaration required the ladder be CALIBRATED IN THE GATE'S UNITS ("刻度实测") with a top
rung at or above 4.0, above M13b's observed maximum of 3.79. m15_structure_filter.py recorded only
the detection fraction per rho, never the measured E[effect] per rho, so "top rung >= 4.0" was never
checked. That is a declared-but-not-implemented probe -- the same class of error as M10b's shape_2
and M11's complete-cases arm -- and it is repaired here rather than reported as if it had run.

COST: $0, offline.
"""
from __future__ import annotations
import json, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from m13_cache import POPS                                                            # noqa: E402
from m13_gc_placement import assemble, match, quantile_matched_delta, DELTA           # noqa: E402
from m12_sequence_intrinsic import synth_col                                          # noqa: E402
from m15_structure_filter import rnet_table, CUT                                      # noqa: E402

RHOS = (0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45, 0.60, 0.80)
DRAWS = 20

C = json.load(open(os.path.join(HERE, "m13_features.json")))
E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
RT = rnet_table()
print("=== M15 P2 补做:梯级的【实测刻度】(wet 单位),声明要求顶档 >= 4.0 ===\n")
print(f"{'人群':12s} " + "  ".join(f"ρ={r:.2f}" for r in RHOS) + "   顶档")
allrows = {}
for nm, fn in POPS:
    rows = C[fn]["rows"]
    for r, e in zip(rows, E[fn]):
        r["seq"] = e["seq"]
    keep = []
    for r in rows:
        v = RT.get((r["puzzle"], r["seq"]))
        if v is None:
            continue
        f1, w = v
        if w is not None and abs(w - r["wet"]) > 1e-6:
            continue
        if f1 >= CUT:
            keep.append(r)
    D = assemble(keep, 0)
    if D is None:
        continue
    scale = []
    for rho in RHOS:
        eff = []
        for t in range(DRAWS):
            col = synth_col(D["y"], D["ci"], rho, random.Random(int(rho * 1e4) + t))
            M = match(D, col, quantile_matched_delta(D, col), use_panel=True)
            if M and M["npair"] >= 10:
                eff.append(float(sum(M["sums"].values()) / sum(M["cnts"].values())))
        scale.append(float(np.mean(eff)) if eff else float("nan"))
    allrows[fn] = {"pop": nm, "scale": [round(x, 3) for x in scale], "top": round(scale[-1], 3)}
    print(f"{nm:12s} " + "  ".join(f"{x:+6.2f}" for x in scale) + f"   **{scale[-1]:+.2f}**")
tops = [v["top"] for v in allrows.values()]
print(f"\n六人群顶档:{tops}")
print(f"最小顶档 {min(tops):.2f}  声明要求 >= 4.0  ⇒ **{'通过' if min(tops) >= 4.0 else '不通过'}**")
json.dump(allrows, open(os.path.join(HERE, "m15_ladder_scale.json"), "w"),
          ensure_ascii=False, indent=1)
