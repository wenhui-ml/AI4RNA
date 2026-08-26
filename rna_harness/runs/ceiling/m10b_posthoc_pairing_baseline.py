#!/usr/bin/env python3
"""POST-HOC, not pre-registered: recompute P3 against the correct independence baseline.

The frozen P3 asked   hw_paired < (hw_a + hw_b)/2      -- the MEAN of the two marginal half-widths.
The independence baseline is   sqrt(hw_a^2 + hw_b^2)   -- what the difference's half-width would be
if the two rhos were uncorrelated. The frozen gate is therefore ~1.41x stricter than "pairing helped":
it tests "pairing helped a LOT". This file separates the two and reports the implied correlation

    rho_boot = (hw_a^2 + hw_b^2 - hw_D^2) / (2 * hw_a * hw_b)

which is the quantity the declaration actually assumed ("strong positive correlation").

The frozen verdict is NOT changed by anything here. Offline, $0.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "m10b_pairwise_restriction.json")))
rows = [x for x in J["rows"] if not x.get("void")]

print("对 (a × b)                       n   hw_a    hw_b    hw_配对  独立基线  比值   隐含相关")
print("-" * 96)
agg = []
for x in rows:
    ha, hb, hd = x["hw_a"], x["hw_b"], x["hw_paired"]
    indep = math.sqrt(ha * ha + hb * hb)
    ratio = hd / indep
    rb = (ha * ha + hb * hb - hd * hd) / (2 * ha * hb)
    agg.append((x["strat"], ratio, rb, hd < indep, hd < x["hw_marg_sum_half"]))
    print(f"[{x['strat']:<5}] {x['pair'][0]:<11}× {x['pair'][1]:<11} {x['n']:3d}"
          f" {ha:.4f}  {hb:.4f}  {hd:.4f}  {indep:.4f}  {ratio:.3f}  {rb:+.3f}")

for lab, sel in (("全部 15 对(可用)", lambda s: True), ("核心 6 对", lambda s: s == "core")):
    g = [a for a in agg if sel(a[0])]
    if not g:
        continue
    print(f"\n{lab}  n={len(g)}")
    print(f"  比 独立基线 sqrt(ha^2+hb^2) 窄 : {sum(1 for a in g if a[3])}/{len(g)}   <- 『配对有没有帮上忙』")
    print(f"  比 冻结基线 (ha+hb)/2      窄 : {sum(1 for a in g if a[4])}/{len(g)}   <- 冻结的 P3(严约 1.41 倍)")
    print(f"  比值 hw_配对/独立基线  中位数  : {sorted(a[1] for a in g)[len(g)//2]:.3f}")
    print(f"  隐含 bootstrap 相关    中位数  : {sorted(a[2] for a in g)[len(g)//2]:+.3f}"
          f"   范围 [{min(a[2] for a in g):+.3f},{max(a[2] for a in g):+.3f}]")
print("\n声明第二节假设的是『强正相关 ⇒ D 的区间远窄于两条边际』。上表是对那句话的直接测量。")
