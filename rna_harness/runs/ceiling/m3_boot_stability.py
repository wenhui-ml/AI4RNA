#!/usr/bin/env python3
"""Is "replication 1/3" a result or a bootstrap RNG draw?

The frozen run reported sample-3 arm 1 as [-0.0001, +0.0327] -- contains zero, counted as a replication
failure. Recomputing the identical estimate gave [+0.0002, +0.0327] -- excludes zero. Same data, same
statistic; the two differ by 0.0003. If that gap is Monte-Carlo noise in the percentile bootstrap, then
K3's replication count was decided by the RNG and not by the data, which is worth knowing before anyone
quotes "1/3".

Recompute g for sample 3 arm 1 once, then run the interval under 200 different bootstrap seeds and count
how often the lower bound lands above zero.
"""
from __future__ import annotations
import json, math, os, statistics as st, sys
sys.path.insert(0, "/data/wenhui/openclaude_rna"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2_heldout_cluster_ci as H
from m3_bca_check import gvec

out = {}
for nm, fn, formable in (("样本3 臂1", "panel_f1_third.json", True),
                         ("样本2(主) 臂1", "panel_f1_fresh.json", True),
                         ("基准4560 臂1", "panel_f1_designs.json", True)):
    g, used = gvec(fn, formable)
    los, his, ex = [], [], 0
    for s in range(200):
        ci, _n = H.boot_ci(g, 1000 + s)
        los.append(ci[0]); his.append(ci[1]); ex += 1 if (ci[0] > 0 or ci[1] < 0) else 0
    m = st.fmean(g)
    print(f"\n=== {nm} (cluster {len(g)},均值 {m:+.4f}) ===")
    print(f"  200 个 bootstrap 种子下,下界:中位 {st.median(los):+.5f}"
          f"  范围 {min(los):+.5f} ~ {max(los):+.5f}  (蒙特卡洛散布 {max(los)-min(los):.5f})")
    print(f"  **区间排除 0 的种子数 {ex}/200 = {ex/2*1:.1f}%**"
          f"  {'⇒ 裁定由 RNG 决定,不由数据决定' if 5 < ex < 195 else '⇒ 裁定稳定'}")
    out[nm] = {"n": len(g), "mean": round(m, 4), "excl_frac": ex / 200,
               "lo_median": round(st.median(los), 5), "lo_min": round(min(los), 5),
               "lo_max": round(max(los), 5)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "m3_boot_stability.json"), "w"), indent=1, ensure_ascii=False)
print("\n写出 m3_boot_stability.json")
