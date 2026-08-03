#!/usr/bin/env python3
"""Read the overnight seed replicates and say whether either gap number was a lucky draw.

THE QUESTION, and it is the same one that already cost this project a published figure. The delivery rate was
quoted as 51/55 = 92.7% until five independent solver seeds showed it was the MAXIMUM of the distribution and
the mean was 87.6%. Both remaining gap numbers -- co-transcriptional 8/24 vs 19/24, switch 35/58 vs 54/58 --
are still single draws from the same nondeterministic solver.

WHAT WOULD CHANGE A CONCLUSION rather than merely a decimal:
  * the gap survives at every seed        -> it is a property of the requirement, and the single-run figure
                                             should be replaced by the mean with its spread
  * the gap vanishes at some seeds        -> the published number was a draw, and the honest report is the
                                             distribution with the fraction of seeds where the gap closes
  * per-target churn is large             -> a stable mean can hide targets flipping in and out, which is a
                                             different finding from a stable mean over stable targets

The third is reported separately for the same reason it was in deliver_seeds: averaging hides it.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = os.path.join(HERE, "seeds")


def load(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(SEEDS, pattern))):
        try:
            out.append((os.path.basename(p), json.load(open(p))))
        except Exception:                                                      # noqa: BLE001
            print(f"  !! 读不出 {p} —— 记为失败,不当成没跑过")
    return out


def spread(name, vals):
    if not vals:
        return "  (无数据)"
    return (f"  {name}: 中位 {st.median(vals):.1%}  极差 {max(vals)-min(vals):.1%}"
            + (f"  标准差 {st.stdev(vals):.1%}" if len(vals) > 1 else "  (单次)"))


def main():
    print("=== 夜间批次汇总 ===\n")

    # ---- 1. co-transcriptional -------------------------------------------------------------------
    ks = load("kinetic_s*.json")
    print(f"[1] 共转录折叠缺口 —— {len(ks)} 个独立种子")
    if ks:
        tool, ours, band_t, band_o = [], [], [], []
        for _, d in ks:
            f = d.get("free", {})
            n = f.get("n") or 0
            if n:
                tool.append(f["tool"] / n)
                ours.append(f["ours"] / n)
            b = d.get("band", {})
            if b.get("tool_n_in_band"):
                band_t.append(b["tool"] / b["tool_n_in_band"])
                band_o.append(b["ours"] / b["n"])
        print(spread("工具自己的答案", tool))
        print(spread("筛同一批池", ours))
        print(spread("带内 工具", band_t))
        print(spread("带内 我们", band_o))
        closed = sum(1 for t, o in zip(tool, ours) if o <= t)
        print(f"  缺口在 {closed}/{len(tool)} 个种子上闭合(闭合 = 我们不优于它)")
        print(f"  单次运行曾报:工具 8/24 = 33.3%,我们 19/24 = 79.2%")

    # ---- 2. two-state switch ----------------------------------------------------------------------
    ss = load("switch_s*.json")
    print(f"\n[2] 两态开关缺口 —— {len(ss)} 个独立种子")
    if ss:
        tool, ours, mt, mo = [], [], [], []
        for _, d in ss:
            f = d.get("free", {})
            if f.get("n"):
                tool.append(f["tool"] / f["n"])
                ours.append(f["ours"] / f["n"])
            for m in d.get("gc_matched", []):
                if m["tolerance"] == 0.0 and m["n"]:
                    mt.append(m["tool"] / m["n"])
                    mo.append(m["ours"] / m["n"])
        print(spread("工具自己的答案", tool))
        print(spread("筛同一批池", ours))
        print(spread("GC 严格配对(容差 0)工具", mt))
        print(spread("GC 严格配对(容差 0)我们", mo))
        closed = sum(1 for t, o in zip(mt, mo) if o <= t)
        print(f"  【决定性的那个控制】在 {closed}/{len(mt)} 个种子上闭合")
        print(f"  单次运行曾报:同 GC 下 35/58 = 60.3% 对 54/58 = 93.1%")

    # ---- 3+4. register-shift decoys ---------------------------------------------------------------
    for tag, label, note in (("shift_higher.json", "higher-order", ""),
                             ("shift_htype.json", "H-type",
                              "  注意:shift_decoy 直接调 _knot_recall,测的是【标定规则】本身 —— 而产品在 H-type\n"
                              "  上已改走严格规则。所以这一格回答的是:标定规则在 H-type 上位置特异吗?\n"
                              "  它此前在 H-type 上的问题是【对外部裁判过松】(29.9% vs 71.1%),那是另一种失效;\n"
                              "  这一格把两种失效分开了。")):
        p = os.path.join(SEEDS, tag)
        print(f"\n[{'3' if 'higher' in tag else '4'}] 移位诱饵 —— {label}")
        if note:
            print(note)
        if not os.path.exists(p):
            print("  (未产出)")
            continue
        d = json.load(open(p))
        t = d.get("total", {})
        print(f"  接受 {t.get('accepted')}/{t.get('n')} "
              f"[{(t.get('lo95') or 0):.1%}, {(t.get('hi95') or 1):.1%}]")
        for k, v in (d.get("by_shift") or {}).items():
            print(f"    位移 {k}: 可构造 {v['n_targets']-v['not_constructible']}/{v['n_targets']},"
                  f" 真做出移位假结 {v['n_valid_decoys']},接受 {v['accepted']}")
        print(f"  对照 kissing:0/79 = 0.0% [0.0%, 4.6%]")

    print("\n读法:极差小 = 是要求的性质;极差大 = 之前那个单次数字是一次抽签,报分布不报点估计。")


if __name__ == "__main__":
    main()
