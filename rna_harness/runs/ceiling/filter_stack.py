#!/usr/bin/env python3
"""Does our three-oracle panel add ANYTHING on top of RibonanzaNet as a filter? If not, it should go.

WHERE THIS COMES FROM. `label_calibration` established that our certificate tiers are informative about the
tube: 51.7% -> 34.1% -> 20.8% failure across not_found / knot_screened / knot_confirmed, monotone, with a
monotone k = 1/2/3 ladder underneath (25.2% / 18.2% / 15.2%) and a target-clustered p = 4.7e-6. Against no
filter at all (29.2%) the label clearly helps.

Then the matched-coverage comparison went the other way. Thresholding on RibonanzaNet's crossed-pair F1
beats the panel at EVERY coverage below 90%: at 60% coverage 18.3% against 23.0%, at 40% 17.3% against
20.8%. A published model anyone can run is a better filter than our panel.

That is not yet a reason to remove the panel. Two filters can each beat the other on their own and still be
complementary. This asks the only question that decides it:

    F1  Does panel AND RNet, at matched coverage, beat RNet alone?
        PASS -> the panel earns its place as a second, partly independent voice.
        KILL -> it is redundant, and a harness that keeps a component contributing nothing is charging the
        user for oracle calls that buy them no accuracy.

    F2  The same, the other way round: does adding RNet on top of the panel help the panel? Reported for
        symmetry so the answer cannot be read as an artefact of which filter is applied first.

    ONE THING THE PANEL DOES THAT RNET CANNOT, kept in view so F1 is not read too widely: RNet is this
    project's only external-ish judge. A harness that filters with RNet has no independent judge left for
    its own designs and must get one elsewhere -- which, on a benchmark whose outcome IS chemical mapping
    and whose filter is a model TRAINED on chemical mapping, is the honest reading of why RNet wins here.

COST: $0.
"""
from __future__ import annotations
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from external_judge import clopper_pearson

D = json.load(open(os.path.join(HERE, "criterion_vs_tube.json")))["designs"]
N, TH = len(D), 80.0


def at_coverage(keys, target):
    """Accept designs passing ALL key thresholds, tuned to land nearest `target` coverage."""
    if len(keys) == 1:
        k = keys[0]
        best = None
        for v in sorted({d[k] for d in D}):
            ds = [d for d in D if d[k] >= v]
            if len(ds) < 100:
                continue
            c = len(ds) / N
            if best is None or abs(c - target) < abs(best[0] - target):
                best = (c, sum(1 for d in ds if d["wet"] < TH) / len(ds), len(ds), {k: v})
        return best
    ka, kb = keys
    best = None
    # Coarse grid rather than every observed value: the full cross product is ~10^6 rescans and the
    # thresholds are continuous, so a 40 x 40 grid over each key's own quantiles resolves the frontier
    # to well under the differences being compared.
    def grid(k):
        v = sorted(d[k] for d in D)
        return sorted({v[min(len(v) - 1, int(len(v) * i / 40))] for i in range(41)})
    for va in grid(ka):
        for vb in grid(kb):
            ds = [d for d in D if d[ka] >= va and d[kb] >= vb]
            if len(ds) < 100:
                continue
            c = len(ds) / N
            if best is None or abs(c - target) < abs(best[0] - target):
                best = (c, sum(1 for d in ds if d["wet"] < TH) / len(ds), len(ds), {ka: va, kb: vb})
    return best


print(f"{N} 条真分子;失败 = 实测 OpenKnot < {TH:.0f}\n")
print(f"  {'目标覆盖':>8s} {'RNet 单独':>20s} {'面板 单独':>20s} {'面板 AND RNet':>22s} {'叠加是否更好':>14s}")
rows = []
for tgt in (0.8, 0.7, 0.6, 0.5, 0.4):
    r = at_coverage(["f1c"], tgt)
    p = at_coverage(["panel_mean"], tgt)
    b = at_coverage(["panel_mean", "f1c"], tgt)
    lo, hi = clopper_pearson(int(round(b[1] * b[2])), b[2])
    gain = r[1] - b[1]
    verdict = "叠加更好" if b[1] < r[1] - 0.01 else ("打平" if abs(b[1] - r[1]) <= 0.01 else "叠加更差")
    print(f"  {tgt:8.0%} {r[0]:9.1%}/{r[1]:<9.1%} {p[0]:9.1%}/{p[1]:<9.1%} "
          f"{b[0]:10.1%}/{b[1]:<10.1%} {verdict:>14s}")
    rows.append({"target": tgt, "rnet": r[:2], "panel": p[:2], "both": b[:2],
                 "both_ci": [lo, hi], "gain_over_rnet": gain, "verdict": verdict})

wins = sum(1 for r in rows if r["verdict"] == "叠加更好")
print(f"\n=== F1 判读 ===")
print(f"  面板叠在 RNet 之上更好的工作点:{wins}/{len(rows)}")
print(f"  平均降低失败率 {st.mean([r['gain_over_rnet'] for r in rows]):+.1%}")
print(f"  -> {'面板有独立贡献,保留' if wins >= 3 else '面板在 RNet 之上是冗余的 —— 它每次调用要花三个氧化物的算力,却买不到准确率'}")
print(f"\n=== 必须同时记住的一条 ===")
print(f"  RNet 训练于化学探测数据,而 OpenKnot Score 本身就是化学探测算出来的。")
print(f"  它在这个结局上是主场。这不算作弊,但它解释了为什么它赢,也意味着")
print(f"  【把 RNet 当过滤器就没有独立裁判了】—— 裁判必须另找,只能是试管。")
json.dump({"threshold": TH, "rows": rows, "panel_adds": wins >= 3},
          open(os.path.join(HERE, "filter_stack.json"), "w"), ensure_ascii=False, indent=1)
print(f"\n  wrote filter_stack.json")
