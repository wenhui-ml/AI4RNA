#!/usr/bin/env python3
"""The panel's score ranks well and its ARGMAX is bad. The candidate explanation is saturation.

RUNS IN the RibonanzaNet venv.

WHAT KILLED THE PREVIOUS FRAMING. I had been calling this a contradiction -- validated classifier, bad
ranker. It is not. Measured on the same judge and the same quantity, the panel's continuous margin ranks
RNet crossed-pair F1 at rho = +0.355 inside our own pools, positive on 51 of 51 targets, ABOVE the +0.213
it manages inside other people's (target, method) cells. Its discrete vote count ranks at +0.299, positive
on 39 of 39. The signal is not weak and it is not inverted. So the loss lives in taking the ARGMAX of it.

THE HYPOTHESIS, and it explains four separate observations at once if true. The panel measures how much of
the pseudoknot a predictor reproduces, and it CAPS: once an oracle recalls every crossing pair, recall is
1.0 and nothing further can be resolved. So the score should discriminate in the bulk and go flat at the
top -- exactly where argmax reads it.

    S1  Rank correlation with RNet F1 INSIDE score strata. Predicts: clearly positive low and mid, ~0 in
        the top stratum. If it stays positive at the top, saturation is the wrong explanation.
    S2  The same for the solver's own ordering, which is not capped and should NOT go flat at the top.
        This is the control that decides whether flatness is a property of the panel or of the targets.
    S3  How crowded the ceiling is: the share of candidates already at panel mean recall = 1.0, and the
        spread of RNet F1 inside that ceiling group. A wide spread inside a group the panel scores
        identically is the loss, quantified.

WHY IT MATTERS BEYOND THIS BUG. If it holds, one mechanism explains: the panel screens well but selects
badly; the loose operating point (0.1) beats the strict one (1.0) on wet lab; unanimity tracks agreement
rather than correctness; and the entropy tie-break decides more than it should. Those were four separate
findings this week.

COST: $0.
"""
from __future__ import annotations
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from order_signal import spearman
from m2r_pairwise import wilcoxon
from pool_rerank_score import _score_chunk
from selector_replay import load_dump, KNOT_VERDICTS

R = json.load(open(os.path.join(HERE, "leaderboard_replay.json")))
T = {p: v["target"] for p, v in R["targets"].items()}
rows = {r["puzzle"]: r for r in R["rows"]}
P = load_dump(os.path.join(HERE, "pool_dump"), T)
elig = [p for p in P if rows.get(p, {}).get("verdict") in KNOT_VERDICTS]
jobs = [((p, i), r["seq"], T[p]) for p in elig for i, r in enumerate(P[p])]
from concurrent.futures import ProcessPoolExecutor
CH = max(1, len(jobs) // 160); sc = {}
print(f"{len(jobs)} 条候选打分\n", flush=True)
with ProcessPoolExecutor(max_workers=40) as ex:
    for res in ex.map(_score_chunk, [jobs[i:i + CH] for i in range(0, len(jobs), CH)]):
        sc.update(dict(res))

BANDS = [(0.0, 0.34, "低"), (0.34, 0.67, "中"), (0.67, 0.999, "高"), (0.999, 1.01, "封顶 (=1.0)")]
band_pm = {b[2]: [] for b in BANDS}
band_ord = {b[2]: [] for b in BANDS}
ceil_share, ceil_spread, ceil_n = [], [], []
for p in elig:
    v = [(i, P[p][i]["mean_recall"], sc[(p, i)]) for i in range(len(P[p])) if (p, i) in sc]
    if len(v) < 20: continue
    for lo, hi, lab in BANDS:
        g = [x for x in v if lo <= x[1] < hi]
        if len(g) < 8: continue
        f1 = [x[2] for x in g]
        if len(set(f1)) < 3: continue
        if len(set(x[1] for x in g)) > 1: band_pm[lab].append(spearman([x[1] for x in g], f1))
        band_ord[lab].append(spearman([x[0] for x in g], f1))
    top = [x for x in v if x[1] >= 0.999]
    if len(top) >= 8:
        ceil_share.append(len(top) / len(v)); ceil_n.append(len(top))
        ceil_spread.append(max(x[2] for x in top) - min(x[2] for x in top))

def show(title, d, alt):
    print(f"=== {title} ===")
    for _, _, lab in BANDS:
        z = d[lab]
        if not z:
            print(f"  {lab:12s} 靶标不足"); continue
        _, pv, _ = wilcoxon(z, alt)
        print(f"  {lab:12s} {len(z):3d} 个靶标  格内 Spearman 中位 {st.median(z):+.3f}  "
              f"为正 {sum(1 for x in z if x>0):3d}/{len(z):<3d}  p = {'n/a' if pv is None else f'{pv:.3g}'}")

show("S1 面板连续边距,按自己的分数分层", band_pm, "greater")
print()
show("S2 求解器排序(不封顶的对照),同样分层 —— 负值才是信号", band_ord, "less")
print(f"\n=== S3 天花板有多拥挤 ===")
if ceil_share:
    print(f"  {len(ceil_share)} 个靶标有 >=8 条候选处在 mean recall = 1.0")
    print(f"  这一档占全池比例中位 {st.median(ceil_share):.1%},条数中位 {st.median(ceil_n):.0f}")
    print(f"  面板给它们【完全相同的分数】,而它们的 RNet F1 跨度中位 {st.median(ceil_spread):.3f}")
    print(f"  -> 这个跨度就是面板在做选择的那一刻【看不见】的质量差")

lo = band_pm["低"] + band_pm["中"]
hi = band_pm["封顶 (=1.0)"] or band_pm["高"]
print(f"\n=== 判读 ===")
if lo and hi:
    print(f"  低/中档 {st.median(lo):+.3f}  ->  顶档 {st.median(hi):+.3f}")
    sat = st.median(hi) < st.median(lo) - 0.10
    print(f"  {'饱和成立 —— 面板的信息恰好在做决定的地方用尽' if sat else '饱和不成立 —— 顶档仍有分辨力,argmax 失败另有原因'}")
json.dump({"band_panel": {k: v for k, v in band_pm.items()},
           "band_order": {k: v for k, v in band_ord.items()},
           "ceiling_share_median": st.median(ceil_share) if ceil_share else None,
           "ceiling_f1_spread_median": st.median(ceil_spread) if ceil_spread else None},
          open(os.path.join(HERE, "panel_saturation.json"), "w"), ensure_ascii=False, indent=1)
print(f"\n  wrote panel_saturation.json")
