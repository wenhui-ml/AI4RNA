#!/usr/bin/env python3
"""The cell that was never computed: `panel_f1` against `shipped`, judged by three never-in-loop predictors.

WHAT THIS IS AND IS NOT. `selector_heldout_f1.json` holds ten pairwise comparisons and this is not one of them,
even though it is the one a product decision needs. It is therefore a COMPLETION of an existing table, not a
blind test: the two arms' medians were already on screen (panel_f1 0.2904, shipped 0.2425 under mean3) before
this file existed. Labelled as such, and the estimator is fixed here before anything is computed.

WHY THIS COMPARISON IS LEGITIMATE WHERE panel-VERSUS-RNet IS NOT. IPknot, Knotty and RNAPKplex are all
thermodynamic pseudoknot predictors, so they share an energy-model family with the panel's pKiss/ProbKnot/
ShapeKnots and not with RibonanzaNet. That is why "the panel looks good under these judges" cannot be read as
"the panel is the better ruler" -- the argument this project used on 2026-08-18 and then had to defend. But
`shipped` and `panel_f1` are BOTH panel-family selectors, so the shared error path largely cancels BETWEEN
them, and what is left is which panel statistic chooses better. That is a question these judges can answer.

    shipped    the confirm count, k = how many oracles reach full crossing recall, ties by ensemble entropy
    panel_f1   the panel's own precision-aware F1 on the crossing pairs -- an observable added 2026-08-19

ESTIMATOR, declared here: the unit is the TARGET (each arm delivers one sequence per target, 51 targets).
Paired difference panel_f1 - shipped per target; median, sign counts, two-sided Wilcoxon, and a bootstrap
interval over targets. Reported for mean3 and for each judge separately, because a result that lives in one
judge is a property of that judge. Multiplicity is stated rather than corrected away: this makes eleven
comparisons in that table, so the p is shown against 0.05 and against 0.05/11.

The 11 targets where the two arms pick the SAME sequence contribute exactly zero and are reported, because a
median over 51 targets of which 11 are structurally zero is easy to misread.

COST: $0. Re-judges 91 unique sequences; the arms themselves were chosen and paid for already.

KNOTTY FAILS ON SOME SEQUENCES (27/91 in the first run), so mean3 needs a rule, and the obvious one is wrong.
Averaging over whatever judges happen to be available makes the paired difference on such a target a 3-judge
mean minus a 2-judge mean -- two different estimators subtracted, which is the same weighting-heterogeneity
error class this project has now retracted four claims for. The declared primary is therefore mean3 over the
per-target INTERSECTION of judges available on BOTH arms; the available-judges variant is printed beside it,
and if the two disagree the disagreement is the result.
"""
from __future__ import annotations

import argparse, json, os, random, statistics as st, sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from selector_heldout import judge_one                                          # noqa: E402

ARMS = ("panel_f1", "shipped")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "selector_heldout_f1.json"))
    ap.add_argument("--run", default=os.path.join(HERE, "leaderboard_replay.json"))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "panel_f1_vs_shipped.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "panel_f1_vs_shipped_judged.json"),
                    help="raw per-(target,sequence) judge table; reused if present so Knotty is paid once")
    a = ap.parse_args()
    from m2r_pairwise import wilcoxon

    S = json.load(open(a.src))
    R = json.load(open(a.run))
    T = {p: v["target"] for p, v in R["targets"].items()}
    ch = {arm: S["chosen"][arm] for arm in ARMS}
    tgts = sorted(set(ch["shipped"]) & set(ch["panel_f1"]))
    same = [p for p in tgts if ch["shipped"][p] == ch["panel_f1"][p]]
    need = sorted({(p, ch[arm][p]) for arm in ARMS for p in tgts if ch[arm][p]})
    print(f"靶标 {len(tgts)};两臂选同一条的 {len(same)} 个(它们的配对差恒为 0);唯一序列 {len(need)} 条")

    J = {}
    if a.cache and os.path.exists(a.cache):
        raw = json.load(open(a.cache))
        J = {tuple(k.split("\t")): v for k, v in raw.items()}
        miss = [x for x in need if tuple(x) not in J]
        print(f"复用判官表 {a.cache}:{len(J)} 条,缺 {len(miss)} 条")
        assert not miss, f"缓存不覆盖本次需要的 {len(miss)} 条,删掉缓存重跑"
    else:
      with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, (puz, seq, out) in enumerate(
                ex.map(judge_one, [(p, s, T[p]) for p, s in need], chunksize=2), 1):
            J[(puz, seq)] = out
            if n % 25 == 0 or n == len(need):
                print(f"    judged {n}/{len(need)}", flush=True)
      if a.cache:
        json.dump({"\t".join(k): v for k, v in J.items()}, open(a.cache, "w"), indent=1)
        print(f"  写出判官表 {a.cache}")
    for name in ("ipknot", "knotty", "pkplex"):
        nf = sum(1 for v in J.values() if v[name] is None)
        print(f"  {name:9s} 失败 {nf}/{len(J)}")

    JUDGES = ("ipknot", "knotty", "pkplex")

    def both(p):
        """judges that returned a score for BOTH arms of this target"""
        ss = [ch[arm][p] for arm in ARMS]
        if not all(ss):
            return ()
        return tuple(j for j in JUDGES if all(J[(p, s_)][j] is not None for s_ in ss))

    def val(arm, p, judge, mode="isect"):
        s_ = ch[arm][p]
        if not s_:
            return None
        if judge != "mean3":
            return J[(p, s_)][judge]
        keep = both(p) if mode == "isect" else tuple(
            j for j in JUDGES if J[(p, s_)][j] is not None)
        xs = [J[(p, s_)][j] for j in keep]
        return st.mean(xs) if xs else None

    asym = [p for p in tgts if all(ch[arm][p] for arm in ARMS) and len(both(p)) < max(
        len([j for j in JUDGES if J[(p, ch[arm][p])][j] is not None]) for arm in ARMS)]
    print(f"  判官可用性两臂不一致的靶标 {len(asym)}/{len(tgts)}"
          f"{'(交集口径把它们拉回同一把尺)' if asym else ''}"
          f"{'  ' + ','.join(asym[:8]) if asym else ''}")

    res = {"n_targets": len(tgts), "n_same_pick": len(same), "n_unique_sequences": len(need),
           "note": "completion of selector_heldout_f1.json's comparison table, not a blind test",
           "judges": {}}
    res["n_asymmetric_judge_availability"] = len(asym)
    print(f"\n{'judge':10s} {'median d':>10s} {'mean d':>10s}  {'better/worse/tie':>18s} {'p2':>10s}  CI(median)")
    for judge in ("mean3", "mean3_avail", "ipknot", "knotty", "pkplex"):
        mode = "avail" if judge == "mean3_avail" else "isect"
        jn = "mean3" if judge == "mean3_avail" else judge
        d = [val("panel_f1", p, jn, mode) - val("shipped", p, jn, mode) for p in tgts
             if val("panel_f1", p, jn, mode) is not None and val("shipped", p, jn, mode) is not None]
        if not d:
            print(f"{judge:10s} 无可用配对"); res["judges"][judge] = {"n": 0}; continue
        w, pv, nz = wilcoxon(d, "two-sided")
        better = sum(1 for x in d if x > 0); worse = sum(1 for x in d if x < 0)
        rng = random.Random(a.seed)
        bs = sorted(st.median([d[rng.randrange(len(d))] for _ in d]) for _ in range(a.draws))
        lo, hi = bs[int(0.025 * a.draws)], bs[min(a.draws - 1, int(0.975 * a.draws))]
        print(f"{judge:10s} {st.median(d):+10.4f} {st.mean(d):+10.4f}  "
              f"{better:5d}/{worse:5d}/{len(d)-better-worse:5d}  "
              f"{(pv if pv is not None else float('nan')):10.4g}  [{lo:+.4f}, {hi:+.4f}]"
              f"{'  CI excludes 0' if (lo > 0 or hi < 0) else '  spans 0'}")
        res["judges"][judge] = {"n": len(d), "median": st.median(d), "mean": st.mean(d),
                                "better": better, "worse": worse, "tie": len(d) - better - worse,
                                "p_two_sided": pv, "ci_median": [lo, hi],
                                "ci_excludes_zero": bool(lo > 0 or hi < 0)}

    m = res["judges"]["mean3"]; pv = m["p_two_sided"]
    print(f"\n=== 判读(主口径 = mean3,单位 = 靶标)===")
    print(f"  多重比较如实说明:这是那张表的第 11 个比较。p = {pv:.4g};"
          f"对 0.05 {'过' if pv is not None and pv < 0.05 else '不过'},"
          f"对 Bonferroni 0.05/11 = {0.05/11:.4f} {'过' if pv is not None and pv < 0.05/11 else '不过'}")
    av = res["judges"].get("mean3_avail", {})
    print(f"  {len(same)}/{len(tgts)} 个靶标两臂选同一条,配对差恒为 0 —— 中位数要带这一点读")
    print(f"  加权口径:主口径 = 两臂共同可用判官的交集(n={m['n']});"
          f"「有几个用几个」口径 n={av.get('n')} 均值 {av.get('mean', float('nan')):+.4f} p={av.get('p_two_sided')}"
          f" —— 判官可用性不一致的靶标 {len(asym)} 个"
          f"{',两口径结论一致' if (m['ci_excludes_zero'] == av.get('ci_excludes_zero')) else ',两口径结论不一致 —— 分歧本身就是结果'}")
    print(f"  共享误差路径:两臂【同为面板家族】,判官是热力学家族,混杂在两臂之间大致抵消。")
    print(f"  这【不是】'面板优于 RNet' 的证据 —— 那个比较在这些判官下被混杂,在试管下已打平(p=0.40)。")
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
