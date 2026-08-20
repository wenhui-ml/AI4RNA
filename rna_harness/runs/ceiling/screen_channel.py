#!/usr/bin/env python3
"""S1: put RibonanzaNet where the panel's gradient is ZERO by construction -- the screen channel.

Declared in `screen_channel_declaration.json`, pushed before a single judge ran.

WHY HERE AND NOT AT THE TOP. Two measured facts had to be put together. (1) RNet knows more: its `f1c`, one
column, reaches net R^2 0.117-0.164 within cell against 0.086-0.103 for all twenty-one thermodynamic columns,
and panel-plus-RNet reaches 0.196, four times the shipped verdict. (2) Swapping rulers AT THE TOP is a
non-event: matched-coverage panel-versus-RNet came out +0.0000, p = 0.40 in the tube, and five separate runs
(A10, A11, A12, A13-A15) found the information sitting at the low end with the top saturated. Together those
say something specific -- RNet's extra knowledge lives in the range where we do not make decisions -- and they
point at the one place RNet does not have to fight the panel for the top:

    the SCREEN CHANNEL. 15 of the 51 comparable targets, where the panel's k is IDENTICALLY ZERO: twelve
    whose pools hold no knot_confirmed candidate at all, three H-types refused by the topology gate. The
    panel's gradient there is zero by construction, and the product's fallback is "the first candidate
    clearing the loose screen" -- the solver's own order. The five targets that deliver 0.000 are in here.

THE CONTROL ARM IS NOT OPTIONAL. `random_screen` -- a random survivor, averaged over ten fixed draws -- is
what separates "RNet chooses well" from "anything beats the solver's order". Without it a win cannot be
attributed.

THE JUDGES ARE BIASED AGAINST THE HYPOTHESIS, WHICH IS WHY THEY ARE USABLE. IPknot, Knotty and RNAPKplex are
thermodynamic; they share an energy-model family with the PANEL and not with RibonanzaNet. A win for RNet
under them is a win under a conservative test. The primary set is FIXED at IPknot + RNAPKplex, both required,
because `mean3`-style "average whatever succeeded" was shown on 2026-08-20 not to be a reproducible quantity:
Knotty is wall-clock bounded and its availability moved from 37 to 36 targets between two runs of the same
code on the same input. Knotty runs anyway, reported as coverage only.

n = 15, AND THAT IS STATED AS A LIMIT NOT A FOOTNOTE. A sign test at n=15 gives p=0.035 at 12/15 and p=0.118
at 11/15. This population can produce directional evidence and nothing stronger.

COST: $0, offline. About 180 unique sequences.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from selector_replay import load_dump, pick, KNOT_VERDICTS, SCREEN                # noqa: E402
from rna_harness.cognitive.certify import calibrated_rule_applies                 # noqa: E402

PRIMARY = ("ipknot", "pkplex")


def judge(item):
    from selector_heldout import judge_one
    return judge_one(item)


def wilcoxon2(d):
    from m2r_pairwise import wilcoxon
    return wilcoxon(d, "two-sided")


def boot(v, draws, seed):
    rng = random.Random(seed); n = len(v)
    if n < 5:
        return None, None, None
    o = sorted(st.median([v[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return st.median(v), o[int(0.025 * draws)], o[min(draws - 1, int(0.975 * draws))]


def line(lab, d, draws, seed):
    if len(d) < 5:
        print(f"  {lab:34s} 靶标 {len(d):3d} —— 太少"); return {"n": len(d), "no_data": True}
    pt, lo, hi = boot(d, draws, seed)
    w, pv, nz = wilcoxon2(d)
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0)
    ex = lo > 0 or hi < 0
    print(f"  {lab:34s} 靶标 {len(d):3d}  中位 {pt:+.4f}  均值 {st.mean(d):+.4f}  "
          f"好/差/平 {pos:3d}/{neg:3d}/{len(d)-pos-neg:3d}  "
          f"p={pv if pv is None else round(pv,4)}  CI [{lo:+.4f}, {hi:+.4f}]"
          f"{'  排除 0' if ex else '  跨过 0'}")
    return {"n": len(d), "median": pt, "mean": st.mean(d), "pos": pos, "neg": neg,
            "p_two_sided": pv, "ci": [lo, hi], "excludes_zero": bool(ex),
            "excludes_zero_positive": bool(ex and lo > 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdraws", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "screen_channel.json"))
    a = ap.parse_args()
    D = json.load(open(os.path.join(HERE, "screen_channel_declaration.json")))

    R = json.load(open(os.path.join(HERE, "leaderboard_replay.json")))
    T = {p: v["target"] for p, v in R["targets"].items()}
    rows = {r["puzzle"]: r for r in R["rows"]}
    P = load_dump(os.path.join(HERE, "pool_dump"), T)
    F = json.load(open(os.path.join(HERE, "selector_heldout_rnet.json")))["f1c"]
    G = {p: not calibrated_rule_applies(t) for p, t in T.items()}
    elig = [p for p in P if rows.get(p, {}).get("verdict") in KNOT_VERDICTS]
    bad = [p for p in elig if (pick(P[p], "shipped", gated=G[p]) or {}).get("seq") != rows[p]["sequence"]]
    print(f"完整性闸门(与 selector_heldout 同一个):可比靶标 {len(elig)},不一致 {len(bad)}")
    assert not bad, "重放没有复刻产品行为,不读任何数字"

    chosen, why = {}, {}
    rng = random.Random(a.seed)
    for p in elig:
        rw, sc = P[p], F.get(p, {})
        conf = [] if G[p] else [r for r in rw if r["knot_confirmed"]]
        if conf:
            continue                                            # 有 confirmed → 不属本人群
        surv = [r for r in rw if r["mean_recall"] >= SCREEN]
        scored = [r for r in surv if r["seq"] in sc]
        if not surv or len(scored) < 2 or len({round(sc[r["seq"]], 6) for r in scored}) < 2:
            why[p] = "RNet 分不出高下,按声明剔除"; continue
        chosen[p] = {"screen_first": surv[0]["seq"],
                     "rnet_screen": max(scored, key=lambda r: sc[r["seq"]])["seq"],
                     "random_screen": [surv[rng.randrange(len(surv))]["seq"] for _ in range(a.rdraws)]}
        why[p] = ("拓扑闸门(H-type)" if G[p] else "池内无 confirmed")
    print(f"筛通道靶标 {len(chosen)};剔除 {sum(1 for v in why.values() if '剔除' in v)}")
    print(f"  原因分布 {dict(collections.Counter(v for v in why.values() if '剔除' not in v))}")

    need = sorted({(p, s) for p, v in chosen.items()
                   for s in ([v["screen_first"], v["rnet_screen"]] + v["random_screen"])})
    print(f"要评判的唯一 (靶标,序列) {len(need)} 条,{a.workers} 进程", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    J = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, (p, s, o) in enumerate(ex.map(judge, [(p, s, T[p]) for p, s in need], chunksize=2), 1):
            J[(p, s)] = o
            if n % 20 == 0 or n == len(need):
                print(f"    judged {n}/{len(need)}", flush=True)
    for nm in ("ipknot", "pkplex", "knotty"):
        f = sum(1 for v in J.values() if v[nm] is None)
        print(f"  {nm:8s} 失败 {f}/{len(J)} = {f/len(J):.1%}")

    def q(p, s):
        v = J[(p, s)]
        xs = [v[k] for k in PRIMARY]
        return None if any(x is None for x in xs) else st.mean(xs)

    per = {}
    for p, v in chosen.items():
        sf, rn = q(p, v["screen_first"]), q(p, v["rnet_screen"])
        rd = [q(p, s) for s in v["random_screen"]]
        rd = [x for x in rd if x is not None]
        if sf is None or rn is None or not rd:
            continue
        per[p] = {"screen_first": sf, "rnet_screen": rn, "random_screen": st.mean(rd), "n_rd": len(rd)}
    print(f"\n  两家主判官都成功、可进检验的靶标 {len(per)}/{len(chosen)}"
          f"  (声明的无数据下限 10 → {'过' if len(per) >= 10 else '不过'})")
    res = {"declaration": D, "n_targets": len(per), "per_target": per,
           "arm_medians": {k: st.median([v[k] for v in per.values()])
                           for k in ("screen_first", "rnet_screen", "random_screen")}}
    print(f"  各臂中位:{ {k: round(v,4) for k,v in res['arm_medians'].items()} }")
    if len(per) < 10:
        res["verdict"] = "no_data"
        print(f"\n  【结论】no_data —— {D['outcomes_fixed_in_advance']['no_data']}")
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False); return 0

    print(f"\n  --- 主检验与必需对照(主口径 = IPknot + RNAPKplex 的平均,两家都必须成功)---")
    res["primary"] = line("rnet_screen − screen_first 【主】", 
                          [v["rnet_screen"] - v["screen_first"] for v in per.values()], a.draws, a.seed)
    res["control"] = line("rnet_screen − random_screen 【对照】",
                          [v["rnet_screen"] - v["random_screen"] for v in per.values()], a.draws, a.seed + 1)
    res["reference"] = line("random_screen − screen_first (参考)",
                            [v["random_screen"] - v["screen_first"] for v in per.values()], a.draws, a.seed + 2)

    P1, C = res["primary"], res["control"]
    v = ("null" if not P1.get("excludes_zero_positive")
         else "rnet_wins_here" if C.get("excludes_zero_positive") else "not_rnet_just_not_solver_order")
    print(f"\n=== 判定(按声明)===")
    print(f"  【结论】{v} —— {D['outcomes_fixed_in_advance'][v]}")
    print(f"  功效如实说明:{D['estimator']['power_stated_upfront']}")
    print(f"  判官的偏见方向:{D['judges']['the_bias_runs_against_the_hypothesis']}")
    print(f"  非盲检:{D['not_a_blind_test']}")
    res["verdict"] = v
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
