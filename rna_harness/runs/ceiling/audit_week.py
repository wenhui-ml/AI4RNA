#!/usr/bin/env python3
"""Adversarial audit of this week's three SHIPPED changes, written as a hostile referee would attack them.

Every experiment this week was checked against ITS OWN pre-registration. That is not the same as checking
the WEEK, and a referee attacks the week: multiplicity across experiments, power of the nulls, confounds
that a single experiment's internal logic cannot see, and claims that are true by construction.

Five objections are raised and answered below. Three stand and are recorded as limitations; one is closed
with an exact unconfounded replay; one forces a claim to be withdrawn from the report.
"""
import json, statistics as st, sys, os
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__))


def sgn(u, d):
    n = u + d
    return sum(comb(n, i) for i in range(u, n + 1)) / 2 ** n if n else float("nan")


def main():
    out = {}
    # ---- OBJECTION 4, closed exactly: same pool, both policies -------------------------------------
    # grade_max_bench ran two arms as two separate solver invocations, and DesiRNA is wall-clock bounded,
    # so the arms saw different candidate pools. The 17-0 direction cannot plausibly come from pool luck,
    # but "implausible" is not "excluded". This closes it: judge_paired.json stores each pool in the order
    # the policy would have consumed it, so applying BOTH policies to the SAME pool is an exact replay with
    # no confound at all. This is the number that should have been quoted for the shipped change.
    P = json.load(open(os.path.join(HERE, "judge_paired.json")))
    per_t = {}
    for r in P["rows"]:
        ds = r.get("designs") or []
        if not ds:
            continue
        first = ds[0]["k_panel"]                      # R0: first candidate clearing the floor
        best = max(d["k_panel"] for d in ds)          # grade-max: best in the SAME pool
        t = r["task_id"]
        a, b = per_t.get(t, (None, 0))
        per_t[t] = (first if a is None else a, max(b, best))
    up = sum(1 for a, b in per_t.values() if b > a)
    dn = sum(1 for a, b in per_t.values() if b < a)
    print("=== 异议4(已关闭):同一个候选池上跑两种策略,精确回放,零混杂 ===")
    print(f"  {len(per_t)} 个靶标:等级提升 {up} 个 = {up/len(per_t):.1%},下降 {dn} 个,符号 p={sgn(up,dn):.3g}")
    print(f"  -> 出厂那条改动应当引用这个数字,而不是 grade_max_bench 里两次独立运行的对比")
    out["objection4_same_pool"] = {"n": len(per_t), "up": up, "down": dn, "p": sgn(up, dn)}

    # ---- OBJECTION 2: multiplicity -----------------------------------------------------------------
    E = json.load(open(os.path.join(HERE, "entropy_select.json")))
    ps = [(j, n, v["p"]) for j, rr in E["res"].items() for n, v in rr.items() if v["up"] + v["dn"] > 0]
    print(f"\n=== 异议2(成立):entropy_select 做了 {len(ps)} 个方向检验 ===")
    print(f"  主检验 p=0.0362;Bonferroni 门槛 0.05/{len(ps)} = {0.05/len(ps):.4f} —— 【过不了族内校正】")
    print(f"  P1 是跑前唯一指定的主检验,其余为描述性 —— 但报告必须写明,否则读起来像筛选")
    out["objection2_multiplicity"] = {"n_tests": len(ps), "bonferroni": 0.05 / len(ps), "p_primary": 0.0362}

    # ---- OBJECTION 3: power of E1-P2 ---------------------------------------------------------------
    import random
    L = json.load(open(os.path.join(HERE, "ladder_confirm.json")))
    n = L["E1"]["n_fresh"]

    def power(true_loss, reps=4000, rate=0.058, seed=7):
        rng = random.Random(seed); hit = 0
        for _ in range(reps):
            lost = gained = 0
            for _ in range(n):
                if rng.random() < true_loss:
                    lost += 1
                elif rng.random() < rate / 2:
                    lost += 1
                elif rng.random() < rate / 2:
                    gained += 1
            if sgn(lost, gained) < 0.05:
                hit += 1
        return hit / reps
    pw = {f"{t:.0%}": power(t) for t in (0.02, 0.05, 0.10)}
    print(f"\n=== 异议3(成立):ladder_confirm 的 E1-P2 是个【检验力有限的 null】 ===")
    print(f"  n={n},观察到 {L['E1']['lost']} 丢 / {L['E1']['gained']} 得,与 5.8% 噪声下的期望 {n*0.058:.1f} 次翻转一致")
    print(f"  真实损失率 2% -> 只有 {pw['2%']:.0%} 把握测出;5% -> {pw['5%']:.0%};10% -> {pw['10%']:.0%}")
    print(f"  -> 「丢证与零不可分辨」只到【10% 以上的损失能被排除】这个强度,报告里当时没写")
    out["objection3_power"] = {"n": n, "power": pw}

    # ---- OBJECTION 1: knife-edge -------------------------------------------------------------------
    A = json.load(open(os.path.join(HERE, "agent_ladder_lottery.json")))
    n4 = sum(1 for r in A["targets"].values() if r["full"] == 4)
    print(f"\n=== 异议1(成立):agent 阶梯 P1 的判定悬在少数几次抽样上 ===")
    print(f"  3 个靶标恰好 4/10,阈值是 <=3/10;各少中一次就从 {A['n_lottery']}/6 变成 {A['n_lottery']+n4}/6,越过 P1 的 4/6")
    print(f"  但保留 @180 的真正理由与阈值无关:286-400 nt 的靶标截断后是 0/10、0/10、1/10")
    out["objection1_knife_edge"] = {"n_at_4": n4, "n_lottery": A["n_lottery"], "flips_verdict": A["n_lottery"] + n4 >= 4}

    # ---- OBJECTION 5: tautology --------------------------------------------------------------------
    K = json.load(open(os.path.join(HERE, "k1_diagnosis.json")))
    s = K["stats"]
    within = max(s[f"k{i}"]["entropy"] for i in (1, 2, 3)) - min(s[f"k{i}"]["entropy"] for i in (1, 2, 3))
    between = s["native"]["entropy"] - max(s[f"k{i}"]["entropy"] for i in (1, 2, 3))
    print(f"\n=== 异议5(成立,并撤回一句话):「我们的设计比天然确定三倍」多半是同义反复 ===")
    print(f"  组内(k1..k3)差 {within:.4f},组间(设计 vs 天然)差 {between:.4f} —— 相差 {between/within:.1f} 倍")
    print(f"  设计是为靶标优化出来的,天然不是;优化过的序列折叠更确定近乎构造使然")
    print(f"  -> 撤回「过度确定是一个发现」的说法。并列打破器建立在【同靶标同等级候选之间】的")
    print(f"     小得多的差上(+0.0283),那一条由前瞻检验支撑,不受本条影响")
    out["objection5_tautology"] = {"within": within, "between": between, "withdrawn": True}

    json.dump(out, open(os.path.join(HERE, "audit_week.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote audit_week.json")


if __name__ == "__main__":
    main()
