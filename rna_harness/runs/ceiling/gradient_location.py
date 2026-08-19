#!/usr/bin/env python3
"""A13: which LAYER of the ruler still has a gradient?

Declared in `gradient_location_declaration.json`, committed and pushed before anything was computed.

THE QUESTION AND WHY IT IS THE RIGHT ONE NOW. Three independent attempts to read a signal at the top of the
ruler returned zero (A11 entropy, A12 the shipped tie-breaker, and the eleventh cell of the selector table).
The next move is therefore not a fourth quantity but a different place to decide. "Do not decide at the top"
has exactly two landing sites, and they are two different engineering projects, so measure which one first:

    the BOUNDARY is the lever   the good candidates sit OUTSIDE the top block, excluded by the grade bar
    the BLOCK is the lever      the good candidates sit INSIDE it and nothing there is readable, so the
                                block has to be made smaller at GENERATION time

MEANS, NOT MAXIMA, AND THAT IS THE WHOLE TRICK. A max over a set is inflated by the set's SIZE alone -- this
project measured that inflation at about +3.67 with its own permutation null. Means are not, so the three
terms below add up EXACTLY to "delivered minus pool average", with no oracle anywhere and no set-size
correction to argue about. The identity is asserted per target, not hoped for.

    A = mean(q | confirmed)  - mean(q | whole pool)      what "confirmed at all" buys        (k >= 1)
    B = mean(q | top block)  - mean(q | confirmed)       what "take the highest grade" buys  (k = max)
    C = q(delivered)         - mean(q | top block)       what the tie-breaker buys
    A + B + C == q(delivered) - mean(q | whole pool)

A is a positive CONTROL: the panel's confirmation boundary is supposed to carry information, and if A is zero
the chain is dead in this population and B/C mean nothing. C is a REPLICATION of A12 on a different quality
variable and should be ~0. Only B is a new question, and its direction is deliberately not declared.

TWO POPULATIONS, BECAUSE EACH ONE'S JUDGE IS THE OTHER'S WEAKNESS. Population 1 is our own 57 pools judged by
RibonanzaNet -- real candidate pools, but that judge's verdict has already once failed to reach the tube
(+0.129 -> +-0.000, p = 0.40). Population 2 is the OpenKnot library judged by MEASUREMENT -- load-bearing,
but its "pool" is a library grouped by method, not one solver run.

COST: $0, entirely offline. Dumped pools, RNet scores already computed, tube scores already measured.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from selector_replay import load_dump, pick, KNOT_VERDICTS, SCREEN               # noqa: E402
from arbitration_confirm import load_fresh                                       # noqa: E402
from tiebreak_direction import grade                                             # noqa: E402
from rna_harness.cognitive.certify import calibrated_rule_applies                # noqa: E402

FLOOR = 20


def boot(vals, draws, rng, stat=st.median):
    v = list(vals)
    if len(v) < 5:
        return None, None, None
    n = len(v)
    o = sorted(stat([v[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return stat(v), o[int(0.025 * draws)], o[min(draws - 1, int(0.975 * draws))]


def line(label, vals, draws, seed):
    if len(vals) < FLOOR:
        print(f"  {label:38s} 单位 {len(vals):4d} —— 低于声明下限 {FLOOR},无数据")
        return {"n": len(vals), "no_data": True}
    pt, lo, hi = boot(vals, draws, random.Random(seed))
    pos = sum(1 for x in vals if x > 0); neg = sum(1 for x in vals if x < 0)
    ex = lo > 0 or hi < 0
    print(f"  {label:38s} 单位 {len(vals):4d}  中位 {pt:+.4f}  均值 {st.mean(vals):+.4f}  "
          f"正/负 {pos:4d}/{neg:4d}  CI [{lo:+.4f}, {hi:+.4f}]"
          f"{'  排除 0' if ex else '  跨过 0'}")
    return {"n": len(vals), "median": pt, "mean": st.mean(vals), "n_pos": pos, "n_neg": neg,
            "ci": [lo, hi], "excludes_zero": bool(ex),
            "excludes_zero_positive": bool(ex and lo > 0)}


def headroom(layers, pool_q, draws, rng):
    """max(layer) minus E[max of |layer| draws from the whole pool] -- what the boundary buys BEYOND size."""
    out = {}
    for nm, qs in layers.items():
        if not qs:
            continue
        n = len(qs)
        nulls = [max(pool_q[rng.randrange(len(pool_q))] for _ in range(n)) for _ in range(draws)]
        out[nm] = max(qs) - st.mean(nulls)
    return out


def report(name, units, draws, seed):
    """units: list of dicts with A/B/C/identity/head_*; prints and returns the row."""
    row = {"n_units": len(units)}
    print(f"\n  --- 主检验(三项严格相加)---")
    for k, lab in (("A", "A  confirmed 边界(k>=1)"), ("B", "B  等级取最高(k=max)"),
                   ("C", "C  打平器")):
        row[k] = line(lab, [u[k] for u in units], draws, seed + ord(k))
    row["total"] = line("A+B+C = 交付 − 池均值", [u["A"] + u["B"] + u["C"] for u in units], draws, seed)
    bad = [u for u in units if abs(u["A"] + u["B"] + u["C"] - u["identity"]) > 1e-9]
    print(f"  恒等式 A+B+C ≡ 交付−池均值:违例 {len(bad)} 个" + ("  ✓" if not bad else "  ✗ 有问题"))
    assert not bad, "分解恒等式不成立,结果不可读"
    hk = [k for k in ("confirmed", "block") if any(k in u.get("head", {}) for u in units)]
    if hk:
        print(f"  --- 次检验 headroom(减掉「n 抽最大」的零)---")
        row["headroom"] = {k: line(f"headroom {k}", [u["head"][k] for u in units if k in u["head"]],
                                   draws, seed + 7) for k in hk}
    return row


def pop1(a):
    R = json.load(open(os.path.join(HERE, "leaderboard_replay.json")))
    T = {p: v["target"] for p, v in R["targets"].items()}
    rows = {r["puzzle"]: r for r in R["rows"]}
    P = load_dump(os.path.join(HERE, "pool_dump"), T)
    F = json.load(open(os.path.join(HERE, "selector_heldout_rnet.json")))["f1c"]
    G = {p: not calibrated_rule_applies(t) for p, t in T.items()}
    elig = [p for p in P if rows.get(p, {}).get("verdict") in KNOT_VERDICTS]
    bad = [p for p in elig if (pick(P[p], "shipped", gated=G[p]) or {}).get("seq") != rows[p]["sequence"]]
    print(f"  完整性闸门(与 selector_heldout 同一个):可比靶标 {len(elig)},不一致 {len(bad)} {bad[:5]}")
    assert not bad, "重放没有复刻产品行为,不读任何数字"
    units, skip = [], collections.Counter()
    rng = random.Random(a.seed)
    for p in elig:
        rw, sc = P[p], F.get(p, {})
        q = {r["seq"]: sc[r["seq"]] for r in rw if r["seq"] in sc}
        pool = [q[r["seq"]] for r in rw if r["seq"] in q]
        conf = [] if G[p] else [r for r in rw if r["knot_confirmed"] and r["seq"] in q]
        if not conf or len(pool) < 2:
            skip["无 confirmed / 池太小"] += 1; continue
        kmax = max(len(r["confirmed_by"]) for r in conf)
        blk = [r for r in conf if len(r["confirmed_by"]) == kmax]
        d = pick(rw, "shipped", gated=G[p])
        if d is None or d["seq"] not in q:
            skip["交付序列无质量分"] += 1; continue
        cq = [q[r["seq"]] for r in conf]; bq = [q[r["seq"]] for r in blk]
        units.append({"A": st.mean(cq) - st.mean(pool), "B": st.mean(bq) - st.mean(cq),
                      "C": q[d["seq"]] - st.mean(bq), "identity": q[d["seq"]] - st.mean(pool),
                      "puzzle": p, "n_block": len(blk), "n_conf": len(conf), "n_pool": len(pool),
                      "head": headroom({"confirmed": cq, "block": bq}, pool, a.hdraws, rng)})
    print(f"  参与检验的靶标 {len(units)};跳过 {dict(skip)}")
    print(f"  层大小中位:顶端块 {st.median([u['n_block'] for u in units]):.0f}  "
          f"confirmed {st.median([u['n_conf'] for u in units]):.0f}  "
          f"整池 {st.median([u['n_pool'] for u in units]):.0f}")
    return units


def pop2(a, fn):
    rec, _sp, _m = load_fresh(os.path.join(HERE, fn), "panel_recall_product")
    cells = collections.defaultdict(list)
    for r in rec:
        cells[(r["puzzle"], r["method"])].append(r)
    units, skip = [], collections.Counter()
    rng = random.Random(a.seed)
    # A12's `equal_grade_blocks` is deliberately NOT reused here. It carries MIN_BLOCK=4 and
    # MIN_DISTINCT_WET=3, and the second of those SELECTS ON THE OUTCOME VARIABLE -- keeping only cells whose
    # wet scores differ would bias B, which is the one term this run exists to measure. MIN_BLOCK=4 would
    # also throw away most cells: the real pools' median block is 3. So the block here is the plain
    # maximal-grade block, with no filter beyond "the cell has at least two molecules and one confirmed".
    for cell, rw in cells.items():
        pool = [r["wet"] for r in rw]
        conf = [r for r in rw if (grade(r) or 0) >= 1]
        if len(pool) < 2 or not conf:
            skip["池太小/无 confirmed"] += 1; continue
        kmax = max(grade(r) for r in conf)
        blk = [r for r in conf if grade(r) == kmax]
        # 交付 = 出厂规则:同等级块内取最高熵。熵不在这些文件里,而 A12 已测「它买不到东西」,
        # 所以这里用【块内均值】代表交付,并把 C 恒等地记为 0 —— 显式记下来,不假装测过。
        cq = [r["wet"] for r in conf]; bq = [r["wet"] for r in blk]
        units.append({"A": st.mean(cq) - st.mean(pool), "B": st.mean(bq) - st.mean(cq), "C": 0.0,
                      "identity": st.mean(bq) - st.mean(pool),
                      "puzzle": cell[0], "n_block": len(blk), "n_conf": len(conf), "n_pool": len(pool),
                      "head": headroom({"confirmed": cq, "block": bq}, pool, a.hdraws, rng)})
    print(f"  参与检验的格 {len(units)};跳过 {dict(skip)}")
    print(f"  层大小中位:顶端块 {st.median([u['n_block'] for u in units]):.0f}  "
          f"confirmed {st.median([u['n_conf'] for u in units]):.0f}  "
          f"格内全部 {st.median([u['n_pool'] for u in units]):.0f}")
    print(f"  【C 恒等为 0 且不是一次测量】:熵不在这些文件里;打平器的效应由 A12 单独测过(null)。")
    return units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--hdraws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "gradient_location.json"))
    a = ap.parse_args()
    D = json.load(open(os.path.join(HERE, "gradient_location_declaration.json")))
    res = {"declaration": D, "populations": {}}

    print("=== 人群1 真实池(57 池),质量 = RibonanzaNet 交叉对 F1 ===")
    u1 = pop1(a)
    res["populations"]["pop1_real_pools_rnet"] = report("pop1", u1, a.draws, a.seed)

    for nm, fn in (("样本2 (160/靶标)", "panel_f1_fresh.json"), ("样本3 (80/靶标)", "panel_f1_third.json")):
        print(f"\n=== 人群2 {nm},质量 = 实测 OpenKnot 试管分 ===")
        u = pop2(a, fn)
        row = report(f"pop2/{nm}", u, a.draws, a.seed)
        print(f"  --- 更保守:按【靶标】聚类(格之间共用靶标)---")
        byp = collections.defaultdict(list)
        for x in u:
            byp[x["puzzle"]].append(x)
        row["by_puzzle"] = {k: line(f"{k} 按靶标", [st.median([y[k] for y in v]) for v in byp.values()],
                                    a.draws, a.seed + 20) for k in ("A", "B")}
        res["populations"][f"pop2_{'fresh' if 'fresh' in fn else 'third'}"] = row

    print(f"\n=== 判定(按声明)===")
    keys = ["pop1_real_pools_rnet", "pop2_fresh", "pop2_third"]
    As = [res["populations"][k]["A"] for k in keys]
    Bs = [res["populations"][k]["B"] for k in keys]
    if any(x.get("no_data") for x in As + Bs):
        v = "no_data"
    elif not all(x.get("excludes_zero_positive") for x in As):
        v = "null"
    elif all(x.get("excludes_zero_positive") for x in Bs):
        v = "both"
    elif not any(x.get("excludes_zero") for x in Bs):
        v = "gradient_at_the_low_end"
    else:
        v = "gradient_at_the_top"
    for k in keys:
        A, B = res["populations"][k]["A"], res["populations"][k]["B"]
        if not A.get("no_data"):
            print(f"  {k:26s} A {A['median']:+.4f} [{A['ci'][0]:+.4f},{A['ci'][1]:+.4f}]"
                  f"{'排除0' if A['excludes_zero'] else '跨0'}   "
                  f"B {B['median']:+.4f} [{B['ci'][0]:+.4f},{B['ci'][1]:+.4f}]"
                  f"{'排除0' if B['excludes_zero'] else '跨0'}")
    print(f"  【结论】{v} —— {D['outcomes_fixed_in_advance'][v]}")
    res["verdict"] = v
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
