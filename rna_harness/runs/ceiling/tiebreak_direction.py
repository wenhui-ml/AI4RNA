#!/usr/bin/env python3
"""A12: the shipped tie-breaker's DIRECTION, on the population the shipped rule actually fires in.

Declared in `tiebreak_direction_declaration.json`, committed and pushed before one entropy was computed.

WHAT IS BEING QUESTIONED. certify.py, among candidates of EQUAL grade k (k = how many panel oracles reach
full crossing recall), prefers the one with HIGHER ensemble entropy. Every number that justifies that rule
came from IPknot/Knotty/RNAPKplex -- judges that share an energy-model family with the panel itself. A11 put
the same quantity against MEASUREMENT for the first time and got four negative point estimates, one of them
significant. But A11's populations were RNet's top tie block and the panel-SCORE top tie block; neither is the
equal-grade block. This run uses the shipped rule's own population, so that a product decision has ground.

THE PRIMARY IS THE DECISION, NOT THE CORRELATION. What the product does is pick the block's argmax entropy;
what it would otherwise do is take an arbitrary member. So the primary statistic is, per cell,
wet(argmax entropy) - mean(wet over the block) -- the second term IS the expected value of a random tie-break.
The rank correlation A11 used is kept as a secondary so the two runs stay comparable.

TWO-SIDED ON PURPOSE. The declared direction is positive, but the interval is two-sided: a significant
NEGATIVE result is the finding this run exists to be able to state.

A NULL DOES NOT LICENSE DELETING THE RULE. The rule's original outcome variable was outside-judge agreement,
not tube score. A null here bounds the rule's claim; it does not refute the evidence for it.

COST: $0, offline. Partition functions inside the tie blocks only.
"""
from __future__ import annotations

import argparse, collections, json, os, statistics as st, sys, random
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from arbitration_confirm import load_fresh                                      # noqa: E402
from top_signal import spearman, ent_worker, MIN_BLOCK, MIN_DISTINCT_WET        # noqa: E402

SAMPLES = (("样本2 (160/靶标)", "panel_f1_fresh.json"),
           ("样本3 (80/靶标)",  "panel_f1_third.json"))


def grade(r):
    """k = how many panel oracles reach FULL crossing recall -- certify.py's own confirm count."""
    po = r.get("per_oracle") or {}
    v = [x for x in po.values() if x]
    if not v:
        return None
    return sum(1 for x in v if (x.get("recall_product") or 0) >= 1.0)


def equal_grade_blocks(rec):
    """Per cell: among knot_confirmed rows, the maximal-grade block. certify.py's best_knot tie set."""
    g = collections.defaultdict(list)
    for r in rec:
        g[(r["puzzle"], r["method"])].append(r)
    out = {}
    for cell, rows in g.items():
        c = [r for r in rows if (grade(r) or 0) >= 1]
        if not c:
            continue
        top = max(grade(r) for r in c)
        blk = [r for r in c if grade(r) == top]
        if len(blk) >= MIN_BLOCK and len({round(r["wet"], 6) for r in blk}) >= MIN_DISTINCT_WET:
            out[cell] = blk
    return out


def boot(vals, draws, rng, stat=st.median):
    v = list(vals)
    if len(v) < 5:
        return None, None, None
    n = len(v)
    o = sorted(stat([v[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return stat(v), o[int(0.025 * draws)], o[min(draws - 1, int(0.975 * draws))]


def line(label, vals, draws, seed, floor):
    rng = random.Random(seed)
    if len(vals) < floor:
        print(f"  {label:34s} 可用格 {len(vals):4d} —— 低于声明的下限 {floor},无数据、无结论")
        return {"n": len(vals), "no_data": True}
    pt, lo, hi = boot(vals, draws, rng)
    mn = st.mean(vals)
    pos = sum(1 for x in vals if x > 0); neg = sum(1 for x in vals if x < 0)
    ex = (lo is not None and (lo > 0 or hi < 0))
    tag = "  区间排除 0" if ex else "  跨过 0"
    sign = "正" if (pt or 0) > 0 else ("负" if (pt or 0) < 0 else "零")
    print(f"  {label:34s} 格 {len(vals):4d}  中位 {pt:+.4f}  均值 {mn:+.4f}  "
          f"正/负 {pos:4d}/{neg:4d}  CI [{lo:+.4f}, {hi:+.4f}]{tag}({sign})")
    return {"n": len(vals), "median": pt, "mean": mn, "n_pos": pos, "n_neg": neg,
            "ci": [lo, hi], "excludes_zero": bool(ex),
            "excludes_zero_positive": bool(ex and lo > 0),
            "excludes_zero_negative": bool(ex and hi < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "tiebreak_direction.json"))
    a = ap.parse_args()
    D = json.load(open(os.path.join(HERE, "tiebreak_direction_declaration.json")))
    FLOOR = 20
    res = {"declaration": D, "no_data_floor_cells": FLOOR, "samples": {}}
    pooled = {"primary": [], "spearman": [], "maxmin": []}

    for nm, fn in SAMPLES:
        rec, _sp, _meta = load_fresh(os.path.join(HERE, fn), "panel_recall_product")
        pop = equal_grade_blocks(rec)
        need = sorted({r["seq"] for blk in pop.values() for r in blk})
        print(f"\n=== {nm}:去重后分子 {len(rec)};同等级并列块 {len(pop)} 格,"
              f"块内唯一序列 {len(need)} 条 ===", flush=True)
        print("  (声明里的格数是【去重前】按原始 designs 数的;load_fresh 会按 (靶标,序列) 合并同一分子、"
              "并剔除 RNet-optimised 那一档,所以这里的格数略有不同。声明的无数据下限是 "
              f"{FLOOR} 格,两批都远在其上。)")
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            E = dict(zip(need, ex.map(ent_worker, need, chunksize=8)))
        bad = [s for s, v in E.items() if v is None or v == float("-inf")]
        print(f"  熵算完;失败 {len(bad)} 条(失败的序列整块不用)")

        prim, spea, mxmn, kdist = [], [], [], collections.Counter()
        for cell, blk in pop.items():
            v = [(E[r["seq"]], r["wet"]) for r in blk if E.get(r["seq"]) not in (None, float("-inf"))]
            if len(v) < MIN_BLOCK or len({round(y, 6) for _e, y in v}) < MIN_DISTINCT_WET:
                continue
            kdist[grade(blk[0])] += 1
            hi = max(v, key=lambda t: t[0])[1]
            lo = min(v, key=lambda t: t[0])[1]
            prim.append(hi - st.mean([y for _e, y in v]))
            mxmn.append(hi - lo)
            s = spearman([e for e, _y in v], [y for _e, y in v])
            if s is not None:
                spea.append(s)
        print(f"  参与检验的格 {len(prim)};顶端 k 分布 {dict(sorted(kdist.items()))}")

        row = {"n_designs_dedup": len(rec), "n_blocks": len(pop), "n_unique_seqs": len(need),
               "n_entropy_failed": len(bad), "k_distribution": dict(sorted(kdist.items()))}
        print("  --- 主检验(声明方向为正)---")
        row["primary_argmax_minus_random"] = line("wet(最高熵) − wet(随机打平)", prim, a.draws, a.seed, FLOOR)
        print("  --- 次检验 ---")
        row["spearman_in_block"] = line("块内 Spearman(熵, wet)", spea, a.draws, a.seed + 1, FLOOR)
        row["argmax_minus_argmin"] = line("wet(最高熵) − wet(最低熵)", mxmn, a.draws, a.seed + 2, FLOOR)
        res["samples"][nm] = row
        pooled["primary"] += prim; pooled["spearman"] += spea; pooled["maxmin"] += mxmn

    print(f"\n=== 合并两批(仅参考;两批各自的结论优先)===")
    res["pooled"] = {
        "primary_argmax_minus_random": line("wet(最高熵) − wet(随机打平)", pooled["primary"], a.draws, a.seed, FLOOR),
        "spearman_in_block": line("块内 Spearman(熵, wet)", pooled["spearman"], a.draws, a.seed + 1, FLOOR),
        "argmax_minus_argmin": line("wet(最高熵) − wet(最低熵)", pooled["maxmin"], a.draws, a.seed + 2, FLOOR)}

    print(f"\n=== 判定(按声明:主检验,【两批都要】,双侧)===")
    ms = [res["samples"][nm]["primary_argmax_minus_random"] for nm, _ in SAMPLES]
    if any(m.get("no_data") for m in ms):
        verdict = "no_data"
    elif all(m.get("excludes_zero_positive") for m in ms):
        verdict = "direction_confirmed"
    elif all(m.get("excludes_zero_negative") for m in ms):
        verdict = "direction_inverted"
    else:
        verdict = "null"
    for nm, _ in SAMPLES:
        m = res["samples"][nm]["primary_argmax_minus_random"]
        if not m.get("no_data"):
            print(f"  {nm}:中位 {m['median']:+.4f}  CI [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}]  "
                  f"{'排除 0' if m['excludes_zero'] else '跨过 0'}")
    print(f"  【结论】{verdict} —— {D['outcomes_fixed_in_advance'][verdict]}")
    print(f"  多重比较:{D['multiplicity_stated_not_corrected']}")
    res["verdict"] = verdict
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
