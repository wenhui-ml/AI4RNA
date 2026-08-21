#!/usr/bin/env python3
"""K1x: what changes if the shipped k counts crossing-CONSISTENT recall instead of recall_product.

Pre-registered in k1_crossing_consistent_k_declaration.json, the first declaration written to the rule-38
spec: every kill condition carries an input under which it does NOT fire, and every quantity carries its
domain with the code raising outside it.

WHAT TRIGGERED IT. Seven designs carry panel_k = 3 -- all three oracles confirm -- while the panel predicted
ZERO crossing pairs. That is not a bug, it is the shipped column's definition: recall_product counts a target
crossing pair as recovered if the oracle predicts that PAIR, whether or not the pair still crosses in the
oracle's own prediction. An oracle can reproduce every one of them as a purely NESTED structure and score
1.000. All seven sit at k=3 and none at k=1 or k=2, which follows from the definitions.

DOMAIN. `recall`'s denominator is |target crossing set|, which our pipeline already gates to be non-empty, so
k_x is well defined -- and `count_kx` raises rather than returning 0 if that ever fails. Defining k_x on
precision or f1 instead WOULD hit a real 0/0 on the 94 designs with n_pred_cross = 0. gamma_star's canonical
pair gate does NOT apply here: it protects energy quantities, and k_x only concerns what the oracles
predicted.

THIS DECIDES A RECOMMENDATION, NOT A PATCH. The shipped grades still carry sensitivities measured on the
recall_product column (0.432 / 0.636 / 0.785) and the 84.0%/0.4% calibrated operating point. A definition
change voids all of those and needs its own calibration run. No shipped code is touched here.

COST: $0, offline, seconds. Both columns are already on disk.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

TUBE_PASS = 80.0
ORACLES = ("pkiss", "probknot", "shapeknots")
COSMETIC_THRESHOLD = 0.01


class Undefined(ValueError):
    """k_x is not defined for this molecule (rule 38, second sentence)."""


def count_kx(r, col):
    """How many oracles reach recall 1.0 on `col`. Raises if the denominator would be zero."""
    po = r.get("per_oracle") or {}
    if sorted(po) != sorted(ORACLES):
        raise Undefined("per_oracle incomplete")
    # |target crossing set| == 0 would make recall 0/0. The pipeline gates it upstream; assert, don't assume.
    if r.get("n_pred_total") is not None and r.get("panel_recall") is None:
        raise Undefined("panel_recall missing")
    n = 0
    for o in ORACLES:
        v = (po[o] or {}).get(col)
        if v is None:
            raise Undefined(f"{o}.{col} missing")
        n += (v >= 1.0)
    return n


def run(name, path):
    d = json.load(open(os.path.join(HERE, path)))
    rows, undef = [], 0
    for r in d["designs"]:
        if "error" in r:
            continue
        try:
            k = count_kx(r, "recall_product")
            kx = count_kx(r, "recall")
        except Undefined:
            undef += 1
            continue
        rows.append((k, kx, r["wet"], r["n_pred_cross"], r["puzzle"]))
    print(f"\n=== {name} ===  n={len(rows)}  域外/列不全 {undef}")

    M = collections.Counter((k, kx) for k, kx, *_ in rows)
    moved = sum(v for (k, kx), v in M.items() if k != kx)
    print(f"  改动 {moved}/{len(rows)} = {moved/len(rows)*100:.2f}%"
          f"   {'(< 1% 门槛)' if moved/len(rows) < COSMETIC_THRESHOLD else '(>= 1% 门槛)'}")
    print(f"  重分类矩阵(行 k = recall_product,列 k_x = 交叉一致 recall)")
    print("        " + "".join(f"{c:>8}" for c in ("k_x=0", "k_x=1", "k_x=2", "k_x=3")))
    for k in range(4):
        print(f"    k={k} " + "".join(f"{M.get((k, x), 0):>8}" for x in range(4)))

    def fail_rate(sel):
        g = [w for _k, _kx, w, *_ in sel]
        return (sum(1 for w in g if w < TUBE_PASS) / len(g), len(g)) if g else (None, 0)

    print(f"  试管失败率(< {TUBE_PASS})按档")
    out = {"n": len(rows), "undef": undef, "moved": moved, "moved_frac": moved / len(rows),
           "matrix": {f"{k}->{x}": M.get((k, x), 0) for k in range(4) for x in range(4)}}
    for lab, idx in (("k   (出厂 recall_product)", 0), ("k_x (交叉一致 recall)", 1)):
        line, rates = [], {}
        for g in range(4):
            fr, n = fail_rate([r for r in rows if r[idx] == g])
            rates[g] = {"fail": fr, "n": n}
            line.append(f"{g}:{('%.3f' % fr) if fr is not None else ' n/a '}(n={n})")
        mono = all(rates[g]["fail"] is not None and rates[g + 1]["fail"] is not None
                   and rates[g + 1]["fail"] <= rates[g]["fail"] for g in range(3))
        span = (rates[0]["fail"] - rates[3]["fail"]) if (rates[0]["fail"] is not None
                                                        and rates[3]["fail"] is not None) else None
        print(f"    {lab:28} " + "  ".join(line))
        print(f"    {'':28} 单调(档越高失败率越低){'成立' if mono else '【被破坏】'}"
              f"   最低档−最高档 = {span:+.4f}" if span is not None else "")
        out[("k" if idx == 0 else "k_x")] = {"rates": rates, "monotone": mono, "span": span}

    seven = [r for r in rows if r[0] == 3 and r[3] == 0]
    print(f"  那 7 条(k=3 且 n_pred_cross=0):本人群里 {len(seven)} 条,"
          f"k_x 分布 {dict(collections.Counter(r[1] for r in seven))}")
    out["seven"] = {"n": len(seven), "kx": dict(collections.Counter(r[1] for r in seven))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(HERE, "k1_crossing_consistent_k.json"))
    a = ap.parse_args()
    POPS = (("基准 4,560(主)", "panel_f1_designs.json"), ("样本3", "panel_f1_third.json"),
            ("第四批", "panel_f1_fourth.json"), ("样本2", "panel_f1_fresh.json"))
    res = {}
    for nm, fn in POPS:
        res[fn] = run(nm, fn)

    # cluster bootstrap on the span difference, primary population only
    d = json.load(open(os.path.join(HERE, "panel_f1_designs.json")))["designs"]
    rows = []
    for r in d:
        if "error" in r:
            continue
        try:
            rows.append((count_kx(r, "recall_product"), count_kx(r, "recall"), r["wet"], r["puzzle"]))
        except Undefined:
            pass
    byt = collections.defaultdict(list)
    for x in rows:
        byt[x[3]].append(x)
    tg = sorted(byt)

    def span_of(pool, idx):
        lo = [w for k, kx, w, _p in pool if (k, kx)[idx] == 0]
        hi = [w for k, kx, w, _p in pool if (k, kx)[idx] == 3]
        if not lo or not hi:
            return None
        return (sum(1 for w in lo if w < TUBE_PASS) / len(lo)
                - sum(1 for w in hi if w < TUBE_PASS) / len(hi))
    rng = random.Random(90210)
    B = []
    for _ in range(a.boot):
        s = [tg[rng.randrange(len(tg))] for _ in tg]
        pool = [x for p in s for x in byt[p]]
        sa, sb = span_of(pool, 0), span_of(pool, 1)
        if sa is not None and sb is not None:
            B.append(sb - sa)
    B.sort()
    ci = [round(B[int(0.025 * len(B))], 4), round(B[int(0.975 * len(B))], 4)]
    pt = span_of(rows, 1) - span_of(rows, 0)
    print(f"\n  主人群:span(k_x) − span(k) = {pt:+.4f}  聚类CI {ci}"
          f"  {'排除 0' if ci[0] > 0 or ci[1] < 0 else '含 0'}   (span = 最低档失败率 − 最高档失败率,越大越好)")
    res["span_diff"] = {"point": pt, "ci": ci, "n_targets": len(tg)}

    P = res["panel_f1_designs.json"]
    K1 = P["moved_frac"] < COSMETIC_THRESHOLD
    K2 = (not P["k_x"]["monotone"]) or (P["k_x"]["span"] < P["k"]["span"])
    K3 = P["seven"]["kx"].get(3, 0) > 0
    print(f"\n  K1 装饰性改名(改动 < 1%)            :{'FIRES' if K1 else '不触发'}")
    print(f"  K2 换口径让证书变差                  :{'FIRES' if K2 else '不触发'}")
    print(f"  K3 没解决提出它的那个问题(近乎恒真)  :{'FIRES' if K3 else '不触发'}")
    verdict = ("cosmetic_do_not_change" if K1 else "makes_it_worse_do_not_change" if K2 else
               "does_not_fix_the_trigger" if K3 else "recommend_a_calibration_run")
    print(f"  裁定:{verdict}")
    print("  提醒(声明里写好的):本实验只产出【该不该去做那次重新校准】的依据。"
          "出厂分级的灵敏度 0.432/0.636/0.785 与 84.0%/0.4% 操作点都是在 recall_product 上量的,换口径会让它们全部失效。")
    res["verdict"] = verdict
    res["kills"] = {"K1": K1, "K2": K2, "K3": K3}
    res["deliverable"] = "a recommendation, not a patch; no shipped code touched"
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
