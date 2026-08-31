#!/usr/bin/env python3
"""M46: is the sign inversion a property of the budget, or of a few targets?

Declaration `m46_inversion_declaration.json`, frozen and pushed at 9b0febac BEFORE this file
existed. Offline, $0.

Imports m44v_verify, NOT m44_decision: m44v already carries an independent reimplementation of
recovery that reproduced M44's published numbers to its rounding, so reusing it keeps this run one
step away from the code whose claim is under test.
"""
from __future__ import annotations
import collections, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m44v_verify as V44

DECL = json.load(open(os.path.join(HERE, "m46_inversion_declaration.json")))
PUB = json.load(open(os.path.join(HERE, "m44_decision.json")))
BUDGETS = V44.BUDGETS
IMPL = V44.IMPL
OUTJ = os.path.join(HERE, "m46_inversion.json")
MIN_TARGETS = 10
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def gap(cells, a, b, sgn, frac, thresh=None):
    """recovery(a) - recovery(b) at one budget. `thresh` overrides the rescue threshold."""
    old = V44.THRESH
    if thresh is not None:
        V44.THRESH = thresh
    try:
        ra = V44.recover(cells, a, sgn, frac)
        rb = V44.recover(cells, b, sgn, frac)
    finally:
        V44.THRESH = old
    return None if (ra is None or rb is None) else ra - rb


def n_scorable(cells, frac, thresh=None):
    old = V44.THRESH
    if thresh is not None:
        V44.THRESH = thresh
    try:
        n = sum(1 for c in cells.values() if V44.informative(c))
    finally:
        V44.THRESH = old
    return n


def main():
    t0 = time.time()
    cells = V44.cells_from_csv()
    sgn = V44.depth_sign(cells)
    by_t = collections.defaultdict(dict)
    for k, c in cells.items():
        by_t[c["puzzle"]][k] = c
    targets = sorted(by_t)
    out = {"declaration": "m46_inversion_declaration.json", "cost": "offline, $0",
           "wording_rules": DECL["wording_rules"], "not_a_test": DECL["not_a_test"],
           "n_cells": len(cells), "n_targets": len(targets), "depth_sign": sgn}
    log(f"格 {len(cells)}  靶标 {len(targets)}  深度方向 {sgn:+.0f}")

    # ---------------------------------------------------------------- V reverse control
    log("\n=== V 反向对照:必须复现 M44 发表的三个 gap")
    rev, worst = {}, 0.0
    for f in BUDGETS:
        mine = gap(cells, "RibonanzaNet", "ViennaRNA", sgn, f)
        pub = (PUB["E1"][str(f)]["RibonanzaNet"]["recovery"]
               - PUB["E1"][str(f)]["ViennaRNA"]["recovery"])
        rev[str(f)] = {"ours": round(mine, 6), "published": round(pub, 6),
                       "abs_diff": round(abs(mine - pub), 7)}
        worst = max(worst, abs(mine - pub))
        log(f"  {int(f*100):3d}%  本次 {mine:+.5f}  M44 {pub:+.5f}  差 {abs(mine-pub):.2e}")
    out["V_reverse_control"] = {"cells": rev, "max_abs_diff": round(worst, 7),
                                "pass": bool(worst < 1e-4)}
    if worst >= 1e-4:
        log("  ** 复现不了 ⇒ 按 kill 作废 **")
        out["void"] = "reverse_control"
        json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 1
    log("  -> 通过")

    # ---------------------------------------------------------------- E1 leave-one-target-out
    log("\n=== E1 留一靶标:逐个删掉一个靶标,重算 gap")
    loo = {}
    for f in BUDGETS:
        vals = {}
        for t in targets:
            sub = {k: c for k, c in cells.items() if c["puzzle"] != t}
            g = gap(sub, "RibonanzaNet", "ViennaRNA", sgn, f)
            if g is not None:
                vals[t] = round(g, 5)
        full = gap(cells, "RibonanzaNet", "ViennaRNA", sgn, f)
        arr = np.array(list(vals.values()), float)
        loo[str(f)] = {"full": round(full, 5), "n_drops": len(vals),
                       "min": round(float(arr.min()), 5), "max": round(float(arr.max()), 5),
                       "sign_stable": bool(np.all(np.sign(arr) == np.sign(full))),
                       "worst_target": min(vals, key=lambda t: abs(vals[t])),
                       "by_target": vals}
        log(f"  {int(f*100):3d}%  全量 {full:+.5f}   留一范围 [{arr.min():+.5f}, {arr.max():+.5f}]"
            f"   符号全部一致:{'是' if loo[str(f)]['sign_stable'] else '**否**'}")
    out["E1_leave_one_target_out"] = loo

    # ---------------------------------------------------------------- E2 per target
    log("\n=== E2 逐靶标:单个靶标上 10% 预算的 gap")
    per = {}
    fmin = min(BUDGETS)
    for t in targets:
        sub = by_t[t]
        if n_scorable(sub, fmin) < 2:
            continue
        g = gap(sub, "RibonanzaNet", "ViennaRNA", sgn, fmin)
        if g is not None:
            per[t] = round(g, 5)
    neg = sum(1 for v in per.values() if v < 0)
    zer = sum(1 for v in per.values() if v == 0)
    out["E2_per_target"] = {"budget": fmin, "n_targets": len(per), "n_negative": neg,
                            "n_zero": zer, "n_positive": len(per) - neg - zer, "values": per}
    log(f"  可算的靶标 {len(per)}:反转(负) {neg},持平 {zer},同向(正) {len(per)-neg-zer}")

    # ---------------------------------------------------------------- E3 budget sweep
    log("\n=== E3 预算扫描:gap 作为预算的连续函数")
    sweep, xs = [], [round(0.05 + 0.025 * i, 3) for i in range(19)]     # 0.05 .. 0.50
    for f in xs:
        ns = n_scorable(cells, f)
        if ns < MIN_TARGETS:
            continue
        g = gap(cells, "RibonanzaNet", "ViennaRNA", sgn, f)
        sweep.append({"budget": f, "gap": round(g, 5), "n_scorable_cells": ns})
    sig = [np.sign(p["gap"]) for p in sweep]
    cross = [i for i in range(1, len(sig)) if sig[i] != sig[i - 1] and sig[i] != 0]
    out["E3_budget_sweep"] = {
        "points": sweep, "n_points": len(sweep), "n_sign_changes": len(cross),
        "crossing_between": ([sweep[cross[0] - 1]["budget"], sweep[cross[0]]["budget"]]
                             if cross else None),
        "W1": "a crossing read off this curve describes this corpus; it carries no interval"}
    log("  " + "  ".join(f"{p['budget']:.3f}:{p['gap']:+.4f}" for p in sweep[:10]))
    log("  " + "  ".join(f"{p['budget']:.3f}:{p['gap']:+.4f}" for p in sweep[10:]))
    log(f"  符号变化次数 {len(cross)}"
        + (f",发生在预算 {out['E3_budget_sweep']['crossing_between']} 之间" if cross else ""))

    # ---------------------------------------------------------------- E4 threshold sweep
    log("\n=== E4 rescue 阈值扫描:反转是否只在一个阈值上成立")
    thr = {}
    for th in (0.3, 0.4, V44.THRESH, 0.6, 0.7):
        ns = n_scorable(cells, fmin, th)
        if ns < MIN_TARGETS:
            thr[str(th)] = {"skipped": "too few scorable cells", "n": ns}
            continue
        g10 = gap(cells, "RibonanzaNet", "ViennaRNA", sgn, fmin, th)
        g50 = gap(cells, "RibonanzaNet", "ViennaRNA", sgn, max(BUDGETS), th)
        thr[str(th)] = {"n_scorable": ns, "gap_low_budget": round(g10, 5),
                        "gap_high_budget": round(g50, 5),
                        "inverts": bool(g10 < 0 < g50)}
        log(f"  阈值 {th}:可计分 {ns:3d}  低预算 {g10:+.5f}  高预算 {g50:+.5f}"
            f"  {'反转' if thr[str(th)]['inverts'] else '不反转'}")
    out["E4_threshold_sweep"] = thr

    # ---------------------------------------------------------------- E5 other pairs
    log("\n=== E5 其它序对:反转是不是这一对独有的")
    pairs = {}
    for a in IMPL:
        for b in IMPL:
            if a >= b:
                continue
            g10 = gap(cells, a, b, sgn, fmin)
            g50 = gap(cells, a, b, sgn, max(BUDGETS))
            pairs[f"{a}|{b}"] = {"low": round(g10, 5), "high": round(g50, 5),
                                 "inverts": bool(g10 * g50 < 0)}
            log(f"  {a:14s} - {b:14s}  低 {g10:+.5f}  高 {g50:+.5f}"
                f"  {'**反转**' if pairs[f'{a}|{b}']['inverts'] else ''}")
    out["E5_other_pairs"] = {"pairs": pairs,
                             "n_inverting": sum(1 for v in pairs.values() if v["inverts"]),
                             "n_pairs": len(pairs)}

    # ---------------------------------------------------------------- staked
    st = {
        "1_sign_stable_under_leave_one_out": bool(loo[str(fmin)]["sign_stable"]),
        "2_exactly_one_crossing": bool(out["E3_budget_sweep"]["n_sign_changes"] == 1),
        "3_another_pair_inverts": bool(out["E5_other_pairs"]["n_inverting"] >= 2),
        "4_survives_another_threshold": bool(
            sum(1 for k, v in thr.items() if v.get("inverts")) >= 2)}
    out["staked"] = st; out["staked_text"] = DECL["staked"]
    log("\n=== 预注册记分")
    for k, v in st.items():
        log(f"  {k:38s} {'成立' if v else '**失败**'}")
    if not st["1_sign_stable_under_leave_one_out"]:
        log("  ** W2 触发:反转由靶标扛着 ⇒ 摘要与结论中基于它的主张必须在同一次编辑里撤回 **")
        out["W2_triggered"] = True

    out["elapsed_s"] = round(time.time() - t0, 1); out["log"] = LOG
    json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {OUTJ}  用时 {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
