#!/usr/bin/env python3
"""M47: formal equivalence, replacing "the interval covers zero".

Declaration `m47_equivalence_declaration.json`, frozen and pushed at 36492016 BEFORE this file
existed. Offline, $0. Imports m44v_verify, which carries the reimplementation of recovery that
reproduced M44 to its rounding, so this stays one step from the code whose claim is under test.

W1: the word equivalent never appears without its margin. W4: TOST bounds a difference; it never
establishes identity.
"""
from __future__ import annotations
import collections, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m44v_verify as V44

DECL = json.load(open(os.path.join(HERE, "m47_equivalence_declaration.json")))
PUB = json.load(open(os.path.join(HERE, "m44_decision.json")))
BUDGETS, IMPL = V44.BUDGETS, V44.IMPL
NBOOT = 600
OUTJ = os.path.join(HERE, "m47_equivalence.json")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def boot_all(arms, sgn, frac, idx, pz, bypz):
    """Paired target-clustered bootstrap of (arm - depth) for EVERY arm at once.

    The depth baseline is recomputed once per resample rather than once per arm-resample. The
    pairing is what makes these differences comparable, and computing depth inside each arm's loop
    was both nine times the work and nine opportunities to pair against a different resample.
    """
    out = {a: [] for a in arms}
    for ix in idx:
        sub = {}
        for j, q in enumerate(ix):
            for k, c in bypz[pz[q]].items():
                sub[(j, k)] = c
        d0 = V44.recover(sub, "depth", sgn, frac)
        if d0 is None:
            continue
        for a in arms:
            v = V44.recover(sub, a, sgn, frac)
            if v is not None:
                out[a].append(v - d0)
    return {a: np.asarray(v, float) for a, v in out.items()}


def delta_star(v):
    """The smallest margin at which two one-sided tests at alpha=0.05 declare equivalence. TOST at
    alpha is the 100(1-2alpha)% interval lying inside +-delta, so delta_star is that interval's
    larger absolute endpoint."""
    lo, hi = float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))
    return max(abs(lo), abs(hi)), lo, hi


def main():
    t0 = time.time()
    cells = V44.cells_from_csv()
    sgn = V44.depth_sign(cells)
    scored = {k: c for k, c in cells.items() if V44.informative(c)}
    out = {"declaration": "m47_equivalence_declaration.json", "cost": "offline, $0",
           "wording_rules": DECL["wording_rules"], "not_a_test": DECL["not_a_test"],
           "n_cells_scored": len(scored)}
    log(f"可计分格 {len(scored)}  深度方向 {sgn:+.0f}")

    # ---------------------------------------------------------------- V: reproduce M44
    log("\n=== V 反向对照:先复现 M44 的十二个点估计")
    worst = 0.0; rev = {}
    for f in BUDGETS:
        for n in IMPL:
            mine = V44.recover(cells, n, sgn, f) - V44.recover(cells, "depth", sgn, f)
            pub = PUB["E1"][str(f)][n]["minus_depth"]
            rev[f"{f}|{n}"] = round(abs(mine - pub), 7); worst = max(worst, abs(mine - pub))
    log(f"  十二个量,最大差 {worst:.2e}  -> {'通过' if worst < 1e-4 else '**未通过**'}")
    out["V_reverse_control"] = {"max_abs_diff": round(worst, 7), "pass": bool(worst < 1e-4),
                               "cells": rev}
    if worst >= 1e-4:
        out["void"] = "reverse_control"
        json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 1

    # ---------------------------------------------------------------- the margin, from the rule
    # depth alone and the corpus geometry. Nothing here touches a model.
    log("\n=== 冻结规则算出 delta_tube(只用免费基线和语料几何,不碰任何模型)")
    lo_b, hi_b = min(BUDGETS), max(BUDGETS)
    r_lo = V44.recover(cells, "depth", sgn, lo_b)
    r_hi = V44.recover(cells, "depth", sgn, hi_b)
    slope = (r_hi - r_lo) / (hi_b - lo_b)
    sizes = [len(c["y"]) for c in scored.values()]
    n_med = float(np.median(sizes))
    delta = slope * (1.0 / n_med)
    out["delta_tube"] = {
        "depth_recovery_at_low_budget": round(r_lo, 5), "at_high_budget": round(r_hi, 5),
        "slope_per_unit_budget": round(slope, 5), "median_cell_size": n_med,
        "one_construct_is_this_much_budget": round(1.0 / n_med, 5), "delta": round(delta, 5)}
    log(f"  深度捞回 {r_lo:.5f}@{lo_b} → {r_hi:.5f}@{hi_b}   斜率 {slope:.5f}/单位预算")
    log(f"  格大小中位 {n_med:.0f} ⇒ 一条构建体 = {1/n_med:.5f} 的预算")
    log(f"  **delta_tube = {delta:.5f}**  (一条额外构建体买到的捞回)")
    if not (0.005 <= delta <= 0.20):
        log("  ** delta_tube 落在 [0.005, 0.20] 之外 ⇒ 按 kill 作废该臂 **")
        out["delta_tube"]["void"] = "outside_acceptable_range"

    # ---------------------------------------------------------------- bootstrap machinery
    pz = sorted({c["puzzle"] for c in cells.values()})
    bypz = collections.defaultdict(dict)
    for k, c in cells.items():
        bypz[c["puzzle"]][k] = c
    rb = np.random.default_rng(V44.SEED)
    idx = [rb.integers(0, len(pz), len(pz)) for _ in range(NBOOT)]

    # ---------------------------------------------------------------- E1 / E3 / E4
    ARMS = list(IMPL) + [f"depth_then_{n}" for n in IMPL] + ["depth"]
    log("\n=== E1/E3/E4  delta_star = TOST 在 alpha=0.05 下宣告等价所需的最小 margin")
    res = {}
    for f in BUDGETS:
        log(f"\n  预算 {int(f*100)}%   (delta_tube = {delta:.5f})")
        BOOT = boot_all(ARMS, sgn, f, idx, pz, bypz)
        for a in ARMS:
            v = BOOT[a]
            if len(v) < 20:
                continue
            ds, lo, hi = delta_star(v)
            pt = V44.recover(cells, a, sgn, f) - V44.recover(cells, "depth", sgn, f)
            eq = bool(ds < delta)
            res[f"{f}|{a}"] = {"point": round(pt, 5), "ci90": [round(lo, 5), round(hi, 5)],
                               "delta_star": round(ds, 5), "equivalent_at_delta_tube": eq}
            tag = ("等价" if eq else "**不等价**") if a != "depth" else "(自比,零点)"
            log(f"    {a:24s} 差 {pt:+.5f}  90%CI [{lo:+.5f},{hi:+.5f}]"
                f"  delta*={ds:.5f}  {tag}")
    out["E1_E3_E4"] = res

    # ---------------------------------------------------------------- staked
    impl_eq = [res[f"{f}|{n}"]["equivalent_at_delta_tube"] for f in BUDGETS for n in IMPL]
    stack_eq = [res[f"{f}|depth_then_{n}"]["equivalent_at_delta_tube"] for f in BUDGETS
                for n in IMPL]
    self_ds = max(res[f"{f}|depth"]["delta_star"] for f in BUDGETS)
    st = {"1_all_implementations_equivalent_at_delta_tube": bool(all(impl_eq)),
          "2_depth_against_itself_below_0001": bool(self_ds < 0.001),
          "3_some_stacked_arm_not_equivalent": bool(not all(stack_eq)),
          "4_delta_tube_in_range": bool(0.01 <= delta <= 0.10)}
    out["staked"] = st; out["staked_text"] = DECL["staked"]
    out["summary"] = {
        "n_implementation_comparisons": len(impl_eq),
        "n_equivalent_at_delta_tube": sum(impl_eq),
        "largest_implementation_delta_star": round(
            max(res[f"{f}|{n}"]["delta_star"] for f in BUDGETS for n in IMPL), 5),
        "depth_self_delta_star": round(self_ds, 6),
        "n_stacked_not_equivalent": sum(1 for x in stack_eq if not x)}
    log("\n=== 预注册记分")
    for k, v in st.items():
        log(f"  {k:48s} {'成立' if v else '**失败**'}")
    if not st["1_all_implementations_equivalent_at_delta_tube"]:
        log("  ** W2 触发:必须逐臂改写,不许整体软化 **"); out["W2_triggered"] = True

    out["elapsed_s"] = round(time.time() - t0, 1); out["log"] = LOG
    json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {OUTJ}  用时 {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
