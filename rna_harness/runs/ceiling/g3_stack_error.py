#!/usr/bin/env python3
"""G3: bound the dG_stack approximation error, on the side where the approximation is TRUE.

Pre-registered in g3_stack_error_declaration.json. Read that first; the kill conditions there are frozen.

THE IDEA. gamma* = F_nested - E_skeleton - dG_stack(crossed) has exactly one approximate term, dG_stack.
Its error cannot be measured on a pseudoknot, because there the residual and the unknown gamma are the same
number. But on a NESTED structure the pseudoknot term is identically zero, so the residual of the SAME
additive decomposition is exactly measurable: both E_full and E_skeleton are exact ViennaRNA evaluations.
That is the true side of the property (rule 29), and void gate 1 asserts the property itself inside the run.

TWO ARMS, because they kill different downstream items:
    A  remove a whole nested helix, size-matched to that molecule's crossing helix -> total error per helix
    B  remove only a helix's outermost base pair                                  -> error per stack

WHAT THIS CANNOT DO. The nested analogue's residual carries the cost of closing a NESTED loop; the
pseudoknot's carries the cost of closing a CROSSING loop. Those are different physical terms. So this bounds
only the components with a nested counterpart (dangling ends / terminal mismatches, coaxial stacking, loop-size
dependence) and gives a LOWER bound on the total variability. The crossing-specific part is gamma by
definition and is out of range. One-sided: this run can kill construction one, it cannot confirm it.

COST: $0, local ViennaRNA only. Needs the `rna` env (RNA 2.7.2). Seconds, single process.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import RNA                                                                     # noqa: E402
from gamma_star import best_wet_per_target, crossing_set, stack_terms          # noqa: E402


def pairs_of(struct: str) -> set:
    """Pair set of a purely nested dot-bracket string, 0-based."""
    stk, out = [], set()
    for i, c in enumerate(struct):
        if c == "(":
            stk.append(i)
        elif c == ")":
            out.add((stk.pop(), i))
    return out


def helices(pairs) -> list:
    """Split a nested pair set into maximal stacked runs. A helix is a list of pairs, outermost first."""
    S = set(pairs)
    starts = [p for p in S if (p[0] - 1, p[1] + 1) not in S]
    out = []
    for (i, j) in sorted(starts):
        h, a, b = [], i, j
        while (a, b) in S:
            h.append((a, b))
            a, b = a + 1, b - 1
        out.append(h)
    return out


def struct_from(pairs, n: int) -> str:
    """Render a nested pair set back to dot-bracket."""
    s = ["."] * n
    for (i, j) in pairs:
        s[i], s[j] = "(", ")"
    return "".join(s)


def balanced(struct: str) -> bool:
    d = 0
    for c in struct:
        if c == "(":
            d += 1
        elif c == ")":
            d -= 1
            if d < 0:
                return False
    return d == 0


def n_lonely(pairs) -> int:
    """Pairs with no stacked neighbour on either side -- ViennaRNA's evaluation of these is not physical."""
    S = set(pairs)
    return sum(1 for (i, j) in S if (i - 1, j + 1) not in S and (i + 1, j - 1) not in S)


def row(puz: str, seq: str, target: str) -> dict:
    from rna_harness.benchmarks.rnainvbench import _round_only
    skel = _round_only(target)
    n = len(seq)
    P = pairs_of(skel)
    cross = crossing_set(target)
    nst_cross = len(stack_terms(seq, cross))
    fc = RNA.fold_compound(seq)
    e_full = RNA.energy_of_struct(seq, skel)

    r = {"puzzle": puz, "L": n, "nst_cross": nst_cross, "e_full_skeleton": round(e_full, 3)}

    # ---- void gate 1: the property itself. Remove NOTHING; the residual must be identically zero. -------
    r["gate1_zero_removal_resid"] = round(e_full - RNA.energy_of_struct(seq, skel) - 0.0, 9)

    H = [h for h in helices(P) if len(h) >= 2]
    if not H:
        r["error"] = "skeleton has no helix with >= 2 stacked pairs"
        return r
    r["n_helices"] = len(H)

    # ---- arm A: whole helix, size-matched to this molecule's crossing helix ------------------------------
    # A helix of k pairs carries k-1 stack terms; match that against the crossing set's stack count.
    h = min(H, key=lambda x: (abs((len(x) - 1) - nst_cross), -len(x)))
    keptA = P - set(h)
    sA = struct_from(keptA, n)
    dgA = sum(v for _, v in stack_terms(seq, h))
    eA = RNA.energy_of_struct(seq, sA)
    r.update({"A_helix_pairs": len(h), "A_nst": len(h) - 1, "A_match_gap": abs((len(h) - 1) - nst_cross),
              "A_dG_stack": round(dgA, 3), "A_e_skel": round(eA, 3),
              "A_resid": round(e_full - eA - dgA, 3),
              "A_n_removed": len(h), "A_balanced": balanced(sA), "A_lonely_left": n_lonely(keptA)})
    r["A_resid_per_stack"] = round(r["A_resid"] / (len(h) - 1), 3) if len(h) > 1 else None

    # ---- arm B: only the outermost base pair of the largest helix ---------------------------------------
    hb = max(H, key=len)
    (i, j) = hb[0]
    keptB = P - {(i, j)}
    sB = struct_from(keptB, n)
    stB = fc.eval_int_loop(i + 1, j + 1, i + 2, j) / 100.0
    eB = RNA.energy_of_struct(seq, sB)
    r.update({"B_stack": round(stB, 3), "B_e_skel": round(eB, 3),
              "B_resid": round(e_full - eB - stB, 3),
              "B_n_removed": 1, "B_balanced": balanced(sB), "B_lonely_left": n_lonely(keptB)})
    return r


def spread(vals, label: str) -> dict:
    """IQR/std/percentiles, with the exact-tie count printed beside them -- rule 35."""
    s = sorted(vals)
    m = len(s)
    if m < 4:
        return {"label": label, "n": m, "too_few": True}
    q = lambda p: s[min(m - 1, int(p * m))]                                    # noqa: E731
    return {"label": label, "n": m, "median": round(st.median(s), 3),
            "mean": round(st.fmean(s), 3), "iqr": round(q(0.75) - q(0.25), 3),
            "std": round(st.pstdev(s), 3), "p10": round(q(0.10), 3), "p25": round(q(0.25), 3),
            "p75": round(q(0.75), 3), "p90": round(q(0.90), 3),
            "min": round(s[0], 3), "max": round(s[-1], 3),
            "exact_zeros": sum(1 for x in s if abs(x) < 1e-12)}


def main() -> int:
    from leaderboard_design import targets
    from gamma_star import gamma_star
    T = targets()
    best = best_wet_per_target()

    R, bad = [], []
    for p in sorted(best):
        tgt = T[p]["target"]
        if not crossing_set(tgt):
            continue
        r = row(p, best[p]["seq"], tgt)
        r["gamma_star"] = gamma_star(best[p]["seq"], tgt)["gamma_star"]
        (bad if "error" in r else R).append(r)

    # ---- void gates ------------------------------------------------------------------------------------
    g1 = [r for r in R if abs(r["gate1_zero_removal_resid"]) > 1e-6]
    g2 = [r for r in R if not (r["A_balanced"] and r["B_balanced"])]
    g3A = [r for r in R if r["A_n_removed"] == 0]
    lonelyA = [r for r in R if r["A_lonely_left"] > 0]
    gates = {"gate1_nonzero": len(g1), "gate2_unbalanced": len(g2), "gate3_noop": len(g3A),
             "gate4_lonely_left": len(lonelyA), "skipped_no_helix": len(bad), "n": len(R)}
    print("void gates:", json.dumps(gates))
    if g1 or g2:
        print("\n  VOID: 被验证的性质在 smoke 里就不成立,或结构非法。不报任何结果。")
        json.dump({"verdict": "void", "gates": gates, "rows": R},
                  open(os.path.join(HERE, "g3_stack_error.json"), "w"), indent=1)
        return 1
    live = [r for r in R if r["A_n_removed"] > 0]
    if len(live) < 20:
        print(f"\n  NO DATA: 可用条数 {len(live)} < 20(声明里的地板)。")
        json.dump({"verdict": "no_data", "gates": gates, "rows": R},
                  open(os.path.join(HERE, "g3_stack_error.json"), "w"), indent=1)
        return 1

    clean = [r for r in live if r["A_lonely_left"] == 0]
    S = {"A_resid": spread([r["A_resid"] for r in live], "A 臂残差(整条螺旋)"),
         "A_resid_clean": spread([r["A_resid"] for r in clean], "A 臂残差(排除留下孤立配对的)"),
         "A_resid_per_stack": spread([r["A_resid_per_stack"] for r in live if r["A_resid_per_stack"] is not None],
                                     "A 臂每堆叠残差"),
         "B_resid": spread([r["B_resid"] for r in live], "B 臂残差(单个最外配对)"),
         "gamma_star": spread([r["gamma_star"] for r in live], "gamma*(比较基准)")}

    print(f"\n{'':34} {'n':>3} {'median':>8} {'mean':>8} {'IQR':>8} {'std':>8} {'p10':>8} {'p90':>8}")
    for k in ("A_resid", "A_resid_clean", "A_resid_per_stack", "B_resid", "gamma_star"):
        d = S[k]
        if d.get("too_few"):
            print(f"{d['label']:34} n={d['n']} 太少")
            continue
        print(f"{d['label']:34} {d['n']:3d} {d['median']:8.3f} {d['mean']:8.3f} {d['iqr']:8.3f} "
              f"{d['std']:8.3f} {d['p10']:8.3f} {d['p90']:8.3f}")

    # ---- the frozen kill conditions --------------------------------------------------------------------
    iqrA, iqrG = S["A_resid"]["iqr"], S["gamma_star"]["iqr"]
    ratio = iqrA / iqrG if iqrG else float("inf")
    medA = S["A_resid"]["median"]
    if iqrA >= iqrG:
        verdict, says = "hard_kill", ("IQR(resid_A) >= IQR(gamma*) ⇒ 构造一分辨率不够,"
                                     "第 1/2/4 项不许跑,必须先修 dG_stack")
    elif ratio > 0.5:
        verdict, says = "soft_kill", ("IQR(resid_A) > 0.5*IQR(gamma*) ⇒ 第 1 项可跑,"
                                      "但必须把噪声底一起报,不许声称小于噪声底的效应")
    else:
        verdict, says = "passes", "近似误差的分子间变动显著小于 gamma* 的变动 ⇒ 第 1 项可跑"
    sign_ok = abs(medA) <= 3.0
    print(f"\n  IQR(resid_A)/IQR(gamma*) = {iqrA:.3f}/{iqrG:.3f} = {ratio:.3f}   -> {verdict}")
    print(f"  {says}")
    print(f"  median(resid_A) = {medA:+.3f} kcal/mol -> 符号检验(第 2 项)"
          f"{'可以报' if sign_ok else '不许报,零点位置不可信'}(闸门 |median| <= 3)")

    out = {"verdict": verdict, "says": says, "iqr_ratio": round(ratio, 3),
           "sign_test_allowed": sign_ok, "median_resid_A": medA,
           "gates": gates, "spreads": S, "rows": R, "skipped": bad,
           "declared_limitation": ("只界住在嵌套世界里有对应物的误差成分;交叉专属的部分按定义就是 gamma,"
                                   "不在射程内。故这是总误差变动性的下界,本实验能打死不能确认。")}
    json.dump(out, open(os.path.join(HERE, "g3_stack_error.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> g3_stack_error.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
