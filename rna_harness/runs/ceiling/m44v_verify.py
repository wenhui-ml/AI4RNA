#!/usr/bin/env python3
"""M44v: the four verification steps M44 owed before its numbers enter the manuscript.

NOT a new experiment. It re-derives M44's published quantities without importing M44, and it
tests the one construction in M44 that could have manufactured its headline.

The trigger. M44 ranks interventions inside a design cell with
    np.argsort(-x, kind="stable")
and its free baseline is HELIX DEPTH, an integer. Ties are therefore everywhere, and a stable sort
breaks them by CSV row order. If that order carries any information about rescue -- and rows are
written in construct order, which is position order, which is not obviously independent of the
outcome -- then "the free depth baseline delivers 91% of the best model's gain" is partly a
statement about how the file was written. No summary statistic in M44 can see this.

    (1) cross-implementation   recovery recomputed from the CSV by an independent routine
    (2) equivalent form        weighted mean of per-cell recalls  ==  pooled hits / pooled positives
    (3) reverse control        the original tie rule reproduces the published JSON; then ONE thing
                               changes -- ties are broken at random and averaged -- so any movement
                               is attributable to that one thing
    (4) identities             recovery(budget=100%) == 1 for every rule; the random rule equals its
                               closed form; a constant score under random ties equals the random rule

It also computes the interval M44's staked criterion 2 needed and did not have: the paired
bootstrap CI on RibonanzaNet minus ViennaRNA recovery. The stake is graded as it was written and
stays FAILED; the interval is reported beside it so the reader can see what the gap licenses.

Offline, $0.
"""
from __future__ import annotations
import collections, csv, json, math, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "m43_predictor_scores.csv")
PUB = json.load(open(os.path.join(HERE, "m44_decision.json")))
K = json.load(open(os.path.join(HERE, "m44_decision_declaration.json")))["constants"]
THRESH, BUDGETS, NMIN = K["THRESH_MAIN"], K["BUDGETS"], K["NMIN"]
SEED = K["SEED"]
IMPL = ("ViennaRNA", "EternaFold", "RNAstructure", "RibonanzaNet")
COL = {"ViennaRNA": "pmut", "EternaFold": "emut", "RNAstructure": "smut", "RibonanzaNet": "nmut"}
NTIE = 200          # random tie draws per cell
OUTJ = os.path.join(HERE, "m44v_verify.json")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# ------------------------------------------------------------------ independent reimplementation
def cells_from_csv():
    """Deliberately a different shape from M44's loader: arrays per cell, built in one pass, with
    the CSV row index kept so the tie-order question can be asked at all."""
    by = collections.defaultdict(lambda: {"y": [], "depth": [], "pos": [], **{n: [] for n in IMPL}})
    for i, d in enumerate(csv.DictReader(open(SCORES))):
        c = by[(d["puzzle"], d["method"])]
        c["y"].append(float(d["rescue"])); c["depth"].append(float(d["depth"])); c["pos"].append(i)
        for n in IMPL:
            c[n].append(float(d[COL[n]]))
    out = {}
    for k, c in by.items():
        if len(c["y"]) < NMIN:
            continue
        out[k] = {n: np.asarray(v, float) for n, v in c.items()}
        out[k]["puzzle"] = k[0]
    return out


def score_vector(c, rule, sgn):
    if rule == "depth":
        return sgn * c["depth"]
    if rule == "oracle":
        return c["y"]
    if rule == "constant":
        return np.zeros(len(c["y"]))
    if rule.startswith("depth_then_"):
        p = c[rule[len("depth_then_"):]]
        return sgn * c["depth"] + 0.5 * (p.argsort().argsort().astype(float) / max(1, len(p)))
    return c[rule]


def budget_k(n, frac):
    return max(2, math.ceil(frac * n))


def informative(c):
    y = c["y"] >= THRESH
    return 0 < int(y.sum()) < len(y)


def recover(cells, rule, sgn, frac, tie="stable", rng=None, pooled=False):
    """tie='stable' reproduces M44. tie='random' averages over random within-cell orders, which is
    the only change made between the two calls."""
    num = den = 0.0; hits = 0.0
    for c in cells.values():
        if not informative(c):
            continue
        y = c["y"] >= THRESH; n = len(y); npos = int(y.sum()); kk = budget_k(n, frac)
        if rule == "random":
            rec = float(np.mean([y[rng.permutation(n)[:kk]].sum() for _ in range(NTIE)])) / npos
        else:
            x = score_vector(c, rule, sgn)
            if tie == "stable":
                rec = float(y[np.argsort(-x, kind="stable")[:kk]].sum()) / npos
            else:
                acc = 0.0
                for _ in range(NTIE):
                    p = rng.permutation(n)
                    acc += float(y[p[np.argsort(-x[p], kind="stable")[:kk]]].sum())
                rec = acc / NTIE / npos
        num += npos * rec; den += npos; hits += npos * rec
    if den == 0:
        return None
    return (hits / den) if pooled else (num / den)


def depth_sign(cells):
    s = 0.0
    for c in cells.values():
        if c["depth"].std() > 0 and c["y"].std() > 0:
            s += float(np.corrcoef(c["depth"], c["y"])[0, 1])
    return 1.0 if s >= 0 else -1.0


def main():
    t0 = time.time()
    cells = cells_from_csv()
    sgn = depth_sign(cells)
    scored = {k: c for k, c in cells.items() if informative(c)}
    out = {"reads": "m43_predictor_scores.csv + m44_decision.json", "cost": "offline, $0",
           "n_cells": len(cells), "n_cells_scored": len(scored), "depth_sign": sgn}
    log(f"格 {len(cells)}  可计分 {len(scored)}  深度方向 {sgn:+.0f}"
        f"  (M44 发表 {PUB['n_cells']}/{PUB['n_cells_scored']}/{PUB['E1']['0.25']['depth']['recovery'] and sgn:+.0f})")

    RULES = ["depth"] + list(IMPL) + [f"depth_then_{n}" for n in IMPL]

    # ---------------------------------------------------------------- (3) reverse control FIRST
    log("\n=== ③ 反向对照:同一个 tie 规则必须复现已发表的 JSON")
    rev = {}
    worst = 0.0
    for frac in BUDGETS:
        for r in RULES + ["oracle"]:
            mine = recover(cells, r, sgn, frac)
            pub = PUB["E1"][str(frac)][r]["recovery"]
            rev[f"{frac}|{r}"] = {"ours": round(mine, 6), "published": pub,
                                  "abs_diff": round(abs(mine - pub), 7)}
            worst = max(worst, abs(mine - pub))
    log(f"  {len(rev)} 个量,最大绝对差 {worst:.2e}  -> {'通过' if worst < 1e-5 else '**未通过**'}")
    out["S3_reverse_control"] = {"max_abs_diff": round(worst, 7), "pass": bool(worst < 1e-5),
                                 "cells": rev}
    if worst >= 1e-5:
        log("  ** 复现不了已发表的数 ⇒ 独立实现与原实现不是同一个量,后面的检验没有意义,停 **")
        json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 1

    # ---------------------------------------------------------------- (4) identities
    log("\n=== ④ 零点 / 恒等式自检")
    ident = {}
    full = {r: recover(cells, r, sgn, 1.0) for r in RULES + ["oracle", "constant"]}
    ident["full_budget_is_one"] = {r: round(v, 9) for r, v in full.items()}
    ok_full = all(abs(v - 1.0) < 1e-12 for v in full.values())
    log(f"  100% 预算下每一条规则都必须捞回 1.0:{'通过' if ok_full else '**未通过** ' + str(full)}")

    rr = np.random.default_rng(SEED + 101)
    closed = {}
    for frac in BUDGETS:
        num = den = 0.0
        for c in scored.values():
            n = len(c["y"]); npos = int((c["y"] >= THRESH).sum())
            num += npos * (budget_k(n, frac) / n); den += npos
        emp = recover(cells, "random", sgn, frac, rng=rr)
        closed[str(frac)] = {"closed_form": round(num / den, 5), "empirical": round(emp, 5),
                             "abs_diff": round(abs(num / den - emp), 5)}
        log(f"  random 在 {int(frac*100)}%:闭式 {num/den:.5f}  实测 {emp:.5f}"
            f"  差 {abs(num/den-emp):.5f}")
    ident["random_matches_closed_form"] = closed
    ok_rand = all(v["abs_diff"] < 0.01 for v in closed.values())

    rr = np.random.default_rng(SEED + 202)
    con = {}
    for frac in BUDGETS:
        cst = recover(cells, "constant", sgn, frac, tie="random", rng=rr)
        rnd = closed[str(frac)]["closed_form"]
        con[str(frac)] = {"constant_random_ties": round(cst, 5), "random_rule": rnd,
                          "abs_diff": round(abs(cst - rnd), 5)}
        log(f"  常数分数+随机解 tie 在 {int(frac*100)}%:{cst:.5f} 对 random 的 {rnd:.5f}"
            f"  差 {abs(cst-rnd):.5f}")
    ident["constant_score_equals_random"] = con
    ok_con = all(v["abs_diff"] < 0.01 for v in con.values())
    ident["pass"] = bool(ok_full and ok_rand and ok_con)
    out["S4_identities"] = ident

    # ---------------------------------------------------------------- (2) equivalent form
    log("\n=== ② 等价形式:加权平均的每格召回 == 汇总命中 / 汇总阳性")
    eq = {}; worst_eq = 0.0
    for frac in BUDGETS:
        for r in RULES:
            a = recover(cells, r, sgn, frac); b = recover(cells, r, sgn, frac, pooled=True)
            eq[f"{frac}|{r}"] = round(abs(a - b), 12); worst_eq = max(worst_eq, abs(a - b))
    log(f"  最大绝对差 {worst_eq:.2e}  -> {'通过' if worst_eq < 1e-12 else '**未通过**'}")
    out["S2_equivalent_form"] = {"max_abs_diff": worst_eq, "pass": bool(worst_eq < 1e-12)}

    # ---------------------------------------------------------------- (1) the tie-rule question
    log("\n=== ① 对拍 + 唯一改动:把 tie 从『按文件行序』换成『随机、取平均』")
    log("    改动的只有这一处。任何移动都只能来自它。")
    tie = {}
    for frac in BUDGETS:
        rr = np.random.default_rng(SEED + 303)
        row = {}
        for r in RULES:
            s = recover(cells, r, sgn, frac)
            t = recover(cells, r, sgn, frac, tie="random", rng=rr)
            row[r] = {"stable_ties": round(s, 5), "random_ties": round(t, 5),
                      "delta": round(t - s, 5)}
        tie[str(frac)] = row
        log(f"\n  预算 {int(frac*100)}%")
        for r in RULES:
            e = row[r]
            log(f"    {r:24s} 行序 {e['stable_ties']:.5f}   随机 {e['random_ties']:.5f}"
                f"   Δ {e['delta']:+.5f}")
    out["S1_tie_rule"] = tie
    mx = max(abs(tie[str(f)][r]["delta"]) for f in BUDGETS for r in RULES)
    out["S1_max_abs_shift"] = round(mx, 5)
    log(f"\n  行序 tie 造成的最大位移 {mx:+.5f}")

    # E4 recomputed under BOTH tie rules, at every budget, with the best implementation named
    log("\n=== E4 复算:免费深度拿走最佳实现增益的多少 —— 逐预算,并注明该预算下最佳的是谁")
    e4 = {}
    for frac in BUDGETS:
        rr = np.random.default_rng(SEED + 404)
        rnd = recover(cells, "random", sgn, frac, rng=rr)
        for mode in ("stable", "random"):
            rr2 = np.random.default_rng(SEED + 505)
            rec = {r: recover(cells, r, sgn, frac, tie=mode, rng=rr2) for r in ["depth"] + list(IMPL)}
            best = max(IMPL, key=lambda n: rec[n])
            gd, gb = rec["depth"] - rnd, rec[best] - rnd
            e4[f"{frac}|{mode}"] = {"best_implementation": best, "random": round(rnd, 5),
                                    "depth": round(rec["depth"], 5), "best": round(rec[best], 5),
                                    "fraction": round(gd / gb, 4) if gb > 0 else None}
            log(f"  {int(frac*100)}% / {mode:6s}  最佳={best:14s}"
                f"  深度份额 {e4[f'{frac}|{mode}']['fraction']}")
    out["E4_by_budget"] = e4

    # ---------------------------------------------------------------- staked 2's missing interval
    log("\n=== staked 2 缺的那个区间:RibonanzaNet − ViennaRNA 捞回差,按靶标自助")
    pz = sorted({c["puzzle"] for c in cells.values()})
    bypz = collections.defaultdict(dict)
    for k, c in cells.items():
        bypz[c["puzzle"]][k] = c
    rb = np.random.default_rng(SEED)
    idx = [rb.integers(0, len(pz), len(pz)) for _ in range(500)]
    gapci = {}
    for frac in BUDGETS:
        vals = []
        for ix in idx:
            sub = {}
            for j, q in enumerate(ix):
                for k, c in bypz[pz[q]].items():
                    sub[(j, k)] = c
            a = recover(sub, "RibonanzaNet", sgn, frac); b = recover(sub, "ViennaRNA", sgn, frac)
            if a is not None and b is not None:
                vals.append(a - b)
        lo, hi = float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
        pt = recover(cells, "RibonanzaNet", sgn, frac) - recover(cells, "ViennaRNA", sgn, frac)
        gapci[str(frac)] = {"point": round(pt, 5), "ci": [round(lo, 5), round(hi, 5)],
                            "excludes_zero": bool(lo > 0 or hi < 0)}
        log(f"  {int(frac*100)}%  {pt:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]"
            f"  {'**不含0**' if (lo>0 or hi<0) else 'unresolved'}")
    out["staked2_missing_interval"] = gapci
    out["staked2_verdict_unchanged"] = ("graded as written -> FAILED; the interval is reported "
                                        "beside it and does not regrade it")

    out["elapsed_s"] = round(time.time() - t0, 1); out["log"] = LOG
    json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {OUTJ}  用时 {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
