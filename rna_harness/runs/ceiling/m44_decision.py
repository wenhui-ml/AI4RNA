#!/usr/bin/env python3
"""M44: what the incumbent benchmark's margin buys at a fixed experimental budget.

Declaration `m44_decision_declaration.json`, frozen and pushed at 3f61ef2c BEFORE this file
existed. Offline, $0 -- reads m43_predictor_scores.csv and folds nothing.

M43 settled the reorder question: no. The intervention readout ranks the four implementations in
the incumbent order. What did not survive is the MARGIN -- 0.27082 best-to-worst on structure
agreement against 0.07377 on intervention response. This run converts that into experiments: at a
fixed budget, how many more rescuing interventions does acting on the incumbent margin recover?

M31 computes recovery at one budget for three rules and reports point estimates with no interval
on any difference. Every difference here carries one, resampled by target.
"""
from __future__ import annotations

import collections, csv, json, math, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DECL = json.load(open(os.path.join(HERE, "m44_decision_declaration.json")))
K = DECL["constants"]
THRESH, BUDGETS, NMIN = K["THRESH_MAIN"], K["BUDGETS"], K["NMIN"]
NBOOT, SEED, NSPLIT, NRAND = K["NBOOT"], K["SEED"], K["NSPLIT"], K["NRANDOM_DRAWS"]
SCORES = os.path.join(HERE, "m43_predictor_scores.csv")
OUTJ = os.path.join(HERE, "m44_decision.json")
IMPL = ("ViennaRNA", "EternaFold", "RNAstructure", "RibonanzaNet")
COL = {"ViennaRNA": "pmut", "EternaFold": "emut", "RNAstructure": "smut", "RibonanzaNet": "nmut"}
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def load_cells():
    rows = list(csv.DictReader(open(SCORES)))
    by = collections.defaultdict(list)
    for d in rows:
        r = {"puzzle": d["puzzle"], "design": (d["puzzle"], d["method"]),
             "rf": float(d["rescue"]), "depth": float(d["depth"])}
        for nm in IMPL:
            r[nm] = float(d[COL[nm]])
        r["oracle"] = r["rf"]
        by[r["design"]].append(r)
    return [(k, v) for k, v in by.items() if len(v) >= NMIN], len(rows)


def depth_sign(cells):
    """M31's convention: the globally most favourable direction for the free baseline, which is
    the side that makes our own contribution look smaller. Recomputed, not imported."""
    s = 0.0
    for _k, rs in cells:
        d = np.array([r["depth"] for r in rs], float); y = np.array([r["rf"] for r in rs], float)
        if d.std() > 0 and y.std() > 0:
            s += float(np.corrcoef(d, y)[0, 1])
    return 1.0 if s >= 0 else -1.0


def recovery(cells, key, sign=1.0, thresh=THRESH, frac=0.25, order=None):
    """M31's topk_recovery, verbatim in behaviour. `order` lets a caller supply a ranking (used
    for the random rule) instead of sorting a score column."""
    num = den = 0.0; rnd = 0.0
    for k, rs in cells:
        y = np.array([r["rf"] >= thresh for r in rs])
        npos, n = int(y.sum()), len(rs)
        if npos == 0 or npos == n:
            continue
        kk = max(2, math.ceil(frac * n))
        if order is None:
            x = sign * np.array([r[key] for r in rs], float)
            top = np.argsort(-x, kind="stable")[:kk]
        else:
            top = order(k, n)[:kk]
        num += npos * (float(y[top].sum()) / npos); den += npos; rnd += npos * (kk / n)
    return (num / den if den else None), (rnd / den if den else None)


def rank_key(rs, rule, sgn):
    if rule == "depth":
        return sgn * np.array([r["depth"] for r in rs], float)
    if rule.startswith("depth_then_"):
        nm = rule[len("depth_then_"):]
        d = sgn * np.array([r["depth"] for r in rs], float)
        p = np.array([r[nm] for r in rs], float)
        rp = p.argsort().argsort().astype(float) / max(1, len(p))
        return d + 0.5 * rp                      # depth first, the model breaks ties only
    return np.array([r[rule] for r in rs], float)


def rec_rule(cells, rule, sgn, frac, rng=None):
    num = den = 0.0
    for _k, rs in cells:
        y = np.array([r["rf"] >= THRESH for r in rs])
        npos, n = int(y.sum()), len(rs)
        if npos == 0 or npos == n:
            continue
        kk = max(2, math.ceil(frac * n))
        if rule == "random":
            hits = [float(y[rng.permutation(n)[:kk]].sum()) / npos for _ in range(NRAND)]
            rec = float(np.mean(hits))
        else:
            x = rank_key(rs, rule, sgn)
            rec = float(y[np.argsort(-x, kind="stable")[:kk]].sum()) / npos
        num += npos * rec; den += npos
    return num / den if den else None


def boot_ci(v, lo=0.025, hi=0.975):
    v = np.asarray([x for x in v if x is not None], float)
    return [None, None] if len(v) < 20 else [round(float(np.quantile(v, lo)), 5),
                                             round(float(np.quantile(v, hi)), 5)]


def main():
    t0 = time.time()
    out = {"declaration": "m44_decision_declaration.json", "cost": "offline, $0",
           "not_a_test": DECL["not_a_test_and_declared_as_such"]}
    cells, nrows = load_cells()
    sgn = depth_sign(cells)
    scored = [(k, rs) for k, rs in cells
              if 0 < sum(r["rf"] >= THRESH for r in rs) < len(rs)]
    log(f"干预 {nrows}  设计格(>={NMIN}) {len(cells)}  可计分的格 {len(scored)}  深度方向 {sgn:+.0f}")
    out["n_interventions"], out["n_cells"], out["n_cells_scored"] = nrows, len(cells), len(scored)
    if len(scored) < 100:
        log("** 可计分格少于 100 ⇒ 按 kill 不报告任何 recovery **")
        out["void"] = "coverage"; json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
        return 0

    # ---------------------------------------------------------------- arm V
    log("\n=== V 臂 验证(先于任何结果)")
    rng = np.random.default_rng(SEED)
    V = {}
    orc = rec_rule(cells, "oracle", sgn, 1.0)
    V["V1_oracle_at_full_budget"] = round(orc, 6)
    log(f"  V1 oracle 在 100% 预算下捞回 {orc:.6f}  -> {'通过' if abs(orc-1.0)<1e-9 else '**未通过**'}")
    rd = rec_rule(cells, "random", sgn, 0.25, rng)
    _r, base = recovery(cells, "depth", sgn, frac=0.25)
    V["V2_random"] = round(rd, 5); V["V2_expected_base_rate"] = round(base, 5)
    log(f"  V2 random 前25% 捞回 {rd:.5f}  对照期望基率 {base:.5f}  "
        f"-> {'通过' if abs(rd-base)<0.02 else '**未通过**'}")
    m31 = json.load(open(os.path.join(HERE, "m31_which_flips_rescue.json")))["C_decision"]
    rep = {}
    for nm, pub in (("ViennaRNA", "P_vienna"), ("RibonanzaNet", "P_neural"),
                    ("depth", "depth_baseline")):
        ours = rec_rule(cells, nm, sgn, 0.25)
        rep[nm] = {"ours": round(ours, 5), "m31": m31[pub]["recovery_topk"],
                   "abs_diff": round(abs(ours - m31[pub]["recovery_topk"]), 6)}
        log(f"  V3 {nm:14s} 本次 {ours:.5f}  M31 发表 {m31[pub]['recovery_topk']:.5f}  "
            f"差 {rep[nm]['abs_diff']:.6f}")
    V["V3_reproduces_m31"] = rep
    worst = max(r["abs_diff"] for r in rep.values())
    V["V3_pass"] = bool(worst < 1e-4)
    if not V["V3_pass"]:
        log(f"  ** V3 未通过(最大差 {worst}) ⇒ 按 kill,本次作废并报告为复现失败 **")
        out["V"] = V; out["void"] = "V3_reproduction_failure"
        json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 0
    out["V"] = V

    # ---------------------------------------------------------------- arm E1
    RULES = ["random", "depth"] + list(IMPL) + [f"depth_then_{n}" for n in IMPL] + ["oracle"]
    pz = sorted({rs[0]["puzzle"] for _k, rs in cells})
    bypz = collections.defaultdict(list)
    for k, rs in cells:
        bypz[rs[0]["puzzle"]].append((k, rs))
    rb = np.random.default_rng(SEED)
    idx = [rb.integers(0, len(pz), len(pz)) for _ in range(NBOOT)]
    log("\n=== E1 固定预算下的捞回率,每一条与免费深度基线的差都带区间")
    E1 = {}
    for frac in BUDGETS:
        rr = np.random.default_rng(SEED + 1)
        pt = {r: rec_rule(cells, r, sgn, frac, rr) for r in RULES}
        # bootstrap the DIFFERENCE from depth, paired on the same resamples
        diffs = {r: [] for r in RULES}
        for ix in idx[:500]:
            sub = [c for q in ix for c in bypz[pz[q]]]
            d0 = rec_rule(sub, "depth", sgn, frac)
            if d0 is None:
                continue
            for r in RULES:
                if r in ("random", "depth"):
                    continue
                v = rec_rule(sub, r, sgn, frac)
                if v is not None:
                    diffs[r].append(v - d0)
        E1[str(frac)] = {r: {"recovery": round(pt[r], 5),
                             "minus_depth": (None if r in ("random", "depth")
                                             else round(pt[r] - pt["depth"], 5)),
                             "minus_depth_ci": boot_ci(diffs[r])} for r in RULES}
        log(f"\n  预算 {int(frac*100)}%")
        for r in RULES:
            e = E1[str(frac)][r]
            tail = "" if e["minus_depth"] is None else (
                f"  减深度 {e['minus_depth']:+.5f}  CI {e['minus_depth_ci']}"
                f"  {'**区间不含0**' if e['minus_depth_ci'][0] is not None and (e['minus_depth_ci'][0]>0 or e['minus_depth_ci'][1]<0) else 'unresolved'}")
            log(f"    {r:22s} {e['recovery']:.5f}{tail}")
    out["E1"] = E1
    return out, cells, scored, sgn, E1, idx, pz, bypz, t0


def arm_E2(cells, sgn, E1):
    """THE TRANSLATION. The incumbent margin against the decision margin, and the decision margin
    restated in constructs, because a reader's budget is counted in tubes and not in fractions."""
    m43 = json.load(open(os.path.join(HERE, "m43_reorder.json")))
    A = m43["R_agree"]
    log("\n=== E2 常规基准的幅度,换算成实验")
    scored = [(k, rs) for k, rs in cells if 0 < sum(r["rf"] >= THRESH for r in rs) < len(rs)]
    npos_total = sum(sum(r["rf"] >= THRESH for r in rs) for _k, rs in scored)
    log(f"  可计分格 {len(scored)}  其中真正 rescue 的干预共 {npos_total}")
    rows = []
    for i in range(len(IMPL)):
        for j in range(len(IMPL)):
            A_, B_ = IMPL[i], IMPL[j]
            if A[A_] <= A[B_]:
                continue                        # report each pair once, leader first
            for frac in BUDGETS:
                e = E1[str(frac)]
                dec = e[A_]["recovery"] - e[B_]["recovery"]
                rows.append({"leader": A_, "trailing": B_, "budget": frac,
                             "incumbent_margin_f1": round(A[A_] - A[B_], 5),
                             "decision_margin_recovery": round(dec, 5),
                             "extra_rescuing_interventions": round(dec * npos_total, 1)})
    for r in rows:
        if r["budget"] == 0.25:
            log(f"  {r['leader']:14s} 领先 {r['trailing']:14s}  常规 F1 差 {r['incumbent_margin_f1']:+.5f}"
                f"  →  25% 预算下捞回差 {r['decision_margin_recovery']:+.5f}"
                f"  ≈ {r['extra_rescuing_interventions']:+.1f} 个干预(全语料)")
    return {"pairs": rows, "n_rescuing_interventions_total": int(npos_total),
            "agreement_from_m43": {k: A[k] for k in IMPL}}


def arm_E3(cells, sgn):
    """HELD OUT. Pick the winning rule on half the targets, apply it to the other half."""
    log("\n=== E3 留出检验(按靶标对半分,在前一半选规则,在后一半用)")
    RULES = ["depth"] + list(IMPL) + [f"depth_then_{n}" for n in IMPL]
    pz = sorted({rs[0]["puzzle"] for _k, rs in cells})
    bypz = collections.defaultdict(list)
    for k, rs in cells:
        bypz[rs[0]["puzzle"]].append((k, rs))
    rng = np.random.default_rng(SEED + 7)
    picked, ins, outs = collections.Counter(), [], []
    for _ in range(NSPLIT):
        p = rng.permutation(len(pz)); h = len(pz) // 2
        A_ = [c for q in p[:h] for c in bypz[pz[q]]]
        B_ = [c for q in p[h:] for c in bypz[pz[q]]]
        sc = {r: rec_rule(A_, r, sgn, 0.25) for r in RULES}
        sc = {r: v for r, v in sc.items() if v is not None}
        if not sc:
            continue
        win = max(sc, key=sc.get)
        v = rec_rule(B_, win, sgn, 0.25)
        if v is None:
            continue
        picked[win] += 1; ins.append(sc[win]); outs.append(v)
    res = {"n_splits": len(ins), "rules_picked": dict(picked),
           "in_sample_mean": round(float(np.mean(ins)), 5),
           "held_out_mean": round(float(np.mean(outs)), 5),
           "shrinkage": round(float(np.mean(ins)) - float(np.mean(outs)), 5),
           "held_out_ci": boot_ci(outs)}
    log(f"  {res['n_splits']} 次划分;被选中的规则 {res['rules_picked']}")
    log(f"  样本内 {res['in_sample_mean']:.5f}  留出 {res['held_out_mean']:.5f}  "
        f"收缩 {res['shrinkage']:+.5f}")
    return res


def run():
    r = main()
    if not isinstance(r, tuple):
        return r
    out, cells, scored, sgn, E1, idx, pz, bypz, t0 = r
    out["E2"] = arm_E2(cells, sgn, E1)
    out["E3_held_out"] = arm_E3(cells, sgn)

    # ---- E4: the free-information floor
    e = E1["0.25"]
    best = max(IMPL, key=lambda n: e[n]["recovery"])
    gd, gb = e["depth"]["recovery"] - e["random"]["recovery"], \
        e[best]["recovery"] - e["random"]["recovery"]
    out["E4_free_information_fraction"] = {
        "budget": 0.25, "best_implementation": best,
        "depth_gain_over_random": round(gd, 5), "best_gain_over_random": round(gb, 5),
        "fraction_of_the_best_gain_delivered_by_depth": round(gd / gb, 4) if gb else None}
    log(f"\n=== E4 免费信息的份额:深度拿走最佳实现({best})对随机增益的 "
        f"{out['E4_free_information_fraction']['fraction_of_the_best_gain_delivered_by_depth']:.1%}")

    # ---- staked
    st = {}
    st["1_no_impl_beats_depth_at_any_budget"] = all(
        (E1[str(f)][n]["minus_depth_ci"][0] is None)
        or (E1[str(f)][n]["minus_depth_ci"][0] <= 0 <= E1[str(f)][n]["minus_depth_ci"][1])
        for f in BUDGETS for n in IMPL)
    am = out["E2"]["agreement_from_m43"]
    gap = am["RibonanzaNet"] - am["ViennaRNA"]
    dm = [abs(E1[str(f)]["RibonanzaNet"]["recovery"] - E1[str(f)]["ViennaRNA"]["recovery"])
          for f in BUDGETS]
    st["2_incumbent_margin_does_not_predict"] = bool(gap > 0.20 and max(dm) < 0.03)
    st["2_detail"] = {"agreement_gap": round(gap, 5), "max_recovery_gap": round(max(dm), 5)}
    st["3_held_out_not_above_in_sample"] = bool(
        out["E3_held_out"]["held_out_mean"] <= out["E3_held_out"]["in_sample_mean"])
    st["4_depth_delivers_at_least_80pc"] = bool(
        (out["E4_free_information_fraction"]["fraction_of_the_best_gain_delivered_by_depth"] or 0) >= 0.80)
    out["staked"] = st
    log("\n=== 预注册记分")
    for k, v in st.items():
        if k.endswith("_detail"):
            continue
        log(f"  {k:44s} {'成立' if v else '**失败**'}")
    out["elapsed_s"] = round(time.time() - t0, 1); out["log"] = LOG
    json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {OUTJ}  用时 {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(run())
