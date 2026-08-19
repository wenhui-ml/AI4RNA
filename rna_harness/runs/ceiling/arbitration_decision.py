#!/usr/bin/env python3
"""A10: does the criterion change a DECISION, and does physical measurement say it changed for the better?

Declared in `arbitration_decision_declaration.json`, committed and pushed before the third sample was folded.
Three things in that file are worth repeating here because they are why the run is shaped this way:

  NO VALID JUDGE EXISTS ON OUR OWN POOLS. The criterion is calibrated against the tube; our designs have no
  tube score and never will. RibonanzaNet sits in the selection loop, so it cannot judge. The three
  thermodynamic predictors are the family the criterion overrules, so scoring the rule with them is not a
  test. Hence: OpenKnot designs, judged by measured target_openknot_score.

  THE CRITERION IS PAIRWISE AND ONE-SIDED, so it induces no total order and cannot be used as a ranking -- a
  pairwise rule may be intransitive. So the object here is a TWO-CANDIDATE CONTEST: the panel's top pick
  against the learned ruler's top pick, one pair per cell, which is also the shape the product decision has.

  THE SAMPLE IS THE THIRD ONE. The cut was fitted on sample 1 and confirmed on sample 2, so a selector built
  from that cut needs molecules neither has seen. `replay_third` excludes both and asserts it.

Ties in a ruler's top pick are broken at random and averaged over 200 draws: a tie block is where a ruler has
no opinion, and keeping it whole is the defect (D2) that made an earlier coverage comparison undefined.

COST: $0 beyond the panel fold.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from arbitration_absolute import ranks01                                      # noqa: E402

RNET_OPTIMISED = "Struct2SeQ-SHAPE"
TUBE_NOISE = 4.1          # independently measured on the confirmatory sample; used only in secondary 3


def load_third(path, panel_key):
    """Rows of the third panel run, deduplicated by molecule, as ruler_gradient.load does it."""
    d = json.load(open(path))
    if d["n_failed"]:
        raise SystemExit(f"{d['n_failed']} designs failed to fold; fix before analysing")
    rows = [r for r in d["designs"] if r["method"] != RNET_OPTIMISED]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["puzzle"], r["seq"])].append(r)
    out, spreads = [], []
    for _, cp in g.items():
        if len(cp) > 1:
            spreads.append(max(x["wet"] for x in cp) - min(x["wet"] for x in cp))
        r = dict(cp[0])
        r["wet"] = sum(x["wet"] for x in cp) / len(cp)
        r["panel"] = r[panel_key]
        out.append(r)
    return out, spreads, d


def contests(rec, cut, draws, rng):
    """One contest per cell: the panel's top pick against the learned ruler's top pick.

    Returns a dict cell -> record. `wet_*` are averaged over random tie-breaks, so a cell whose ruler ties at
    the top contributes the average outcome of the choices that ruler leaves open rather than a lucky one.
    """
    g = collections.defaultdict(list)
    for r in rec:
        g[(r["puzzle"], r["method"])].append(r)
    out, n_small = {}, 0
    for k, rows in g.items():
        if len(rows) < 3:
            n_small += 1; continue
        vf, vp = [x["f1c"] for x in rows], [x["panel"] for x in rows]
        if st.pstdev(vf) == 0 or st.pstdev(vp) == 0:
            n_small += 1; continue          # a ruler with no spread in this cell has no top pick to defend
        rf, rp = ranks01(vf), ranks01(vp)
        for x, a_, b_ in zip(rows, rf, rp):
            x["rf"], x["rp"] = a_, b_
        top_f = [x for x in rows if x["f1c"] == max(vf)]
        top_p = [x for x in rows if x["panel"] == max(vp)]
        acc = {"panel": [], "learned": [], "arb": [], "changed": [], "d": [], "same": []}
        for _ in range(draws):
            a = rng.choice(top_f)                     # learned ruler's pick, ties broken at random
            b = rng.choice(top_p)                     # panel's pick
            same = a["seq"] == b["seq"]
            d = abs(a["rf"] - b["rf"]) - abs(a["rp"] - b["rp"])
            lic = (not same) and (d > cut)
            acc["panel"].append(b["wet"])
            acc["learned"].append(a["wet"])
            acc["arb"].append(a["wet"] if lic else b["wet"])
            acc["changed"].append(1.0 if lic else 0.0)
            acc["same"].append(1.0 if same else 0.0)
            if not same:
                acc["d"].append(d)
        out[k] = {"wet_panel": st.mean(acc["panel"]), "wet_learned": st.mean(acc["learned"]),
                  "wet_arb": st.mean(acc["arb"]), "p_changed": st.mean(acc["changed"]),
                  "p_same": st.mean(acc["same"]),
                  "d_mean": (st.mean(acc["d"]) if acc["d"] else None), "n_designs": len(rows)}
    return out, n_small


def boot_diff(cells, key_a, key_b, draws, rng):
    """Cluster bootstrap of the paired difference key_a - key_b; the unit is the cell, one vote each."""
    ks = list(cells)
    if not ks:
        return None, None, None
    dv = {k: cells[k][key_a] - cells[k][key_b] for k in ks}
    pt = st.mean(dv.values())
    n = len(ks)
    out = sorted(sum(dv[ks[rng.randrange(n)]] for _ in range(n)) / n for _ in range(draws))
    return pt, out[int(0.025 * draws)], out[min(draws - 1, int(0.975 * draws))]


def boot_frac(vals, draws, rng):
    """Cluster bootstrap of a mean over cells (each cell one value in [0,1])."""
    v = list(vals)
    if not v:
        return None, None, None
    n = len(v)
    out = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return st.mean(v), out[int(0.025 * draws)], out[min(draws - 1, int(0.975 * draws))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--third", default=os.path.join(HERE, "panel_f1_third.json"))
    ap.add_argument("--declaration", default=os.path.join(HERE, "arbitration_decision_declaration.json"))
    ap.add_argument("--tie-draws", type=int, default=200)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "arbitration_decision.json"))
    a = ap.parse_args()
    D = json.load(open(a.declaration))
    CUTS = {"panel_recall_product": 0.16406, "panel_f1": 0.2313432835820895}
    res = {"declaration": D, "cuts_reused": CUTS}

    for col in ("panel_recall_product", "panel_f1"):
        tag = "出厂列(主)" if col == "panel_recall_product" else "观测列(次)"
        rec, spreads, meta = load_third(a.third, col)
        cells, n_small = contests(rec, CUTS[col], a.tie_draws, random.Random(a.seed))
        print(f"\n=== {tag} {col};切点复用 {CUTS[col]:.5f}(未重拟合)===")
        print(f"  第三批 {meta['n']} 条(每靶标 {meta.get('third_per_target')}),去重 {len(rec)} 条分子;"
              f"可用格 {len(cells)}(丢弃 {n_small}:不足 3 条或某把尺子无方差)")
        # THE POWER FLOOR IS REPORTED BEFORE ANY INTERVAL, as declared.
        n_changed = sum(1 for v in cells.values() if v["p_changed"] > 0.5)
        n_same = sum(1 for v in cells.values() if v["p_same"] > 0.5)
        print(f"  两头名【相同】的格 {n_same};判据【真正改变了决定】的格 {n_changed}"
              f"(声明门槛 30){'  ✓' if n_changed >= 30 else '  ✗ 低于门槛'}")
        row = {"n_cells": len(cells), "n_dropped": n_small, "n_same_pick": n_same, "n_changed": n_changed}
        if n_changed < 30:
            print("  按声明:改变决定的格 < 30 → 三臂几乎相同,本检验记无数据。")
            row["verdict"] = "no_data_as_declared"; res[col] = row; continue

        rng = random.Random(a.seed + 1)
        for nm, ka, kb in (("arbitrated − panel_only  【主检验】", "wet_arb", "wet_panel"),
                           ("arbitrated − learned_only 【次1】", "wet_arb", "wet_learned"),
                           ("learned_only − panel_only (参考)", "wet_learned", "wet_panel")):
            pt, lo, hi = boot_diff(cells, ka, kb, a.draws, rng)
            ok = lo is not None and lo > 0
            print(f"  {nm}: {pt:+.3f} 分  聚类 CI [{lo:+.3f},{hi:+.3f}]"
                  f"{'  ✓ 排除 0 且为正' if ok else ('  ✓ 排除 0 但为负' if hi < 0 else '  跨过 0')}")
            row[ka + "_minus_" + kb] = {"diff": pt, "ci": [lo, hi], "excludes_zero_positive": bool(ok)}

        # secondary 2: licensed vs unlicensed, level-invariant contrast, only where the picks differ
        diff_cells = {k: v for k, v in cells.items() if v["p_same"] <= 0.5}
        lic = [1.0 if v["wet_learned"] > v["wet_panel"] else 0.0
               for v in diff_cells.values() if v["p_changed"] > 0.5]
        unl = [1.0 if v["wet_learned"] > v["wet_panel"] else 0.0
               for v in diff_cells.values() if v["p_changed"] <= 0.5]
        print(f"  【次2】两头名不同的格 {len(diff_cells)};试管更喜欢学习型头名的比例:")
        for nm, v in (("许可组", lic), ("不许可组", unl)):
            if len(v) < 10:
                print(f"      {nm} n={len(v)} —— 太少,不给数"); row[nm] = {"n": len(v)}; continue
            pt, lo, hi = boot_frac(v, a.draws, random.Random(a.seed + 2))
            print(f"      {nm} n={len(v):3d}  {pt:.1%} [{lo:.1%},{hi:.1%}]"
                  f"{'  ✓ 上越 0.5' if lo > 0.5 else '  跨 0.5'}")
            row[nm] = {"n": len(v), "frac": pt, "ci": [lo, hi], "clears_above": bool(lo > 0.5)}

        # secondary 3: restricted to cells the tube can actually referee, applied symmetrically
        ref = {k: v for k, v in diff_cells.items() if abs(v["wet_learned"] - v["wet_panel"]) > TUBE_NOISE}
        l2 = [1.0 if v["wet_learned"] > v["wet_panel"] else 0.0
              for v in ref.values() if v["p_changed"] > 0.5]
        u2 = [1.0 if v["wet_learned"] > v["wet_panel"] else 0.0
              for v in ref.values() if v["p_changed"] <= 0.5]
        print(f"  【次3】只看 |试管差| > {TUBE_NOISE} 的格({len(ref)} 个):"
              f"许可组 {(sum(l2)/len(l2)) if l2 else float('nan'):.1%} (n={len(l2)}),"
              f"不许可组 {(sum(u2)/len(u2)) if u2 else float('nan'):.1%} (n={len(u2)})")
        row["refereeable"] = {"n": len(ref), "licensed_n": len(l2), "unlicensed_n": len(u2),
                              "licensed_frac": (sum(l2) / len(l2)) if l2 else None,
                              "unlicensed_frac": (sum(u2) / len(u2)) if u2 else None}
        res[col] = row

    m = res.get("panel_recall_product", {})
    print(f"\n=== 判定(按声明,主口径 = 出厂列 × 每格一票)===")
    if m.get("verdict") == "no_data_as_declared":
        print("  无数据 → 无结论;④「判据能改变一个决定」这一行仍不计分。")
    else:
        p1 = m.get("wet_arb_minus_wet_panel", {}).get("excludes_zero_positive")
        p2 = m.get("wet_arb_minus_wet_learned", {}).get("excludes_zero_positive")
        print(f"  主检验(判据改变决定且试管说更好): {'✓ 成立' if p1 else '✗ 不成立'}")
        print(f"  次1(组合打过现成模型单用):        {'✓ 成立' if p2 else '✗ 不成立'}")
        if not p1:
            print("  按声明:④ 的「判据能改变一个决定」记 ❌(测过、未拿到),不是不计分。")
            print("  A6/A7 不因此失效 —— 那是关于相悖配对的,这是关于决定的。")
        res["A10_PRIMARY_PASS"] = bool(p1); res["A10_BEATS_OFFTHESHELF"] = bool(p2)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
