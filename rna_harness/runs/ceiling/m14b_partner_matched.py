#!/usr/bin/env python3
"""M14b: hold the partner's base identical, vary the strength of the pair being broken.

Declaration: m14b_partner_matched_declaration.json, frozen and pushed at 6a082284 BEFORE this file
existed (the fourteenth). Blind on the main arm; M14's four arms are not.

M14 died because the partner's OWN base identity was collinear with arm (C/G/U/A across the four
arms), so Y1 magnitudes were never comparable. This round removes that by construction:

  partner G : break G-C (n=3014) against break G-U (n=138)
  partner U : break A-U (n=1092) against break G-U (n=137)

Identical partner base => identical intrinsic reactivity, dynamic range and normalisation, so
chemistry predicts NO difference. Thermodynamics has G-C > A-U > G-U, so it predicts the stronger
pair hurts more.

The null is fixed too: scramble the pair-type LABEL inside a partner stratum, leaving molecule and
partner base untouched -- destroying the association under test rather than the baseline.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from m14_mutation_intervention import build, boot, DISTAL, DELTAS, DRAWS, PASS      # noqa: E402

R_SCRAM, NBOOT = 12, 2000
STRATA = (("partner_G", "G", "CG", "GU"), ("partner_U", "U", "AU", "GU"))


def pairtype(r):
    return "".join(sorted({r["b0"], r["bp"]})) if r["paired"] else None


def baselines(recs):
    """within-molecule baseline: mean delta at position j over OTHER mutants of the same parent
    whose mutation is unpaired and >= DISTAL nt from j."""
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["puzzle"], r["method"])].append(r)
    out = {}
    for k, rs in by.items():
        far = [r for r in rs if not r["paired"]]
        for r in rs:
            j = r["j"]
            if j is None:
                continue
            v = [x["mr"][j] - x["pr"][j] for x in far
                 if abs(x["i"] - j) >= DISTAL and x["mr"][j] is not None and x["pr"][j] is not None]
            out[(k, j)] = (float(np.mean(v)), len(v)) if v else (None, 0)
    return out


def collect(recs, base):
    rows = []
    for r in recs:
        if not r["paired"] or r["Y1"] is None:
            continue
        b, nb = base.get(((r["puzzle"], r["method"]), r["j"]), (None, 0))
        rows.append({"puzzle": r["puzzle"], "method": r["method"], "bp": r["bp"],
                     "pt": pairtype(r), "raw": r["Y1"],
                     "adj": (r["Y1"] - b) if b is not None else None, "nbase": nb,
                     "parent_j": r["pr"][r["j"]], "dECS": r["dECS"]})
    return rows


def diff_boot(A, B, seed, nb=NBOOT):
    """cluster bootstrap over puzzle of mean(A) - mean(B)."""
    def agg(rs):
        d = collections.defaultdict(lambda: [0.0, 0])
        for x, t in rs:
            d[t][0] += x; d[t][1] += 1
        return d
    da, db = agg(A), agg(B)
    T = sorted(set(da) | set(db))
    sa = np.array([da[t][0] for t in T]); ca = np.array([da[t][1] for t in T], float)
    sb = np.array([db[t][0] for t in T]); cb = np.array([db[t][1] for t in T], float)
    obs = float(sa.sum() / max(ca.sum(), 1) - sb.sum() / max(cb.sum(), 1))
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(T), size=(nb, len(T)))
    na, da2 = sa[ix].sum(1), ca[ix].sum(1)
    nbb, db2 = sb[ix].sum(1), cb[ix].sum(1)
    ok = (da2 > 0) & (db2 > 0)
    D = np.sort(na[ok] / da2[ok] - nbb[ok] / db2[ok])
    lo, hi = float(D[int(.025 * len(D))]), float(D[int(.975 * len(D))])
    return {"diff": round(obs, 4), "ci": [round(lo, 4), round(hi, 4)],
            "half": round((hi - lo) / 2, 4), "excl_pos": bool(lo > 0), "excl_neg": bool(hi < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m14b_partner_matched.json"))
    a = ap.parse_args()
    print("=== M14b:伙伴碱基逐字配平之后,拆更强的对是不是更伤 ===")
    print("  声明 6a082284(代码存在之前已推送)。主臂对我是盲的。离线 $0。")
    print("  化学预测:同一伙伴碱基 ⇒ 无差别。热力学预测:G-C > A-U > G-U。\n")
    recs, _p = build()
    base = baselines(recs)
    rows = collect(recs, base)
    nb_ok = [r["nbase"] for r in rows if r["adj"] is not None]
    print(f"配对位突变体 {len(rows)};可扣基线的 {len(nb_ok)}"
          f"(每条平均可用远端未配对突变体 {np.mean(nb_ok):.1f} 个)\n")

    OUT = {"strata": {}}
    for name, bp, strong, weak in STRATA:
        S = [r for r in rows if r["bp"] == bp and r["pt"] == strong]
        W = [r for r in rows if r["bp"] == bp and r["pt"] == weak]
        print(f"=== {name}(伙伴碱基 = {bp})===  拆 {strong} n={len(S)}   拆 {weak} n={len(W)}")
        if len(W) < 30:
            print("  弱对臂样本不足\n"); continue
        # shape_4: are the two groups starting from the same place?
        pjS = [r["parent_j"] for r in S if r["parent_j"] is not None]
        pjW = [r["parent_j"] for r in W if r["parent_j"] is not None]
        print(f"  形状四自检:父本 j 位反应性均值  拆{strong} {np.mean(pjS):+.4f}   "
              f"拆{weak} {np.mean(pjW):+.4f}   差 {np.mean(pjS)-np.mean(pjW):+.4f}")
        ent = {"n_strong": len(S), "n_weak": len(W),
               "parent_j_strong": round(float(np.mean(pjS)), 4),
               "parent_j_weak": round(float(np.mean(pjW)), 4)}
        for key, lab in (("raw", "扣前"), ("adj", "扣基线后")):
            A = [(r[key], r["puzzle"]) for r in S if r[key] is not None]
            B = [(r[key], r["puzzle"]) for r in W if r[key] is not None]
            if len(B) < 30:
                continue
            ba = boot([x for x, _t in A], [t for _x, t in A], a.seed)
            bb = boot([x for x, _t in B], [t for _x, t in B], a.seed)
            d = diff_boot(A, B, a.seed)
            ent[key] = {"strong": ba, "weak": bb, "diff": d}
            print(f"  {lab}: 拆{strong} {ba['mean']:+.4f}  拆{weak} {bb['mean']:+.4f}  "
                  f"**差 {d['diff']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]** "
                  f"半宽 {d['half']:.4f}  "
                  f"{'排除0' if d['excl_pos'] else ('负向排除0' if d['excl_neg'] else '含0')}")
        # P2: scramble the pair-type label inside the stratum
        allr = [(r["adj"] if r["adj"] is not None else r["raw"], r["puzzle"]) for r in S + W]
        nS = len(S); pos = neg = 0
        for t in range(R_SCRAM):
            rng = np.random.default_rng(5000 + a.seed + t)
            idx = rng.permutation(len(allr))
            A = [allr[q] for q in idx[:nS]]; B = [allr[q] for q in idx[nS:]]
            d = diff_boot(A, B, a.seed + t, nb=500)
            pos += d["excl_pos"]; neg += d["excl_neg"]
        ent["P2_scramble"] = {"pos": pos, "neg": neg}
        print(f"  P2 标签对调 {R_SCRAM} 次:正向排除0 {pos}  负向 {neg}  "
              f"{'通过' if pos <= 1 else '**不通过**'}")
        # shape_3: leave-one-puzzle-out on the difference
        A = [(r["adj"] if r["adj"] is not None else r["raw"], r["puzzle"]) for r in S]
        B = [(r["adj"] if r["adj"] is not None else r["raw"], r["puzzle"]) for r in W]
        pz = sorted({t for _x, t in B})
        loo = [diff_boot([x for x in A if x[1] != q], [x for x in B if x[1] != q],
                         a.seed, nb=200)["diff"] for q in pz if sum(1 for x in B if x[1] != q) >= 20]
        ent["loo_range"] = [round(min(loo), 4), round(max(loo), 4)] if loo else None
        print(f"  留一 puzzle 极差 {ent['loo_range']}  (弱对臂分布在 {len(pz)} 个 puzzle 上)")
        # P3 ladder AT THIS SAMPLE SIZE
        lad = {}
        for dd in DELTAS:
            hit = 0
            for t in range(DRAWS):
                rng = np.random.default_rng(int(dd * 1e4) + t + a.seed)
                ia = rng.integers(0, len(A), len(A)); ib = rng.integers(0, len(B), len(B))
                A2 = [A[q] for q in ia]
                B2 = [(B[q][0] - dd, B[q][1]) for q in ib]
                hit += diff_boot(A2, B2, a.seed + t, nb=300)["excl_pos"]
            lad[dd] = round(hit / DRAWS, 3)
        thr = next((d for d in DELTAS if lad[d] >= PASS), None)
        ent["ladder"] = lad; ent["threshold"] = thr
        print(f"  P3 梯级(在 n={len(B)} 上做):" +
              "  ".join(f"δ={d:.2f}:{lad[d]:.2f}" for d in DELTAS))
        print(f"     检出阈值 {thr}  顶档 {max(DELTAS)}\n")
        OUT["strata"][name] = ent

    # verdict
    ds = {k: v.get("adj", {}).get("diff") for k, v in OUT["strata"].items()}
    have = {k: v for k, v in ds.items() if v}
    p2ok = all(v["P2_scramble"]["pos"] <= 1 for v in OUT["strata"].values())
    if not p2ok:
        vd = "void_probe_failed"
    elif len(have) < 2:
        vd = "no_data"
    elif all(v["excl_pos"] for v in have.values()):
        vd = "pair_strength_ordering_holds"
    elif all(not v["excl_pos"] and not v["excl_neg"] for v in have.values()):
        vd = "no_strength_ordering"
    elif len({math.copysign(1, v["diff"]) for v in have.values()}) > 1:
        vd = "inconsistent"
    else:
        vd = "under_replicated"
    print("裁定:", vd)
    for k, v in have.items():
        t = OUT["strata"][k]["threshold"]
        print(f"  {k}: 差 {v['diff']:+.4f} 半宽 {v['half']:.4f};检出阈值 {t}"
              f"  => {'有功率' if (t is not None and abs(v['diff']) >= t) else '**功率不足,K2 的读法是【看不见】而不是【没有】**'}")
    OUT["verdict"] = vd
    json.dump(OUT, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
