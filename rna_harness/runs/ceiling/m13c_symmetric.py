#!/usr/bin/env python3
"""M13c shape_2: the one disclosed loosening, and it is a SYMMETRISATION, not a knob turn.

The frozen declaration forbids judging K1 when the C arm gets under a fifth of the P arm's pairs,
and requires either a single disclosed loosening of the two arms' COMMON stringency or a verdict of
`underpowered_or_inseparable`. The first run had P/C pair ratios of 4.6 to 25.5.

The asymmetry is structural, not a tuning choice. gc_paired and the GC-pair fraction correlate 0.91,
so demanding a raw gap of 0.05 costs the C arm about 2.4 residual SDs and the P arm about 0.9. A
common raw delta is therefore NOT a common stringency.

The symmetrisation: express each arm's delta in units of ITS OWN residual SD given the variable that
arm holds fixed, and give both arms the same k. Nothing else changes. k is swept and every k is
reported, so there is no k to pick after the fact.

COST: $0, offline.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from m12_sequence_intrinsic import parse_pairs                                        # noqa: E402
from m13_cache import POPS                                                            # noqa: E402
from m13c_pairing_vs_composition import (assemble, match, point, null_dist, pval,      # noqa: E402
                                         cluster_ci, MIN_PAIRS, MIN_TGT, MAX_SMD, P_GATE)

KS = (0.4, 0.6, 0.8, 1.0)


def resid_sd(D, key, hold):
    """SD of treat after removing its within-cell linear dependence on the held variable."""
    num = den = 0.0; res = []
    for c in D["cells"]:
        t, h = c[key] - c[key].mean(), c[hold] - c[hold].mean()
        num += float((t * h).sum()); den += float((h * h).sum())
    b = num / den if den else 0.0
    for c in D["cells"]:
        t, h = c[key] - c[key].mean(), c[hold] - c[hold].mean()
        res.append(t - b * h)
    return float(np.std(np.concatenate(res))) or 1.0, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "m13c_symmetric.json"))
    a = ap.parse_args()
    print("=== M13c 形状二:对称化后的两臂(声明允许的唯一一次放宽,且是对称化) ===")
    print("  两臂 delta 一律取 k × 【该臂处理列在配平其持有变量之后的残差 SD】。k 全程扫描,全部报出。\n")
    from leaderboard_design import targets as load_targets
    T = load_targets(); PM = {p: sorted(parse_pairs(d["target"])) for p, d in T.items()}
    C = json.load(open(os.path.join(HERE, "m13_features.json")))
    E = json.load(open(os.path.join(HERE, "m13b_extra.json")))
    D = {fn: assemble(C[fn]["rows"], E[fn], PM, a.seed) for _nm, fn in POPS}
    for nm, fn in POPS:
        sp, bp = resid_sd(D[fn], "pair", "gcp"); sc, bc = resid_sd(D[fn], "gcp", "pair")
        print(f"  {nm:12s} 残差SD:P臂(pair|gcp) {sp:.4f}   C臂(gcp|pair) {sc:.4f}   比 {sp/sc:.2f}")
    print()
    OUT = {}
    for k in KS:
        print(f"===== k = {k} 残差SD =====")
        row = {}
        for nm, fn in POPS:
            d = D[fn]; ent = {}
            for lab, key, hold in (("P", "pair", "gcp"), ("C", "gcp", "pair")):
                sd, _b = resid_sd(d, key, hold)
                delta = k * sd
                M = match(d, [c[key] for c in d["cells"]], delta, hold)
                if M is None or M["npair"] < 10:
                    ent[lab] = None; continue
                obs = point(M); g = float(M["gap"].mean())
                null = null_dist(d, key, delta, hold, a.seed, a.nperm)
                smd = max(abs(float(M["dgcf"].mean())) / d["sd_gcf"],
                          abs(float(M["dpB"].mean())) / d["sp"],
                          abs(float(M["hgap"].mean())) / ({"gcp": d["sd_gcp"], "pair": d["sd_pair"]}[hold]))
                ent[lab] = {"delta": round(delta, 4), "effect": round(obs, 3), "gap": round(g, 5),
                            "slope": round(obs / g, 2), "p": round(pval(obs, null), 5),
                            "z": round((obs - float(null.mean())) / (float(null.std()) or 1), 2),
                            "npair": M["npair"], "ntgt": M["ntgt"], "smd": round(smd, 4),
                            "ci": cluster_ci(M, a.seed),
                            "usable": bool(M["npair"] >= MIN_PAIRS and M["ntgt"] >= MIN_TGT
                                           and smd <= MAX_SMD)}
            row[fn] = {"pop": nm, **ent}
            s = []
            for lab in ("P", "C"):
                r = ent[lab]
                s.append(f"{lab}: " + ("—" if r is None else
                         f"斜率{r['slope']:+7.2f} 效应{r['effect']:+6.2f} p{r['p']:.4f} "
                         f"n{r['npair']:5d}/{r['ntgt']:3d} SMD{r['smd']:.3f}"
                         f"{'' if r['usable'] else '(不可用)'}"))
            print(f"  {nm:12s} " + "   ".join(s))
        usP = [f for f in row if row[f]["P"] and row[f]["P"]["usable"]]
        usC = [f for f in row if row[f]["C"] and row[f]["C"]["usable"]]
        nP = sum(1 for f in usP if row[f]["P"]["p"] < P_GATE)
        nC = sum(1 for f in usC if row[f]["C"]["p"] < P_GATE)
        mP = float(np.median([row[f]["P"]["slope"] for f in usP])) if usP else None
        mC = float(np.median([row[f]["C"]["slope"] for f in usC])) if usC else None
        rp = [row[f]["P"]["npair"] for f in row if row[f]["P"]]
        rc = [row[f]["C"]["npair"] for f in row if row[f]["C"]]
        pr = float(np.median(rp)) / max(1.0, float(np.median(rc))) if rp and rc else None
        print(f"    -> 可用 P {len(usP)}/6 C {len(usC)}/6;p<{P_GATE}: P {nP} C {nC};"
              f"中位斜率 P {mP} C {mC};配对数比 P/C {pr:.2f}" if pr else "")
        OUT[str(k)] = {"pops": row, "usable_P": len(usP), "usable_C": len(usC),
                       "sig_P": nP, "sig_C": nC, "median_slope_P": mP, "median_slope_C": mC,
                       "npair_ratio": pr}
        print()
    json.dump(OUT, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
