#!/usr/bin/env python3
"""Audit A1/A4/A5b before the arbitration result is allowed to stand. Nothing here is a new claim.

WHY. Two results need to survive a code review before they mean anything:

    A4 FAILED, and failure was NOT expected -- "follow the more confident ruler" is the natural rule. A test
       that fails because it had no power is not a negative result, it is a missing measurement. The reported
       -1.5% [-10.7, +7.4] is uninterpretable until the rule's FLIP RATE is known: if `conf_learned >=
       conf_panel` holds on almost every pair, FOLLOW_CONFIDENT *is* FOLLOW_LEARNED and the difference is
       arithmetically pinned near zero.
    A5b's TOP BIN LANDED WHERE I WANTED IT. 75.1% [64.9, 82.8] is the number that turned a constant into a
       one-sided criterion, so it gets the harder look, not the easier one.

THREE WAYS THE TOP BIN COULD BE AN ARTEFACT, each testable:

    C3  WHICH SIDE MOVES. The bin variable is d = conf_learned - conf_panel. d rises when the learned ruler
        separates the pair more, AND when the panel separates it less. Those are different findings: "trust a
        confident learned ruler" vs "the panel's indifference is readable". The code never decomposed it.
        Fixed by binning on conf_learned alone and conf_panel alone, dev-fitted cuts, cluster CIs.
    C4  |dz| IS NOT CONFIDENCE, IT IS delta/sigma_cell. A cell where the panel is nearly constant has a tiny
        sigma, so a numerically negligible panel difference standardises into a huge "confidence". The panel
        column carries more tie/saturation structure than f1c (36.1% of designs sit at f1c = 1.000 exactly,
        and that block is EXCLUDED here because dr = 0, which biases what survives). Fixed by repeating
        everything with in-cell RANK differences, which are immune to sigma.
    C5  RAW MAGNITUDE PER BIN. If the top bin is where |delta panel| is numerically negligible, then the
        honest sentence is about the panel having no opinion, not about magnitude.

AND ONE ON A1 ITSELF:

    C1  boot_ci weights cells by their PAIR COUNT, and a 20-design cell contributes 190 pairs while a
        3-design cell contributes 3. So 57.6% could be a property of a few large cells. Reported next to the
        unweighted mean-of-cell-fractions, which gives every cell one vote.

C0 reproduces the published A1/A4/A5b numbers from the same inputs first; if C0 does not match, nothing below
is worth reading. COST: $0.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

TUBE_NOISE = 3.0
RNET_OPTIMISED = "Struct2SeQ-SHAPE"


def ranks01(vals):
    """In-cell average ranks scaled to [0,1]; immune to the cell's scale and to sigma blow-up."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = r / max(1, len(vals) - 1)
        i = j + 1
    return out


def cell_pairs(rows, margin):
    """Discordant pairs of one cell, carrying BOTH confidence definitions and the raw magnitudes."""
    if len(rows) < 3:
        return []
    stat = {}
    for key in ("panel", "f1c"):
        vals = [r[key] for r in rows]
        m, s = st.mean(vals), st.pstdev(vals)
        if s == 0:
            return []
        stat[key] = s
        rk = ranks01(vals)
        for r, q in zip(rows, rk):
            r["z_" + key] = (r[key] - m) / s
            r["rk_" + key] = q
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            x, y = rows[i], rows[j]
            dp, dr, dw = x["panel"] - y["panel"], x["f1c"] - y["f1c"], x["wet"] - y["wet"]
            if dp == 0 or dr == 0 or (dp > 0) == (dr > 0) or abs(dw) <= margin:
                continue
            out.append({"learned_right": (dr > 0) == (dw > 0),
                        "cl": abs(x["z_f1c"] - y["z_f1c"]), "cp": abs(x["z_panel"] - y["z_panel"]),
                        "rl": abs(x["rk_f1c"] - y["rk_f1c"]), "rp": abs(x["rk_panel"] - y["rk_panel"]),
                        "raw_p": abs(dp), "raw_r": abs(dr),
                        "sig_p": stat["panel"], "sig_r": stat["f1c"]})
    return out


def cluster(percell, draws, rng, unweighted=False):
    """(point, lo, hi) for P(learned_right); the resampled unit is the cell. Never the pair."""
    ks = list(percell)
    if not ks:
        return None, None, None
    def est(sel):
        if unweighted:
            fs = [sum(1 for p in percell[k] if p["learned_right"]) / len(percell[k]) for k in sel]
            return st.mean(fs) if fs else None
        num = sum(1 for k in sel for p in percell[k] if p["learned_right"])
        den = sum(len(percell[k]) for k in sel)
        return num / den if den else None
    pt = est(ks)
    out = [v for v in (est([ks[rng.randrange(len(ks))] for _ in ks]) for _ in range(draws)) if v is not None]
    out.sort()
    return pt, out[int(0.025 * len(out))], out[min(len(out) - 1, int(0.975 * len(out)))]


def binned(hold, dev, kl, kp, draws, rng, label, res):
    """Dev-fitted quartile cuts on (hold key), holdout accuracy per bin with cluster CIs.

    kl/kp name the two confidence columns; when kp is None the bin variable is kl alone.
    """
    def d(p):
        return p[kl] - p[kp] if kp else p[kl]
    dv = sorted(d(p) for v in dev.values() for p in v)
    cuts = [dv[len(dv) // 4], dv[len(dv) // 2], dv[3 * len(dv) // 4]]
    def binof(p):
        x = d(p)
        return 0 if x <= cuts[0] else (1 if x <= cuts[1] else (2 if x <= cuts[2] else 3))
    print(f"\n  [{label}]  开发半四分位边界 {[round(c, 3) for c in cuts]}")
    row = {"cuts_from_dev": cuts, "bins": []}
    for b in range(4):
        pc = {k: [p for p in v if binof(p) == b] for k, v in hold.items()}
        pc = {k: v for k, v in pc.items() if v}
        n = sum(len(v) for v in pc.values())
        if n < 20:
            print(f"    第{b+1}档 n={n} —— 太少,不给数")
            row["bins"].append({"n": n}); continue
        pt, lo, hi = cluster(pc, draws, rng)
        # C1's lesson applied to the bins themselves: a bin's pair-weighted fraction is a property of its
        # largest cells, so every bin is also reported with one vote per cell. A bin only counts if BOTH
        # weightings clear 0.5 -- the weaker of the two governs the claim.
        ptu, lou, hiu = cluster(pc, draws, random.Random(12345 + b), unweighted=True)
        allp = [p for v in pc.values() for p in v]
        cl = lo > 0.5 or hi < 0.5
        clu = lou > 0.5 or hiu < 0.5
        print(f"    第{b+1}档 n={n:5d} 格{len(pc):3d}  加权 {pt:.1%} [{lo:.1%},{hi:.1%}]{'✓' if cl else '跨0.5'}"
              f"   每格一票 {ptu:.1%} [{lou:.1%},{hiu:.1%}]{'✓' if clu else '跨0.5'}"
              f"   |Δ面板|中位 {st.median([p['raw_p'] for p in allp]):.4f}"
              f"  |Δ学习|中位 {st.median([p['raw_r'] for p in allp]):.4f}"
              f"  σ面板中位 {st.median([p['sig_p'] for p in allp]):.4f}")
        row["bins"].append({"n": n, "n_cells": len(pc), "frac": pt, "cluster_ci": [lo, hi],
                            "ci_clears_half": cl,
                            "frac_unweighted": ptu, "cluster_ci_unweighted": [lou, hiu],
                            "ci_clears_half_unweighted": clu,
                            "median_raw_dpanel": st.median([p["raw_p"] for p in allp]),
                            "median_raw_dlearned": st.median([p["raw_r"] for p in allp]),
                            "median_sigma_panel": st.median([p["sig_p"] for p in allp])})
    res[label] = row
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-key", default="panel_f1")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "arbitration_audit.json"))
    a = ap.parse_args()
    rng = random.Random(a.seed)

    from arbitration import load
    rec, spreads = load(a.panel_key)
    rec = [r for r in rec if r["method"] != RNET_OPTIMISED]
    groups = collections.defaultdict(list)
    for r in rec:
        groups[(r["puzzle"], r["method"])].append(r)
    cells = {k: v for k, v in ((k, cell_pairs(v, TUBE_NOISE)) for k, v in groups.items()) if v}
    npairs = sum(len(v) for v in cells.values())
    print(f"{len(rec)} 条分子;有相悖配对的格子 {len(cells)},配对 {npairs}")

    res = {"n_cells": len(cells), "n_pairs": npairs}

    # ---- C0 reproduce the published A1 ---------------------------------------------------------------
    pub = json.load(open(os.path.join(HERE, "arbitration.json")))
    a1 = [p for p in pub["points"] if abs(p["margin"] - TUBE_NOISE) < 1e-9][0]
    pt, lo, hi = cluster(cells, a.draws, random.Random(a.seed))
    ok0 = abs(pt - a1["frac_learned"]) < 5e-4 and npairs == a1["n_pairs"] and len(cells) == a1["n_cells"]
    print(f"\n=== C0 复现 A1 ===")
    print(f"  已发表 {a1['frac_learned']:.1%}  配对 {a1['n_pairs']}  格 {a1['n_cells']}")
    print(f"  本次   {pt:.1%}  配对 {npairs}  格 {len(cells)}   {'一致 ✓' if ok0 else '不一致 ✗ 下面不用看了'}")
    res["C0_reproduces_A1"] = bool(ok0)
    if not ok0:
        json.dump(res, open(a.out, "w"), indent=1); return 1

    # ---- C1 is A1 a few large cells? ----------------------------------------------------------------
    sizes = sorted((len(v) for v in cells.values()), reverse=True)
    ptu, lou, hiu = cluster(cells, a.draws, random.Random(a.seed), unweighted=True)
    top10 = sum(sizes[:10]) / npairs
    print(f"\n=== C1 A1 是不是几个大格子撑起来的 ===")
    print(f"  每格配对数 中位 {st.median(sizes):.0f}  最大 {sizes[0]}  前10格占全部配对 {top10:.1%}")
    print(f"  按配对加权(已发表口径) {pt:.1%} [{lo:.1%},{hi:.1%}]")
    print(f"  每格一票(未加权)       {ptu:.1%} [{lou:.1%},{hiu:.1%}]"
          f"{'  ✓ 同样越过 0.5' if lou > 0.5 else '  跨过 0.5 —— A1 依赖加权'}")
    res["C1"] = {"pairs_per_cell_median": st.median(sizes), "max": sizes[0], "top10_share": top10,
                 "weighted": [pt, lo, hi], "unweighted": [ptu, lou, hiu],
                 "unweighted_clears_half": bool(lou > 0.5)}

    # ---- split, exactly as A4 did -------------------------------------------------------------------
    def half(k):
        return int(hashlib.sha1(f"{a.seed}|{k[0]}|{k[1]}".encode()).hexdigest(), 16) % 2
    dev = {k: v for k, v in cells.items() if half(k) == 0}
    hold = {k: v for k, v in cells.items() if half(k) == 1}
    print(f"\n  开发半 {len(dev)} 格 / {sum(len(v) for v in dev.values())} 配对;"
          f"留出半 {len(hold)} 格 / {sum(len(v) for v in hold.values())} 配对")

    # ---- C2 A4's power ------------------------------------------------------------------------------
    print(f"\n=== C2 A4 到底有没有功效 ===")
    res["C2"] = {}
    for nm, kl, kp in (("z(已发表)", "cl", "cp"), ("秩(尺度免疫)", "rl", "rp")):
        flat = [p for v in hold.values() for p in v]
        flip = [p for p in flat if not (p[kl] >= p[kp])]          # where FOLLOW_CONFIDENT departs
        fr = len(flip) / len(flat)
        acc_learned_on_flip = sum(1 for p in flip if p["learned_right"]) / max(1, len(flip))
        gain = fr * (1 - 2 * acc_learned_on_flip)                 # exact algebraic gain of flipping there
        print(f"  {nm}: 规则改变决定的配对 {len(flip)}/{len(flat)} = {fr:.1%}")
        print(f"       在这些配对上「跟学习型」正确率 {acc_learned_on_flip:.1%}"
              f"  → 翻过去的代数增益 {gain:+.1%}")
        print(f"       该规则能拿到的最大增益(哪怕翻对100%) = {fr * (1 - acc_learned_on_flip):+.1%}"
              f" ;能造成的最大损失 = {-fr * acc_learned_on_flip:+.1%}")
        res["C2"][nm] = {"flip_rate": fr, "n_flip": len(flip),
                         "acc_follow_learned_on_flipped": acc_learned_on_flip, "algebraic_gain": gain,
                         "max_possible_gain": fr * (1 - acc_learned_on_flip)}

    # ---- C3/C4/C5 decompose the bin variable -------------------------------------------------------
    print(f"\n=== C3/C4/C5 顶档到底是谁在动(边界一律取自开发半)===")
    res["bins"] = {}
    r_zd  = binned(hold, dev, "cl", "cp", a.draws, rng, "z差 (已发表 A5b)", res["bins"])
    r_zl  = binned(hold, dev, "cl", None, a.draws, rng, "z 只看学习型自信", res["bins"])
    r_zp  = binned(hold, dev, "cp", None, a.draws, rng, "z 只看面板自信", res["bins"])
    r_rd  = binned(hold, dev, "rl", "rp", a.draws, rng, "秩差 (σ免疫版 A5b)", res["bins"])
    r_rl  = binned(hold, dev, "rl", None, a.draws, rng, "秩 只看学习型自信", res["bins"])
    r_rp  = binned(hold, dev, "rp", None, a.draws, rng, "秩 只看面板自信", res["bins"])

    def top(r):
        b = r["bins"][3]
        return b.get("frac"), b.get("ci_clears_half"), b.get("frac_unweighted"), \
            b.get("ci_clears_half_unweighted")
    print(f"\n=== 审计小结 ===")
    print(f"  A1 每格一票版本: {'站住' if res['C1']['unweighted_clears_half'] else '站不住'}")
    for nm, r in (("z差(已发表 A5b)", r_zd), ("秩差(σ免疫)", r_rd),
                  ("只看学习型自信 z", r_zl), ("只看面板自信 z", r_zp),
                  ("只看学习型自信 秩", r_rl), ("只看面板自信 秩", r_rp)):
        f_, c_, fu, cu = top(r)
        both = c_ and cu
        print(f"  {nm:20s} 顶档 加权 {f_:.1%}{'✓' if c_ else '跨'}  每格一票 {fu:.1%}{'✓' if cu else '跨'}"
              f"   {'两种加权都站住 ✓' if both else '只在一种加权下成立'}")
    res["survives_both_weightings"] = {
        nm: bool(top(r)[1] and top(r)[3]) for nm, r in
        (("z_diff", r_zd), ("rank_diff", r_rd), ("z_learned_only", r_zl), ("z_panel_only", r_zp),
         ("rank_learned_only", r_rl), ("rank_panel_only", r_rp))}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
