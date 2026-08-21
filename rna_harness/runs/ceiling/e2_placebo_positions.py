#!/usr/bin/env python3
"""E2: E1's result, judged by a gate that matches what E1 said it was testing.

Pre-registered in e2_placebo_positions_declaration.json (rule 38/39/40/41 spec).

WHY THIS EXISTS. E1's K1 and K2 excluded zero on all four populations in both variants -- sixteen of
sixteen -- and its own K3 fired, because K3 compared E's STANDALONE strength against a trivial GC count while
the declared purpose was incremental contribution. A block that is weak alone but adds on top of B+C is
complementary information, not a rider: a rider adds nothing there, and E added +0.012 to +0.034. Rather than
re-read E1 on my own say-so -- the fourth mis-specified kill condition today, and each one noticed only when
the result was favourable -- E1 keeps its verdict and this run re-judges with the gate written correctly.

K3 IS NOT DELETED, IT IS DEMOTED, and replaced by the question it was gesturing at and never asked: is the
increment specific to the PSEUDOKNOT ARMS, or does any ensemble summary of a floppy molecule do as well?
E_rand computes the identical five statistics over a RANDOM position set of the same size, seeded
deterministically from the sequence. If E and E_rand add the same, the signal is "this ensemble is loose",
which has nothing to do with a pseudoknot. Same pattern as J1's directional negative control.

acc_arms_vs_bg was an ADJUSTMENT, not a control -- it cannot answer "would the same statistic computed
elsewhere do as well". This can.

COST: $0, offline. Two pf() passes are not needed -- one fold serves both blocks.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from r1_info_ladder import r2_within, demean_by_cell                             # noqa: E402
from e1_ensemble_marginals import base_block, load_pop, E_FREE, E_SKEL, CANON, FLOOR_CELLS  # noqa: E402


class Undefined(ValueError):
    """This molecule cannot supply a placebo position set of the required size."""


def stats_at(unp, ent, pos, bg):
    """The five E_FREE statistics evaluated over an arbitrary position set (0-based)."""
    acc = [unp[i] for i in pos]
    return [st.fmean(acc), min(acc), st.fmean(acc) - bg, st.fmean(ent[i + 1] for i in pos), None]


def marginals2(item):
    """E (crossing arms) and E_rand (placebo positions of the same count) from ONE fold."""
    puz, seq, target = item
    import RNA
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from gamma_star import crossing_set
    try:
        cross = crossing_set(target)
        if not cross:
            return {"puzzle": puz, "error": "target has no crossing pairs"}
        n = len(seq)
        fc = RNA.fold_compound(seq)
        fc.pf()
        bpp = fc.bpp()
        paired = [0.0] * (n + 2)
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                p = bpp[i][j]
                if p:
                    paired[i] += p
                    paired[j] += p
        unp = [max(0.0, min(1.0, 1.0 - paired[i])) for i in range(1, n + 1)]
        ent = fc.positional_entropy()
        bg = st.fmean(unp)
        mbd = fc.mean_bp_distance()
        arms = sorted({x for p in cross for x in p})
        # ---- domain gate: the placebo needs the same NUMBER of distinct positions -------------------
        if len(arms) >= n:
            raise Undefined(f"crossing arms {len(arms)} >= L {n}; cannot draw a distinct placebo set")
        rng = random.Random(int(hashlib.sha256(seq.encode()).hexdigest()[:16], 16))
        rpos = sorted(rng.sample(range(n), len(arms)))
        E = stats_at(unp, ent, arms, bg); E[4] = mbd
        R = stats_at(unp, ent, rpos, bg); R[4] = mbd
        S = set(cross); gc = 0
        for (i, j) in sorted(S):
            if (i + 1, j - 1) in S:
                for (x, y) in ((i, j), (i + 1, j - 1)):
                    if {seq[x], seq[y]} == {"G", "C"}:
                        gc += 1
        out = {"puzzle": puz, "L": n, "n_arms": len(arms), "E": E, "E_rand": R,
               "C": [float(gc), (seq.count("G") + seq.count("C")) / n],
               "acc_arms": E[0], "acc_rand": R[0]}
        skel = _round_only(target)
        bad = [(i, j) for (i, j) in pk_pairs(skel) if frozenset({seq[i], seq[j]}) not in CANON]
        if bad:
            out["skel_undefined"] = len(bad)
        else:
            out["E_skel"] = [fc.ensemble_defect(skel), fc.pr_structure(skel)]
        return out
    except Undefined as exc:
        return {"puzzle": puz, "error": f"undefined: {exc}"}
    except Exception as exc:                                                     # noqa: BLE001
        return {"puzzle": puz, "error": repr(exc)[:140]}


def run_variant(name, rows, cols_of, blocks, nperm, draws, seed):
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["puzzle"], r["method"])].append(r)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 3]
    print(f"    {name}: 分子 {len(rows)},格 {len(rws)}", flush=True)
    if len(rws) < FLOOR_CELLS:
        return {"no_data": True, "n_cells": len(rws)}
    y, X, ci = demean_by_cell(rws, cols_of)
    NT = X.shape[1]
    if len(y) - NT - len(rws) <= 0:
        return {"no_data": True, "dof": len(y) - NT - len(rws)}
    rng = np.random.default_rng(seed)
    obs, nul = {}, {}
    for nm, idx in blocks.items():
        Xa = X[:, idx]
        obs[nm] = r2_within(y, Xa)
        vv = []
        for _ in range(nperm):
            yp = y.copy()
            for c in range(ci.max() + 1):
                m = ci == c
                yp[m] = rng.permutation(yp[m])
            vv.append(r2_within(yp, Xa))
        nul[nm] = float(np.mean(vv))
    net = {k: obs[k] - nul[k] for k in blocks}
    viol = [(a, b) for a, b in (("B", "BE"), ("B", "BC"), ("BC", "BCE"), ("BC", "BCR"))
            if a in blocks and b in blocks and obs[b] < obs[a] - 1e-12]
    print(f"      {'块':8}" + "".join(f"{k:>9}" for k in blocks))
    print(f"      {'净R2':8}" + "".join(f"{net[k]:9.4f}" for k in blocks))
    print(f"      自检(嵌套单调):违反 {len(viol)}")
    if viol:
        return {"estimator_broken": True, "violations": viol}
    prng = random.Random(seed)

    def inc(a, b, sel):
        yy, XX, _ = demean_by_cell(sel, cols_of)
        ra, rb = r2_within(yy, XX[:, blocks[a]]), r2_within(yy, XX[:, blocks[b]])
        return None if (ra is None or rb is None) else rb - ra
    out = {"n_cells": len(rws), "n_mol": int(len(y)), "net": net, "r2": obs, "null": nul}
    # K1, K2, the demoted standalone comparison, and the PAIRED placebo difference
    offE, offR = nul["BCE"] - nul["BC"], nul["BCR"] - nul["BC"]
    ptE, ptR = inc("BC", "BCE", rws) - offE, inc("BC", "BCR", rws) - offR
    bsE, bsR, bsD, bs1 = [], [], [], []
    off1 = nul["BE"] - nul["B"]
    for _ in range(draws):
        sel = [rws[prng.randrange(len(rws))] for _ in range(len(rws))]
        vE, vR, v1 = inc("BC", "BCE", sel), inc("BC", "BCR", sel), inc("B", "BE", sel)
        if vE is None or vR is None or v1 is None:
            continue
        bsE.append(vE - offE); bsR.append(vR - offR); bsD.append((vE - offE) - (vR - offR)); bs1.append(v1 - off1)

    def ci95(v):
        s = sorted(v)
        return [round(s[int(0.025 * len(s))], 4), round(s[min(len(s) - 1, int(0.975 * len(s)))], 4)]
    rows_out = (("K1  净dR2(B+E - B)", inc("B", "BE", rws) - off1, ci95(bs1)),
                ("K2  净dR2(B+C+E - B+C)", ptE, ci95(bsE)),
                ("    安慰剂 净dR2(B+C+E_rand - B+C)", ptR, ci95(bsR)),
                ("K3new 配对差 E - E_rand", ptE - ptR, ci95(bsD)))
    for lab, pt, c in rows_out:
        excl = c[0] > 0 or c[1] < 0
        print(f"      {lab:36}{pt:+.4f}  CI [{c[0]:+.4f}, {c[1]:+.4f}]{'  排除0' if excl else '  含0'}", flush=True)
    out["K1"] = {"delta": inc("B", "BE", rws) - off1, "ci": ci95(bs1)}
    out["K2"] = {"delta": ptE, "ci": ci95(bsE)}
    out["placebo"] = {"delta": ptR, "ci": ci95(bsR)}
    out["K3new_paired"] = {"delta": ptE - ptR, "ci": ci95(bsD)}
    # demoted observable: standalone strength, reported not gated
    out["demoted_Eonly_vs_Conly"] = {"E_only_net": net["Eonly"], "C_only_net": net["Conly"]}
    print(f"      (降级为观测量)E 单独净R2 {net['Eonly']:.4f}  vs  C 单独 {net['Conly']:.4f}"
          f"   —— 第 41 条:这条不再是打死条件")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "e2_placebo_positions.json"))
    a = ap.parse_args()
    from leaderboard_design import targets
    T = targets()
    POPS = (("基准4560(主)", "panel_f1_designs.json"), ("样本3", "panel_f1_third.json"),
            ("第四批", "panel_f1_fourth.json"), ("样本2", "panel_f1_fresh.json"))
    from concurrent.futures import ProcessPoolExecutor
    res = {}
    for nm, fn in POPS:
        print(f"\n=== {nm} ===", flush=True)
        rec, _s, _m = load_pop(fn)
        keep = [r for r in rec if base_block(r) is not None]
        jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
        M = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, m in enumerate(ex.map(marginals2, jobs, chunksize=8), 1):
                M.append(m)
                if i % 3000 == 0:
                    print(f"    边际 {i}/{len(jobs)}", flush=True)
        err = [m for m in M if "error" in m]
        undef = sum(1 for m in err if m["error"].startswith("undefined"))
        print(f"  失败 {len(err)}(其中定义域抛出 {undef});骨架不可形成 "
              f"{sum(1 for m in M if 'skel_undefined' in m)}", flush=True)
        acc_a = [m["acc_arms"] for m in M if "error" not in m]
        acc_r = [m["acc_rand"] for m in M if "error" not in m]
        print(f"  安慰剂健全性:交叉臂平均可及性 中位 {st.median(acc_a):.4f} vs 随机位置 {st.median(acc_r):.4f}"
              f"  —— 两者若几乎相同,K3new 事前就没有分辨空间", flush=True)
        rowsF, rowsU, cF, cU = [], [], {}, {}
        for r, m in zip(keep, M):
            if "error" in m:
                continue
            B = base_block(r)
            rf = dict(r); cF[id(rf)] = B + m["C"] + m["E"] + m["E_rand"]; rowsF.append(rf)
            if "skel_undefined" not in m:
                ru = dict(r); cU[id(ru)] = B + m["C"] + m["E"] + m["E_skel"] + m["E_rand"]; rowsU.append(ru)
        nB, nC, nE, nEs = len(base_block(keep[0])), 2, len(E_FREE), len(E_SKEL)
        iB = list(range(nB)); iC = list(range(nB, nB + nC))
        iE = list(range(nB + nC, nB + nC + nE))
        blkF = {"B": iB, "Conly": iC, "Eonly": iE, "BC": iB + iC, "BE": iB + iE, "BCE": iB + iC + iE,
                "BCR": iB + iC + list(range(nB + nC + nE, nB + nC + 2 * nE))}
        iEs = list(range(nB + nC + nE, nB + nC + nE + nEs))
        iR2 = list(range(nB + nC + nE + nEs, nB + nC + 2 * nE + nEs))
        blkU = {"B": iB, "Conly": iC, "Eonly": iE + iEs, "BC": iB + iC, "BE": iB + iE + iEs,
                "BCE": iB + iC + iE + iEs, "BCR": iB + iC + iR2}
        print(f"  --- E_free(全人群 n={len(rowsF)}) ---", flush=True)
        rf = run_variant("E_free", rowsF, cF, blkF, a.nperm, a.draws, a.seed)
        print(f"  --- E_full(限骨架可形成 n={len(rowsU)}) ---", flush=True)
        ru = run_variant("E_full", rowsU, cU, blkU, a.nperm, a.draws, a.seed)
        res[fn] = {"E_free": rf, "E_full": ru, "n_failed": len(err), "n_domain_undefined": undef,
                   "acc_arms_median": st.median(acc_a), "acc_rand_median": st.median(acc_r)}
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== 判定(冻结的打死条件)===")
    P = res["panel_f1_designs.json"]
    def ex(d, k):
        c = d[k]["ci"]; return c[0] > 0 or c[1] < 0
    verdicts = {}
    for var in ("E_free", "E_full"):
        d = P[var]
        if d.get("no_data") or d.get("estimator_broken"):
            print(f"  {var}: 不可用"); continue
        K1 = not ex(d, "K1"); K2 = not ex(d, "K2"); K3 = not ex(d, "K3new_paired")
        reps = sum(1 for fn in ("panel_f1_third.json", "panel_f1_fourth.json", "panel_f1_fresh.json")
                   if not res[fn][var].get("no_data") and ex(res[fn][var], "K2"))
        K5 = ex(d, "K2") and reps < 2
        print(f"  {var}: K1 无增量 {'FIRES' if K1 else '不触发'} | K2 相对GC对照无增量 {'FIRES' if K2 else '不触发'}"
              f" | K3new 无假结特异性 {'FIRES' if K3 else '不触发'} | K5 复现不足 {'FIRES' if K5 else '不触发'}"
              f" (复现 {reps}/3)")
        verdicts[var] = ("dead" if K1 else "subsumed_by_GC" if K2 else
                        "not_pseudoknot_specific" if K3 else "underreplicated" if K5 else "survives")
        d["kills"] = {"K1": K1, "K2": K2, "K3new": K3, "K5": K5, "replications": reps}
    K4 = len(set(verdicts.values())) > 1
    print(f"  K4 两版结论相反:{'FIRES' if K4 else '不触发'}   裁定 {verdicts}")
    res["verdicts"] = verdicts; res["K4_variants_disagree"] = K4
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
