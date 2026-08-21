#!/usr/bin/env python3
"""E1: feed the pipeline ENSEMBLE MARGINALS instead of only functions of predicted pair sets.

Pre-registered in e1_ensemble_marginals_declaration.json (rule 38/39/40 spec).

WHY. All fifteen shipped columns plus f1c are functions of a PREDICTED PAIR SET -- the pipeline collapses the
Boltzmann ensemble into a point estimate at its second rung and does bookkeeping on the point estimate from
there. That is the one Deng move this project has never used: change the representation so the difficulty
becomes bookkeeping over the index set, and do NOT collapse the distribution. Today's failure is the cleanest
illustration -- gamma* took a single scalar out of the ensemble (F_nested, a free energy) and threw every
marginal away, which is why a count of G-C pairs matched it: a count is at least a local property of the
right positions, and a global free energy carries no positional information at all.

WHAT IT CANNOT DO, stated before the numbers: these columns are Turner-family, sharing the panel's energy
parameters, so they are not an independent judge. This targets capability (0.196 -> 0.811), not independence.

DOMAIN, PER COLUMN (rule 40). Five columns need only that positions exist and are defined on the whole
population. Two -- ensemble_defect and pr_structure against the target skeleton -- require the design to
actually FORM that skeleton with canonical pairs, which 22.5% of designs do not, so they are withheld. Both
variants are reported and a disagreement between them is a kill condition, not a footnote.

COST: $0, offline. pf() + bpp() is ~0.12 s at 240 nt. Shared box: default 10 workers, ViennaRNA is ~1 core
per worker with no fork multiplier (unlike pKiss's Perl wrapper or DesiRNA's 10 replicas).
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from arbitration_confirm import load_fresh                                       # noqa: E402
from r1_info_ladder import r2_within, demean_by_cell                             # noqa: E402

ORACLES = ("pkiss", "probknot", "shapeknots")
FINE = ("recall", "precision", "f1", "n_pred_cross", "n_pred_total")
CANON = {frozenset("GC"), frozenset("AU"), frozenset("GU")}
FLOOR_CELLS = 20
E_FREE = ("acc_arms", "acc_arms_min", "acc_arms_vs_bg", "entropy_cross", "mean_bp_dist")
E_SKEL = ("ens_defect_skel", "pr_skel")
TESTS = (("K1  净dR2(B+E - B)", "B", "BE"),
         ("K2  净dR2(B+C+E - B+C)", "BC", "BCE"),
         ("    净dR2(B+C - B) 对照自己的增量", "B", "BC"),
         ("K3  E 单独 vs C 单独(第36a条)", "Conly", "Eonly"))


def marginals(item):
    """Ensemble marginals for one design. E_FREE always; E_SKEL only when the skeleton is formable."""
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
        arms = sorted({x for p in cross for x in p})
        acc = [unp[i] for i in arms]
        bg = st.fmean(unp)
        ent = fc.positional_entropy()
        out = {"puzzle": puz, "L": n, "acc_arms": st.fmean(acc), "acc_arms_min": min(acc),
               "acc_arms_vs_bg": st.fmean(acc) - bg,
               "entropy_cross": st.fmean(ent[i + 1] for i in arms),
               "mean_bp_dist": fc.mean_bp_distance()}
        S = set(cross); gc = 0
        for (i, j) in sorted(S):
            if (i + 1, j - 1) in S:
                for (x, y) in ((i, j), (i + 1, j - 1)):
                    if {seq[x], seq[y]} == {"G", "C"}:
                        gc += 1
        out["gc_cross"] = float(gc)
        out["gc_frac"] = (seq.count("G") + seq.count("C")) / n
        skel = _round_only(target)
        bad = [(i, j) for (i, j) in pk_pairs(skel) if frozenset({seq[i], seq[j]}) not in CANON]
        if bad:
            out["skel_undefined"] = len(bad)
        else:
            out["ens_defect_skel"] = fc.ensemble_defect(skel)
            out["pr_skel"] = fc.pr_structure(skel)
        return out
    except Exception as exc:                                                     # noqa: BLE001
        return {"puzzle": puz, "error": repr(exc)[:140]}


def _dedup(rows, panel_key):
    """load_fresh's dedup, inline, so no temp file is written into the repo.

    Kept byte-compatible with arbitration_confirm.load_fresh: drop the RNet-optimised method, group by
    (puzzle, seq), average `wet` over replicates, assert the panel score agrees across replicates.
    """
    from arbitration_confirm import RNET_OPTIMISED
    rows = [r for r in rows if r["method"] != RNET_OPTIMISED]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["puzzle"], r["seq"])].append(r)
    out, spreads = [], []
    for _kk, cp in g.items():
        if len(cp) > 1:
            spreads.append(max(x["wet"] for x in cp) - min(x["wet"] for x in cp))
            assert len({round(x[panel_key], 9) for x in cp}) == 1, "same molecule, different panel score"
        r = dict(cp[0])
        r["wet"] = sum(x["wet"] for x in cp) / len(cp)
        r["panel"] = r[panel_key]
        out.append(r)
    return out, spreads, {}


def load_pop(fn):
    """load_fresh, but tolerate the base sample carrying no `seq`.

    panel_f1_designs.json (the 4,560 seed-0 sample) stores no sequence, while third/fourth/fresh do. The
    sequences come from replay_sample(0, 80) and the join is by ORDER -- element-wise asserted here on
    (puzzle, method, wet, f1c), the same assertion G2 and G5 ran with 0 mismatches. Never assumed.
    """
    path = os.path.join(HERE, fn)
    raw = json.load(open(path))["designs"]
    if raw and "seq" not in raw[0]:
        from novelty_stratified_ruler import replay_sample
        jobs, _ = replay_sample(0, 80)
        if len(jobs) != len(raw):
            raise RuntimeError(f"order join impossible: {len(jobs)} vs {len(raw)}")
        bad = [k for k, ((p, _s, f1c, wet, m), d) in enumerate(zip(jobs, raw))
               if p != d["puzzle"] or m != d["method"] or abs(wet - d["wet"]) > 1e-6
               or abs(f1c - d["f1c"]) > 1e-6]
        if bad:
            raise RuntimeError(f"order join broken at {bad[:5]}")
        print(f"  顺序连接断言({fn} 无 seq 列):{len(raw)} 条,不一致 0", flush=True)
        for (_p, sq, _f, _w, _m), d in zip(jobs, raw):
            d["seq"] = sq
        return _dedup(raw, "panel_recall_product")
    return load_fresh(path, "panel_recall_product")


def base_block(r):
    """B = F2 (k dummies + per-oracle recall_product + the fifteen fine columns) + f1c -- the 0.196 bar."""
    po = r.get("per_oracle") or {}
    if sorted(po) != sorted(ORACLES):
        return None
    rp = [po[o].get("recall_product") for o in ORACLES]
    f1c = r.get("f1c")
    if any(x is None for x in rp) or f1c is None:
        return None
    k = sum(1 for x in rp if x >= 1.0)
    B = [1.0 if k == j else 0.0 for j in (1, 2, 3)] + [float(x) for x in rp]
    for c in FINE:
        v = [po[o].get(c) for o in ORACLES]
        if any(x is None for x in v):
            return None
        B += [float(x) for x in v]
    return B + [float(f1c)]


def analyse(name, rows, cols_of, blocks, nperm, draws, seed):
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["puzzle"], r["method"])].append(r)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 3]
    print(f"    {name}: 分子 {len(rows)},格 {len(rws)}", flush=True)
    if len(rws) < FLOOR_CELLS:
        print(f"      格数 < {FLOOR_CELLS} -> 无数据"); return {"no_data": True, "n_cells": len(rws)}
    y, X, ci = demean_by_cell(rws, cols_of)
    NT = max(max(v) for v in blocks.values()) + 1
    dof = len(y) - NT - len(rws)
    if dof <= 0:
        print(f"      自由度 {dof} <= 0 -> 无数据"); return {"no_data": True, "dof": dof}
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
    print(f"      {'块':14}{'列数':>5}{'样本内R2':>10}{'置换零':>9}{'净R2':>9}")
    for nm in blocks:
        print(f"      {nm:14}{len(blocks[nm]):5d}{obs[nm]:10.4f}{nul[nm]:9.4f}{net[nm]:9.4f}")
    viol = [(a, b) for a, b in (("B", "BE"), ("B", "BC"), ("BC", "BCE")) if obs[b] < obs[a] - 1e-12]
    print(f"      自检(嵌套 ⇒ 样本内 R2 单调不减):违反 {len(viol)}" + ("" if not viol else f" {viol} 中止"))
    if viol:
        return {"estimator_broken": True, "violations": viol}
    prng = random.Random(seed)

    def inc(a, b, sel):
        yy, XX, _ = demean_by_cell(sel, cols_of)
        ra, rb = r2_within(yy, XX[:, blocks[a]]), r2_within(yy, XX[:, blocks[b]])
        return None if (ra is None or rb is None) else rb - ra
    out = {"n_cells": len(rws), "n_mol": int(len(y)), "dof": int(dof), "net": net, "r2": obs, "null": nul}
    for lab, a, b in TESTS:
        off = nul[b] - nul[a]
        pt = inc(a, b, rws) - off
        bs = []
        for _ in range(draws):
            sel = [rws[prng.randrange(len(rws))] for _ in range(len(rws))]
            v = inc(a, b, sel)
            if v is not None:
                bs.append(v - off)
        bs.sort()
        lo, hi = bs[int(0.025 * len(bs))], bs[min(len(bs) - 1, int(0.975 * len(bs)))]
        excl = lo > 0 or hi < 0
        print(f"      {lab:34}{pt:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]{'  排除0' if excl else '  含0'}", flush=True)
        out[f"{a}->{b}"] = {"delta": pt, "ci": [lo, hi], "excludes_zero": bool(excl), "positive": bool(lo > 0)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "e1_ensemble_marginals.json"))
    a = ap.parse_args()
    from leaderboard_design import targets
    T = targets()
    POPS = (("基准4560(主)", "panel_f1_designs.json"), ("样本3", "panel_f1_third.json"),
            ("第四批", "panel_f1_fourth.json"), ("样本2", "panel_f1_fresh.json"))
    from concurrent.futures import ProcessPoolExecutor
    res = {}
    for nm, fn in POPS:
        print(f"\n=== {nm} ===", flush=True)
        rec, _sp, _m = load_pop(fn)
        keep = [r for r in rec if base_block(r) is not None]
        jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
        M = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, m in enumerate(ex.map(marginals, jobs, chunksize=8), 1):
                M.append(m)
                if i % 2000 == 0:
                    print(f"    边际 {i}/{len(jobs)}", flush=True)
        err = sum(1 for m in M if "error" in m)
        nodef = sum(1 for m in M if "skel_undefined" in m)
        print(f"  失败 {err};骨架不可形成(E_SKEL 不予定义){nodef} = {nodef/max(1,len(M))*100:.1f}%", flush=True)
        rows_free, rows_full, cf, cu = [], [], {}, {}
        for r, m in zip(keep, M):
            if "error" in m:
                continue
            B = base_block(r); C = [m["gc_cross"], m["gc_frac"]]; Ef = [m[k] for k in E_FREE]
            r2 = dict(r); cf[id(r2)] = B + C + Ef; rows_free.append(r2)
            if "skel_undefined" not in m:
                r3 = dict(r); cu[id(r3)] = B + C + Ef + [m[k] for k in E_SKEL]; rows_full.append(r3)
        nB = len(base_block(keep[0])); nC = 2; nEf = len(E_FREE); nEs = len(E_SKEL)
        iB = list(range(nB)); iC = list(range(nB, nB + nC))
        iEf = list(range(nB + nC, nB + nC + nEf))
        iEs = list(range(nB + nC + nEf, nB + nC + nEf + nEs))
        blk_free = {"B": iB, "Conly": iC, "Eonly": iEf, "BC": iB + iC, "BE": iB + iEf, "BCE": iB + iC + iEf}
        blk_full = {"B": iB, "Conly": iC, "Eonly": iEf + iEs, "BC": iB + iC,
                    "BE": iB + iEf + iEs, "BCE": iB + iC + iEf + iEs}
        print(f"  --- E_free(5 列,全人群 n={len(rows_free)}) ---", flush=True)
        rf = analyse("E_free", rows_free, cf, blk_free, a.nperm, a.draws, a.seed)
        print(f"  --- E_full(7 列,限骨架可形成 n={len(rows_full)}) ---", flush=True)
        ru = analyse("E_full", rows_full, cu, blk_full, a.nperm, a.draws, a.seed)
        res[fn] = {"E_free": rf, "E_full": ru, "n_failed": err, "n_skel_undefined": nodef}
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== 判定(按冻结的打死条件,主人群)===")
    P = res["panel_f1_designs.json"]
    def fired(d, key):
        return d.get(key) is None or not d[key]["positive"]
    for var in ("E_free", "E_full"):
        d = P[var]
        if d.get("no_data") or d.get("estimator_broken"):
            print(f"  {var}: {'无数据' if d.get('no_data') else '估计量坏了'}"); continue
        k1, k2, k3 = fired(d, "B->BE"), fired(d, "BC->BCE"), fired(d, "Conly->Eonly")
        print(f"  {var}:  K1 集合边际无增量 {'FIRES' if k1 else '不触发'}"
              f" | K2 相对 GC 对照无增量 {'FIRES' if k2 else '不触发'}"
              f" | K3 E 单独不胜 C 单独 {'FIRES' if k3 else '不触发'}")
        d["kills"] = {"K1": k1, "K2": k2, "K3": k3}
    ka, kb = P["E_free"].get("kills"), P["E_full"].get("kills")
    if ka and kb:
        k4 = ka["K2"] != kb["K2"]
        print(f"  K4 两版结论相反(由那 22.5% 驱动):{'FIRES' if k4 else '不触发'}")
        P["K4_variants_disagree"] = k4
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
