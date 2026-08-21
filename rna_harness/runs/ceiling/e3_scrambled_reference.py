#!/usr/bin/env python3
"""E3: is it THIS structure that matters, or any formable nested structure of the same size?

Pre-registered in e3_scrambled_reference_declaration.json (rule 38/39/40/41 spec).

WHAT E2 LEFT UNMEASURED. E2's equal-width placebo refuted my hypothesis -- arm accessibility carries nothing
a random position set does not. But adding ensemble_defect(skeleton) and pr_structure(skeleton) doubled the
treatment on all four populations while the placebo did not move, which pointed the signal at those two
target-REFERENCING columns. That was an inference: E2's placebo had no skeleton-referencing counterpart, so
its gate compared "arms plus skeleton" against "random positions" rather than "target reference" against
"non-target reference".

THIS RUN NARROWS THE TREATMENT TO THOSE TWO COLUMNS and gives the placebo a matched counterpart: the same two
quantities evaluated against a RANDOM nested structure the sequence can actually form, with the same pair
count as the target skeleton. Equal width by construction, matched on pair count and formability, unmatched
only on the one dimension under test -- which structure is being referenced.

FORMABILITY IS NOT DECORATION. A placebo structure containing non-canonical pairs would make pr_structure
identically zero for a trivial reason, the placebo would degenerate to a constant, and K2 would "pass" for a
stupid reason. Void gate 1 checks the placebo actually varies within cells and voids the run if it does not,
rather than reading a degenerate control as a pass.

COST: $0, offline. One fold per molecule serves both blocks.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from r1_info_ladder import r2_within, demean_by_cell                             # noqa: E402
from e1_ensemble_marginals import base_block, load_pop, CANON, FLOOR_CELLS       # noqa: E402

MIN_LOOP = 4


class Undefined(ValueError):
    """This molecule cannot supply a size-matched formable random nested structure."""


def random_formable_nested(seq, n_pairs, rng, tries=40000):
    """A random nested structure the SEQUENCE CAN FORM, with exactly n_pairs pairs.

    Greedy: draw (i, j), require canonical pairing, a minimum loop, both ends free, and no crossing with any
    pair already chosen. Raises rather than returning a smaller structure -- matching the pair count is what
    makes ensemble_defect and pr_structure comparable in scale between the two blocks.
    """
    n = len(seq)
    chosen = []
    used = set()
    for _ in range(tries):
        if len(chosen) == n_pairs:
            break
        i = rng.randrange(0, n - MIN_LOOP - 1)
        j = rng.randrange(i + MIN_LOOP + 1, n)
        if i in used or j in used:
            continue
        if frozenset({seq[i], seq[j]}) not in CANON:
            continue
        ok = True
        for (a, b) in chosen:
            if (a < i < b < j) or (i < a < j < b):        # crossing
                ok = False
                break
        if not ok:
            continue
        chosen.append((i, j))
        used.add(i)
        used.add(j)
    if len(chosen) != n_pairs:
        raise Undefined(f"only reached {len(chosen)}/{n_pairs} formable non-crossing pairs")
    s = ["."] * n
    for (i, j) in chosen:
        s[i], s[j] = "(", ")"
    return "".join(s)


def work(item):
    puz, seq, target = item
    import RNA
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from gamma_star import crossing_set
    try:
        if not crossing_set(target):
            return {"puzzle": puz, "error": "target has no crossing pairs"}
        skel = _round_only(target)
        sp = pk_pairs(skel)
        bad = [(i, j) for (i, j) in sp if frozenset({seq[i], seq[j]}) not in CANON]
        if bad:
            return {"puzzle": puz, "error": "undefined: skeleton not formable"}
        fc = RNA.fold_compound(seq)
        mfe_s, _mfe_e = fc.mfe()
        fc.pf()
        S = [fc.ensemble_defect(skel), fc.pr_structure(skel)]
        # THE PLACEBO, amended before any outcome was observed (see the declaration's amendment block).
        # The nested MFE, not a random formable structure: pr_structure of a RANDOM nested structure is
        # 1e-27..1e-35 against the target skeleton's 1e-7..3e-1, because a specific structure among
        # exponentially many has negligible probability unless it sits near the mode. That made the original
        # placebo degenerate BY CONSTRUCTION, and the target-vs-random comparison vacuous -- it would only
        # have said "a designed structure beats a random one". ensemble_defect(MFE) is additionally the exact
        # alternative hypothesis this run exists to control for: generic ensemble CONCENTRATION. And it is
        # conservative, since the MFE is the most probable structure, so the placebo is favoured.
        P = [fc.ensemble_defect(mfe_s), fc.pr_structure(mfe_s)]
        # the superseded block, kept as an observable where definable, never as a gate
        R = None
        try:
            rng = random.Random(int(hashlib.sha256((seq + "|E3").encode()).hexdigest()[:16], 16))
            rs = random_formable_nested(seq, len(sp), rng)
            R = [fc.ensemble_defect(rs), fc.pr_structure(rs)]
        except Undefined:
            pass
        cross = crossing_set(target); cs = set(cross); gc = 0
        for (i, j) in sorted(cs):
            if (i + 1, j - 1) in cs:
                for (x, y) in ((i, j), (i + 1, j - 1)):
                    if {seq[x], seq[y]} == {"G", "C"}:
                        gc += 1
        return {"puzzle": puz, "S": S, "S_mfe": P, "S_rand_superseded": R, "n_skel_pairs": len(sp),
                "mfe_is_skel": mfe_s == skel,
                "C": [float(gc), (seq.count("G") + seq.count("C")) / len(seq)],
                "ed_skel": S[0], "pr_skel": S[1], "ed_mfe": P[0], "pr_mfe": P[1],
                "ed_rand": (R[0] if R else None), "pr_rand": (R[1] if R else None)}
    except Undefined as exc:
        return {"puzzle": puz, "error": f"undefined: {exc}"}
    except Exception as exc:                                                     # noqa: BLE001
        return {"puzzle": puz, "error": repr(exc)[:140]}


def run_pop(nm, fn, nperm, draws, seed, workers):
    from leaderboard_design import targets
    T = targets()
    print(f"\n=== {nm} ===", flush=True)
    rec, _s, _m = load_pop(fn)
    keep = [r for r in rec if base_block(r) is not None]
    jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
    from concurrent.futures import ProcessPoolExecutor
    M = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, m in enumerate(ex.map(work, jobs, chunksize=8), 1):
            M.append(m)
            if i % 3000 == 0:
                print(f"    {i}/{len(jobs)}", flush=True)
    unf = sum(1 for m in M if m.get("error", "").startswith("undefined: skeleton"))
    nog = sum(1 for m in M if m.get("error", "").startswith("undefined: only"))
    other = sum(1 for m in M if "error" in m) - unf - nog
    print(f"  骨架不可形成(定义域外){unf};抽不出同尺寸可形成随机结构(抛出){nog};其他失败 {other}", flush=True)
    rows, cols = [], {}
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rr = dict(r)
        cols[id(rr)] = base_block(r) + m["C"] + m["S"] + m["S_mfe"]
        rr["_m"] = m
        rows.append(rr)
    if not rows:
        print("  无可用分子"); return {"no_data": True}
    print(f"  可用 {len(rows)}", flush=True)
    rr_ = [r for r in rows if r["_m"]["ed_rand"] is not None]
    print(f"  健全性:ensemble_defect 靶标 {st.median([r['_m']['ed_skel'] for r in rows]):.4f}"
          f" vs MFE {st.median([r['_m']['ed_mfe'] for r in rows]):.4f};"
          f"  pr 靶标 {st.median([r['_m']['pr_skel'] for r in rows]):.3e}"
          f" vs MFE {st.median([r['_m']['pr_mfe'] for r in rows]):.3e}", flush=True)
    print(f"  (被取代的随机块,仅观测 n={len(rr_)}):ed {st.median([r['_m']['ed_rand'] for r in rr_]):.4f}"
          f"  pr {st.median([r['_m']['pr_rand'] for r in rr_]):.3e}  <- 构造性退化,不作为闸门"
          if rr_ else "  (被取代的随机块无可定义分子)", flush=True)
    print(f"  MFE 恰好就是靶标骨架的分子数:{sum(1 for r in rows if r['_m']['mfe_is_skel'])}"
          f"/{len(rows)}  <- 这些分子上安慰剂与处理组重合,必须报出", flush=True)
    nB, nC, nS = len(base_block(keep[0])), 2, 2
    iB = list(range(nB)); iC = list(range(nB, nB + nC))
    iS = list(range(nB + nC, nB + nC + nS)); iR = list(range(nB + nC + nS, nB + nC + 2 * nS))
    assert len(iS) == len(iR) == 2, "void gate 2: blocks must be equal width"
    blocks = {"B": iB, "BC": iB + iC, "BCS": iB + iC + iS, "BCR": iB + iC + iR,
              "Sonly": iS, "Ronly": iR}     # iR now holds the MFE placebo
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["puzzle"], r["method"])].append(r)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 3]
    print(f"  格 {len(rws)}")
    if len(rws) < FLOOR_CELLS:
        print(f"  格数 < {FLOOR_CELLS} -> 无数据"); return {"no_data": True, "n_cells": len(rws)}
    # ---- void gate 1: the placebo must actually vary within cells ---------------------------------
    okc = 0
    for _c, g in rws:
        a = sorted(x["_m"]["ed_mfe"] for x in g); b = sorted(x["_m"]["pr_mfe"] for x in g)
        q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]                       # noqa: E731
        # amended: '> 0' was too weak -- 1e-34 and 1e-27 differ numerically and are both physically zero
        if (q(a, .75) - q(a, .25)) > 1e-4 and (q(b, .75) - q(b, .25)) > 1e-9:
            okc += 1
    frac = okc / len(rws)
    print(f"  闸门1 安慰剂在格内有方差的格占比 {frac:.3f}(要求 >= 0.90)"
          f"  {'通过' if frac >= 0.90 else '  VOID —— 退化的安慰剂不算通过'}", flush=True)
    if frac < 0.90:
        return {"void": True, "placebo_varying_cell_frac": frac, "n_cells": len(rws)}
    y, X, ci = demean_by_cell(rws, cols)
    rng = np.random.default_rng(seed)
    obs, nul = {}, {}
    for k, idx in blocks.items():
        Xa = X[:, idx]
        obs[k] = r2_within(y, Xa)
        vv = []
        for _ in range(nperm):
            yp = y.copy()
            for c in range(ci.max() + 1):
                m2 = ci == c
                yp[m2] = rng.permutation(yp[m2])
            vv.append(r2_within(yp, Xa))
        nul[k] = float(np.mean(vv))
    net = {k: obs[k] - nul[k] for k in blocks}
    viol = [(a, b) for a, b in (("B", "BC"), ("BC", "BCS"), ("BC", "BCR")) if obs[b] < obs[a] - 1e-12]
    print(f"  {'块':8}" + "".join(f"{k:>9}" for k in blocks))
    print(f"  {'净R2':8}" + "".join(f"{net[k]:9.4f}" for k in blocks))
    print(f"  闸门5 嵌套单调:违反 {len(viol)}")
    if viol:
        return {"estimator_broken": True, "violations": viol}
    prng = random.Random(seed)

    def inc(a, b, sel):
        yy, XX, _ = demean_by_cell(sel, cols)
        ra, rb = r2_within(yy, XX[:, blocks[a]]), r2_within(yy, XX[:, blocks[b]])
        return None if (ra is None or rb is None) else rb - ra
    offS, offR = nul["BCS"] - nul["BC"], nul["BCR"] - nul["BC"]
    ptS, ptR = inc("BC", "BCS", rws) - offS, inc("BC", "BCR", rws) - offR
    bsS, bsR, bsD = [], [], []
    for _ in range(draws):
        sel = [rws[prng.randrange(len(rws))] for _ in range(len(rws))]
        vS, vR = inc("BC", "BCS", sel), inc("BC", "BCR", sel)
        if vS is None or vR is None:
            continue
        bsS.append(vS - offS); bsR.append(vR - offR); bsD.append((vS - offS) - (vR - offR))

    def ci95(v):
        s = sorted(v)
        return [round(s[int(0.025 * len(s))], 4), round(s[min(len(s) - 1, int(0.975 * len(s)))], 4)]
    out = {"n_cells": len(rws), "n_mol": int(len(y)), "net": net, "placebo_varying_cell_frac": frac,
           "K1": {"delta": ptS, "ci": ci95(bsS)},
           "placebo": {"delta": ptR, "ci": ci95(bsR)},
           "K2_paired": {"delta": ptS - ptR, "ci": ci95(bsD)},
           "medians": {"ed_skel": st.median([r["_m"]["ed_skel"] for r in rows]),
                       "pr_skel": st.median([r["_m"]["pr_skel"] for r in rows]),
                       "ed_mfe": st.median([r["_m"]["ed_mfe"] for r in rows]),
                       "pr_mfe": st.median([r["_m"]["pr_mfe"] for r in rows]),
                       # the superseded random block is only defined on part of the population
                       "ed_rand_superseded": (st.median([r["_m"]["ed_rand"] for r in rr_]) if rr_ else None),
                       "pr_rand_superseded": (st.median([r["_m"]["pr_rand"] for r in rr_]) if rr_ else None),
                       "n_rand_definable": len(rr_)},
           "mfe_equals_skeleton": sum(1 for r in rows if r["_m"]["mfe_is_skel"]),
           "observables_not_gates": {"Sonly_net": net["Sonly"], "Ronly_net": net["Ronly"]}}
    for lab, pt, c in (("K1  净dR2(B+C+S - B+C)  靶标引用", ptS, ci95(bsS)),
                       ("    安慰剂(MFE) 净dR2(B+C+P - B+C)", ptR, ci95(bsR)),
                       ("K2  配对差 靶标引用 - MFE引用", ptS - ptR, ci95(bsD))):
        print(f"  {lab:38}{pt:+.4f}  CI [{c[0]:+.4f}, {c[1]:+.4f}]"
              f"{'  排除0' if (c[0] > 0 or c[1] < 0) else '  含0'}", flush=True)
    print(f"  (观测量,非闸门)S 单独净R2 {net['Sonly']:.4f}  vs  MFE引用单独 {net['Ronly']:.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "e3_scrambled_reference.json"))
    a = ap.parse_args()
    POPS = (("基准4560(主)", "panel_f1_designs.json"), ("样本3", "panel_f1_third.json"),
            ("第四批", "panel_f1_fourth.json"), ("样本2", "panel_f1_fresh.json"))
    res = {}
    for nm, fn in POPS:
        res[fn] = run_pop(nm, fn, a.nperm, a.draws, a.seed, a.workers)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("\n=== 判定(冻结的打死条件)===")
    P = res["panel_f1_designs.json"]
    if P.get("void") or P.get("no_data") or P.get("estimator_broken"):
        v = "void" if P.get("void") else ("no_data" if P.get("no_data") else "estimator_broken")
        print(f"  主人群 {v} —— 不报任何结论")
    else:
        def ex(d, k):
            c = d[k]["ci"]; return c[0] > 0 or c[1] < 0
        K1 = not ex(P, "K1")
        K2 = not ex(P, "K2_paired")
        reps = sum(1 for fn in ("panel_f1_third.json", "panel_f1_fourth.json", "panel_f1_fresh.json")
                   if not (res[fn].get("void") or res[fn].get("no_data")) and ex(res[fn], "K2_paired"))
        K3 = ex(P, "K2_paired") and reps < 2
        print(f"  K1 两根列无增量               :{'FIRES' if K1 else '不触发'}")
        print(f"  K2 引用靶标不比引用随机结构强 :{'FIRES' if K2 else '不触发'}")
        print(f"  K3 复现不足                   :{'FIRES' if K3 else '不触发'}  (复现 {reps}/3)")
        v = ("no_increment" if K1 else "reference_does_not_matter" if K2 else
             "underreplicated" if K3 else "target_reference_matters")
        P["kills"] = {"K1": K1, "K2": K2, "K3": K3, "replications": reps}
    print(f"  裁定:{v}")
    res["verdict"] = v
    res["boundary"] = "仍在 Turner 族内,不是独立判官,不修问题 (b);抬能力侧,不抬判官独立性。"
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
