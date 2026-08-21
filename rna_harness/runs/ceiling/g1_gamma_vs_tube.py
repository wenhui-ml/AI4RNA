#!/usr/bin/env python3
"""G1: does gamma* predict in-vitro outcome -- and does it beat a trivial stability prior?

Pre-registered in g1_gamma_vs_tube_declaration.json. The kill conditions there are frozen.

THE REAL QUESTION is not "is gamma* correlated with the tube score". gamma* = F_nested - E_skeleton -
dG_stack, so a design with a weak intended skeleton gets a large gamma* automatically -- gamma* could be a
proxy for generic misfolding propensity with nothing pseudoknot-specific in it. This project has already been
caught by exactly that once (retrieval warm-start was NULL against a generic GC-stiffness prior). So three
NESTED predictors are compared, each removing one more term:

    M0  F_nested                                  trivial stability prior
    M1  F_nested - E_skeleton                     the nested-world margin, NO pseudoknot term
    M2  F_nested - E_skeleton - dG_stack = gamma*  the full construction

M2 - M1 is the only place the pseudoknot-specific term can show up. If that difference brackets zero, the
pseudoknot content of the construction is not measurable, whatever M2's own correlation looks like.

WITHIN TARGET, always: the target sets both gamma*'s scale (7.5 for P01, 43.6 for P19) and the tube score's.
Cluster bootstrap over TARGETS, mean not median (rule 35: with a few dozen clusters a median CI gets pinned
by ties). Paired differences drawn from the SAME resample.

COST: $0, local ViennaRNA only. Needs the `rna` env. ~4 min at 8 workers; gamma* is ~0.05 s per molecule and
this box is shared, so the default is deliberately small (the 160-core cap is a hard ceiling, and ViennaRNA
pf() is ~1 core per worker with no fork multiplier).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

NOISE_FLOOR = 2.300          # IQR(resid_A) measured by G3. Frozen input, not re-derived here.
GAMMA_BIAS = 2.4             # gamma* is over-stated by this much (g3_audit.md section five).


def work(item):
    puz, seq, target, wet = item
    import RNA
    from gamma_star import crossing_set, stack_terms
    from rna_harness.benchmarks.rnainvbench import _round_only
    try:
        cross = crossing_set(target)
        if not cross:
            return {"puzzle": puz, "error": "target has no crossing pairs"}
        f = RNA.fold_compound(seq).pf()[1]
        e = RNA.energy_of_struct(seq, _round_only(target))
        dg = sum(v for _, v in stack_terms(seq, cross))
    except Exception as exc:                                                   # noqa: BLE001
        return {"puzzle": puz, "error": repr(exc)[:120]}
    return {"puzzle": puz, "seq": seq, "wet": wet, "L": len(seq),
            "M0": round(f, 3), "M1": round(f - e, 3), "M2": round(f - e - dg, 3),
            "F_nested": round(f, 3), "E_skeleton": round(e, 3), "dG_stack": round(dg, 3)}


def spearman(x, y):
    """Spearman rho with mid-ranks for ties. None when either side has no variance."""
    n = len(x)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    a, b = ranks(x), ranks(y)
    ma, mb = st.fmean(a), st.fmean(b)
    va = sum((p - ma) ** 2 for p in a)
    vb = sum((q - mb) ** 2 for q in b)
    if va == 0 or vb == 0:
        return None
    return sum((p - ma) * (q - mb) for p, q in zip(a, b)) / (va * vb) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-target", type=int, default=80)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--floor", type=int, default=20, help="minimum valid designs per target")
    ap.add_argument("--out", default=os.path.join(HERE, "g1_gamma_vs_tube.json"))
    a = ap.parse_args()

    from leaderboard_design import targets
    from novelty_stratified_ruler import replay_sample
    T = targets()
    jobs_raw, _ = replay_sample(a.seed, a.per_target)
    jobs = [(p, s, T[p]["target"], wet) for p, s, _f1c, wet, _m in jobs_raw]
    print(f"{len({j[0] for j in jobs})} 个靶标,{len(jobs)} 条设计,{a.workers} 进程", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    R = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(work, jobs, chunksize=16), 1):
            R.append(r)
            if n % 800 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}", flush=True)

    bad = [r for r in R if "error" in r]
    ok = [r for r in R if "error" not in r]
    print(f"\n  void gate 2 -- 成功 {len(ok)},失败/无交叉 {len(bad)}"
          + (f" ({collections.Counter(r['error'] for r in bad).most_common(1)[0][0]})" if bad else ""))

    # ---- void gate 1: identity against the stored gamma* record ---------------------------------------
    rec = json.load(open(os.path.join(HERE, "gamma_star_reproduce.json")))["rows"]
    from gamma_star import best_wet_per_target, crossing_set
    best = best_wet_per_target()
    mism = []
    for p, want in rec.items():
        seq = best[p]["seq"]
        got = work((p, seq, T[p]["target"], 0.0))
        if "error" in got or abs(got["M2"] - want["gamma_star"]) > 0.011:
            mism.append((p, got.get("M2"), want["gamma_star"]))
    print(f"  void gate 1 -- 对已存 γ* 记录的恒等校验:{len(rec)} 行,不一致 {len(mism)}")
    if mism:
        print("\n  VOID: 重算的 γ* 与已存记录不一致。不报任何结果。")
        json.dump({"verdict": "void", "identity_mismatches": mism}, open(a.out, "w"), indent=1)
        return 1

    by = collections.defaultdict(list)
    for r in ok:
        by[r["puzzle"]].append(r)
    small = {p: len(v) for p, v in by.items() if len(v) < a.floor}
    keep = {p: v for p, v in by.items() if len(v) >= a.floor}
    print(f"  每靶标地板 {a.floor}:保留 {len(keep)} 个靶标,剔除 {len(small)} 个 {small or ''}")

    tot = sum(len(v) for v in keep.values())
    dom = {p: round(len(v) / tot, 3) for p, v in keep.items() if len(v) / tot > 0.15}
    print(f"  void gate 4 -- 单靶标占比 > 15% 的:{dom or '无'}")

    # ---- the gate G3's audit says does not transfer ----------------------------------------------------
    iqrs = []
    for p, v in keep.items():
        g = sorted(r["M2"] for r in v)
        m = len(g)
        iqrs.append(g[min(m - 1, int(0.75 * m))] - g[min(m - 1, int(0.25 * m))])
    med_iqr = st.median(iqrs)
    below = med_iqr < NOISE_FLOOR
    print(f"\n  靶标内 IQR(γ*) 中位 = {med_iqr:.3f}  vs G3 噪声底 {NOISE_FLOOR:.3f}"
          f"  -> {'在噪声底之下' if below else '在噪声底之上'}")
    print(f"    (每靶标 IQR 分布:min {min(iqrs):.2f}  p25 {sorted(iqrs)[len(iqrs)//4]:.2f} "
          f" med {med_iqr:.2f}  p75 {sorted(iqrs)[3*len(iqrs)//4]:.2f}  max {max(iqrs):.2f})")

    # ---- per-target Spearman for each nested model -----------------------------------------------------
    per = {}
    nodef = collections.Counter()
    for p, v in sorted(keep.items()):
        wet = [r["wet"] for r in v]
        row = {"n": len(v)}
        for m in ("M0", "M1", "M2"):
            rho = spearman([r[m] for r in v], wet)
            if rho is None:
                nodef[m] += 1
            row[m] = None if rho is None else round(rho, 4)
        per[p] = row
    print(f"  void gate 3 -- Spearman 无定义(预测量在该靶标内无方差)的靶标数:{dict(nodef) or '无'}")

    usable = [p for p, r in per.items() if all(r[m] is not None for m in ("M0", "M1", "M2"))]
    print(f"  三个模型都有定义的靶标:{len(usable)}/{len(per)}")

    def means(ps):
        return {m: st.fmean(per[p][m] for p in ps) for m in ("M0", "M1", "M2")}

    obs = means(usable)
    rng = random.Random(12345)
    B = {k: [] for k in ("M0", "M1", "M2", "M2-M1", "M1-M0")}
    for _ in range(a.boot):
        s = [usable[rng.randrange(len(usable))] for _ in usable]
        mm = means(s)
        for m in ("M0", "M1", "M2"):
            B[m].append(mm[m])
        B["M2-M1"].append(mm["M2"] - mm["M1"])
        B["M1-M0"].append(mm["M1"] - mm["M0"])

    def ci(v):
        s = sorted(v)
        return [round(s[int(0.025 * len(s))], 4), round(s[int(0.975 * len(s))], 4)]

    pooled = {m: round(spearman([r[m] for r in ok if r["puzzle"] in keep],
                                [r["wet"] for r in ok if r["puzzle"] in keep]) or 0.0, 4)
              for m in ("M0", "M1", "M2")}

    print(f"\n{'模型':30} {'靶标内均值 ρ':>13}  {'cluster bootstrap 95%':>24}   合池ρ(混杂)")
    lbl = {"M0": "M0  F_nested(平凡稳定性)", "M1": "M1  F−E_skel(嵌套余量)",
           "M2": "M2  γ*(完整构造)"}
    for m in ("M0", "M1", "M2"):
        c = ci(B[m])
        print(f"{lbl[m]:30} {obs[m]:+13.4f}  [{c[0]:+.4f}, {c[1]:+.4f}]{'':>6}   {pooled[m]:+.4f}")
    for d in ("M2-M1", "M1-M0"):
        c = ci(B[d])
        excl = (c[0] > 0) or (c[1] < 0)
        print(f"{'  差值 ' + d:30} {st.fmean(B[d]):+13.4f}  [{c[0]:+.4f}, {c[1]:+.4f}]"
              f"{'':>6}   {'排除 0' if excl else '含 0'}")

    # ---- frozen kill conditions -----------------------------------------------------------------------
    cM2, c21, c10 = ci(B["M2"]), ci(B["M2-M1"]), ci(B["M1-M0"])
    K1 = not (cM2[0] > 0 or cM2[1] < 0)
    K2 = not (c21[0] > 0 or c21[1] < 0)
    K3 = not (c10[0] > 0 or c10[1] < 0)
    print(f"\n  K1 γ* 无预测力(区间含 0)          : {'FIRES' if K1 else '不触发'}")
    print(f"  K2 dG_stack 无贡献(M2−M1 含 0)    : {'FIRES' if K2 else '不触发'}")
    print(f"  K3 嵌套余量不胜平凡稳定性(M1−M0)  : {'FIRES' if K3 else '不触发'}")
    if obs["M2"] < 0:
        print("  注意:方向与跑前预测的【正号】相反。这需要解释,不许改口叙述。")

    verdict = ("dead" if K1 else
               "no_pseudoknot_content" if K2 else
               "restated_stability_prior" if K3 else "survives")
    print(f"\n  裁定:{verdict}"
          + ("(并且整个检验处在噪声底之下)" if below else ""))

    json.dump({"verdict": verdict, "kills": {"K1": K1, "K2": K2, "K3": K3},
               "within_target_mean_rho": obs, "ci": {k: ci(v) for k, v in B.items()},
               "diff_means": {d: round(st.fmean(B[d]), 4) for d in ("M2-M1", "M1-M0")},
               "pooled_rho_CONFOUNDED": pooled,
               "within_target_iqr_gamma": {"median": round(med_iqr, 3), "noise_floor": NOISE_FLOOR,
                                           "below_noise_floor": below,
                                           "all": [round(x, 3) for x in sorted(iqrs)]},
               "gamma_bias_from_g3": GAMMA_BIAS,
               "n_targets": len(usable), "n_designs": tot, "per_target": per,
               "excluded_small": small, "dominant_targets": dom,
               "spearman_undefined": dict(nodef), "n_failed": len(bad),
               "identity_mismatches": mism, "boot": a.boot},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
