#!/usr/bin/env python3
"""M31r: review M31 before its verdict is read, and ask the question K2 should have asked.

M31 returned reframe_supported on seven of seven staked predictions. It should not have been
read that way, and the fault is in the kill condition I wrote, not in the run.

  K2 compared the best model's top-k gain (+9.44 points) against a free helix-depth baseline
  (+8.59 points) as two point estimates with no interval on their difference. That is the same
  mistake this project retracted in M28: two numbers compared without asking whether the
  difference resolves. On the ranking statistic itself the two are indistinguishable --
  P_neural +0.32947 [0.271, 0.391] against depth +0.32252 [0.266, 0.374].

So "the model beats a free baseline" is not supported. But it is also not the right question.
Depth predicts rescue because deeper pairs are more often genuinely paired; an experimenter
holds depth for free and would use it. The question that decides the reframe is whether the
model adds anything ON TOP of depth. That is a partial rank correlation, not a comparison of
magnitudes, and it needs no fitting.

  R1  the paired interval K2 was missing: (model rho - depth rho), resampled together.
  R2  within-cell partial Spearman of predictor against rescue, GIVEN depth, on all 4317.
  R3  a top-k ranker that uses depth first and the model only to break its ties -- no fitting,
      no parameters -- against depth alone.
  R4  R2 repeated on the two control populations M30 and M26d already defined.

No new claim is made here. This decides whether M31's verdict may be read.
"""
from __future__ import annotations

import collections, csv, json, math, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26
import m31_which_flips_rescue as M31

NBOOT, NPERM = 2000, 2000
MODELS = (("P_vienna", "pmut"), ("P_neural", "nmut"))
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def cz(v):
    v = np.asarray(v, float) - np.mean(v)
    n = float(np.sqrt((v * v).sum()))
    return v / n if n > 0 else None


def partial_in_cell(rs, xkey, ykey="rf", zkey="depth"):
    """Spearman(x, y | z) inside one cell, on ranks. None when a column is constant."""
    x = cz(rankdata([r[xkey] for r in rs]))
    y = cz(rankdata([r[ykey] for r in rs]))
    z = cz(rankdata([r[zkey] for r in rs]))
    if x is None or y is None:
        return None
    if z is None:                      # depth constant here: nothing to partial out
        return float(x @ y)
    rxy, rxz, ryz = float(x @ y), float(x @ z), float(y @ z)
    d = math.sqrt(max(1 - rxz ** 2, 0.0) * max(1 - ryz ** 2, 0.0))
    return (rxy - rxz * ryz) / d if d > 1e-12 else None


def agg(cells, fn):
    num = den = 0.0
    for _k, rs in cells:
        v = fn(rs)
        if v is None or not np.isfinite(v):
            continue
        num += len(rs) * v; den += len(rs)
    return (num / den) if den else None


def boot(cells, fns, seed, nboot=NBOOT):
    """puzzle-clustered bootstrap, every statistic on the SAME resample."""
    pz = sorted({rs[0]["puzzle"] for _k, rs in cells})
    byp = collections.defaultdict(list)
    for k, rs in cells:
        byp[rs[0]["puzzle"]].append((k, rs))
    rng = np.random.default_rng(seed)
    out = collections.defaultdict(list)
    for _ in range(nboot):
        ix = rng.integers(0, len(pz), len(pz))
        sub = [c for q in ix for c in byp[pz[q]]]
        vals = {nm: agg(sub, fn) for nm, fn in fns.items()}
        for nm, v in vals.items():
            if v is not None:
                out[nm].append(v)
        for a, b in (("P_neural", "depth"), ("P_vienna", "depth"),
                     ("P_neural", "P_vienna")):
            if vals.get(a) is not None and vals.get(b) is not None:
                out[f"{a}-{b}"].append(vals[a] - vals[b])
    return {k: M31.boot_ci(v) for k, v in out.items()}


def main():
    t0 = time.time()
    log("=== M31r:先复查,再判 ===")
    log("  不产生新主张。决定 M31 的裁定能不能读。")
    log("  M31 的 K2 把『模型 +9.44 点 vs 免费深度基线 +8.59 点』当成两个点估计比了大小,")
    log("  没给差算区间 —— 这正是本项目在 M28 里撤回过的那个错误。\n")
    OUT = {}

    recs, _dr = M26.build()
    z = np.load(M31.NEURAL_CACHE, allow_pickle=True)
    pn = {k: v for k, v in zip(z["keys"], z["mats"])}
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    del pn, z
    cells = M31.make_cells(recs)
    log(f"  {len(recs)} 次干预,{len(cells)} 个设计格(与 M31 相同)\n")

    # ---------------- R1: the interval K2 was missing
    log("=== R1 K2 缺的那个区间:模型与免费基线之差,在同一次重抽样上算 ===")
    plain = {nm: (lambda rs, k=key: (lambda x, y: float(x @ y) if x is not None and y is not None
                                     else None)(cz(rankdata([r[k] for r in rs])),
                                                cz(rankdata([r["rf"] for r in rs]))))
             for nm, key in MODELS}
    plain["depth"] = lambda rs: (lambda x, y: float(x @ y) if x is not None and y is not None
                                 else None)(cz(rankdata([r["depth"] for r in rs])),
                                            cz(rankdata([r["rf"] for r in rs])))
    obs = {nm: agg(cells, fn) for nm, fn in plain.items()}
    ci = boot(cells, plain, seed=320)
    for nm in ("P_vienna", "P_neural", "depth"):
        log(f"  {nm:10s} rho {obs[nm]:+.5f}  CI {ci[nm]}")
    for pair in ("P_neural-depth", "P_vienna-depth"):
        a, b = pair.split("-")
        zero = ci[pair][0] is not None and ci[pair][0] <= 0 <= ci[pair][1]
        log(f"  差 {pair:18s} {obs[a]-obs[b]:+.5f}  CI {ci[pair]}  含 0 {zero}")
    K2r = bool(ci["P_neural-depth"][0] is not None
               and ci["P_neural-depth"][0] <= 0 <= ci["P_neural-depth"][1]
               and ci["P_vienna-depth"][0] <= 0 <= ci["P_vienna-depth"][1])
    log(f"  **K2 重判(两个模型都打不出免费基线之外)= {K2r}**")
    OUT["R1_vs_free_baseline"] = {"rho": {k: round(v, 5) for k, v in obs.items()},
                                  "ci": {k: v for k, v in ci.items()},
                                  "K2_reissued": K2r}

    # ---------------- R2: the question that actually decides it
    log("\n=== R2 在深度之上,模型还加不加得出东西(偏秩相关,不拟合任何参数)===")
    pfns = {nm: (lambda rs, k=key: partial_in_cell(rs, k)) for nm, key in MODELS}
    pobs = {nm: agg(cells, fn) for nm, fn in pfns.items()}
    pci = boot(cells, pfns, seed=321)
    # rule 80, with depth held: permute the outcome INSIDE (cell, depth) blocks
    perm = {}
    for nm, key in MODELS:
        rp = np.random.default_rng(322); ge = 0
        blocks = []
        for _k, rs in cells:
            by = collections.defaultdict(list)
            for r in rs:
                by[r["depth"]].append(r)
            blocks.append((rs, list(by.values())))
        for _ in range(NPERM):
            num = den = 0.0
            for rs, bl in blocks:
                shuffled = []
                for g in bl:
                    y = [r["rf"] for r in g]
                    y = [y[i] for i in rp.permutation(len(y))]
                    shuffled += [{**r, "rf": v} for r, v in zip(g, y)]
                v = partial_in_cell(shuffled, key)
                if v is not None and np.isfinite(v):
                    num += len(rs) * v; den += len(rs)
            if den and abs(num / den) >= abs(pobs[nm]):
                ge += 1
        perm[nm] = (ge + 1) / (NPERM + 1)
    for nm, _k in MODELS:
        zero = pci[nm][0] is not None and pci[nm][0] <= 0 <= pci[nm][1]
        log(f"  {nm:10s} 偏 rho {pobs[nm]:+.5f}  CI {pci[nm]}  含 0 {zero}  "
            f"置换 p {perm[nm]:.5f}(置换在(格,深度)块内做)")
    K5 = bool(all(pci[nm][0] is not None and pci[nm][0] <= 0 <= pci[nm][1]
                  for nm, _k in MODELS))
    log(f"  **K5(按住深度后模型什么也不剩 ⇒ 转向被否)= {K5}**")
    OUT["R2_partial_given_depth"] = {"rho": {k: round(v, 5) for k, v in pobs.items()},
                                     "ci": {k: pci[k] for k, _ in MODELS},
                                     "p_perm": {k: round(v, 5) for k, v in perm.items()},
                                     "K5_nothing_beyond_depth": K5}

    # ---------------- R3: a ranker that uses depth first, model only to break ties
    log("\n=== R3 深度优先、模型只用来打破深度的平手 —— 不拟合,不调参 ===")

    def recov(cells, keyfn, thresh=M31.THRESH_MAIN, frac=M31.TOPK_FRAC):
        num = den = 0.0
        for _k, rs in cells:
            y = np.array([r["rf"] >= thresh for r in rs])
            npos, n = int(y.sum()), len(rs)
            if npos == 0 or npos == n:
                continue
            kk = max(2, math.ceil(frac * n))
            order = keyfn(rs)
            num += npos * float(y[order[:kk]].sum()) / npos; den += npos
        return (num / den) if den else None

    def by_depth(rs):
        return np.lexsort((np.zeros(len(rs)), -np.array([r["depth"] for r in rs], float)))

    def by_depth_then(key):
        def f(rs):
            d = -np.array([r["depth"] for r in rs], float)
            m = -np.array([r[key] for r in rs], float)
            return np.lexsort((m, d))            # primary d, ties broken by m
        return f

    rfns = {"depth_only": by_depth}
    for nm, key in MODELS:
        rfns[f"depth_then_{nm}"] = by_depth_then(key)
    robs = {nm: recov(cells, fn) for nm, fn in rfns.items()}
    pz = sorted({rs[0]["puzzle"] for _k, rs in cells})
    byp = collections.defaultdict(list)
    for k, rs in cells:
        byp[rs[0]["puzzle"]].append((k, rs))
    rng = np.random.default_rng(323)
    dif = collections.defaultdict(list)
    for _ in range(NBOOT):
        ix = rng.integers(0, len(pz), len(pz))
        sub = [c for q in ix for c in byp[pz[q]]]
        base = recov(sub, rfns["depth_only"])
        for nm, _k in MODELS:
            v = recov(sub, rfns[f"depth_then_{nm}"])
            if v is not None and base is not None:
                dif[nm].append(v - base)
    for nm, _k in MODELS:
        c = M31.boot_ci(dif[nm])
        zero = c[0] is not None and c[0] <= 0 <= c[1]
        log(f"  深度优先 {robs['depth_only']:.5f} -> 用 {nm} 打破平手 "
            f"{robs[f'depth_then_{nm}']:.5f}   增 "
            f"{100*(robs[f'depth_then_{nm}']-robs['depth_only']):+.2f} 点  CI {c}  含 0 {zero}")
    OUT["R3_tiebreak"] = {"recovery": {k: round(v, 5) for k, v in robs.items()},
                          "ci_gain": {nm: M31.boot_ci(dif[nm]) for nm, _k in MODELS}}

    # ---------------- R4: the two control populations
    log("\n=== R4 在两套已有对照上重问 R2 ===")
    R4 = {}
    try:
        import m30_selection_audit as M30S
        pool, tested = M30S.read_pool(), M30S.read_tested()
        bs = collections.defaultdict(list)
        for p in pool:
            bs[p["design_sequence"]].append(p)
        okc = {k: bool(max(r["RNet_F1"] for r in bs[v]) >= M30S.CUT)
               for k, v in tested.items() if v in bs}
        fb = [(k, rs) for k, rs in cells if okc.get(k) is False]
        o = {nm: agg(fb, fn) for nm, fn in pfns.items()}
        c = boot(fb, pfns, seed=324, nboot=1000)
        for nm, _k in MODELS:
            log(f"  [fallback {len(fb)} 格] {nm:10s} 偏 rho {o[nm]:+.5f}  CI {c[nm]}")
        R4["selection_fallback"] = {"n_cells": len(fb),
                                    "rho": {k: round(v, 5) for k, v in o.items()},
                                    "ci": {k: c[k] for k, _ in MODELS}}
    except Exception as e:                                                      # noqa: BLE001
        log(f"  fallback 取用失败 {repr(e)[:70]}"); R4["selection_fallback"] = {"void": repr(e)[:200]}
    try:
        import m26c_leakage as LEAK, m26d_clean_subset as M26D
        train = set()
        for rid in LEAK.TRAIN_DEPOSITS:
            u = LEAK.rdat_url(rid)
            if u:
                train.update(LEAK.stream_sequences(u, lambda *a: None))
        csv.field_size_limit(10 ** 9)
        win = {}
        for d in csv.DictReader(open(M26.SRC)):
            try:
                win[(d["puzzle"], d["method"])] = (int(float(d["sub_start"])) - 1,
                                                   int(float(d["sub_end"])))
            except (ValueError, KeyError):
                continue
        q = {k: rs[0]["wt_seq"][win[k][0]:win[k][1]] for k, rs in cells if k in win}
        best, _p = M26D.max_identity(q, train, lambda *a: None)
        leaked = {k for k, v in best.items() if v >= M26D.THR}
        cl = [(k, rs) for k, rs in cells if k not in leaked]
        o = {nm: agg(cl, fn) for nm, fn in pfns.items()}
        c = boot(cl, pfns, seed=325, nboot=1000)
        for nm, _k in MODELS:
            log(f"  [去泄漏 {len(cl)} 格] {nm:10s} 偏 rho {o[nm]:+.5f}  CI {c[nm]}")
        R4["leakage_clean"] = {"n_cells": len(cl), "n_leaked": len(leaked),
                               "rho": {k: round(v, 5) for k, v in o.items()},
                               "ci": {k: c[k] for k, _ in MODELS}}
    except Exception as e:                                                      # noqa: BLE001
        log(f"  去泄漏取用失败 {repr(e)[:70]}"); R4["leakage_clean"] = {"void": repr(e)[:200]}
    OUT["R4_controls"] = R4

    OUT["verdict"] = ("reframe_refused_nothing_beyond_depth" if K5 else
                      "reframe_supported_as_increment_over_depth")
    log(f"\n  重判 {OUT['verdict']}   ({time.time()-t0:.0f}s)")
    log("  注:M31 押的第 5 条『打得过深度基线』按 R1 重判;转向的依据改成 R2/R3,")
    log("      即『在深度之上还加得出东西』,而不是『比深度强』。")
    json.dump(OUT, open(os.path.join(HERE, "m31r_review.json"), "w"), ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m31r_review.log"), "w").write("\n".join(LOG) + "\n")
    log("写出 m31r_review.json / .log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
