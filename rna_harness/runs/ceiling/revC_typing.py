#!/usr/bin/env python3
"""REVIEW C: is the "bounded number of types" proposal a real claim or a property of the threshold?

WHAT I SAID. Cluster the successful designs of a target; if the cluster count saturates as designs
accumulate, the unexplained variance is "k types" and a criterion only needs to name the type.

WHAT IS WRONG WITH IT AS STATED. Cluster count at a fixed distance threshold grows sublinearly for
almost any distribution with bounded support, so saturation alone is not evidence of anything. The
Szemeredi framing adds nothing: the regularity lemma is about graphs and its bounds are astronomical.
Two things have to be added before the claim means something, and both are here:

  (a) a NULL: the same clustering applied to random compatible sequences for the same target, which
      have no structure to find, must NOT saturate the same way;
  (b) RELEVANCE: cluster identity must predict the wet readout within the target, otherwise a bounded
      number of types is a fact about sequence space and not about the 76.3%.

Offline, $0.
"""
from __future__ import annotations
import argparse, collections, json, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import load_pop                                     # noqa: E402
from revA_lll_proxy import random_compatible                                   # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("样本3", "panel_f1_third.json"),
        ("第四批", "panel_f1_fourth.json"))


def ham(a, b):
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)


def nclusters(seqs, thr):
    """single linkage at thr, union-find."""
    p = list(range(len(seqs)))
    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            if len(seqs[i]) == len(seqs[j]) and ham(seqs[i], seqs[j]) <= thr:
                a, b = f(i), f(j)
                if a != b:
                    p[a] = b
    return len({f(i) for i in range(len(seqs))}), [f(i) for i in range(len(seqs))]


def eta2(lab, y):
    """fraction of within-target wet variance explained by cluster identity."""
    g = collections.defaultdict(list)
    for l, v in zip(lab, y):
        g[l].append(v)
    if len(g) < 2:
        return 0.0
    m = st.fmean(y); tot = sum((v - m) ** 2 for v in y)
    if tot <= 0:
        return 0.0
    bet = sum(len(v) * (st.fmean(v) - m) ** 2 for v in g.values())
    return bet / tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--min-designs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "revC_typing.json"))
    a = ap.parse_args()
    print("=== REVIEW C:『k 个类型』是真断言还是阈值的性质 ===")
    print("  声明:事后复核,不是预注册实验。加了两样我上一条漏掉的东西:零模型 与 相关性。\n")
    from leaderboard_design import targets as load_targets
    T = load_targets()
    OUT = {}
    for nm, fn in POPS:
        rec, _s, _m = load_pop(fn)
        by = collections.defaultdict(list)
        for r in rec:
            if r.get("seq") and r.get("wet") is not None:
                by[r["puzzle"]].append(r)
        rows = []
        print(f"=== {nm} ===")
        print(f"  {'靶标':<26s} {'设计':>4s} {'真簇':>5s} {'零簇':>5s} {'真/零':>6s} {'eta^2':>7s} {'零eta^2':>8s}")
        rng = random.Random(a.seed)
        for puz, g in sorted(by.items()):
            if len(g) < a.min_designs:
                continue
            g = sorted(g, key=lambda r: -r["wet"])[:max(a.min_designs, len(g) // 2)]
            seqs = [r["seq"] for r in g]; y = [float(r["wet"]) for r in g]
            if len({len(s) for s in seqs}) != 1:
                continue
            k, lab = nclusters(seqs, a.thr)
            tgt = T[puz]["target"]
            nl = [random_compatible(tgt, rng) for _ in range(len(seqs))]
            if len({len(s) for s in nl}) != 1 or len(nl[0]) != len(seqs[0]):
                nl = None
            k0, lab0 = (nclusters(nl, a.thr) if nl else (None, None))
            e = eta2(lab, y)
            e0 = eta2(lab0, y) if lab0 else None
            rows.append({"puzzle": puz, "n": len(seqs), "k": k, "k_null": k0,
                         "eta2": round(e, 4), "eta2_null": None if e0 is None else round(e0, 4)})
            print(f"  {puz:<26s} {len(seqs):4d} {k:5d} {('--' if k0 is None else k0):>5} "
                  f"{('--' if not k0 else f'{k/k0:.2f}'):>6} {e:7.4f} "
                  f"{('--' if e0 is None else f'{e0:.4f}'):>8}", flush=True)
        if rows:
            ks = [r["k"] / r["n"] for r in rows]
            es = [r["eta2"] for r in rows]
            e0s = [r["eta2_null"] for r in rows if r["eta2_null"] is not None]
            print(f"  簇数/设计数 中位 {st.median(ks):.3f}(1.0 = 每条设计自成一簇 = 没有类型)")
            print(f"  eta^2 中位 {st.median(es):.4f}   零模型 eta^2 中位 "
                  f"{('--' if not e0s else f'{st.median(e0s):.4f}')}")
            OUT[fn] = {"pop": nm, "rows": rows, "median_k_over_n": round(st.median(ks), 4),
                       "median_eta2": round(st.median(es), 4),
                       "median_eta2_null": None if not e0s else round(st.median(e0s), 4)}
        print()
    if OUT:
        kk = st.median([v["median_k_over_n"] for v in OUT.values()])
        ee = st.median([v["median_eta2"] for v in OUT.values()])
        nn = [v["median_eta2_null"] for v in OUT.values() if v["median_eta2_null"] is not None]
        v = ("no_typing__k_scales_with_n" if kk > 0.7 else
             "types_exist_but_do_not_predict_wet" if ee < 0.10 or (nn and ee <= st.median(nn) * 1.5) else
             "types_exist_and_predict_wet")
        print(f"=== 裁定 ===\n  **{v}**  (簇数/设计数 {kk:.3f},eta^2 {ee:.4f})")
        OUT["_summary"] = {"verdict": v, "median_k_over_n": kk, "median_eta2": ee}
    json.dump(OUT, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
