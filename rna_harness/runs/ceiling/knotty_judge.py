#!/usr/bin/env python3
"""A FIFTH judge, chosen because it is the one that can KILL last night's result.

WHAT IT IS AIMED AT. `judge_paired.py` found that panel agreement predicts outside agreement (+36.1 points,
within-target paired), and then found that almost all of that comes from ONE panel member: ShapeKnots
confirmation predicts the outside judge at +42.7 points, while consensus without ShapeKnots is worth +5.3.
The shipping consequence was large -- ShapeKnots-confirmed reaches 95.5% precision at 50.9% recall against
panel-unanimity's 97.9% at 9.3%.

THE OBJECTION WE RAISED AGAINST OURSELVES. ShapeKnots (RNAstructure) and RNAPKplex (ViennaRNA) are different
packages, but they are the SAME KIND of algorithm: find helices, then assemble a pseudoknot from them, scored
on Turner-family energies. Two heuristics of one family agreeing is weaker evidence than it looks. None of
the numbers above can tell method kinship apart from joint correctness.

WHY KNOTTY, and not simply another binary. It differs on BOTH axes at once:

    ShapeKnots / RNAPKplex   heuristic helix assembly     Turner-family parameters
    Knotty                   exact sparsified DP over     HotKnots DP09 parameters
                             a broad pseudoknot class

so an agreement between ShapeKnots and Knotty cannot be attributed to a shared search heuristic OR to shared
energy parameters. (IPknot was the alternative -- probabilities + integer programming -- and is a reasonable
third if this one is inconclusive.)

PRE-REGISTERED, before Knotty judged a single sequence:

    P1  The ShapeKnots edge REPLICATES under the new judge: within-target paired, ShapeKnots-confirmed minus
        not-confirmed > 0 by Wilcoxon over at least 8 paired targets.
        KILL: not significant -> the edge was kinship with RNAPKplex, and the shipping recommendation in
        weekly-report 4.15 is RETRACTED, not softened.

    P2  The two outside judges are NOT redundant: Cohen's kappa between RNAPKplex and Knotty on the same
        sequences is < 0.8. If kappa is >= 0.8 they are one source wearing two names, and P1 passing would
        prove much less than it appears to. This is the measurement that the whole objection asks for, and
        it is reported whichever way it lands.

    P3  Panel unanimity replicates too (the headline of 4.15's primary test), same paired form.

    THE HONEST n IS PAIRED TARGETS, never designs -- the replication level this project has got wrong twice.

FAILING LOUD. The wrapper is self-tested on known answers BEFORE any judging, because four wrapper defects in
this project (ct2dot line concatenation, RNAduplex last-opener matching, bifold multi-structure blocks, a
missing ProbKnot --sequence) each produced a confident wrong number rather than an error. `None` from the
judge means IT FAILED TO RUN and is never counted as a rejection.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

KNOTTY_HOME = os.environ.get("KNOTTY_HOME", "/data/wenhui/pk_judges/Knotty")
KNOTTY = os.path.join(KNOTTY_HOME, "build", "knotty")
MIN_PAIRED = 8


def knotty_pairs(seq: str, timeout_s: float = 300.0):
    """Fold with the fifth judge and return its pair set. None means it failed to run, never a rejection.

    CONDA_PREFIX is cleared deliberately: Knotty's getParamPath() prefers $CONDA_PREFIX/share/simfold/params
    and only falls back to a path relative to the working directory. With a conda env active it reads a
    parameter file that is not there and dies -- loudly, which is why this is safe, but it must be handled.
    """
    from rna_harness.benchmarks.rnainvbench import pk_pairs
    if not os.path.exists(KNOTTY):
        return None
    env = dict(os.environ)
    env.pop("CONDA_PREFIX", None)
    try:
        r = subprocess.run([KNOTTY, seq], capture_output=True, text=True,
                           timeout=timeout_s, cwd=KNOTTY_HOME, env=env)
    except Exception:                                                          # noqa: BLE001
        return None
    for ln in (r.stdout or "").splitlines():
        if not ln.startswith("RES:"):
            continue
        body = ln[4:].strip().split()[0]
        if set(body) <= set(".()[]{}<>") and len(body) == len(seq):
            return pk_pairs(body)
    return None


def self_test() -> bool:
    """Known answers first. A wrapper that silently mis-parses is the failure mode this project actually has."""
    from rna_harness.instrument import InstrumentError, check_pair_set
    cases = [
        ("GGCGCGGCACCGUCCGCGGAACAAACGG", True, "known H-type pseudoknot -> crossing pairs expected"),
        ("GCAACGAUGACAUACAUCGCUAGUCGACGC", False, "the tool's own example -> nested, no crossing pairs"),
        ("A" * 28, False, "poly-A -> no pairs at all; a negative control that MUST be able to pass"),
    ]
    ok = True
    print("=== wrapper 自检(先验工具,再用它判任何东西)===")
    for seq, want_cross, why in cases:
        ps = knotty_pairs(seq)
        if ps is None:
            print(f"  ✗ 跑不动: {why}")
            ok = False
            continue
        try:                          # check_pair_set RAISES; it does not return a list of complaints
            check_pair_set(ps, len(seq))
            prob = ""
        except InstrumentError as e:
            prob = str(e)
        crossing = any(i < k < j < l for (i, j) in ps for (k, l) in ps)
        good = (crossing == want_cross) and not prob
        ok = ok and good
        print(f"  {'✓' if good else '✗'} {len(ps):2d} 对,交叉={crossing}(期望 {want_cross})"
              f"{'  结构检查: ' + prob if prob else ''}   {why}")
    dyn = len({len(knotty_pairs(s) or ()) for s, _, _ in cases})
    print(f"  {'✓' if dyn > 1 else '✗'} 动态范围:三个用例给出 {dyn} 种不同的配对数"
          f"(全相同则说明它对输入不敏感,任何下游结论都无效)")
    return ok and dyn > 1


def work(job):
    seq, timeout_s = job
    t0 = time.time()
    ps = knotty_pairs(seq, timeout_s)
    return seq, (None if ps is None else sorted(ps)), round(time.time() - t0, 2)


def cohen_kappa(a, b):
    """Agreement between two binary judges, corrected for agreement by chance."""
    n = len(a)
    if not n:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def paired(by, lab, hi, lo, quiet=False):
    """Within-target paired test. The cluster is the TARGET; several designs of one target are not
    several independent observations."""
    import numpy as np
    from scipy.stats import wilcoxon
    P = []
    for t, v in by.items():
        A = [d["ext"] for d in v if hi(d)]
        B = [d["ext"] for d in v if lo(d)]
        if A and B:
            P.append((t, st.mean(A), st.mean(B)))
    if len(P) < MIN_PAIRED:
        if not quiet:
            print(f"  {lab:34s} 只有 {len(P)} 个配对靶标 < {MIN_PAIRED},不检验")
        return {"label": lab, "n_paired": len(P), "insufficient": True}
    d = np.array([p[1] for p in P]) - np.array([p[2] for p in P])
    p = float(wilcoxon([x[1] for x in P], [x[2] for x in P], alternative="greater")[1])
    if not quiet:
        print(f"  {lab:34s} {len(P):3d} 靶标  差 {d.mean():+6.1%}  p={p:.3g}"
              f"   好{int((d > 0).sum())} 差{int((d < 0).sum())} 平{int((d == 0).sum())}")
    return {"label": lab, "n_paired": len(P), "gap": float(d.mean()), "p": p,
            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "judge_paired.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "knotty_judge.json"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()

    if not self_test():
        print("\n!! wrapper 自检未通过 —— 不判任何序列。先修 wrapper。")
        sys.exit(1)

    from rna_harness.benchmarks.rnainvbench import _round_only, load_pseudobase, pk_pairs
    D = json.load(open(a.src))
    tgt = {str(t.task_id): str(t.target) for t in load_pseudobase()}
    rows = [r for r in D["rows"] if not (r.get("error") or r.get("skipped"))]
    seqs = sorted({d["seq"] for r in rows for d in r.get("designs", []) if d["external_ran"]})
    print(f"\n=== 用第五裁判 Knotty 重判 {len(seqs)} 条唯一序列(来自 {len(rows)} 次设计)===")
    print(f"    RNAPKplex 与 ShapeKnots 同属【先找螺旋再拼】启发式;Knotty 是精确 DP + HotKnots DP09 参数\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    got, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, (s, a.timeout)) for s in seqs]
        for n, f in enumerate(as_completed(futs), 1):
            s, ps, _w = f.result()
            got[s] = ps
            if n % 500 == 0 or n == len(seqs):
                print(f"    {n}/{len(seqs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    n_fail = sum(1 for v in got.values() if v is None)
    print(f"  跑通 {len(got)-n_fail}/{len(got)}(失败 {n_fail} 条 —— 记为【未跑】,绝不当作否决)")

    # ---- assemble: each design now carries BOTH outside verdicts ----------------------------------------
    by, both = defaultdict(list), []
    for r in rows:
        target = tgt.get(r["task_id"])
        if not target:
            continue
        cross = pk_pairs(target) - pk_pairs(_round_only(target))
        for d in r.get("designs", []):
            if not d["external_ran"]:
                continue
            kp = got.get(d["seq"])
            if kp is None:
                continue
            ext = bool(set(map(tuple, kp)) >= cross)
            by[r["task_id"]].append({**d, "ext": ext})
            both.append((int(d["external"]), int(ext)))
    print(f"  两个外部裁判都判过的设计 {len(both)} 条,来自 {len(by)} 个靶标")

    # ---- P2 first: are the two outside judges one source wearing two names? -----------------------------
    A = [x[0] for x in both]
    B = [x[1] for x in both]
    k = cohen_kappa(A, B)
    print(f"\n=== P2 两个外部裁判是不是同一个来源 ===")
    print(f"  RNAPKplex 确认率 {st.mean(A):.1%}   Knotty 确认率 {st.mean(B):.1%}   "
          f"逐条一致 {sum(1 for x, y in both if x == y)/len(both):.1%}")
    print(f"  Cohen's kappa = {k:.3f}   -> {'两者高度冗余,P1 通过的说服力大打折扣' if k >= 0.8 else '不冗余,是两个可用的独立视角'}"
          f"  (预注册门槛 <0.8)")

    sk = lambda d: "shapeknots" in d["confirmed_by"]                            # noqa: E731
    print(f"\n=== P1 / P3 同靶标配对,标签换成 Knotty ===")
    r1 = paired(by, "P1 ShapeKnots 确认 vs 未确认", sk, lambda d: not sk(d))
    r3 = paired(by, "P3 面板一致(3家) vs 分裂", lambda d: d["k_panel"] == 3, lambda d: 1 <= d["k_panel"] <= 2)
    rc = paired(by, "对照 家数2vs1(都排除 SK)", lambda d: d["k_panel"] == 2 and not sk(d),
                lambda d: d["k_panel"] == 1 and not sk(d))

    print(f"\n=== 出厂规则在新裁判下的精确率/召回 ===")
    allv = [d for v in by.values() for d in v]
    pos = sum(d["ext"] for d in allv)
    ship = {}
    for lab, f in (("面板一致(现行)", lambda d: d["k_panel"] == 3), ("ShapeKnots 确认", sk)):
        hi = [d["ext"] for d in allv if f(d)]
        lo = [d["ext"] for d in allv if not f(d)]
        ship[lab] = {"n": len(hi), "precision": st.mean(hi) if hi else None,
                     "base": st.mean(lo) if lo else None, "recall": sum(hi) / max(1, pos)}
        print(f"  {lab:18s} 命中 {len(hi):5d} 条  精确率 {st.mean(hi):6.1%}(未命中 {st.mean(lo):6.1%})"
              f"  召回 {sum(hi)/max(1,pos):6.1%}")

    ok1 = bool(r1.get("p", 1) < 0.05 and r1.get("gap", 0) > 0)
    print(f"\n=== 预注册判决 ===")
    print(f"  P1 {'通过 —— ShapeKnots 的优势在一个算法类完全不同的裁判上复现,不是方法亲缘' if ok1 else '未通过 —— 撤回 4.15 的出厂规则建议'}")
    print(f"  P2 kappa={k:.3f} {'< 0.8,两个裁判不冗余' if k < 0.8 else '>= 0.8,两个裁判高度冗余'}")

    json.dump({"n_seqs": len(seqs), "n_failed": n_fail, "n_designs": len(both), "n_targets": len(by),
               "rate_pkplex": st.mean(A), "rate_knotty": st.mean(B), "kappa": k,
               "P1": r1, "P3": r3, "control": rc, "shipping": ship, "P1_passes": ok1,
               "knotty": {s: v for s, v in got.items()}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
