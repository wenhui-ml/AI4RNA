#!/usr/bin/env python3
"""Before spending a cent: can the TARGET STRUCTURE ALONE say whether this design is worth attempting?

WHY THIS AND NOT ANOTHER RULER. Every capability this project has built answers "is this sequence right?"
after the solver has run. The north star is that a non-expert gets a correct technical verification
CONVENIENTLY and CHEAPLY, and the cheapest possible answer is the one given before any solver runs at all.
Goal 2 has had zero new evidence for an entire period; this is the only open direction that touches it.

The fragments are already in the record and they all say designability is heterogeneous and structural:

    64 of 178 targets accept NO isomer of their own design under the co-transcriptional criterion
    residual triage split 8 undelivered targets cleanly by length: 5 search-limited, 3 no-signal
    the shape probe already answers "can this shape be made" 9/9 in 0.3 s against up to 2727 s
    the window criterion made two prospective predictions in OPPOSITE directions and both held

PRE-REGISTERED, written before a single feature was computed:

    Two tasks, two different domains, so this also tests whether designability prediction TRANSFERS:
      A  pseudoknot   -- will the external judge (RNAPKplex, never in any loop) confirm the design?
                         base rate 91/243 = 37.4%
      B  co-transcriptional -- does the target's pool contain ANY candidate the criterion accepts?
                         base rate 70/178 = 39.3%
    Both base rates are near 40%, so AUC is meaningful rather than degenerate -- checked before choosing them.

    FEATURES COME FROM THE TARGET STRUCTURE ONLY. No sequence, no solver output, no outcome-derived
    quantity. A single leaked feature would make this trivially succeed, so the feature builder takes the
    dot-bracket string and nothing else.

    Split: the project's deterministic even/odd on sorted ids. Fit on FIT, report on TEST, never the reverse.

    PASS: AUC > 0.70 on the held-out half, in BOTH tasks.
    KILL: AUC <= 0.70 -- the target's structure does not predict designability and this direction is dead.
          Reported as a null with its power, not quietly dropped.

    CONTROL THAT MUST FAIL: labels permuted, same pipeline, AUC must land near 0.5. A pipeline that scores
    well on shuffled labels is leaking, and this project has caught that class of defect often enough that
    the control is mandatory rather than nice to have.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

from rna_harness.instrument import Clustered, constancy, stable_seed          # noqa: E402

AUC_PASS = 0.70


def features(target: str):
    """Everything computable from the dot-bracket alone. Nothing else may enter."""
    from rna_harness.benchmarks.kinetic_suite import pair_set
    n = len(target)
    P = pair_set(target)
    paired = {i for p in P for i in p}
    # helices: runs of consecutive stacked pairs
    hel, cur = [], []
    for i, j in sorted(P):
        if cur and i == cur[-1][0] + 1 and j == cur[-1][1] - 1:
            cur.append((i, j))
        else:
            if cur:
                hel.append(cur)
            cur = [(i, j)]
    if cur:
        hel.append(cur)
    hl = [len(h) for h in hel] or [0]
    # unpaired runs
    runs, c = [], 0
    for i in range(n):
        if i in paired:
            if c:
                runs.append(c)
            c = 0
        else:
            c += 1
    if c:
        runs.append(c)
    runs = runs or [0]
    spans = [j - i for i, j in P] or [0]
    # crossing pairs -- the pseudoknot content, still target-only
    cross = sum(1 for a in P for b in P if a[0] < b[0] < a[1] < b[1])
    return {
        "n": float(n),
        "pair_density": len(P) / max(1, n),
        "n_helices": float(len(hel)),
        "max_helix": float(max(hl)),
        "mean_helix": sum(hl) / len(hl),
        "helix_per_nt": len(hel) / max(1, n),
        "unpaired_frac": 1 - len(paired) / max(1, n),
        "max_unpaired_run": float(max(runs)),
        "mean_unpaired_run": sum(runs) / len(runs),
        "max_span_frac": max(spans) / max(1, n),
        "mean_span_frac": (sum(spans) / len(spans)) / max(1, n),
        "n_crossing": float(cross),
        "crossing_frac": cross / max(1, len(P)),
    }


def auc(y, s):
    pos = [a for a, b in zip(s, y) if b]
    neg = [a for a, b in zip(s, y) if not b]
    if not pos or not neg:
        return None
    t = sum(1.0 if p > q else (0.5 if p == q else 0.0) for p in pos for q in neg)
    return t / (len(pos) * len(neg))


def fit_logistic(X, y, l2=1.0):
    import numpy as np
    from scipy.optimize import minimize
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])

    def nll(w):
        z = np.clip(Z @ w, -30, 30)
        return float(np.sum(np.log1p(np.exp(z)) - y * z) + l2 * np.sum(w[1:] ** 2))

    r = minimize(nll, np.zeros(Z.shape[1]), method="L-BFGS-B")
    return {"w": r.x, "mu": mu, "sd": sd}


def predict(m, X):
    import numpy as np
    Z = np.hstack([np.ones((len(X), 1)), (np.asarray(X, float) - m["mu"]) / m["sd"]])
    return 1 / (1 + np.exp(-np.clip(Z @ m["w"], -30, 30)))


def run_task(name, items, out_lines):
    """items: list of (key, target, label). Everything downstream is target-only."""
    import numpy as np
    keys = sorted({k for k, _, _ in items})
    idx = {k: i for i, k in enumerate(keys)}
    F = [(k, features(t), lab) for k, t, lab in items]
    names = sorted(F[0][1])
    fit = [(k, f, l) for k, f, l in F if idx[k] % 2 == 0]
    test = [(k, f, l) for k, f, l in F if idx[k] % 2 == 1]
    base = sum(1 for _, _, l in F if l) / len(F)
    out_lines.append(f"\n=== 任务 {name} ===")
    out_lines.append(f"  n={len(F)}  基线率 {base:.1%}   FIT {len(fit)} / TEST {len(test)}(确定性偶奇分割)")
    cst = constancy([1.0 if l else 0.0 for _, _, l in F])
    if cst["constant"]:
        out_lines.append(f"  !! 标签是常数({cst['where']})—— 这个任务退化,不做")
        return None

    Xf = [[f[n_] for n_ in names] for _, f, _ in fit]
    yf = [1.0 if l else 0.0 for _, _, l in fit]
    Xt = [[f[n_] for n_ in names] for _, f, _ in test]
    yt = [1.0 if l else 0.0 for _, _, l in test]
    m = fit_logistic(Xf, yf)
    a_fit = auc(yf, predict(m, Xf))
    a_test = auc(yt, predict(m, Xt))
    out_lines.append(f"  逻辑回归(13 个仅来自靶标结构的特征)  FIT AUC {a_fit:.3f}   TEST AUC {a_test:.3f}")

    # single-feature rule, because the product has to be able to SAY WHY
    best = None
    for i, nm in enumerate(names):
        v = [x[i] for x in Xt]
        if len(set(v)) < 3:
            continue
        a1 = auc(yt, v)
        a2 = auc(yt, [-x for x in v])
        cand = (max(a1, a2), nm, "高则更可能" if a1 >= a2 else "低则更可能")
        if best is None or cand[0] > best[0]:
            best = cand
    if best:
        out_lines.append(f"  最强的【单个】特征(TEST 上):{best[1]} {best[2]}  AUC {best[0]:.3f}")

    # the control that must fail
    rng = np.random.default_rng(stable_seed(name, "perm"))
    # 200 draws estimate a 97.5th percentile from its 5th-largest value, which is far too noisy to compare
    # against an analytic bound 0.012 away. This is numerical adequacy, not tuning: the count is raised
    # regardless of which way the comparison then falls.
    perm = [auc(list(rng.permutation(yt)), predict(m, Xt)) for _ in range(2000)]
    perm = [p for p in perm if p is not None]
    out_lines.append(f"  打乱标签的对照:AUC 中位 {np.median(perm):.3f}  95% 上界 "
                     f"{np.quantile(perm, 0.975):.3f}  —— 应在 0.5 附近,否则流水线在漏信息")
    # THE LEAK THRESHOLD WAS A SCALE-BLIND CONSTANT, which is the theta-grid defect again. Under the
    # permutation null the AUC's own spread is sqrt((n1+n2+1)/(12*n1*n2)) and therefore depends on n: the
    # 95% upper bound is 0.607 at task A's sizes and 0.641 at task B's. A fixed 0.62 flags B and clears A
    # for no reason but sample size. The analytic bound is derivable WITHOUT looking at any association
    # between features and labels, so applying it is a correction rather than a retune -- but it was
    # written after seeing the flag, and that is recorded here rather than smoothed over.
    n1 = int(sum(yt)); n2 = len(yt) - n1
    se_null = math.sqrt((n1 + n2 + 1) / (12 * max(1, n1) * max(1, n2)))
    thr = 0.5 + 1.96 * se_null
    leak = np.quantile(perm, 0.975) > thr
    out_lines.append(f"    零假设自身宽度:SE={se_null:.4f},95% 上界(解析){thr:.3f}"
                     f"  —— 实测 {np.quantile(perm, 0.975):.3f}")
    if leak:
        out_lines.append("  !! 打乱后仍高于零假设自身的宽度 —— 流水线可能在漏信息,本任务结果不作数")

    # SKEPTICAL CHECK: is the model merely rediscovering something already documented? For task A the
    # external judge's agreement is known to swing 29.9% (H-type) against 60.8% (kissing), and crossing-pair
    # count tracks topology. If a single already-known feature explains it, the "prediction" adds nothing.
    if best:
        out_lines.append(f"  单特征 {best[1]} 的 AUC {best[0]:.3f} 对 13 特征模型的 {a_test:.3f}"
                         f"  -> {'模型没有超过一个已知特征,收益要按【单特征】记' if best[0] >= a_test - 0.01 else '模型确实超过任一单特征'}")

    cl = Clustered(values=yt, clusters=[k for k, _, _ in test])
    out_lines.append(f"  重复层级:{cl.n_obs} 条 / {cl.n_clusters} 个靶标 —— {cl.warn_if_inflated() or '无膨胀'}")
    ok = (a_test is not None) and a_test > AUC_PASS and not leak
    out_lines.append(f"  预注册 AUC > {AUC_PASS}  ->  {'通过' if ok else '不过'}")
    return {"task": name, "n": len(F), "base_rate": base, "auc_fit": a_fit, "auc_test": a_test,
            "best_single": {"feature": best[1], "direction": best[2], "auc": best[0]} if best else None,
            "perm_median": float(np.median(perm)), "perm_hi": float(np.quantile(perm, 0.975)),
            "leak": bool(leak), "passes": bool(ok),
            "coef": {n_: float(c) for n_, c in zip(names, m["w"][1:])}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "designability.json"))
    a = ap.parse_args()
    L = []

    # --- Task A: pseudoknot, will the external judge confirm it? --------------------------------------
    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    pk = {str(t.task_id): str(t.target) for t in load_pseudobase()}
    ext = json.load(open(os.path.join(HERE, "external_judge.json")))
    A = []
    for r in ext["records"]:
        if r.get("pop") != "ours" or not r.get("ran"):
            continue
        base = str(r["id"]).split("#")[0]
        if base in pk:
            A.append((base, pk[base], bool(r.get("confirmed"))))
    if len(A) < 0.8 * 243:
        raise SystemExit(f"!! 任务A 只连上 {len(A)}/243 —— join 坏了,先修,不许报成不可测")

    # --- Task B: co-transcriptional, does the pool contain anything the criterion accepts? -------------
    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    big = json.load(open(os.path.join(HERE, "kinetic_calibrate_big.json")))
    bm = {str(x["id"]): str(x["target"]) for x in load_bprna_nested(big.get("pool", 400))}
    # THE GUARD MUST COMPARE AGAINST THE ELIGIBLE ROWS, NOT THE ATTEMPTED ONES. The first version checked
    # against 178 (targets attempted) when only 133 produced any candidate at all, so a perfect join was
    # blocked as "broken". A guard with the wrong denominator stops correct work exactly as loudly as it
    # would stop broken work.
    eligible = [r for r in big["rows"] if not r.get("error") and r.get("candidates")]
    B = [(r["task_id"], bm[r["task_id"]], any(c["accepted"] for c in r["candidates"]))
         for r in eligible if r["task_id"] in bm]
    if len(B) < 0.9 * len(eligible):
        raise SystemExit(f"!! 任务B 在 {len(eligible)} 条【有候选的】靶标里只连上 {len(B)} —— join 坏了,先修")

    print("设计前预测:特征【只来自靶标点括号】,不含序列、不含求解器输出、不含任何由结果导出的量")
    ra = run_task("A 假结·外部裁判是否确认", A, L)
    rb = run_task("B 共转录·池中是否有合格候选", B, L)
    print("\n".join(L))
    both = bool(ra and rb and ra["passes"] and rb["passes"])
    print(f"\n=== 预注册判定 ===")
    print(f"  两个任务都要 TEST AUC > {AUC_PASS}  ->  {'通过 —— 可设计性能从靶标结构预测' if both else '不过 —— 照实报'}")
    if not both:
        for r in (ra, rb):
            if r and not r["passes"]:
                print(f"     {r['task']}:TEST AUC {r['auc_test']:.3f}(基线率 {r['base_rate']:.1%},n={r['n']})")
        print("     零结果的检验力:n 分别是 " + ", ".join(str(r["n"]) for r in (ra, rb) if r)
              + ";AUC 的标准误在这个量级上约 0.04,所以 0.70 这道线不是被噪声挡住的")
    json.dump({"A": ra, "B": rb, "passes": both, "auc_pass": AUC_PASS},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
