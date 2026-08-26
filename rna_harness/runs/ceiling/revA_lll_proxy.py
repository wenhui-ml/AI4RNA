#!/usr/bin/env python3
"""REVIEW A: does the Lovász-Local-Lemma proposal survive its own two load-bearing claims?

The proposal was: bad event = "a window forms a spurious helix of length >= l"; these have bounded
dependency, so LLL gives existence and Moser-Tardos gives the repair loop, with a threshold
l* ~ log n / log(8/3).

Two claims carry it, and both are checked here rather than asserted.

  CLAIM 1 (arithmetic). l* ~ log(n)/log(8/3). I quoted l ~ 5-6 at n=100 and dropped the log(d) term.
  CLAIM 2 (the chain). LLL bounds the PROXY (no long spurious helix). The objective is MFE == T, which
      depends on every position and therefore has dependency degree n -- LLL does not apply to it
      directly. So the proposal only survives if the proxy actually predicts the objective.

CLAIM 2 is the one that can kill the proposal, and it is measured here: draw random compatible
sequences for real nested targets (the skeletons of the crossing pool, where ViennaRNA MFE is the
correct oracle), record the longest spurious helix and whether the MFE equals the target, and report
P(MFE=T | L_spur < l) against P(MFE=T) plus an AUC.

Offline, $0.
"""
from __future__ import annotations
import argparse, json, math, os, random, statistics as st, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

P1 = 6 / 16                       # WC + GU: 6 of 16 ordered base pairs can pair
PAIRABLE = {"A": "U", "U": "AG", "G": "CU", "C": "G"}


def lll_threshold(n):
    """smallest l with e * p * (d+1) <= 1, p = P1**l, d ~ 2*l*n bad events sharing a position."""
    l = 1
    while math.e * (P1 ** l) * (2 * l * n + 1) > 1:
        l += 1
        if l > 60:
            return None
    return l


def pairs_of(db):
    st_, out = [], set()
    for i, c in enumerate(db):
        if c == "(":
            st_.append(i)
        elif c == ")":
            if st_:
                out.add((st_.pop(), i))
    return out


def random_compatible(db, rng):
    """uniform over sequences compatible with db: a canonical pair at every paired position."""
    n = len(db); s = ["N"] * n
    P = pairs_of(db)
    for i, j in P:
        a, b = rng.choice([("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")])
        s[i], s[j] = a, b
    for k in range(n):
        if s[k] == "N":
            s[k] = rng.choice("ACGU")
    return "".join(s)


def longest_spurious(seq, P, minloop=3):
    """longest run of stacked canonical pairs (i+k, j-k) containing no pair of the target."""
    n = len(seq); best = 0
    for i in range(n):
        for j in range(i + minloop + 1, n):
            if seq[j] not in PAIRABLE[seq[i]]:
                continue
            if i and j + 1 < n and seq[j + 1] in PAIRABLE[seq[i - 1]]:
                continue                                          # not a maximal helix start
            k, native = 0, False
            while (i + k < j - k - minloop and seq[j - k] in PAIRABLE[seq[i + k]]):
                if (i + k, j - k) in P:
                    native = True
                k += 1
            if not native and k > best:
                best = k
    return best


def one(item):
    import RNA
    tgt, draws, seed = item
    P = pairs_of(tgt); rng = random.Random(seed)
    rows = []
    for _ in range(draws):
        s = random_compatible(tgt, rng)
        mfe = RNA.fold(s)[0]
        rows.append((longest_spurious(s, P), 1 if mfe == tgt else 0))
    return rows


def auc(x, y):
    """AUC of -x (shorter spurious helix should mean higher success) against binary y."""
    pos = [x[i] for i in range(len(y)) if y[i] == 1]
    neg = [x[i] for i in range(len(y)) if y[i] == 0]
    if not pos or not neg:
        return None
    w = sum((1.0 if a < b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "revA_lll_proxy.json"))
    a = ap.parse_args()

    print("=== REVIEW A:LLL 提案的两条承重断言 ===\n")
    print("--- 断言 1(算术):l* 的正确值 ---")
    print(f"  {'n':>6s} {'l*':>4s}   e*p*(d+1)<=1, p=(6/16)^l, d~2*l*n")
    TH = {}
    for n in (30, 50, 100, 200, 400):
        TH[n] = lll_threshold(n)
        print(f"  {n:6d} {TH[n]:4d}")
    print(f"\n  **我上一条说『n=100 时 l≈5-6』。正确值是 {TH[100]}。漏掉了 log(d) 那一项。**")
    print(f"  渐近 l* = log(n)/log(8/3) + log(2e*l*)/log(8/3);第二项在 n<=400 时贡献约 4,比第一项还大。\n")

    print("--- 断言 2(链条):代理事件预不预测目标事件 ---")
    from leaderboard_design import targets as load_targets
    from rna_harness.benchmarks.rnainvbench import _round_only
    T = load_targets()
    skels = []
    for puz, d in sorted(T.items()):
        s = _round_only(d["target"])
        if "(" in s and 20 <= len(s) <= 150:
            skels.append((puz, s))
    rng = random.Random(a.seed)
    rng.shuffle(skels)
    skels = skels[:a.targets]
    print(f"  用 {len(skels)} 个真实靶标的嵌套骨架(ViennaRNA MFE 在这里是正确的裁判),每个抽 {a.draws} 条随机相容序列")

    from concurrent.futures import ProcessPoolExecutor
    jobs = [(s, a.draws, a.seed + 977 * k) for k, (_p, s) in enumerate(skels)]
    allrows, per = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for (puz, s), rows in zip(skels, ex.map(one, jobs)):
            allrows += rows
            hit = sum(y for _x, y in rows) / len(rows)
            per.append({"puzzle": puz, "n": len(s), "l_star": lll_threshold(len(s)),
                        "p_hit": round(hit, 4),
                        "med_spur": st.median([x for x, _y in rows]),
                        "auc": auc([x for x, _y in rows], [y for _x, y in rows])})
            au = per[-1]["auc"]
            austr = "--" if au is None else f"{au:.3f}"
            print(f"    {puz:<28s} n={len(s):3d} l*={per[-1]['l_star']:2d}  P(MFE=T)={hit:.3f}"
                  f"  中位最长伪螺旋={per[-1]['med_spur']:.1f}  AUC={austr}", flush=True)

    X = [x for x, _y in allrows]; Y = [y for _x, y in allrows]
    base = sum(Y) / len(Y)
    A = auc(X, Y)
    Astr = "--" if A is None else f"{A:.4f}"
    print(f"\n  合并 {len(allrows)} 条:P(MFE=T) = **{base:.4f}**,最长伪螺旋的 AUC = **{Astr}**")
    print(f"  {'阈值 l':>7s} {'P(L_spur < l)':>14s} {'P(MFE=T | L_spur<l)':>21s} {'提升':>8s}")
    cond = {}
    for l in range(3, 13):
        sel = [Y[i] for i in range(len(Y)) if X[i] < l]
        if len(sel) < 30:
            continue
        p = sum(sel) / len(sel)
        cond[l] = {"frac": round(len(sel) / len(Y), 4), "p_hit": round(p, 4)}
        print(f"  {l:7d} {len(sel)/len(Y):14.4f} {p:21.4f} {p-base:+8.4f}")

    aucs = [d["auc"] for d in per if d["auc"] is not None]
    verdict = ("proxy_predicts_objective" if (A is not None and A >= 0.65)
               else "proxy_weakly_predicts" if (A is not None and A >= 0.55)
               else "proxy_does_not_predict_objective__proposal_A_broken_as_stated")
    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  逐靶标 AUC 中位数 {st.median(aucs):.4f}(n={len(aucs)}),"
          f"其中 >0.65 的 {sum(1 for v in aucs if v > 0.65)}/{len(aucs)}")
    json.dump({"verdict": verdict, "thresholds": TH, "pooled_auc": A, "base_rate": base,
               "conditional": cond, "per_target": per,
               "n_rows": len(allrows), "n_targets": len(per)},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
