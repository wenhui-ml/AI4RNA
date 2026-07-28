#!/usr/bin/env python3
"""Priority 2: a learned abstention policy on the measured DesiRNA ladder, against every control that
could explain it away. Zero new solver calls -- all 400 cells already exist in sota_curve/.

THE DECISION PROBLEM, stated at its true size. Run the cheapest tier (t=5s) on all 100 targets: 72 solve for
$0.211. That leaves 28. Of those, 12 eventually solve at t<=180s and 16 never do. Every dollar of the
remaining bill is decided by one binary call per target: escalate, or abstain.

    escalate a target that never solves -> pay c15+c60+c180 = $0.03997 for nothing
    abstain on a target that would solve -> lose a solve

So the whole problem is 28 samples with 12 positives, and the metric that matters is PRECISION AT 100%
RECALL: with zero solves lost, how many of the 16 hopeless can you identify? Each one caught saves $0.03997,
which is 4.0% of the total bill. This is a small, honest problem and it is reported as such -- n=28 will not
support a confident claim, which is exactly why the controls below matter more than the learner.

CONTROLS, all of which must be beaten before "learning" means anything:
  CASCADE        escalate everything. Zero knowledge. The baseline the audit established must be used.
  LENGTH(theta)  abstain if len > theta. ZERO learning, one number, and 11 of the 16 hopeless are >250nt --
                 so this is the control most likely to explain the learner away.
  BP(theta)      abstain if the bp-distance reached at t=5 exceeds theta. ZERO learning, one OBSERVED number.
                 This separates "the information source is the cheap tier's residual" from "the learner
                 learned something": today's tally on this branch is 0/6 for re-analysing existing signals
                 and 3/3 for adding an information source, so this control is the real test.
  ORACLE-abstain escalate exactly the 12. Upper bound for any abstention policy.
  ORACLE-tier    additionally know each target's minimum solving tier (frontier.py's oracle). Not reachable
                 by abstention alone; shown to keep the two upper bounds distinct.

LEARNERS: k-NN (the same mechanism as the archived abstention arms in cognitive/router.py) and a logistic
regression, each with and without bp@t=5, all evaluated LEAVE-ONE-OUT over the 28. A permutation test on the
labels says whether any apparent gain survives chance at this sample size.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
RNG = np.random.default_rng(20260728)


def load():
    by = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(HERE, "sota_curve", "*.json")):
        d = json.load(open(f))
        by[d["t"]][d["id"]] = d
    B = sorted(by)
    ids = sorted(by[B[0]])
    cost = {t: sum(by[t][i].get("cost_usd", 0.0) for i in ids) / len(ids) for t in B}
    return by, B, ids, cost


def logistic(X, y, iters=600, lr=0.6, l2=1.0):
    """Plain gradient-descent logistic regression. l2 is deliberately strong: 28 samples, ~16 features."""
    Xs = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        g = Xs.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]] / len(y)
        w -= lr * g
    return w


def predict_logistic(w, X):
    return 1.0 / (1.0 + np.exp(-np.hstack([np.ones((len(X), 1)), X]) @ w))


def knn_score(Xtr, ytr, x, k=5):
    d = np.sqrt(((Xtr - x) ** 2).sum(1))
    idx = np.argsort(d)[:min(k, len(d))]
    return float(ytr[idx].mean())


def loo_scores(X, y, kind, k=5):
    """Leave-one-out predicted P(eventually solves) for each of the n targets."""
    n = len(y)
    out = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        Xtr, ytr = X[m], y[m]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        if kind == "knn":
            out[i] = knn_score((Xtr - mu) / sd, ytr, (X[i] - mu) / sd, k)
        else:
            out[i] = predict_logistic(logistic((Xtr - mu) / sd, ytr), ((X[i] - mu) / sd)[None, :])[0]
    return out


def frontier_from_scores(scores, y, esc_cost, base, full_esc):
    """Sweep the escalate-if-score>=tau threshold. Returns [(lost, cost, n_abstained, tau)] on the Pareto set.
    A target is escalated iff score >= tau; abstaining on y=1 loses a solve."""
    pts = []
    for tau in sorted(set(np.r_[scores, 0.0, 1.0 + 1e-9])):
        esc = scores >= tau
        lost = int(((~esc) & (y == 1)).sum())
        cost = base + float(esc_cost[esc].sum())
        pts.append((lost, cost, int((~esc).sum()), float(tau)))
    best = {}
    for lost, cost, nab, tau in pts:
        if lost not in best or cost < best[lost][0]:
            best[lost] = (cost, nab, tau)
    return sorted((l, v[0], v[1], v[2]) for l, v in best.items())


def main():
    by, B, ids, cost = load()
    t0, rest = B[0], B[1:]
    base = cost[t0] * len(ids)
    solved_t0 = {i: bool(by[t0][i].get("solved")) for i in ids}
    U = [i for i in ids if not solved_t0[i]]

    def esc_cost_of(i):
        """Cost of escalating target i through the remaining tiers until it solves (or through all of them)."""
        c = 0.0
        for t in rest:
            c += cost[t]
            if by[t][i].get("solved"):
                return c
        return c

    y = np.array([int(any(by[t][i].get("solved") for t in rest)) for i in U])
    ec = np.array([esc_cost_of(i) for i in U])
    full = sum(cost[t] for t in rest)
    n_solved_total = sum(solved_t0.values()) + int(y.sum())

    print(f"Priority 2: abstention on the measured DesiRNA ladder, {len(ids)} Eterna100 targets, "
          f"tiers {B} s")
    print(f"  tier-1 (t={t0}s) solves {sum(solved_t0.values())}/100 for ${base:.3f}; "
          f"{len(U)} left, {int(y.sum())} eventually solve, {int((y==0).sum())} never")
    print(f"  escalating a hopeless target costs the full ladder ${full:.5f} "
          f"= {100*full/ (base + ec.sum()):.1f}% of the cascade bill, each\n")

    from rna_harness.cognitive.features import features
    from rna_harness.benchmarks.eterna100 import load_eterna100
    tgt = {p["id"]: p["target"] for p in load_eterna100(version="V2")}
    FN = sorted(k for k in features(tgt[U[0]]) if k not in ("has_pseudoknot", "n_pk_brackets"))
    Xf = np.array([[float(features(tgt[i])[k]) for k in FN] for i in U])
    bp = np.array([[float(by[t0][i].get("bp") or 0)]] for i in U).reshape(len(U), 1) \
        if False else np.array([[float(by[t0][i].get("bp") or 0)] for i in U])
    Xfb = np.hstack([Xf, bp])

    casc = base + float(ec.sum())
    orac_ab = base + float(ec[y == 1].sum())
    need = {}
    for i in ids:
        for t in B:
            if by[t][i].get("solved"):
                need[i] = t
                break
    orac_tier = sum(cost[need[i]] for i in need) + cost[t0] * (len(ids) - len(need))

    rows = []
    print(f"  {'policy':<30} {'lost':>5} {'$ cost':>8} {'vs CASCADE':>11} {'abstained':>10} "
          f"{'of prize':>9}")

    def emit(name, pts, only_lost=(0, 1, 2)):
        for lost, c, nab, tau in pts:
            if lost in only_lost:
                cap = (casc - c) / (casc - orac_ab) if casc > orac_ab else float("nan")
                print(f"  {name + (f' (lost {lost})' if lost else ''):<30} {lost:>5} {c:>8.4f} "
                      f"{casc/c:>10.3f}x {nab:>10} {100*cap:>8.1f}%")
                rows.append({"policy": name, "lost": lost, "cost": c, "vs_cascade": casc / c,
                             "abstained": nab, "tau": tau, "prize_captured": cap})

    print(f"  {'CASCADE (zero knowledge)':<30} {0:>5} {casc:>8.4f} {1.0:>10.3f}x {0:>10} {0.0:>8.1f}%")
    rows.append({"policy": "CASCADE", "lost": 0, "cost": casc, "vs_cascade": 1.0, "abstained": 0,
                 "prize_captured": 0.0})

    emit("LENGTH(theta)", frontier_from_scores(-Xf[:, FN.index("len")], y, ec, base, full))
    emit("BP(theta) @t=5", frontier_from_scores(-bp[:, 0], y, ec, base, full))
    for kind, lbl in (("knn", "kNN"), ("logit", "logistic")):
        emit(f"{lbl}, features", frontier_from_scores(loo_scores(Xf, y, kind), y, ec, base, full))
        emit(f"{lbl}, features+bp", frontier_from_scores(loo_scores(Xfb, y, kind), y, ec, base, full))

    print(f"  {'ORACLE-abstain (bound)':<30} {0:>5} {orac_ab:>8.4f} {casc/orac_ab:>10.3f}x "
          f"{int((y==0).sum()):>10} {100.0:>8.1f}%")
    print(f"  {'ORACLE-tier (frontier.py)':<30} {0:>5} {orac_tier:>8.4f} {casc/orac_tier:>10.3f}x "
          f"{'-':>10} {'n/a':>9}")

    print(f"\n=== the only question that matters at zero solves lost ===")
    print(f"  of the {int((y==0).sum())} hopeless targets, how many can each rule catch "
          f"WITHOUT abstaining on a solvable one?")
    z = [r for r in rows if r["lost"] == 0 and r["policy"] != "CASCADE"]
    for r in sorted(z, key=lambda r: -r["abstained"]):
        print(f"    {r['policy']:<26} catches {r['abstained']:>2}/{int((y==0).sum())}  "
              f"saves ${casc - r['cost']:.4f}  ({r['vs_cascade']:.3f}x)")
    if not z:
        print("    NONE. Every rule that abstains at all abstains on a solvable target.")

    # Exact null for "catch k hopeless at zero loss". A policy abstains on the top-k of its own ordering;
    # under a UNIFORMLY RANDOM ordering of the n decision targets, achieving k catches with zero loss means
    # the first k in that order are all hopeless. P = prod_{j<k} (H-j)/(n-j). This is the same class of check
    # that killed the bp-ranking heuristic earlier today, and it must be applied to the trivial rules too --
    # they are orderings as well.
    H, N = int((y == 0).sum()), len(y)
    print(f"\n=== exact null: is a zero-loss catch of k better than a RANDOM abstention order? ===")
    print(f"  (H={H} hopeless, {int(y.sum())} solvable, n={N}; P = prod_(j<k) (H-j)/(n-j))")
    from math import prod
    for r in sorted(z, key=lambda r: -r["abstained"]):
        k = r["abstained"]
        pv = prod((H - j) / (N - j) for j in range(k)) if k else 1.0
        print(f"    {r['policy']:<26} k={k:<2} p = {pv:.3f}   "
              f"{'significant' if pv < 0.05 else 'NOT distinguishable from random'}")
    exp_before_first = H / (int(y.sum()) + 1)
    print(f"  expected catches before the first solvable, under random order = {exp_before_first:.2f}")

    NP = int(os.environ.get("ABSTAIN_PERMS", "0"))
    print(f"\n=== permutation test (is any zero-loss gain distinguishable from chance at n={len(U)}?) ===")
    best_real = max((r["abstained"] for r in z), default=0)
    if best_real > 0 and NP > 0:
        for _ in range(NP):
            ys = RNG.permutation(y)
            got = 0
            for kind in ("knn", "logit"):
                for X in (Xf, Xfb):
                    f = frontier_from_scores(loo_scores(X, ys, kind), ys, ec, base, full)
                    got = max(got, max((nab for lost, c, nab, tau in f if lost == 0), default=0))
            hits += int(got >= best_real)
        print(f"  best real zero-loss catch = {best_real}; label-shuffled matched or beat it "
              f"{hits}/{NP} times -> p = {(hits+1)/(NP+1):.3f}")
    elif best_real == 0:
        print("  not run: no rule achieves a zero-loss saving, so there is nothing to test.")
    else:
        print(f"  skipped (ABSTAIN_PERMS=0); best real zero-loss catch = {best_real}")

    json.dump({"tiers": B, "cost_per_target": cost, "n": len(ids), "n_decision": len(U),
               "n_positive": int(y.sum()), "n_hopeless": int((y == 0).sum()),
               "cascade": casc, "oracle_abstain": orac_ab, "oracle_tier": orac_tier,
               "solved_total": n_solved_total, "rows": rows},
              open(os.path.join(HERE, "abstain.json"), "w"), indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/abstain.json")


if __name__ == "__main__":
    main()
