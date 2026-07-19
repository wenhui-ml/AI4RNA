"""W1 -- Goodhart under ACTUAL policy optimization (REINFORCE), closing the paper's 'never runs RL' gap.

The paper's controlled-Goodhart experiment holds a fixed Metropolis kernel and varies only the reward. A
reviewer's #1 objection: the paper is framed around RLVR yet never runs RL. This does. We train a genuine
policy-gradient learner (REINFORCE with a moving-average baseline + entropy bonus) whose reward is the
dial-able ladder, and ask the same question: does optimizing the CHEAP reward drive the proxy up while
robustness stalls/falls, while the STRONG reward lifts both?

Policy (structure-aware, factorized, numpy-only -- no framework dep): for a target dot-bracket we sample each
base PAIR (i,j) jointly from a categorical over the 6 Watson--Crick/wobble pairs and each UNPAIRED position
from a categorical over {A,C,G,U}. This respects the pairing graph (so exact-match is reachable) while
staying a clean product-of-categoricals policy with an exact REINFORCE gradient.

Reward arms (identical policy, init, budget, schedule -- only the reward differs):
  * CHEAP  : r = 1 - bp_distance(MFE(x), target)/n     -- the exact-match proxy; continuous, robustness-agnostic.
  * STRONG : r = P(target | x)  (ViennaRNA partition function; fc.pr_structure)  -- rewards robustness directly.
Metrics at budget checkpoints (argmax policy): proxy = exact-match rate; robust = P(target)>=0.9 rate.
18 EteRNA100-V2 targets (n<=50) x seeds = the paper's Goodhart cell, now under RL."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import RNA

from ..benchmarks.eterna100 import load_eterna100

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "rl_goodhart")
WC = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]  # 6 pair identities
NT = ["A", "C", "G", "U"]


def _pairs(target):
    st, pairs = [], []
    for i, c in enumerate(target):
        if c == "(":
            st.append(i)
        elif c == ")":
            pairs.append((st.pop(), i))
    unpaired = [i for i, c in enumerate(target) if c == "."]
    return pairs, unpaired


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _sample(pair_p, unp_p, rng):
    pa = np.array([rng.choice(6, p=pair_p[k]) for k in range(len(pair_p))], dtype=int) if len(pair_p) else np.zeros(0, int)
    ua = np.array([rng.choice(4, p=unp_p[k]) for k in range(len(unp_p))], dtype=int) if len(unp_p) else np.zeros(0, int)
    return pa, ua


def _build(target, pairs, unpaired, pa, ua):
    seq = ["A"] * len(target)
    for k, (i, j) in enumerate(pairs):
        a, b = WC[pa[k]]
        seq[i], seq[j] = a, b
    for k, pos in enumerate(unpaired):
        seq[pos] = NT[ua[k]]
    return "".join(seq)


def _r_cheap(seq, target):
    """Paper-faithful cheap arm: exact-match, then a continuous MFE-stability term that does not saturate
    (deeper MFE free energy keeps paying), which is robustness-agnostic -- it rewards a stable MFE fold, not
    a dominant Boltzmann ensemble. Fixed a priori (the paper's choice), NOT tuned to produce a collapse."""
    fold, e = RNA.fold(seq)
    n = max(1, len(target))
    base = 1.0 - RNA.bp_distance(fold, target) / n         # match signal (=1 at exact match)
    return base + (-e) / (2.0 * n)                          # + non-saturating MFE-stability term


def _r_strong(seq, target):
    fc = RNA.fold_compound(seq)
    fc.pf()
    return float(fc.pr_structure(target))


def _eval(target, pairs, unpaired, pair_logits, unp_logits):
    """Argmax (greedy) policy sequence -> (exact_match, P(target))."""
    pa = pair_logits.argmax(1) if len(pair_logits) else np.zeros(0, int)
    ua = unp_logits.argmax(1) if len(unp_logits) else np.zeros(0, int)
    seq = _build(target, pairs, unpaired, pa, ua)
    match = int(RNA.fold(seq)[0] == target)
    return match, _r_strong(seq, target)


def _run_unit(target, arm, seed, batch, budget, lr, ent, checkpoints):
    """One (target, arm, seed) REINFORCE run; return metrics at each checkpoint."""
    rng = np.random.default_rng(1000 + seed)
    RNA.init_rand(seed)
    pairs, unpaired = _pairs(target)
    np_, nu = len(pairs), len(unpaired)
    pair_logits = np.zeros((np_, 6)); unp_logits = np.zeros((nu, 4))
    baseline = 0.0; evals = 0; ck = {}; ck_i = 0
    rfun = _r_cheap if arm == "cheap" else _r_strong
    while evals < budget:
        pair_p = _softmax(pair_logits) if np_ else np.zeros((0, 6))
        unp_p = _softmax(unp_logits) if nu else np.zeros((0, 4))
        gp = np.zeros_like(pair_logits); gu = np.zeros_like(unp_logits); Rsum = 0.0
        for _ in range(batch):
            pa, ua = _sample(pair_p, unp_p, rng)
            seq = _build(target, pairs, unpaired, pa, ua)
            R = rfun(seq, target); evals += 1; Rsum += R
            adv = R - baseline
            for k in range(np_):
                oh = np.zeros(6); oh[pa[k]] = 1.0
                gp[k] += adv * (oh - pair_p[k])
            for k in range(nu):
                oh = np.zeros(4); oh[ua[k]] = 1.0
                gu[k] += adv * (oh - unp_p[k])
        gp /= batch; gu /= batch
        # entropy bonus (encourage exploration): dH/dtheta = p*(a - <a>_p), a = -(log p + 1)
        if np_:
            a = -(np.log(pair_p + 1e-12) + 1.0); gp += ent * pair_p * (a - (pair_p * a).sum(1, keepdims=True))
        if nu:
            a = -(np.log(unp_p + 1e-12) + 1.0); gu += ent * unp_p * (a - (unp_p * a).sum(1, keepdims=True))
        pair_logits += lr * gp; unp_logits += lr * gu
        baseline = 0.9 * baseline + 0.1 * (Rsum / batch)
        while ck_i < len(checkpoints) and evals >= checkpoints[ck_i]:
            m, p = _eval(target, pairs, unpaired, pair_logits, unp_logits)
            ck[checkpoints[ck_i]] = {"match": m, "ptarget": p}; ck_i += 1
    while ck_i < len(checkpoints):  # budget exhausted before a checkpoint (small targets)
        m, p = _eval(target, pairs, unpaired, pair_logits, unp_logits)
        ck[checkpoints[ck_i]] = {"match": m, "ptarget": p}; ck_i += 1
    return target, arm, seed, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.02)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[200, 1000, 5000, 20000])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--robust-theta", type=float, default=0.9)
    ap.add_argument("--nmin", type=int, default=0)
    ap.add_argument("--nmax", type=int, default=50)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    targets = [x["target"] for x in load_eterna100(version="V2") if a.nmin <= len(x["target"]) <= a.nmax]
    print(f"W1 RL-Goodhart: {len(targets)} targets (n<=50) x {a.seeds} seeds x 2 arms; "
          f"REINFORCE batch={a.batch} budget={a.budget} lr={a.lr} ent={a.ent}", flush=True)

    jobs = [(t, arm, s) for t in targets for arm in ("cheap", "strong") for s in range(a.seeds)]
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_run_unit, t, arm, s, a.batch, a.budget, a.lr, a.ent, a.checkpoints) for t, arm, s in jobs]
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result()); done += 1
            if done % 30 == 0:
                print(f"  {done}/{len(jobs)} units done", flush=True)

    # aggregate: per arm, per checkpoint -> proxy (match rate) + robust rate
    agg = {"cheap": {}, "strong": {}}
    for _, arm, _, ck in results:
        for b, m in ck.items():
            d = agg[arm].setdefault(b, {"n": 0, "match": 0, "robust": 0})
            d["n"] += 1; d["match"] += m["match"]; d["robust"] += int(m["ptarget"] >= a.robust_theta)
    print(f"\n=== W1: GOODHART UNDER RL (REINFORCE; {len(targets)}x{a.seeds}={len(targets)*a.seeds} units/cell) ===", flush=True)
    print(f"  {'budget':>7} | cheap: proxy(match)  robust(P>= {a.robust_theta}) | strong: robust", flush=True)
    rows = []
    for b in a.checkpoints:
        c = agg["cheap"].get(b, {}); s = agg["strong"].get(b, {})
        cp = 100 * c.get("match", 0) / max(1, c.get("n", 1)); cr = 100 * c.get("robust", 0) / max(1, c.get("n", 1))
        sr = 100 * s.get("robust", 0) / max(1, s.get("n", 1))
        rows.append({"budget": b, "cheap_proxy": cp, "cheap_robust": cr, "strong_robust": sr})
        print(f"  {b:>7} |        {cp:5.1f}%        {cr:5.1f}%       |        {sr:5.1f}%", flush=True)
    # headline
    last = rows[-1]
    print(f"\n  -> cheap arm: proxy {rows[0]['cheap_proxy']:.0f}->{last['cheap_proxy']:.0f}% while robustness "
          f"{rows[0]['cheap_robust']:.0f}->{last['cheap_robust']:.0f}%  "
          f"({'GOODHART (proxy up, robustness stalls/falls)' if last['cheap_robust'] <= max(r['cheap_robust'] for r in rows) else 'no divergence'})", flush=True)
    print(f"  -> strong arm robustness {rows[0]['strong_robust']:.0f}->{last['strong_robust']:.0f}% "
          f"(control: optimizing the true objective builds robustness)", flush=True)
    json.dump({"config": vars(a), "n_targets": len(targets), "rows": rows},
              open(os.path.join(a.outdir, "rl_goodhart.json"), "w"), indent=1)
    print(f"  wrote {os.path.join(a.outdir, 'rl_goodhart.json')}", flush=True)


if __name__ == "__main__":
    main()
