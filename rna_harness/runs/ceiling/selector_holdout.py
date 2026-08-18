#!/usr/bin/env python3
"""Was `screen_first` chosen on the same data it is now credited on? A holdout says.

RUNS IN THE RibonanzaNet venv, on the CERTIFY_POOL_DUMP pools.

THE PROBLEM THIS EXISTS FOR. `selector_replay` compared eight rules and `screen_first` came out best. The
leaderboard re-score then showed delivery rising 0.826 -> 0.955 and the per-method tally going 3-5-4 to
6-3-3. Those are NOT two results. Both read the same RibonanzaNet F1 on the same 51 targets, so the second
cannot confirm the first -- picking the argmax of eight rules and quoting its effect on the data that picked
it is exactly the error that produced this project's retracted +0.250 headline. The leaderboard number says
how big the effect is IF real; it does not say it is real.

THE TEST. 200 random halves of the targets. On the CHOOSING half, take whichever rule has the best mean
delivered F1. On the HELD-OUT half, measure THAT rule against `shipped`. A rule that only wins because eight
were tried will not survive; a rule that is genuinely better will.

    S1  PRIMARY. Median held-out gain of the chosen rule over `shipped`, and the fraction of splits where
        it is positive. PASS: gain > 0 and positive on > 75% of splits.
    S2  How often `screen_first` is the one chosen. If it is chosen rarely, the replay's winner was a
        coin-flip among near-equivalent rules and should not be singled out by name.
    S3  `screen_first` specifically, held out: fix the rule in advance, measure only on the held-out half.
        This is the version that answers "should we ship THIS rule", separate from "does rule search help".

COST: $0.
"""
from __future__ import annotations
import argparse, json, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from pool_rerank_score import _score_chunk
from selector_replay import load_dump, pick, KNOT_VERDICTS

RULES = ["k_first", "k_lowent", "k_recall", "recall", "screen_first"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "pool_dump"))
    ap.add_argument("--run", default=os.path.join(HERE, "leaderboard_replay.json"))
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "selector_holdout.json"))
    a = ap.parse_args()
    from m2r_pairwise import wilcoxon
    from rna_harness.cognitive.certify import calibrated_rule_applies

    R = json.load(open(a.run))
    T = {p: v["target"] for p, v in R["targets"].items()}
    rows = {r["puzzle"]: r for r in R["rows"]}
    P = load_dump(a.dump, T)
    G = {p: not calibrated_rule_applies(t) for p, t in T.items()}
    elig = sorted(p for p in P if rows.get(p, {}).get("verdict") in KNOT_VERDICTS)

    jobs = [((p, i), r["seq"], T[p]) for p in elig for i, r in enumerate(P[p])]
    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 4)); sc = {}
    print(f"{len(elig)} 个靶标,{len(jobs)} 条候选打分中\n", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, [jobs[i:i + CH] for i in range(0, len(jobs), CH)]):
            sc.update(dict(res))

    f1 = {}
    for rule in ["shipped"] + RULES:
        f1[rule] = {}
        for p in elig:
            c = pick(P[p], rule, gated=G[p])
            idx = P[p].index(c) if c else None
            f1[rule][p] = sc.get((p, idx)) if idx is not None else None
    ps = [p for p in elig if all(f1[r].get(p) is not None for r in f1)]
    print(f"可比靶标 {len(ps)}\n")

    rng = random.Random(a.seed)
    gains, chosen, sf_gains = [], [], []
    for _ in range(a.splits):
        q = ps[:]; rng.shuffle(q)
        h = len(q) // 2
        A, B = q[:h], q[h:]                                  # A chooses, B measures
        best = max(RULES, key=lambda r: st.mean(f1[r][p] - f1["shipped"][p] for p in A))
        chosen.append(best)
        gains.append(st.mean(f1[best][p] - f1["shipped"][p] for p in B))
        sf_gains.append(st.mean(f1["screen_first"][p] - f1["shipped"][p] for p in B))

    import collections
    cnt = collections.Counter(chosen)
    pos = sum(1 for g in gains if g > 0)
    print(f"=== S1 主检验:在【没参与挑规则】的那一半上,被挑中的规则还剩多少 ===")
    print(f"  留出增益中位 {st.median(gains):+.4f};为正的 {pos}/{a.splits} = {pos/a.splits:.1%}")
    s1 = st.median(gains) > 0 and pos / a.splits > 0.75
    print(f"  {'通过 —— 不是八选一挑出来的运气' if s1 else 'KILL —— 规则搜索的收益在未见靶标上消失'}")
    print(f"\n=== S2 被挑中的是谁 ===")
    for r, n in cnt.most_common():
        print(f"  {r:14s} {n:4d}/{a.splits}  {n/a.splits:.1%}")
    print(f"\n=== S3 事先钉死 `screen_first`,只在留出半边量 ===")
    pos3 = sum(1 for g in sf_gains if g > 0)
    print(f"  留出增益中位 {st.median(sf_gains):+.4f};为正的 {pos3}/{a.splits} = {pos3/a.splits:.1%}")
    json.dump({"n_targets": len(ps), "splits": a.splits,
               "S1_median_gain": st.median(gains), "S1_positive_frac": pos / a.splits, "S1_pass": bool(s1),
               "S2_chosen": dict(cnt),
               "S3_screenfirst_median_gain": st.median(sf_gains), "S3_positive_frac": pos3 / a.splits,
               "full_data_means": {r: st.mean(f1[r][p] - f1["shipped"][p] for p in ps) for r in RULES}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
