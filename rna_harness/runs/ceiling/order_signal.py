#!/usr/bin/env python3
"""Why does "take the first candidate clearing the screen" beat our whole grading apparatus?

RUNS IN THE RibonanzaNet venv, on the CERTIFY_POOL_DUMP pools.

THE RESULT THAT PROMPTED IT. Replaying eight selection rules on the pools the product actually screened,
`screen_first` -- ignore the confirm rung entirely, take the first candidate whose mean crossing recall
clears 0.1 -- delivers a median of 0.917 against the shipped rule's 0.800, better on 24 of 51 targets and
worse on 8 (Wilcoxon p = 1.5e-4, under a Bonferroni threshold of 0.025/7 = 0.0036). `random` over the same
eligible set is a clean null (-0.013, p = 0.73), so this is not "any change helps".

THE HYPOTHESIS, and it is mechanical rather than statistical. "First" is not arbitrary. certify.py builds
each tier's candidate list as [res.best_sequence] + res.candidates, and DesiRNA returns its results ordered
by its OWN objective. So `screen_first` is not "an arbitrary survivor" -- it is THE SOLVER'S OWN TOP-RANKED
CANDIDATE, filtered by a loose screen. If that is right, RNet F1 should DECLINE with pool position, and the
first position should be measurably better than the pool as a whole.

    O1  Within target, Spearman between pool index and RNet crossed-pair F1. Negative => order is
        informative and `first` means something.
    O2  Index 0 against the target's pool median, paired across targets.
    KILL: no order signal -> `screen_first` wins for some other reason and the mechanism here is wrong,
    which gets reported rather than replaced with a nicer story.

COST: $0. Local CPU.
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from pool_rerank_score import _score_chunk
from selector_replay import load_dump, KNOT_VERDICTS


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1): r[s[k]] = avg
            i = j + 1
        return r
    a, b = rank(x), rank(y); n = len(x)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "pool_dump"))
    ap.add_argument("--run", default=os.path.join(HERE, "leaderboard_replay.json"))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "order_signal.json"))
    a = ap.parse_args()
    from m2r_pairwise import wilcoxon

    R = json.load(open(a.run))
    T = {p: v["target"] for p, v in R["targets"].items()}
    rows = {r["puzzle"]: r for r in R["rows"]}
    P = load_dump(a.dump, T)
    elig = [p for p in P if rows.get(p, {}).get("verdict") in KNOT_VERDICTS]
    jobs = [((p, i), r["seq"], T[p]) for p in elig for i, r in enumerate(P[p])]
    print(f"{len(elig)} 个靶标,{len(jobs)} 条候选\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 4)); sc = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, [jobs[i:i + CH] for i in range(0, len(jobs), CH)]):
            sc.update(dict(res))

    rho, first_vs_med, per = [], [], {}
    for p in elig:
        v = [(i, sc[(p, i)]) for i in range(len(P[p])) if (p, i) in sc]
        if len(v) < 10: continue
        r = spearman([i for i, _ in v], [f for _, f in v])
        rho.append(r)
        med = st.median([f for _, f in v])
        first_vs_med.append(v[0][1] - med)
        per[p] = {"rho": r, "first": v[0][1], "pool_median": med, "n": len(v)}

    _, p1, _ = wilcoxon(rho, "less")
    _, p2, _ = wilcoxon(first_vs_med, "greater")
    print(f"=== O1 池内位置 vs RNet F1 的 Spearman({len(rho)} 个靶标)===")
    print(f"  中位 rho = {st.median(rho):+.3f};为负的 {sum(1 for x in rho if x < 0)}/{len(rho)},p = {p1:.3g}")
    print(f"  {'排序有信息 —— 越靠前越好' if st.median(rho) < 0 and p1 < 0.025 else 'KILL —— 排序没有信息,screen_first 赢在别的原因'}")
    print(f"\n=== O2 第一条 vs 该池中位 ===")
    print(f"  中位差 {st.median(first_vs_med):+.3f};更好的 {sum(1 for x in first_vs_med if x > 0)}/{len(first_vs_med)},p = {p2:.3g}")
    print(f"\n  解读:pool 顺序 = [res.best_sequence] + res.candidates,而 DesiRNA 按自己的目标排序。")
    print(f"  所以 `screen_first` 不是『随便一条过筛的』,是【求解器自己排第一的那条】加一道松筛。")
    json.dump({"O1_rho_median": st.median(rho), "O1_p": p1, "O1_negative": sum(1 for x in rho if x < 0),
               "O2_first_minus_median": st.median(first_vs_med), "O2_p": p2,
               "n_targets": len(rho), "per_target": per}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
