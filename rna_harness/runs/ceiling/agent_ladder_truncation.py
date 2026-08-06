#!/usr/bin/env python3
"""Does the agent-facing ladder's last rung earn its keep -- on NESTED targets, where the last one did not?

WHY THIS IS NOT A COPY OF THE LAST RESULT. certify_design's ladder dropped its 180s tier this week: on 131
fresh pseudobase targets that saved 40.8% of solver clock while losing nothing detectable. `tools.py` builds
a SEPARATE ladder for the agent -- [desirna@5s, @15s, @60s, @180s] -- and it was deliberately left alone,
because the measurement was on certify's ladder and a different surface is a different claim. It is also a
different POPULATION: certify was measured on pseudoknots, and this ladder was chosen on Eterna100 and
bpRNA, which are nested. Nested targets are adjudicated by one exact criterion rather than a three-oracle
panel, so nothing about the pseudoknot result transfers by argument. It has to be measured.

NO NEW COMPUTE IS NEEDED, and that is a property of the thing being measured rather than a shortcut. The
ladder is a FIXED-ORDER cascade that stops at the first success, and every run on disk stored its full
per-rung trace (tool, solved, cost). So "what would have happened without the last rung" is not a
simulation: a target is solved iff one of the first three rungs solved it, and its cost is exactly the
prefix of rung costs actually paid. 400 runs already on disk, nothing re-run, nothing modelled.

DECLARED BEFORE COMPUTING IT. The data exists, but this quantity has never been computed from it, so the
thresholds go in first:

    P1  Dropping desirna@180s saves >= 30% of total cascade cost.
    P2  It loses <= 5% of solves AND the loss is not significant across seeds by a directional test.
    KILL: either fails -> the agent ladder keeps its last rung, and this week's certify change is recorded
          as specific to the pseudoknot path rather than a general fact about budget ladders.

    THE CLUSTER IS THE TARGET. Eterna100 has 3 seeds per target; counting 300 runs as 300 observations is
    the pseudo-replication this project has already got wrong twice. Per-target rates first, seeds used
    only to test stability.

    Both populations reported SEPARATELY. Pooling Eterna100 (curated hard puzzles, 75th pct 258 nt) with
    bpRNA (natural distribution, 75th pct 127 nt) would hide exactly the length effect that decides whether
    a long-budget rung matters.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

DROP = "desirna@180s"
SOURCES = {"eterna100": "verify_ladder", "bprna_nested": "verify_ladder_bprna"}


def replay(row, drop_rung: str):
    """Exact counterfactual for a stop-at-first-success cascade: keep the prefix of rungs, drop the rest."""
    kept = [t for t in row.get("trace") or [] if not t["tool"].startswith(drop_rung)]
    cost, solved = 0.0, False
    for t in kept:
        cost += t.get("cost_usd") or 0.0
        if t.get("solved"):
            solved = True
            break
    return solved, cost


def full(row):
    cost, solved = 0.0, False
    for t in row.get("trace") or []:
        cost += t.get("cost_usd") or 0.0
        if t.get("solved"):
            solved = True
            break
    return solved, cost


def sign_p(up, dn):
    n = up + dn
    return sum(comb(n, i) for i in range(up, n + 1)) / 2 ** n if n else float("nan")


def main():
    out = {}
    for name, sub in SOURCES.items():
        files = sorted(glob.glob(os.path.join(HERE, sub, "p*.json")))
        by = defaultdict(list)
        for f in files:
            try:
                r = json.load(open(f))
            except Exception:                                                  # noqa: BLE001
                continue
            if r.get("trace"):
                by[r["id"]].append(r)
        if not by:
            print(f"=== {name}:没有数据,跳过 ===")
            continue
        n_runs = sum(len(v) for v in by.values())
        print(f"\n=== {name}:{len(by)} 个靶标 / {n_runs} 次运行(种子是重复,簇是靶标)===")

        # per-target rates, seeds averaged inside the target -- the cluster is the target
        sf, st_, cf, ct = [], [], [], []
        lost_t, gain_t = 0, 0
        for tid, runs in by.items():
            F = [full(r) for r in runs]
            T = [replay(r, DROP) for r in runs]
            sf.append(st.mean([1.0 * s for s, _ in F]))
            st_.append(st.mean([1.0 * s for s, _ in T]))
            cf.append(st.mean([c for _, c in F]))
            ct.append(st.mean([c for _, c in T]))
            if sf[-1] > st_[-1]:
                lost_t += 1
            elif st_[-1] > sf[-1]:
                gain_t += 1
        save = 1 - sum(ct) / sum(cf) if sum(cf) else 0.0
        loss = (sum(sf) - sum(st_)) / sum(sf) if sum(sf) else 0.0
        p = sign_p(lost_t, gain_t)
        print(f"  完整阶梯  解出 {sum(sf):5.1f}/{len(by)}   成本 ${sum(cf):.4f}")
        print(f"  去掉 @180 解出 {sum(st_):5.1f}/{len(by)}   成本 ${sum(ct):.4f}")
        print(f"  -> 省下成本 {save:.1%}(门槛 ≥30%);丢解 {loss:.1%}(门槛 ≤5%)")
        print(f"  -> 只靠 @180 才解出的靶标 {lost_t} 个,反向 {gain_t} 个,符号 p={p:.3g}")
        p1, p2 = save >= 0.30, (loss <= 0.05 and (p >= 0.05 or lost_t == 0))
        print(f"  -> P1 {'通过' if p1 else '未通过'}   P2 {'通过' if p2 else '未通过'}   "
              f"判定:{'可以去掉' if p1 and p2 else '保留 @180'}")
        out[name] = {"n_targets": len(by), "n_runs": n_runs, "solved_full": sum(sf),
                     "solved_trunc": sum(st_), "cost_full": sum(cf), "cost_trunc": sum(ct),
                     "save": save, "loss": loss, "lost_targets": lost_t, "gained_targets": gain_t,
                     "sign_p": p, "P1": p1, "P2": p2, "drop_ok": bool(p1 and p2)}

        # where the money goes, since that is what decides whether a saving was even available
        paid, solved_by = 0.0, 0
        for runs in by.values():
            for r in runs:
                for t in r["trace"]:
                    if t["tool"].startswith(DROP):
                        paid += t.get("cost_usd") or 0.0
                        solved_by += 1 if t.get("solved") else 0
        print(f"  @180 这一档:总共被付了 ${paid:.4f},换回 {solved_by} 次成功"
              f"(占该档花费的 {'∞' if not paid else f'${paid/max(1,solved_by):.4f}/次'})")

    if len(out) == 2:
        a, b = out["eterna100"], out["bprna_nested"]
        print(f"\n=== 两个总体是否给出同一个答案 ===")
        print(f"  eterna100(难题集,长尾重):省 {a['save']:.1%} 丢 {a['loss']:.1%} -> "
              f"{'去掉' if a['drop_ok'] else '保留'}")
        print(f"  bprna(自然分布,更短)   :省 {b['save']:.1%} 丢 {b['loss']:.1%} -> "
              f"{'去掉' if b['drop_ok'] else '保留'}")
        print(f"  -> {'两个总体一致' if a['drop_ok'] == b['drop_ok'] else '两个总体【不一致】—— 这一档的价值取决于靶标分布,不能一刀切'}")

    json.dump(out, open(os.path.join(HERE, "agent_ladder_truncation.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n  wrote agent_ladder_truncation.json")


if __name__ == "__main__":
    main()
