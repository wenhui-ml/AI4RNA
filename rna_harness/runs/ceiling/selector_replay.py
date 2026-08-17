#!/usr/bin/env python3
"""Replay every selection rule against the pool the product ACTUALLY screened, on all 57 targets.

RUNS IN THE RibonanzaNet venv, on a dump written by a prior `leaderboard_design.py` run with
CERTIFY_POOL_DUMP set. Stage A is that run; there is no separate collector, and that is the point.

WHERE THIS COMES FROM. `frontier_check.json` asked where the shipped product stands against the published
field per target: we beat every method's per-target median on 1 of 57, median gap -0.155 against us. Five
targets are not a gap but a hole -- delivered crossed-pair F1 exactly 0.000, four of them carrying
`knot_confirmed`, on targets where Eterna and MPNN reach a MEDIAN of 1.000. Re-collected pools for those
five held a 1.000 candidate every time and nothing was missing from generation, and the loss localised to
two steps of the shipped rule:

    argmax |confirmed_by|  collapses ~156 confirmed candidates (RNet F1 median 0.634) to 3
    the entropy tie-break  then picks the WORST of those -- last place on P02 (3/3), P03 (4/4) and P14
                           (2/2), which happens by chance about 1 time in 24. The fourth target, P09, had a
                           one-candidate subset where "last place" carries no information and is excluded

WHY THAT EARLIER RUN WAS NOT ENOUGH, and why this script exists. Those pools were re-collected by a
standalone helper, and the shipped ladder is WALL-CLOCK limited: how many candidates a tier returns depends
on machine load. The product reported 302-412 screened where the re-run produced 312-451, so the two pools
are neither equal nor nested, and "the product passed over a 1.000" was not something that data could
establish -- only that the same generator reliably produces one. CERTIFY_POOL_DUMP records each candidate
inside the product's own process, in the same run, so this replay is against what was really on the table.

    INTEGRITY GATE, checked before any number is reported. The `shipped` rule re-implemented here must
    reproduce the sequence the run actually delivered, on every target. If it does not, the replay does not
    model the product and no comparison below means anything, so the script refuses to print them.

    ARMS. All read only quantities the product already computed, so any of them is free to ship.
      shipped        argmax |confirmed_by|, ties to HIGHER ensemble entropy -- today's rule
      k_first        argmax |confirmed_by|, ties by arrival -- isolates the entropy tie-break alone
      k_lowent       argmax |confirmed_by|, ties to LOWER entropy -- the tie-break reversed
      k_recall       argmax |confirmed_by|, ties by mean crossing recall
      recall         argmax mean crossing recall over all confirmed -- ignores k entirely
      screen_first   first candidate clearing the screen -- today's `knot_screened` rule
      random         uniform over the same eligible set, averaged over draws -- the honest control
      oracle         pool best by RNet. AN UPPER BOUND, NOT A REACHABLE TARGET: max-of-outcome inflates by
                     about +3.67 on this project's own synthetic null, so gains are read against `random`.

    THE PRIMARY COMPARISON IS `k_first` AGAINST `shipped`, fixed here before the run. It is the one arm that
    changes exactly one thing -- it deletes the entropy tie-break and keeps everything else -- and the
    tie-break is the step the zero-delivery diagnosis already localised. The other six arms are exploratory
    and are read as directions, not as findings; with seven arms against one baseline, the smallest p among
    them is not a p-value for anything. `random` is the control that says whether a rule beats no rule.

    WHAT A NULL HERE WOULD MEAN, written down so it cannot be reinterpreted afterwards. If no rule beats
    `shipped`, then the five zero-delivery targets are a rare tail rather than a systematic defect, the
    -0.155 gap to the field is NOT mostly a selection gap, and the search for headroom moves back to
    generation. That outcome gets reported as plainly as a win.

COST: $0. No solver, no API. Local RNet inference over the dumped pools.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from pool_rerank_score import _score_chunk                                     # noqa: E402

SCREEN = 0.1
KNOT_VERDICTS = ("knot_confirmed", "knot_screened")


def _mean_recall(r):
    v = [x for x in r["per_oracle"].values() if x is not None]
    return sum(v) / len(v) if v else 0.0


def load_dump(d, targets):
    """One jsonl per target, keyed by sha1 of the target string, deduplicated by sequence in arrival order."""
    out = {}
    for p, t in targets.items():
        f = os.path.join(d, hashlib.sha1(t.encode()).hexdigest()[:16] + ".jsonl")
        if not os.path.exists(f):
            continue
        seen, rows = set(), []
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:                                                  # noqa: BLE001
                continue
            if r["seq"] in seen:
                continue
            seen.add(r["seq"])
            r["mean_recall"] = _mean_recall(r)
            rows.append(r)
        if rows:
            out[p] = rows
    return out


def pick(rows, rule, rng=None, gated=False):
    """Return the chosen row, or None when the rule has nothing eligible.

    `gated` is calibrated_rule_applies() coming back False: the crossing-stem rule was never calibrated on
    this target's topology, so certify.py refuses the `knot_confirmed` rung no matter how many candidates
    the panel confirmed, and the target falls through to the screen. Nine of the 57 are H-type and gated
    this way, and on two of them the pool holds 78 and 131 confirmed candidates that are passed over by
    design. Omitting this was a real defect in the first version of this replay -- the integrity gate caught
    it on the partial run, which is the only reason it was run early.
    """
    conf = [] if gated else [r for r in rows if r["knot_confirmed"]]
    if conf:
        kmax = max(len(r["confirmed_by"]) for r in conf)
        top = [r for r in conf if len(r["confirmed_by"]) == kmax]
        if rule == "shipped":
            # certify.py keeps an incumbent and replaces it only on STRICTLY higher entropy, so among
            # equal-entropy candidates the earliest wins. Reproduced exactly, not approximated.
            best = top[0]
            for r in top[1:]:
                if r["entropy"] > best["entropy"]:
                    best = r
            return best
        if rule == "k_first":
            return top[0]
        if rule == "k_lowent":
            return min(top, key=lambda r: r["entropy"])
        if rule == "k_recall":
            return max(top, key=lambda r: r["mean_recall"])
        if rule == "recall":
            return max(conf, key=lambda r: r["mean_recall"])
        if rule == "random":
            return rng.choice(top)
        if rule == "oracle":
            return None                       # handled by the caller, which has the RNet scores
        if rule == "screen_first":
            pass                              # a confirmed target never reaches the screen rung
    surv = [r for r in rows if r["mean_recall"] >= SCREEN]
    if not surv:
        return None
    if rule == "random":
        return rng.choice(surv)
    if rule == "oracle":
        return None
    if rule in ("recall", "k_recall"):
        return max(surv, key=lambda r: r["mean_recall"])
    return surv[0]                            # shipped / k_first / k_lowent / screen_first: first clearing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--run", default=os.path.join(HERE, "leaderboard_replay.json"))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "selector_replay.json"))
    a = ap.parse_args()

    R = json.load(open(a.run))
    T = {p: v["target"] for p, v in R["targets"].items()}
    rows = {r["puzzle"]: r for r in R["rows"]}
    P = load_dump(a.dump, T)
    from rna_harness.cognitive.certify import calibrated_rule_applies
    G = {p: not calibrated_rule_applies(t) for p, t in T.items()}      # topology gate, per target
    print(f"落盘池 {len(P)}/{len(T)} 个靶标,候选合计 {sum(len(v) for v in P.values())}", flush=True)

    # ---- INTEGRITY GATE ---------------------------------------------------------------------------
    elig = [p for p in P if rows.get(p, {}).get("verdict") in KNOT_VERDICTS]
    bad = []
    for p in elig:
        got = pick(P[p], "shipped", gated=G[p])
        if got is None or got["seq"] != rows[p]["sequence"]:
            bad.append(p)
    print(f"=== 完整性闸门:重实现的 shipped 规则能否复现真实交付的那条序列 ===")
    print(f"  可比靶标 {len(elig)},不一致 {len(bad)}  {bad[:8]}")
    if bad:
        print(f"  -> 重放并未复刻产品行为。按预先写下的规矩,不打印任何对比数字。")
        json.dump({"integrity_pass": False, "mismatched": bad}, open(a.out, "w"), ensure_ascii=False)
        return
    print(f"  -> 通过。下面的每一条都是同一批候选上的重放。")
    ng = [p for p in elig if G[p]]
    blocked = sum(1 for p in ng for r in P[p] if r["knot_confirmed"])
    print(f"  其中 {len(ng)} 个被拓扑闸门挡在 knot_confirmed 之外(H-type,交叉茎规则未在该拓扑标定),"
          f"这些靶标上共 {blocked} 条【已被面板确认】的候选按设计被跳过,交付走筛通道的第一条。\n")

    # ---- SCORE EVERY DUMPED CANDIDATE -------------------------------------------------------------
    jobs = [((p, i), r["seq"], T[p]) for p in elig for i, r in enumerate(P[p])]
    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 4))
    sc = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, [jobs[i:i + CH] for i in range(0, len(jobs), CH)]):
            sc.update(dict(res))
    f1 = {p: {P[p][i]["seq"]: sc[(p, i)] for i in range(len(P[p])) if (p, i) in sc} for p in elig}
    print(f"打分完成 {len(sc)} 条\n", flush=True)

    RULES = ["shipped", "k_first", "k_lowent", "k_recall", "recall", "screen_first", "random", "oracle"]
    rng = random.Random(a.seed)
    per = collections.defaultdict(dict)
    for p in elig:
        for rule in RULES:
            if rule == "oracle":
                per[rule][p] = max(f1[p].values()) if f1[p] else None
            elif rule == "random":
                # pick() can return None when a rule has nothing eligible; an eligible target always has
                # one here, but the guard is written rather than assumed.
                draws = [pick(P[p], "random", rng, gated=G[p]) for _ in range(a.draws)]
                v = [f1[p].get(r["seq"]) for r in draws if r is not None]
                v = [x for x in v if x is not None]
                per[rule][p] = st.mean(v) if v else None
            else:
                r = pick(P[p], rule, gated=G[p])
                per[rule][p] = f1[p].get(r["seq"]) if r else None

    from m2r_pairwise import wilcoxon
    ps = [p for p in elig if all(per[r].get(p) is not None for r in RULES)]
    base = [per["shipped"][p] for p in ps]
    print(f"=== 交付质量,重放在同一批候选上({len(ps)} 个靶标)===")
    print(f"  {'规则':14s} {'交付中位':>9s} {'对 shipped':>11s} {'更好':>9s} {'p':>10s}")
    out = {}
    for rule in RULES:
        v = [per[rule][p] for p in ps]
        if rule == "shipped":
            print(f"  {rule:14s} {st.median(v):9.3f} {'—':>11s} {'—':>9s} {'—':>10s}")
            out[rule] = {"median": st.median(v)}
            continue
        d = [x - y for x, y in zip(v, base)]
        w, pv, _ = wilcoxon(d, "greater")
        print(f"  {rule:14s} {st.median(v):9.3f} {st.median(d):+11.3f} "
              f"{sum(1 for x in d if x > 0):5d}/{len(d):<3d} {'n/a' if pv is None else f'{pv:.3g}':>10s}")
        out[rule] = {"median": st.median(v), "vs_shipped": st.median(d), "p": pv,
                     "wins": sum(1 for x in d if x > 0), "n": len(d)}

    zero = {r: sum(1 for p in ps if per[r][p] == 0.0) for r in RULES}
    print(f"\n  交付 F1 = 0.000 的靶标数:" + "  ".join(f"{r} {zero[r]}" for r in RULES))
    print(f"\n=== 读法 ===")
    print(f"  oracle 是上界不是目标:本项目自己的合成零效应检验里,取最大值会凭空抬高约 +3.67。")
    print(f"  任何规则的收益跟 random 比,不跟 oracle 比。")
    best = max((r for r in RULES if r not in ("shipped", "oracle")),
               key=lambda r: out[r]["vs_shipped"])
    print(f"  最好的可实现规则:`{best}`,对出厂 {out[best]['vs_shipped']:+.3f},"
          f"p = {out[best]['p']};random 对出厂 {out['random']['vs_shipped']:+.3f}")

    json.dump({"integrity_pass": True, "n_targets": len(ps), "draws": a.draws, "screen": SCREEN,
               "rules": out, "zero_delivery": zero, "per_target": {r: per[r] for r in RULES}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
