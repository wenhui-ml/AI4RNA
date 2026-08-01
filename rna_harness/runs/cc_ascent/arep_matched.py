#!/usr/bin/env python3
"""R9 M4: make the positive control's comparison PAIRED and difficulty-matched.

Why this file exists. The positive control's headline is a carry-over of 0.302 for the frozen agent-written
operators against 0.091 for the hand-built panel operator. As printed by arep_stats.py that is a comparison of
two CONDITIONAL rates on two DIFFERENT denominators -- each arm's own successes. Table 11's own caption
forbids exactly this reading for the transplant arms ("a conditional rate can be inflated by collapsing its
own denominator"), so applying a weaker standard to the arm that carries the paper's discriminating-power
claim is not defensible. carryover.py already does the matched version for gamer-vs-panel; this does it for
agent-vs-hand-built, which was the gap.

The matched statistic. Both arms optimize the SAME predicate (pKiss AND ProbKnot) and both are scored by
ShapeKnots, which is held out of both loops. So restricting to the (cell, target) pairs where BOTH arms
satisfy that shared predicate gives a genuinely paired comparison: same target, same nominal seed, same
execution index, same in-loop bar cleared by both, and the only question left is whether the held-out
predictor confirms. Exact two-sided McNemar on the discordant pairs.

Pairing note. The agent operators are bit-identical across the three executions inside a seed (arep_stats.py),
the hand-built operator is not, so each agent cell is paired against a distinct hand-built execution. That
averages the hand-built arm's execution noise into the comparison rather than letting one draw stand for it.

Also reports the per-model split, because "agent-written operators reach 0.302" is a claim about the two
models of six that ran without runtime failures -- a model-level survivorship selection, not only a
target-level one -- and the paper's quantifier has to say so.

Offline over stored sweeps; no oracle calls.
"""
import json, math, os, glob

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
SEEDS, EXECS = [0, 1, 2, 3, 4], [0, 1, 2]
OPS = ["gemini31pro", "kimik3"]


def clopper(k, n, upper=False):
    if n == 0:
        return float("nan")
    if not upper and k == 0:
        return 0.0
    if upper and k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        m = (lo + hi) / 2
        tail = sum(math.comb(n, i) * m ** i * (1 - m) ** (n - i)
                   for i in (range(k, n + 1) if not upper else range(0, k + 1)))
        if (tail < 0.025) == (not upper):
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(2 * sum(math.comb(n, k) * 0.5 ** n for k in range(m + 1)), 1.0)


def load_agent(op):
    """-> {(s,e): {target: sk_confirmed}} over targets where pKiss AND ProbKnot both hold."""
    out = {}
    for s in SEEDS:
        for e in EXECS:
            p = os.path.join(f"arep_{op}_dev", f"s{s}_e{e}.json")
            if not os.path.exists(p):
                continue
            out[(s, e)] = {r["id"]: bool(r.get("shapeknots"))
                           for r in json.load(open(p))["rows"]
                           if r.get("pkiss") and r.get("probknot")}
    return out


def load_handbuilt():
    """The hand-built panel operator from the same 15 (seed, execution) cells; same predicate."""
    out = {}
    for s in SEEDS:
        for e in EXECS:
            p = os.path.join(f"gvp_e{e}_s{s}", "sweep.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            if len(d.get("rows") or []) < 60:
                continue
            per = {}
            for r in d["rows"]:
                v = r["arms"].get("panel_auto")
                if v is None or v.get("error") or not v.get("solved"):
                    continue
                per[r["id"]] = bool((v.get("panel") or {}).get("shapeknots"))
            out[(s, e)] = per
    return out


def report(label, agent_cells, hand):
    """Unmatched rates on each arm's own successes, then the paired difficulty-matched McNemar."""
    ak = an = hk = hn = 0
    for cell, per in agent_cells.items():
        an += len(per); ak += sum(per.values())
        h = hand.get(cell, {})
        hn += len(h); hk += sum(h.values())
    n = b = c = mk_a = mk_h = 0
    for cell, per in agent_cells.items():
        h = hand.get(cell)
        if not h:
            continue
        for t in set(per) & set(h):
            n += 1
            mk_a += per[t]; mk_h += h[t]
            b += int(per[t] and not h[t])
            c += int(h[t] and not per[t])
    print(f"\n  {label}")
    print(f"    unmatched (each arm on its own successes)")
    print(f"      agent      {ak:4d}/{an:<5d} = {ak/max(1,an):.3f}  "
          f"[{clopper(ak,an):.3f},{clopper(ak,an,True):.3f}]")
    print(f"      hand-built {hk:4d}/{hn:<5d} = {hk/max(1,hn):.3f}  "
          f"[{clopper(hk,hn):.3f},{clopper(hk,hn,True):.3f}]")
    if not n:
        print("    difficulty-matched: no overlapping cells")
        return
    print(f"    difficulty-matched (both arms cleared pKiss AND ProbKnot on the same target/cell), n={n}")
    print(f"      agent      {mk_a:4d}/{n:<5d} = {mk_a/n:.3f}  "
          f"[{clopper(mk_a,n):.3f},{clopper(mk_a,n,True):.3f}]")
    print(f"      hand-built {mk_h:4d}/{n:<5d} = {mk_h/n:.3f}  "
          f"[{clopper(mk_h,n):.3f},{clopper(mk_h,n,True):.3f}]")
    print(f"      paired McNemar b={b} (agent only), c={c} (hand-built only), exact p={mcnemar(b,c):.4g}; "
          f"risk difference {(b-c)/n:+.4f}")


def main():
    print("=" * 100)
    print("POSITIVE CONTROL, PAIRED: carry-over to held-out ShapeKnots, agent-written vs hand-built")
    print("=" * 100)
    print("Both arms optimize pKiss AND ProbKnot and are scored by ShapeKnots, held out of both loops.")
    print("Matched cells are (nominal seed, execution index); matched units are targets both arms solved.")

    hand = load_handbuilt()
    print(f"\n  hand-built cells loaded: {len(hand)}")
    pooled = {}
    for op in OPS:
        a = load_agent(op)
        print(f"  {op} cells loaded: {len(a)}")
        report(op, a, hand)
        for cell, per in a.items():
            pooled.setdefault(cell, {}).update({(op, t): v for t, v in per.items()})

    # Pooling across models pools the ROWS, so pair each model's cell against the same hand-built cell.
    n = b = c = mk_a = mk_h = 0
    for op in OPS:
        for cell, per in load_agent(op).items():
            h = hand.get(cell)
            if not h:
                continue
            for t in set(per) & set(h):
                n += 1
                mk_a += per[t]; mk_h += h[t]
                b += int(per[t] and not h[t])
                c += int(h[t] and not per[t])
    print(f"\n  POOLED over the two models, difficulty-matched, n={n}")
    print(f"    agent      {mk_a:4d}/{n:<5d} = {mk_a/n:.3f}  "
          f"[{clopper(mk_a,n):.3f},{clopper(mk_a,n,True):.3f}]")
    print(f"    hand-built {mk_h:4d}/{n:<5d} = {mk_h/n:.3f}  "
          f"[{clopper(mk_h,n):.3f},{clopper(mk_h,n,True):.3f}]")
    print(f"    paired McNemar b={b} (agent only), c={c} (hand-built only), exact p={mcnemar(b,c):.4g}; "
          f"risk difference {(b-c)/n:+.4f}")
    print("\n  Note on the quantifier. These are the 2 of 6 invented operators with no runtime failures;")
    print("  the other 4 time out on 19-32 of 60 targets and are excluded. The claim this supports is")
    print("  'some agent-written operators carry over better', not a property of agent-written operators.")


if __name__ == "__main__":
    main()
