"""A benchmark for the AGENT LOOP, with an answer key we compute instead of search for.

WHY THIS EXISTS BEFORE ANY NEW CAPABILITY. Points 1 and 2 of the product table carry numbers -- 83.3 +/- 1.25
solved, $1.10 +/- 0.02, 9.5 s +/- 0.07, replicated over three seeds and two populations. Point 3 (scope and
open-problem reasoning) carries one screenshot of one successful trajectory. Without a metric, "we expanded
the reasoning" is exactly the kind of claim that collapsed five times today under measurement, so the metric
comes first.

WHAT MAKES THIS CHEAP. Constrained-design feasibility is EXACTLY COMPUTABLE: given a shape and a set of pinned
positions, whether the pins can co-exist follows from the pair table. So the answer key is generated, not
solved for -- unlike Eterna100, which needed people. Generating the suite costs nothing and running it costs
about $0.001 per item.

THREE CLASSES, each with a DIFFERENT correct behaviour, because a single "solve rate" would hide the thing
worth measuring:

  direct          the pins are compatible with the obvious shape.
                  correct = certified, and cheaply. Tests that the easy path stays easy.
  restructurable  the pins conflict with the obvious shape, but the request does NOT fix the total length, so
                  moving the pinned regions out of the paired positions solves it.
                  correct = certified. This is the reasoning case: the agent must change the PROBLEM, not
                  buy more budget.
  impossible      the pins conflict AND the request fixes the total length, so no restructure is available.
                  correct = refuse, and say which positions conflict. An agent that burns ten steps here is
                  worse than one that stops at three with a reason.

SHAPE FAMILIES. Hairpins, multiway junctions and H-type pseudoknots -- the three the assembler can build.
The junction and knot families exist so that unlocking them cannot silently regress: they are judged by
different oracles (exact MFE for the two nested families, the calibrated knot rule for the third), so a
change that breaks routing shows up here rather than in a user's answer.

AND ONE COLUMN THAT MATTERS MORE THAN THE SCORE. On a restructurable item, did the agent AVOID the conflict up
front, or hit it and RECOVER? Only the second is Claude Code's mechanism, and this project has never measured
it -- the one live trajectory we have is an avoidance. A loop that can only anticipate is a smarter pipeline.
"""
from __future__ import annotations

from typing import Dict, List

WC = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def _hairpin(stem: int, loop: int) -> str:
    return "(" * stem + "." * loop + ")" * stem


def _junction(closing: int, branches, spacer: int = 2) -> str:
    inner = ["(" * b[0] + "." * b[1] + ")" * b[0] for b in branches]
    return "(" * closing + ("." * spacer) + ("." * spacer).join(inner) + ("." * spacer) + ")" * closing


def _knot(s1: int, s2: int, l1: int, l2: int, l3: int) -> str:
    return "(" * s1 + "." * l1 + "[" * s2 + "." * l2 + ")" * s1 + "." * l3 + "]" * s2


def pair_table(target: str):
    """0-based partner index per position, -1 when unpaired. Handles every bracket family, so the same
    feasibility rule covers nested shapes and pseudoknots without a special case."""
    pt = [-1] * len(target)
    stacks = {}
    close = {")": "(", "]": "[", "}": "{", ">": "<"}
    for i, ch in enumerate(target):
        if ch in "([{<":
            stacks.setdefault(ch, []).append(i)
        elif ch in close:
            st = stacks.get(close[ch]) or []
            if st:
                j = st.pop()
                pt[i], pt[j] = j, i
    return pt


def feasible(target: str, constraints) -> dict:
    """The answer key, computed exactly: can these pins co-exist on this shape?"""
    r = ["N"] * len(target)
    for c in constraints:
        for k, ch in enumerate(c["motif"]):
            r[c["at"] + k] = ch
    pt = pair_table(target)
    conflicts = [{"i": i, "j": j, "bases": f"{r[i]}-{r[j]}"}
                 for i, j in enumerate(pt)
                 if j > i and r[i] in "ACGU" and r[j] in "ACGU" and (r[i], r[j]) not in WC]
    return {"feasible": not conflicts, "conflicts": conflicts, "restraint": "".join(r), "n": len(target)}


def _conflicting(target: str, m: int):
    """Pin G on BOTH SIDES of a real stacked stem, so the pins are forced to pair and cannot.

    An earlier version pinned the two ENDS of the sequence, which only conflicts when the ends pair with each
    other. That is true of a hairpin and false of an H-type knot, where position 0 pairs with a ')' in the
    middle and the last position pairs with a '[' -- so four "conflicting" pseudoknot items were actually
    feasible and the suite's own sanity check flagged them. Reading the pair table works for every topology.
    """
    pt = pair_table(target)
    for i0, j0 in enumerate(pt):
        if j0 > i0 and all(pt[i0 + k] == j0 - k for k in range(m)):
            return [{"motif": "G" * m, "at": i0}, {"motif": "G" * m, "at": j0 - m + 1}]
    return None


def _in_loop(target: str, m: int):
    """A pin inside the FIRST hairpin loop -- unpaired, so it cannot conflict with anything."""
    pt = pair_table(target)
    for i, j in enumerate(pt):
        if j > i and all(pt[k] < 0 for k in range(i + 1, j)) and (j - i - 1) >= m:
            return [{"motif": "A" * m, "at": i + 1}]
    return [{"motif": "A" * m, "at": next(k for k, c in enumerate(target) if c == ".")}]


def _shapes():
    """(family, target, the request in words). The words must pin the shape down COMPLETELY.

    They did not, and it cost two wrong scores. "一个 H 型假结,两个交叉茎分别 4 对和 4 对" plus "total 27 nt"
    does NOT determine a single string -- the three loop lengths are still free. The agent used that freedom
    exactly as it should: it kept both stems, kept the length, redistributed the loops so the pinned bases no
    longer paired, and got certified on two items whose answer key said "impossible". The key was wrong, not
    the agent. A hairpin has no such freedom (stem + loop + length is one string), which is why the hairpin-
    only version of this suite never exposed it.
    """
    out = []
    for stem, loop in ((6, 4), (5, 6)):
        out.append(("hairpin", _hairpin(stem, loop), f"一个茎 {stem} 对、环 {loop} nt 的发夹"))
    for closing, branches, spacer in ((5, [(4, 4), (4, 4)], 2), (4, [(3, 4), (3, 4), (3, 4)], 1)):
        br = "、".join(f"茎 {b[0]} 对环 {b[1]} nt" for b in branches)
        out.append(("junction", _junction(closing, branches, spacer),
                    f"一个多分支结构：闭合茎 {closing} 对，里面并排 {len(branches)} 个发夹（{br}），"
                    f"每段之间隔 {spacer} nt"))
    for s1, s2, l1, l2, l3 in ((4, 4, 4, 3, 4), (5, 4, 4, 3, 5)):
        out.append(("pseudoknot", _knot(s1, s2, l1, l2, l3),
                    f"一个 H 型假结：茎1 {s1} 对、茎2 {s2} 对，三段环依次 {l1} nt、{l2} nt、{l3} nt"))
    return out


SHAPES = _shapes()


def build(n_per_class: int = 6):
    """Deterministic: same input, same items, every time. No sampling, nothing to re-roll."""
    items = []
    for k in range(n_per_class):
        fam, t, phrase = SHAPES[k % len(SHAPES)]
        c = _in_loop(t, 3)
        items.append({"id": f"direct-{k}", "class": "direct", "family": fam, "expect": "certified",
                      "target": t, "constraints": c,
                      "request": (f"给我这样一个结构:{phrase}，"
                                  f"并且第 {c[0]['at'] + 1} 位开始的 3 个碱基必须固定为 {c[0]['motif']}"),
                      "why": "the pinned bases sit in an unpaired loop, so nothing can conflict"})
    for k in range(n_per_class):
        fam, t, phrase = SHAPES[(k + 2) % len(SHAPES)]
        cs = _conflicting(t, 4)
        items.append({"id": f"restructurable-{k}", "class": "restructurable", "family": fam,
                      "expect": "certified", "target": t, "constraints": cs,
                      # NO HINT THAT THE LENGTH IS FREE. Saying "总长不限,可以加长" out loud raised the
                      # score and destroyed the measurement: recovery went to 0 and anticipation to 6/6,
                      # because an announced freedom is planned around, not discovered. The silence is the
                      # experiment -- what is being measured is whether the loop finds the room by hitting
                      # the wall. (The over-refusal this briefly caused was a prompt bug, fixed there.)
                      "request": (f"我要这样一个结构:{phrase}，"
                                  f"第 {cs[0]['at'] + 1} 位起的 4 个碱基必须是 GGGG，"
                                  f"第 {cs[1]['at'] + 1} 位起的 4 个也必须是 GGGG"),
                      "why": ("those pinned positions pair with each other in the obvious shape and G cannot "
                              "pair with G; the length is not fixed, so spacers resolve it")})
    for k in range(n_per_class):
        fam, t, phrase = SHAPES[(k + 4) % len(SHAPES)]
        cs = _conflicting(t, 4)
        items.append({"id": f"impossible-{k}", "class": "impossible", "family": fam, "expect": "refused",
                      "target": t, "constraints": cs,
                      "request": (f"我要这样一个结构:{phrase}，总长必须正好 {len(t)} nt，"
                                  f"一个碱基都不能多，并且第 {cs[0]['at'] + 1} 位起的 4 个碱基必须是 GGGG，"
                                  f"第 {cs[1]['at'] + 1} 位起的 4 个也必须是 GGGG"),
                      "why": ("the pinned positions pair with each other in the ONLY shape the request permits, "
                              "and G-G is not a pair; the length is fixed, so nothing can move")})
    return items


if __name__ == "__main__":
    S = build()
    print(f"{len(S)} items")
    for it in S:
        f = feasible(it["target"], it["constraints"])
        mark = "OK " if ((it["class"] == "direct") == f["feasible"]) else "!! "
        print(f"  {mark}{it['id']:<18} {it['family']:<11} expect={it['expect']:<10} "
              f"naive feasible={f['feasible']}  n={f['n']}")
