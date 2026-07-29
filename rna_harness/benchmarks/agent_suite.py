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

AND ONE COLUMN THAT MATTERS MORE THAN THE SCORE. On a restructurable item, did the agent AVOID the conflict up
front, or hit it and RECOVER? Only the second is Claude Code's mechanism, and this project has never measured
it -- the one live trajectory we have is an avoidance. A loop that can only anticipate is a smarter pipeline.
"""
from __future__ import annotations

from typing import Dict, List

WC = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def _hairpin(stem: int, loop: int) -> str:
    return "(" * stem + "." * loop + ")" * stem


def _pairs(stem: int, loop: int):
    n = 2 * stem + loop
    return [(i, n - 1 - i) for i in range(stem)]


def _conflicting_motifs(stem: int, loop: int, m: int):
    """A pair of motifs of length m at the two ends whose pinned positions cannot pair."""
    a, b = "G" * m, "G" * m                       # G-G is not a pair in any orientation
    n = 2 * stem + loop
    return [{"motif": a, "at": 0}, {"motif": b, "at": n - m}]


def _compatible_motif(stem: int, loop: int, m: int):
    """A motif pinned inside the LOOP: unpaired, so it cannot conflict with anything."""
    return [{"motif": ("A" * m), "at": stem + max(0, (loop - m) // 2)}]


def build(n_per_class: int = 6) -> List[Dict]:
    """A deterministic suite. Same input, same items, every time -- no sampling, nothing to re-roll."""
    items: List[Dict] = []
    shapes = [(4, 5), (5, 4), (6, 4), (6, 6), (7, 4), (5, 6), (8, 4), (4, 8)]

    for k in range(n_per_class):
        stem, loop = shapes[k % len(shapes)]
        n = 2 * stem + loop
        m = min(3, loop)
        c = _compatible_motif(stem, loop, m)
        items.append({
            "id": f"direct-{k}", "class": "direct", "expect": "certified",
            "request": (f"给我一个茎 {stem} 对、环 {loop} nt 的发夹，"
                        f"并且第 {c[0]['at'] + 1} 位开始的 {m} 个碱基必须固定为 {c[0]['motif']}"),
            "shape": {"stem": stem, "loop": loop}, "constraints": c,
            "why": f"the pinned bases sit in the loop, which is unpaired, so nothing can conflict",
        })

    for k in range(n_per_class):
        stem, loop = shapes[(k + 2) % len(shapes)]
        m = min(4, stem)
        cs = _conflicting_motifs(stem, loop, m)
        items.append({
            "id": f"restructurable-{k}", "class": "restructurable", "expect": "certified",
            "request": (f"我要一个茎 {stem} 对、环 {loop} nt 的发夹，"
                        f"序列开头必须是 {cs[0]['motif']}，结尾也必须是 {cs[1]['motif']}"),
            "shape": {"stem": stem, "loop": loop}, "constraints": cs,
            "why": ("in the plain hairpin those pinned positions pair with each other and G cannot pair with "
                    "G; the total length is not fixed, so spacers at both ends resolve it"),
        })

    for k in range(n_per_class):
        stem, loop = shapes[(k + 4) % len(shapes)]
        n = 2 * stem + loop
        m = min(4, stem)
        cs = _conflicting_motifs(stem, loop, m)
        items.append({
            "id": f"impossible-{k}", "class": "impossible", "expect": "refused",
            "request": (f"我要一个茎 {stem} 对、环 {loop} nt 的发夹，总长必须正好 {n} nt，一个碱基都不能多，"
                        f"并且序列开头必须是 {cs[0]['motif']}，结尾也必须是 {cs[1]['motif']}"),
            "shape": {"stem": stem, "loop": loop}, "constraints": cs,
            "why": (f"positions 0..{m-1} pair with {n-1}..{n-m}, both pinned to G, and G-G is not a pair; "
                    f"the length is fixed so no spacer can be added"),
        })
    return items


def feasible(stem: int, loop: int, constraints: List[Dict]) -> Dict:
    """The answer key, computed exactly: can these pins co-exist on this shape?"""
    n = 2 * stem + loop
    r = ["N"] * n
    for c in constraints:
        for k, ch in enumerate(c["motif"]):
            r[c["at"] + k] = ch
    conflicts = [{"i": i, "j": j, "bases": f"{r[i]}-{r[j]}"}
                 for i, j in _pairs(stem, loop)
                 if r[i] in "ACGU" and r[j] in "ACGU" and (r[i], r[j]) not in WC]
    return {"feasible": not conflicts, "conflicts": conflicts, "restraint": "".join(r), "n": n}


if __name__ == "__main__":
    S = build()
    print(f"{len(S)} items")
    for it in S:
        f = feasible(it["shape"]["stem"], it["shape"]["loop"], it["constraints"])
        mark = "OK " if ((it["class"] == "direct") == f["feasible"]) else "!! "
        print(f"  {mark}{it['id']:<18} expect={it['expect']:<10} naive-shape feasible={f['feasible']}")
        print(f"      {it['request']}")
