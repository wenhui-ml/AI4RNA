"""A benchmark whose SHAPES WE DID NOT DESIGN.

WHY THIS EXISTS. Every number behind "解题范围更大 / 对未知开放问题有更精准的推理" came from a suite this
project generated: our shapes, our perturbations, our computed answer key. That is a valid test of the
reasoning mechanism and worthless as evidence about problems we did not anticipate, because nothing in it
could ever surprise us. This module removes the half we control that matters most -- the structures come from
nature.

SOURCE. The same bpRNA-derived nested pool that already backs benchmarks/bprna_nested.py: RNAInvBench's
inverse_rna_folding train split, filtered to nested + valid, deduplicated by dot-bracket, sorted by Id, taken
at an even stride. Deterministic, declared before the run, and unrelated to how the results come out.

WHAT MAKES THESE HARDER THAN OUR OWN SHAPES, concretely: our generator can only build five element types
(hairpin, spacer, internal loop, multiloop, H-type pseudoknot), and almost no natural structure is one of
those. A bulge two thirds of the way up a stem, a three-way junction whose branches are themselves
interrupted, an 11-nt loop closed by a single pair -- these are ordinary in bpRNA and unbuildable here. Until
`use_target` existed they could not enter the agent loop at all, which is itself the sharpest measurement of
how wide "范围" actually was.

THREE CLASSES. The first is a scope measurement, the other two are scored against an exact key:

  plain       just the structure. NOT scored right/wrong: a natural structure carries no guarantee that any
              sequence folds to it under the Turner model, so "expect certified" would not be a valid key.
              What it measures is the LOSS from going through the agent instead of calling the ladder
              directly -- the loop is only worth having if that loss is small.
  pinned      a motif pinned at unpaired positions, taken from the structure itself. Exactly feasible, so
              `certified` is the right answer and anything else is a miss.
  conflicted  the same motif pinned on BOTH sides of a real stem, so the two pinned regions must pair and
              cannot, plus an explicit total-length lock so no restructure is available. `refused` is the
              right answer, and naming the conflicting positions is the behaviour under test.

WHAT IS STILL OURS, stated so it is not mistaken for external: the perturbation rule and the wording. Only
the shapes are nature's. That is a smaller claim than "an external benchmark" and a much larger one than the
generated suite, which had nothing external in it at all.

SELECTION, declared: length 20..80 nt so the design cost stays bounded, and only structures for which
ViennaRNA's own inverse fold reproduces the shape -- otherwise `pinned`'s answer key would be asserting that
a sequence exists when none may. Both filters report how many they dropped; a benchmark that quietly discards
what it cannot handle reports the survivors as if they were the population.
"""
from __future__ import annotations

from typing import Dict, List, Optional

MIN_N, MAX_N = 20, 80
PIN = 3          # length of the feasible pin
CPIN = 4         # length of each conflicting pin


def pair_table(target: str) -> List[int]:
    pt = [-1] * len(target)
    stacks: Dict[str, List[int]] = {}
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
    """The answer key, computed exactly from the pair table -- never searched for."""
    from rna_harness.cognitive.agent import WC
    r = ["N"] * len(target)
    for c in constraints:
        for k, ch in enumerate(c["motif"]):
            r[c["at"] + k] = ch
    pt = pair_table(target)
    conflicts = [{"i": i, "j": j} for i, j in enumerate(pt)
                 if j > i and r[i] in "ACGU" and r[j] in "ACGU" and (r[i], r[j]) not in WC]
    return {"feasible": not conflicts, "conflicts": conflicts, "restraint": "".join(r)}


def _unpaired_run(target: str, m: int) -> Optional[int]:
    """Start of the first run of m unpaired positions -- a pin there cannot conflict with anything."""
    pt = pair_table(target)
    run = 0
    for i, j in enumerate(pt):
        run = run + 1 if j < 0 else 0
        if run >= m:
            return i - m + 1
    return None


def _stacked_stem(target: str, m: int):
    """Both sides of a real stacked stem of at least m pairs: pinning G on each side forces G to pair with G."""
    pt = pair_table(target)
    for i0, j0 in enumerate(pt):
        if j0 > i0 and i0 + m <= len(pt) and all(pt[i0 + k] == j0 - k for k in range(m)) \
                and j0 - m + 1 > i0 + m - 1:
            return [{"motif": "G" * m, "at": i0}, {"motif": "G" * m, "at": j0 - m + 1}]
    return None


def build(n_per_class: int = 8, pool: int = 400, path: Optional[str] = None) -> List[Dict]:
    """Deterministic. Returns items plus, on the first item, the accounting of what selection dropped."""
    from .bprna_nested import load_bprna_nested
    from ..cognitive.agent import designable_probe

    rows = load_bprna_nested(pool, path=path)
    dropped = {"length": 0, "no_unpaired_run": 0, "no_stem": 0, "not_designable": 0}
    kept, examined = [], 0
    for r in rows:
        examined += 1
        t = str(r["target"])
        if not (MIN_N <= len(t) <= MAX_N):
            dropped["length"] += 1
            continue
        if _unpaired_run(t, PIN) is None:
            dropped["no_unpaired_run"] += 1
            continue
        if _stacked_stem(t, CPIN) is None:
            dropped["no_stem"] += 1
            continue
        if designable_probe(t).get("designable") is not True:
            dropped["not_designable"] += 1
            continue
        kept.append(r)
        if len(kept) >= n_per_class:
            break

    items = []
    for k, r in enumerate(kept[:n_per_class]):
        t = str(r["target"])
        items.append({"id": f"plain-{k}", "class": "plain", "expect": None, "target": t,
                      "source_id": r["id"], "constraints": [],
                      "request": f"我有一个结构 {t} ，请给我一条能折叠成它的 RNA 序列",
                      "why": "not scored right/wrong -- it measures what the loop costs against a direct call"})
    for k, r in enumerate(kept[:n_per_class]):
        t = str(r["target"])
        at = _unpaired_run(t, PIN)
        c = [{"motif": "A" * PIN, "at": at}]
        items.append({"id": f"pinned-{k}", "class": "pinned", "expect": "certified", "target": t,
                      "source_id": r["id"], "constraints": c,
                      "request": (f"我有一个结构 {t} ，请给我一条能折叠成它的 RNA 序列，"
                                  f"并且第 {at + 1} 位起的 {PIN} 个碱基必须是 {'A' * PIN}"),
                      "why": "the pinned bases sit on unpaired positions, so nothing can conflict"})
    for k, r in enumerate(kept[:n_per_class]):
        t = str(r["target"])
        cs = _stacked_stem(t, CPIN)
        items.append({"id": f"conflicted-{k}", "class": "conflicted", "expect": "refused", "target": t,
                      "source_id": r["id"], "constraints": cs,
                      "request": (f"我有一个结构 {t} ，请给我一条能折叠成它的 RNA 序列，总长必须正好 "
                                  f"{len(t)} nt，一个碱基都不能多；第 {cs[0]['at'] + 1} 位起的 {CPIN} 个碱基"
                                  f"必须是 {'G' * CPIN}，第 {cs[1]['at'] + 1} 位起的 {CPIN} 个也必须是 "
                                  f"{'G' * CPIN}"),
                      "why": "those pinned regions pair with each other and G cannot pair with G; length locked"})
    if items:
        # `examined`, not len(rows): the scan stops as soon as enough survive, so quoting the whole pool
        # would make the filters look far more selective than they were.
        items[0]["selection"] = {"pool_size": len(rows), "examined": examined, "kept": len(kept),
                                 "dropped": dropped}
    return items


if __name__ == "__main__":
    S = build()
    print(f"{len(S)} items from structures this project did not design")
    print(f"  selection: {S[0].get('selection')}\n")
    for it in S:
        f = feasible(it["target"], it["constraints"])
        want_feasible = it["class"] != "conflicted"
        mark = "OK " if f["feasible"] == want_feasible else "!! "
        print(f"  {mark}{it['id']:<14} bpRNA {it['source_id']:<16} n={len(it['target']):<3} "
              f"expect={str(it['expect']):<10} naive feasible={f['feasible']}")
    print("\n  a shape our own generator cannot build, for contrast:")
    print(f"    {S[0]['target']}")
