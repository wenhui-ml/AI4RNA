"""designaudit.corpora -- adapters that turn a published probing corpus into constructs.

An adapter's whole job is to say, for one corpus, where the design window is, which construct is
the wild type, what the declared target structure is, and what the measured outcome is called.
`mining.mine` does the rest and is corpus-agnostic.

There is one adapter here, for the OpenKnot 7a mutate-map-rescue release, because that is the one
corpus we have verified end to end. It is written out in full rather than hidden behind a config
so the next adapter can be written by copying it: the four things it supplies are the four things
any probing corpus must be interrogated for, and a corpus that cannot supply them cannot be mined
-- which is a finding about that corpus, not a gap in this module.

The window matters more than it looks. Constructs in these libraries carry adapters, barcodes and
padding outside the designed region, so two constructs differing only by a compensatory flip
inside the window can differ at a dozen positions outside it. Mining whole sequences silently
returns nothing, and "nothing" is indistinguishable from "this corpus has no interventions"
unless the window is declared. We produced that false negative once on RMDB before catching it.
"""
from __future__ import annotations

import csv

OPEN = "([{<" + "".join(chr(c) for c in range(ord("A"), ord("Z") + 1))
CLOSE = ")]}>" + "".join(chr(c) for c in range(ord("a"), ord("z") + 1))


def parse_pairs(dbn):
    """0-based i -> j for the round-bracket layer and every pseudoknot layer the string uses."""
    stacks, pairs = {}, {}
    for k, ch in enumerate(dbn):
        if ch in OPEN:
            stacks.setdefault(OPEN.index(ch), []).append(k)
        elif ch in CLOSE:
            st = stacks.get(CLOSE.index(ch))
            if st:
                i = st.pop()
                pairs[i] = k
                pairs[k] = i
    return pairs


def openknot_m2r(path, outcome="rescue_factor", sn_filter=True):
    """OpenKnot 7a mutate-map-rescue -> constructs for `mining.mine`.

    Wild types are the rows with no mutation declared, one per (puzzle, method) design cell. The
    window is the corpus's own sub_start/sub_end, converted to 0-based half-open. The
    signal-to-noise filter is applied to mutants only: dropping a wild type would orphan every
    mutant in its cell and turn a quality cut into a selection on the design.
    """
    csv.field_size_limit(10 ** 9)
    out = []
    for d in csv.DictReader(open(path)):
        is_wt = d["mutA"] in ("", "NA", "nan") and d["mutB"] in ("", "NA", "nan")
        if sn_filter and not is_wt and d.get("SN_filter") not in ("1", "1.0"):
            continue
        try:
            ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        except (ValueError, KeyError):
            continue
        y = d.get(outcome, "")
        sn = d.get("signal_to_noise")
        out.append({
            "design": (d["puzzle"], d["method"]),
            "sequence": d["sequence"],
            "window": (ss - 1, se),
            "pairs": parse_pairs(d["target_structure"]),
            "is_wild_type": is_wt,
            "outcome": (None if y in ("", "NA", "nan") else float(y)),
            "confidence": (float(sn) if sn not in (None, "", "NA", "nan") else 1.0),
            "meta": {"puzzle": d["puzzle"], "method": d["method"],
                     "target_structure": d["target_structure"]},
        })
    return out


def annotate_context(interventions):
    """depth and crossing for each intervention, from the target structure it was mined under.

    depth is how many pairs stack contiguously on the shallower side; crossed says whether any
    other declared pair crosses this one. Both are properties of the design and not of the
    measurement, so they are legitimate stratifying covariates -- conditioning on them cannot
    condition on the outcome.
    """
    for x in interventions:
        pairs = parse_pairs(x.meta.get("target_structure", ""))
        a, b = x.i, x.j
        dn = 0
        while pairs.get(a - dn - 1) == b + dn + 1:
            dn += 1
        up = 0
        while pairs.get(a + up + 1) == b - up - 1:
            up += 1
        x.meta["depth"] = min(dn, up)
        x.meta["depth3"] = min(min(dn, up), 3)
        x.meta["crossed"] = bool(any(p < a < q < b or a < p < b < q
                                     for p, q in pairs.items() if p < q))
    return interventions
