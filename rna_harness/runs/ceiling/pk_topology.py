#!/usr/bin/env python3
"""领域 A:把假结结果按【拓扑类】重新切分 —— RNA 领域的人认这个分类,而基准编号他们不认。

WHY. The pseudoknot result -- the standard single-oracle tool signs 96 certificates an independent panel
cannot confirm, while we sign 16 it can and refuse 80 -- is this project's strongest evidence in a domain
where the FIELD cannot get accurate answers (three predictors disagree on more than 90% of sequences). But it
has only ever been reported as one number over a benchmark set. A domain reader asks a different question:
WHICH pseudoknots?

Pseudobase as shipped in RNAInvBench carries only index, sequence restraint and structure -- no biological
family labels -- so family reporting is not available and is not invented here. What IS available is the
TOPOLOGY, which is a property of the target and is computed exactly:

    helices          maximal runs of stacked pairs, across all bracket families
    crossings        which helices cross each other (i < k < j < l)
    H-type           exactly two helices, crossing once, and the second helix's 5' side sits inside the
                     first's loop while its 3' side sits outside the first entirely -- the classical
                     "hairpin loop pairs with a downstream single strand"
    kissing          two hairpins that each close, with a third helix joining their two loops
    higher-order     anything with more crossing helices than that

This is a classification a structural RNA person recognises, and it is derived, not annotated.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)


def helices(target: str):
    """Maximal stacked runs, across every bracket family."""
    from rna_harness.benchmarks.agent_suite import pair_table
    pt = pair_table(target)
    out = []
    for i, j in enumerate(pt):
        if j > i and not (i > 0 and pt[i - 1] == j + 1):
            k = 0
            while i + k < len(pt) and pt[i + k] == j - k and j - k > i + k:
                k += 1
            out.append((i, i + k - 1, j - k + 1, j))       # 5' start, 5' end, 3' start, 3' end
    return out


def classify_topology(target: str) -> dict:
    H = helices(target)
    cross = [(a, b) for a in range(len(H)) for b in range(a + 1, len(H))
             if H[a][0] < H[b][0] < H[a][3] < H[b][3] or H[b][0] < H[a][0] < H[b][3] < H[a][3]]
    kind = "nested" if not cross else "higher-order"
    if len(cross) == 1:
        a, b = cross[0]
        h1, h2 = (H[a], H[b]) if H[a][0] < H[b][0] else (H[b], H[a])
        inside_loop = h1[1] < h2[0] < h1[2]          # second helix starts inside the first's loop
        outside = h2[3] > h1[3]                      # and ends beyond the first entirely
        kind = "H-type" if (inside_loop and outside) else "crossing (non-H)"
    elif len(cross) == 2:
        kind = "kissing / chain"
    return {"n_helices": len(H), "n_crossings": len(cross), "topology": kind}


def main():
    root = os.path.join(HERE, "agent")
    pk = {r["task_id"]: r for r in json.load(open(os.path.join(root, "pk_desirna.json")))["rows"]}
    pan = {r["task_id"]: r for r in json.load(open(os.path.join(root, "desirna_panel_pool.json")))["rows"]}
    ids = [t for t in pk if t in pan]
    rows = []
    for t in ids:
        top = classify_topology(pk[t]["target"])
        p = pan[t]
        confirmed = bool(p.get("forms_T_pkiss") and p.get("forms_T_probknot") and p.get("forms_T_shapeknots"))
        single = bool(p.get("forms_T_pkiss"))
        rows.append({"task_id": t, "n": pk[t]["n"], "n_pk_pairs": pk[t]["n_pk_pairs"],
                     **top, "single_oracle_claims": single, "panel_confirms": confirmed})

    print(f"  {len(rows)} 个假结靶标,按拓扑类切分(分类由靶结构算出,非标注)\n")
    print(f"  {'拓扑类':<20} {'条数':>5} {'中位长度':>9} {'单裁声称':>10} {'面板确认':>10} {'无法确认':>10}")
    out = {}
    for kind, _ in Counter(r["topology"] for r in rows).most_common():
        sub = [r for r in rows if r["topology"] == kind]
        L = sorted(r["n"] for r in sub)
        sc = sum(1 for r in sub if r["single_oracle_claims"])
        pc = sum(1 for r in sub if r["panel_confirms"])
        print(f"  {kind:<20} {len(sub):>5} {L[len(L)//2]:>9} {sc:>10} {pc:>10} {sc - pc:>10}")
        out[kind] = {"n": len(sub), "median_len": L[len(L) // 2], "single_claims": sc,
                     "panel_confirms": pc, "unconfirmable": sc - pc}
    print(f"\n  {'':<20} {'':>5} {'':>9} {sum(r['single_oracle_claims'] for r in rows):>10} "
          f"{sum(r['panel_confirms'] for r in rows):>10} "
          f"{sum(r['single_oracle_claims'] and not r['panel_confirms'] for r in rows):>10}  <- 合计")
    print("\n  螺旋数分布:", dict(Counter(r["n_helices"] for r in rows).most_common(6)))
    print("  交叉数分布:", dict(Counter(r["n_crossings"] for r in rows).most_common(6)))
    json.dump({"n": len(rows), "by_topology": out, "rows": rows},
              open(os.path.join(HERE, "pk_topology.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote rna_harness/runs/ceiling/pk_topology.json")


if __name__ == "__main__":
    main()
