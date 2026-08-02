#!/usr/bin/env python3
"""Does anyone ASK for the requirements nobody's tool EXPOSES? The ambiguity the option surface cannot settle.

WHERE THIS COMES FROM. The external requirement distribution (external_requirements.py) found a clean split:
every requirement this project measured as FREE is one somebody built an option for, and every requirement it
measured as a GAP -- except composition -- is one NOBODY built an option for. That is the shape the window
criterion predicts, arrived at by a route entirely independent of our benchmarks.

But it leaves one reading open, and the option surface cannot close it:

    nobody exposes it   ==   genuine white space        (tool authors cannot deliver it)
                        OR   nobody wants it            (tool authors were never asked)

Those imply opposite roadmaps. If it is white space, co-transcriptional folding and two-state switches are
the product. If nobody wants them, three proven gaps are facts about our own imagination and the honest move
is to say so and stop.

WHAT SEPARATES THEM is demand rather than supply: how often the requirement is NAMED in the literature. A
tool's option surface records what its authors could build; a paper's title and abstract record what someone
set out to do.

THE PROTOCOL, and its one weakness stated plainly. Query terms are declared in this file BEFORE any count is
read, one obvious term per requirement type, and the exact query string is printed with every number so the
whole thing is re-runnable by anyone against the same public API. What I still choose is the terms, and no
protocol removes that; printing them is the mitigation, not a cure.

Counts are scoped two ways on purpose:
    bare      the term anywhere in PubMed        how much the concept is talked about at all
    in-design the term AND an RNA-design phrase  how much it is talked about BY PEOPLE DESIGNING RNA

The second is the one that matters, and the first is kept because a term that is rare everywhere and a term
that is common but never appears in design papers are different findings.

READ IT AS ORDINAL, NOT ABSOLUTE. Hit counts depend on indexing and phrasing, so 400 versus 40 is a signal
and 44 versus 40 is not. Nothing here is a p-value.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# The RNA-design scope, declared once and applied identically to every term.
DESIGN = ('("RNA design" OR "RNA sequence design" OR "inverse folding" OR "sequence design" '
          'OR "RNA engineering")')

# ONE OBVIOUS TERM PER REQUIREMENT TYPE, fixed before any count is read. Where a concept genuinely has two
# standard names in the literature both are included, because picking the rarer one would understate demand
# for a requirement we happen to have measured as a gap.
TERMS = {
    "composition-GC":      '("GC content")',
    "temperature":         '("melting temperature" OR "thermal stability")',
    "accessibility":       '("accessibility" AND RNA)',
    "forbidden-motif":     '("sequence motif" AND RNA)',
    "ensemble-strictness": '("ensemble defect" OR "ensemble diversity")',
    "co-transcriptional":  '("cotranscriptional folding" OR "co-transcriptional folding" '
                           'OR "kinetic trap" OR "folding pathway")',
    "multi-state":         '("riboswitch" OR "toehold switch" OR "RNA switch")',
    "set-level":           '("orthogonal" AND ("RNA" OR "riboregulator"))',
    "pseudoknot":          '("pseudoknot")',
}

# What this project concluded, so the two columns sit side by side rather than in separate tables.
OURS = {
    "composition-GC": "缺口", "co-transcriptional": "缺口", "multi-state": "缺口", "set-level": "缺口",
    "temperature": "null", "accessibility": "null", "ensemble-strictness": "null",
    "forbidden-motif": "null", "pseudoknot": "判据缺失的领域",
}
EXPOSED = {  # from external_requirements.json: does any of the 5 external tools offer it
    "composition-GC": "有", "co-transcriptional": "无", "multi-state": "无", "set-level": "无",
    "temperature": "有", "accessibility": "有", "ensemble-strictness": "有",
    "forbidden-motif": "有", "pseudoknot": "有",
}


def count(term: str, retries: int = 3):
    """PubMed hit count for one query. None on failure -- never silently 0, which would read as no demand."""
    url = f"{EUTILS}?{urllib.parse.urlencode({'db': 'pubmed', 'retmode': 'json', 'rettype': 'count', 'term': term})}"
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                d = json.loads(r.read().decode())
            c = d.get("esearchresult", {}).get("count")
            if c is not None:
                return int(c)
        except Exception:                                                      # noqa: BLE001
            time.sleep(2)
    return None


def main():
    print("文献需求量:每个要求类型被【点名】的频次(查询串跑前声明,原样打印)\n")
    print(f"  设计域限定 = {DESIGN}")
    # THE DENOMINATOR. Without it "36 hits" is unreadable -- it could be 36 of 40 or 36 of 40,000. The scope
    # is computed here rather than quoted, so every share below is a number this script produced.
    total = count(DESIGN)
    print(f"  设计域总量 = {total} 篇\n")
    rows = []
    print(f"  {'要求类型':<20}{'我们的结论':<16}{'外部暴露':<10}{'PubMed 全库':>12}{'设计域内':>10}{'占设计域':>10}")
    for name, term in TERMS.items():
        bare = count(term)
        time.sleep(0.4)                    # be polite to a public endpoint
        scoped = count(f"{term} AND {DESIGN}")
        time.sleep(0.4)
        rows.append({"type": name, "term": term, "bare": bare, "in_design": scoped,
                     "ours": OURS.get(name), "externally_exposed": EXPOSED.get(name)})
        b = "查询失败" if bare is None else f"{bare:,}"
        s = "查询失败" if scoped is None else f"{scoped:,}"
        share = f"{scoped/total:.2%}" if (scoped is not None and total) else "-"
        print(f"  {name:<20}{OURS.get(name, ''):<16}{EXPOSED.get(name, ''):<10}{b:>12}{s:>10}{share:>10}")

    print(f"\n  逐条查询串(可复跑):")
    for r in rows:
        print(f"    {r['type']:<20}{r['term']}")

    # THE QUESTION THIS FILE EXISTS FOR
    gaps_unexposed = [r for r in rows if r["ours"] == "缺口" and r["externally_exposed"] == "无"]
    print(f"\n  === 关键:三条【无人暴露】的缺口,文献里有没有人点名要? ===")
    for r in gaps_unexposed:
        print(f"    {r['type']:<20}全库 {r['bare']}    设计域内 {r['in_design']}")
    nulls = [r for r in rows if r["ours"] == "null" and r["in_design"] is not None]
    if nulls and all(r["in_design"] is not None for r in gaps_unexposed):
        med_null = sorted(r["in_design"] for r in nulls)[len(nulls) // 2]
        print(f"\n    对照:我们测出 null 的那几类,设计域内命中中位 {med_null}")
        print(f"    读法(序数,不是显著性):缺口类若明显高于此,\"真空白\"成立;"
              f"若明显低于此,三条缺口与实践无关,必须照实说。")
    json.dump({"design_scope": DESIGN, "design_scope_total": total, "rows": rows},
              open(os.path.join(HERE, "literature_demand.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote literature_demand.json")


if __name__ == "__main__":
    main()
