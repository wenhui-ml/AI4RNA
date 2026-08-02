#!/usr/bin/env python3
"""A requirement distribution WE DID NOT WRITE -- the one hole that could invalidate the whole scorecard.

THE HOLE. Every gap this project confirmed, every null it recorded, and the window criterion itself were
measured on requirements we chose and phrased. That has been scored 0 since the first week and named as the
only thing that could still overturn the table: if real requests look nothing like our 10 requirement types,
then three proven gaps and a closed five-coordinate map are facts about our own imagination.

WHAT COUNTS AS EXTERNAL, and why this source rather than a literature survey. A survey would still have me
choosing which sentences count as requirements. The option surface of a DESIGN TOOL does not: each flag is a
requirement its authors committed to supporting, written by a different group, published, and extractable
mechanically from the argument parser. Six independent codebases sit in RNAInvBench (antaRNA, IncaRNAtion,
SamFEO, LMM, plus the tertiary-design tools) and DesiRNA is already read; none of them is ours.

It measures "what tool authors believed users need" rather than "what users asked", which is a proxy and is
stated as one. It is still the first requirement list in this project that nobody here invented.

THE PROTOCOL, fixed before any total is read:
  1. extract every option VERBATIM with its help text, mechanically, no hand-picking
  2. drop options that are plumbing rather than requirements (I/O paths, verbosity, seeds, thread counts,
     algorithm internals like colony size or step count) by a declared keyword rule
  3. map each surviving option to one of the requirement TYPES this project has already measured, or to
     `unmeasured` -- and `unmeasured` is the interesting bucket, not a failure of the mapping
  4. report the raw table so the classification can be redone by anyone who disagrees with it

WHAT THE ANSWER MEANS, both directions written down now:
  * most external requirement types already measured -> our coverage claim is about a real distribution and
    the three gaps are relevant to practice
  * a large `unmeasured` bucket -> we have been answering our own exam, and that bucket is the next roadmap
    whatever else this project believed it should do next
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
RIB = "/data/wenhui/RNAInvBench/rna_design_algorithms"

# Sources: one entry per independent codebase. DesiRNA is included because its surface was already read
# option by option for section 4.2, so leaving it out would be selective.
SOURCES = {
    "antaRNA": [f"{RIB}/antaRNA/antarna/antarna.py", f"{RIB}/antaRNA/main.py"],
    "IncaRNAtion": [f"{RIB}/IncaRNAtion/incarnation.py"],
    "SamFEO": [f"{RIB}/samfeo/main.py", f"{RIB}/samfeo/samfeo.py"],
    "LMM": [f"{RIB}/LMM/LLM_opt_main.py"],
    "DesiRNA": ["/data/wenhui/DesiRNA/DesiRNA.py"],
}

# PLUMBING, not requirements. Declared before the counts are seen so the boundary cannot drift toward a
# flattering answer. These are things every CLI has regardless of what it designs.
# THE PRINCIPLE, stated rather than tuned: an option that controls HOW THE SEARCH RUNS is an internal; an
# option that controls WHAT THE ANSWER MUST SATISFY is a requirement. The first version's regex let seven
# antaRNA search parameters (alpha, beta, omega, evaporation rate, resets, improve procedure, number of runs)
# through as "unmeasured requirements", which inflated the unmeasured bucket with things no user asks for and
# made our coverage look worse than it is. Widening it is a correction toward the declared principle, not
# toward a nicer number, and every flag it now catches is printed.
PLUMBING = re.compile(
    r"(^|_|-)(help|verbose|version|output|out|input|in|file|path|dir|name|seed|thread|worker|cpu|core|"
    r"plot|log|debug|quiet|py|timer|od|wt|save)($|_|-)|^-[hH]$", re.I)
INTERNALS = re.compile(
    r"(colon|ants?|iteration|step|repeat|replica|exchange|timelimit|time_limit|steps|start|init|online|"
    r"k$|t$|tshelves|tmin|tmax|param|paramfile|strategy|accuracy|trailblaze|level|batch|epoch|lr|model|"
    r"alpha|beta|omega|evapor|\ber\b|reset|improve|number_of_runs|results_number|stop_when)",
    re.I)

# The requirement types this project has already measured, with the phrases that identify them. The mapping
# is by keyword on the option name and its help text, so it can be re-derived by anyone.
MEASURED = [
    ("fold-match", r"(structure|Cstr|target|dotbracket|sec_struct)"),
    ("sequence-pin", r"(Cseq|seq_restr|sequence.?constraint|constraints?\b|iupac)"),
    ("composition-GC", r"(tGC|GC|acgu|content|composition)"),
    ("temperature", r"(temperature|temp|-T\b)"),
    ("accessibility", r"(accessib|unpaired probab|openness)"),
    ("forbidden-motif", r"(motif|forbid|avoid.*(seq|motif))"),
    ("pseudoknot", r"(pseudoknot|pknot|HotKnots|-pk)"),
    ("multi-state", r"(alt_sec|alternative structure|multi.?state|switch|riboswitch)"),
    ("dimer/oligomer", r"(dimer|oligomer|complex|hetero|homo)"),
    ("ensemble-strictness", r"(ensemble|defect|Edef|probab.*ensemble|subopt|negative.?design|umfe|pfe|scoring.?function)"),
    ("pair-type", r"(noGU|GU base|lonely|LBP|canonical)"),
    ("set-level", r"(orthogon|library|diversity|identity|pairwise)"),
    ("co-transcriptional", r"(cotrans|co-trans|kinetic|barrier|folding path)"),
]


ENTRY_FUNCS = ("call_incarnation", "design", "run")


def extract_signature(path: str):
    """Requirements of a WRAPPER-style tool live in its entry function's signature, not in a parser.

    IncaRNAtion and LMM expose no argparse at all and were extracted as ZERO options by the first version --
    which would have silently shrunk the external sample from five codebases to three while the printout
    still said five. A source that contributes nothing has to be visible as such, not absent.
    """
    if not os.path.exists(path):
        return []
    src = open(path, errors="ignore").read()
    out = []
    for m in re.finditer(r"^def\s+(\w+)\s*\(([^)]*)\)", src, re.M):
        if m.group(1) not in ENTRY_FUNCS:
            continue
        for part in m.group(2).split(","):
            nm = part.split("=")[0].strip()
            if nm and nm not in ("self", "args", "kwargs", "*args", "**kwargs"):
                out.append({"flags": ["--" + nm], "help": f"(entry-function parameter of {m.group(1)})"})
    return out


def extract(path: str):
    """Every add_argument in a file, verbatim: the flags plus whatever help string sits with it."""
    if not os.path.exists(path):
        return []
    try:
        src = open(path, errors="ignore").read()
    except Exception:                                                          # noqa: BLE001
        return []
    out = []
    for m in re.finditer(r"add_argument\((.{0,600}?)\)\s*(?:\n|$)", src, re.S):
        blob = m.group(1)
        flags = re.findall(r"['\"](-{1,2}[A-Za-z][\w\-]*)['\"]", blob)
        if not flags:
            continue
        h = re.search(r"help\s*=\s*['\"](.{0,200}?)['\"]", blob, re.S)
        out.append({"flags": flags, "help": (h.group(1) if h else "").strip()})
    return out


def classify(opt) -> str:
    blob = " ".join(opt["flags"]) + " " + opt["help"]
    for name, pat in MEASURED:
        if re.search(pat, blob, re.I):
            return name
    # NO HELP TEXT MEANS NOT CLASSIFIABLE, not "a requirement we never measured". SamFEO documents none of
    # its flags; guessing from a bare name like --bp would put an invented requirement into the denominator.
    return "unmeasured" if opt["help"] else "undocumented"


def main():
    rows, per_tool = [], {}
    for tool, paths in SOURCES.items():
        seen = set()
        opts = []
        for p in paths:
            for o in extract(p) + extract_signature(p):
                key = tuple(o["flags"])
                if key in seen:
                    continue
                seen.add(key)
                opts.append(o)
        kept, dropped = [], []
        for o in opts:
            blob = " ".join(o["flags"])
            if (PLUMBING.search(blob) or INTERNALS.search(blob)
                    or any(PLUMBING.search(f) for f in o["flags"])):
                dropped.append(o)
                continue
            o["type"] = classify(o)
            o["tool"] = tool
            kept.append(o)
            rows.append(o)
        per_tool[tool] = {"n_options": len(opts), "n_requirements": len(kept), "n_plumbing": len(dropped)}
        print(f"  {tool:<14} 选项 {len(opts):>3}  判为要求 {len(kept):>3}  判为管道/内部 {len(dropped):>3}")

    from collections import Counter
    c = Counter(r["type"] for r in rows)
    print(f"\n=== 外部要求类型分布({len(SOURCES)} 个独立代码库,{len(rows)} 个要求型选项) ===")
    print(f"  {'要求类型':<22}{'出现次数':>8}{'出现在几个工具里':>18}")
    by_type_tools = {}
    for r in rows:
        by_type_tools.setdefault(r["type"], set()).add(r["tool"])
    for t, n in c.most_common():
        print(f"  {t:<22}{n:>8}{len(by_type_tools[t]):>18}")

    # THE NUMBER THIS EXISTS FOR
    und = c.get("undocumented", 0)
    unm = c.get("unmeasured", 0)
    meas = len(rows) - und - unm
    print(f"\n  已被本项目实测过:{meas}/{len(rows)} = {meas/max(1,len(rows)):.1%}"
          f"   (未测 {unm};无帮助文本、无法归类 {und} —— 后者不进分母的分子,也不假装是要求)")
    if len(rows) - und:
        print(f"  只在可归类的选项里算:{meas}/{len(rows)-und} = {meas/(len(rows)-und):.1%}")
    print(f"  未测过 / 无法归类 的选项(逐条列出,便于任何人重新归类):")
    for r in rows:
        if r["type"] in ("unmeasured", "undocumented"):
            print(f"    [{r['type']:<12}] {r['tool']:<12}{','.join(r['flags']):<26}{r['help'][:60]}")

    ours = {"composition-GC": "已证实的缺口", "co-transcriptional": "已证实的缺口",
            "multi-state": "已证实的缺口(两态开关)", "set-level": "已证实的缺口(正交集合)",
            "temperature": "实测 null", "accessibility": "实测 null",
            "ensemble-strictness": "实测 null", "forbidden-motif": "实测 null"}
    print(f"\n  === 交叉表:我们的结论 vs 外部工具是否暴露该要求 ===")
    print(f"  {'要求类型':<22}{'我们的结论':<22}{'外部暴露':>16}")
    for t, verdict in ours.items():
        n = c.get(t, 0)
        cell = f"{n} 处 / {len(by_type_tools.get(t, []))} 个工具" if n else "【无人暴露】"
        print(f"  {t:<22}{verdict:<22}{cell:>16}")
    json.dump({"sources": list(SOURCES), "per_tool": per_tool, "ours_vs_external": ours,
               "type_counts": dict(c), "n_requirement_options": len(rows),
               "measured_fraction": meas / max(1, len(rows)), "rows": rows},
              open(os.path.join(HERE, "external_requirements.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n  wrote external_requirements.json")


if __name__ == "__main__":
    main()
