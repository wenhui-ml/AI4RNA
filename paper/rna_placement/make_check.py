#!/usr/bin/env python3
"""Fail the build if a measured number was typed into the manuscript by hand.

make_numbers.py exists so that every measured quantity reaches LaTeX through \\jdg{}. That
discipline is worth nothing without a check, so this is the check. It was written because
make_numbers.py's own docstring claimed it existed before it did.

A decimal literal in the .tex is allowed only where it cannot be a measurement:
  - inside a comment
  - a LaTeX length or width (0.86\\linewidth, boxrule=0.6pt, ...)
  - a package/colour/geometry argument
  - a threshold the manuscript states as a rule rather than a result (0.005, 0.2 SD, 0.80)
Everything else is a violation and the exit code is nonzero.

COST: $0, offline.
"""
from __future__ import annotations

import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "R1_placement.tex")

# thresholds and design constants the paper states as rules, not as measurements
# design constants the paper states as rules: the two calipers, the treatment threshold,
# the significance gate, the ladder self-check bar and the corpus authors' own filter cut
ALLOWED = {"0.005", "0.05", "0.02", "0.2", "0.80", "0.0", "0.5",
           "0.4", "0.6", "0.8", "1.0", "2.0",   # the four stringency rungs k
           "2.7",                               # ViennaRNA version 2.7.2, not a measurement
           "1.96",                              # the normal quantile in a stated interval
           "1.5",                               # the declared collinearity bar (M21 K-conditions)
           "0.6",                               # the declared |rho_s| bar (M22 A arm)
           "0.010", "0.002",                    # the declared type-I band (M23 K1/K2)
           "0.85",                              # the declared affine-h bar (M62 S4)
           "0.20"}                              # the declared SMD bar (M64 S9)      # the four stringency rungs k
CTX = re.compile(r"(linewidth|textwidth|pt\b|em\b|in\}|HTML|\\setcounter|fraction|"
                 r"\\definecolor|width=|scale=|arc=|boxrule|margin=|\\usepackage|"
                 r"documentclass|xytext|figsize)")
NUM = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w])")



WORDS = {"four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
# Spelled-out counts are invisible to the decimal scan above, and two kinds were wrong here: R1
# said "a sevenfold inflation" where the measured ratio is 8.9, and "twenty targets" / "nine design
# methods" / "six populations" / "four rounds" were counts over a released file typed as words.
# A different release is a different corpus, so these must come through \jdg{} like any other
# measurement.
#
# The first version of this check flagged any spelled-out cardinal matching a macro's value and
# was useless: "four rules are enforced as code", "eight in ten", "all five of the following" are
# structural and it caught ten of them for every real one. It now fires only on nouns that name a
# quantity OF THE CORPUS -- the things that change when the release changes -- plus "-fold", which
# is always a ratio of two measured numbers.
CORPUS_NOUNS = r"""targets? puzzles? methods? populations? rounds? designs? cells? predictors?
    libraries library deposits? families family interventions? constructs? flips? mutants?
    sequences? pairs? strata stratum bins? parameter""".split()
WORD_RE = re.compile(r"\b(" + "|".join(WORDS) + r")(?:(fold)\b|[- ](" +
                     "|".join(CORPUS_NOUNS) + r")\b)", re.I)


def word_counts(code, defined):
    """A corpus count, or a ratio, spelled out as a word instead of coming through \jdg{}."""
    out = []
    for m in WORD_RE.finditer(code):
        w = m.group(1).lower()
        noun = "fold" if m.group(2) else m.group(3).lower()
        if noun == "fold" or str(WORDS[w]) in defined:
            out.append(f"{m.group(1)} {noun}")
    return out



# ---------------------------------------------------------------------------
# Citation integrity. Added after a key cited in the body but absent from refs.bib
# printed as "[? ?]" on page 6 of a committed PDF and no check saw it: the build
# does not fail on an undefined citation, it just typesets a question mark.
# ---------------------------------------------------------------------------
def undefined_citations() -> list:
    tex = open(TEX).read()
    bib = open(os.path.join(HERE, "refs.bib")).read()
    keys = set(re.findall(r"^@[a-zA-Z]+\{([^,]+),", bib, re.M))
    cited = set()
    for grp in re.findall(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        cited |= {k.strip() for k in grp.split(",") if k.strip()}
    return sorted(cited - keys)



# ---------------------------------------------------------------------------
# A p-value macro that carries its own relation ("<0.001") must not be preceded by one.
# "$p=\jdg{X}$" typesets "p =< 0.001" and "$p\le\jdg{X}$" typesets "p <= <0.001"; eleven such
# sites reached a built PDF. The converse also fails: a macro carrying a bare number needs the
# relation the site was about to supply, so dropping "=" everywhere is not the fix either.
# ---------------------------------------------------------------------------
def pvalue_relations() -> list:
    tex = open(TEX).read()
    nums = open(os.path.join(HERE, "figs",
                             "nm_numbers.tex" if "NM" in os.path.basename(TEX)
                             else "rna_numbers.tex")).read()
    carries = {m.group(1) for m in re.finditer(r"\\csname\s+jdg@(\w+)\\endcsname\{<", nums)}
    out = []
    for m in re.finditer(r"(=|\\le|\\ge|<|>)\s*\\jdg\{(\w+)\}", tex):
        if m.group(2) in carries:
            out.append(f"{m.group(2)}: preceded by {m.group(1)!r} and already carries '<'")
    for m in re.finditer(r"\bp\s*\\jdg\{(\w+)\}", tex):
        if m.group(1) not in carries:
            out.append(f"{m.group(1)}: written as 'p<macro>' but carries no relation")
    return out


def main() -> int:
    bad, badw = [], []
    defined = set(re.findall(r"\\csname\s+jdg@\w+\\endcsname\{([^}]*)\}",
                             open(os.path.join(os.path.dirname(TEX), "figs", "rna_numbers.tex")).read()))
    started = False
    for i, line in enumerate(open(TEX), 1):
        if "\\begin{document}" in line:
            started = True
        if not started:
            continue
        # an ESCAPED percent is text, not a comment; splitting on a bare "%"
        # truncated every line containing "5\\%" and hid what came after it
        code = re.split(r"(?<!\\\\)%", line)[0] if not line.lstrip().startswith("%") else ""
        if not code.strip() or CTX.search(code):
            continue
        for m in NUM.finditer(code):
            if m.group(1) in ALLOWED:
                continue
            bad.append((i, m.group(1), code.strip()[:96]))
        for w in word_counts(code, defined):
            badw.append((i, w, code.strip()[:96]))
    n_jdg = len(re.findall(r"\\jdg\{", open(TEX).read()))
    print(f"\\jdg{{}} uses in the manuscript: {n_jdg}")
    if badw:
        print(f"\n**{len(badw)} corpus count spelled out as a word:**")
        for ln, w, ctx in badw:
            print(f"  line {ln}: {w}   |  {ctx}")
    if bad:
        print(f"\n**{len(bad)} hand-typed decimal(s) in the manuscript body:**")
        for ln, v, ctx in bad:
            print(f"  line {ln}: {v}   |  {ctx}")
    pv_bad = pvalue_relations()
    if pv_bad:
        print(f"\n**{len(pv_bad)} p-value relation defect(s):**")
        for b in pv_bad:
            print(f"  {b}")
    missing_keys = undefined_citations()
    if missing_keys:
        print(f"\n**{len(missing_keys)} citation key(s) cited in the body and absent from refs.bib:**")
        for k in missing_keys:
            print(f"  {k}   -> would typeset as [?]")
    if bad or badw or missing_keys or pv_bad:
        print("\nEvery measured number must come from make_numbers.py via \\jdg{}, "
              "and every cited key must exist in refs.bib.")
        return 1
    print("clean: no hand-typed measured number in the manuscript body")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
