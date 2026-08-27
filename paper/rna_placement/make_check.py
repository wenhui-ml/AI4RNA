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
           "0.4", "0.6", "0.8", "1.0", "2.0"}      # the four stringency rungs k
CTX = re.compile(r"(linewidth|textwidth|pt\b|em\b|in\}|HTML|\\setcounter|fraction|"
                 r"\\definecolor|width=|scale=|arc=|boxrule|margin=|\\usepackage|"
                 r"documentclass|xytext|figsize)")
NUM = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w])")


def main() -> int:
    bad = []
    started = False
    for i, line in enumerate(open(TEX), 1):
        if "\\begin{document}" in line:
            started = True
        if not started:
            continue
        code = line.split("%")[0] if not line.lstrip().startswith("%") else ""
        if not code.strip() or CTX.search(code):
            continue
        for m in NUM.finditer(code):
            if m.group(1) in ALLOWED:
                continue
            bad.append((i, m.group(1), code.strip()[:96]))
    n_jdg = len(re.findall(r"\\jdg\{", open(TEX).read()))
    print(f"\\jdg{{}} uses in the manuscript: {n_jdg}")
    if bad:
        print(f"\n**{len(bad)} hand-typed decimal(s) in the manuscript body:**")
        for ln, v, ctx in bad:
            print(f"  line {ln}: {v}   |  {ctx}")
        print("\nEvery measured number must come from make_numbers.py via \\jdg{}.")
        return 1
    print("clean: no hand-typed measured number in the manuscript body")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
