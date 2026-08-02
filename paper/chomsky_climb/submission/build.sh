#!/usr/bin/env bash
# Rebuild every submission artifact from source. Run from paper/chomsky_climb/.
#
# Nothing here is hand-assembled: each artifact is regenerated from chomsky_climb.tex and from the logged
# runs under rna_harness/runs/cc_ascent, so a stale artifact is impossible as long as this script is what
# produced it. Re-run after any edit to the manuscript or any new sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(cd ../.. && pwd)
SUB=submission

echo "== 0. build the manuscript =="
tectonic -X compile chomsky_climb.tex --keep-logs >/dev/null 2>&1
[ "$(grep -ci overfull chomsky_climb.log)" -eq 0 ] || { echo "  ABORT: overfull boxes present"; exit 1; }

echo "== 1. arXiv source package =="
rm -rf $SUB/arxiv; mkdir -p $SUB/arxiv/figs
cp chomsky_climb.tex references.bib chomsky_climb.bbl $SUB/arxiv/
# only the figures the manuscript actually includes; arXiv rejects unused large files and we ship none
python3 - <<'PY'
import re, shutil, os
t = open("chomsky_climb.tex").read()
for f in {u.replace(".pdf","") for u in re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}', t)}:
    shutil.copy(f"figs/{f}.pdf", f"submission/arxiv/figs/{f}.pdf")
    print("   fig:", f)
PY
# arXiv runs its own LaTeX; the .bbl must ship because it will not run bibtex for us.
cat > $SUB/arxiv/00README.XXX <<'RM'
% arXiv build directives.
% Engine: pdflatex (the source uses no XeTeX/LuaTeX-only features).
% chomsky_climb.bbl is included deliberately: arXiv does not run bibtex, so the bibliography
% must be pre-built. references.bib is shipped for completeness only.
RM

echo "== 2. code and data supplement (AAAI: <= 50 MB) =="
rm -f $SUB/aaai/code_and_data.zip
( cd "$ROOT" && zip -qr "$OLDPWD/$SUB/aaai/code_and_data.zip" \
    rna_harness/runs/cc_ascent \
    rna_harness/cognitive/skills/pk_decompose_panel.py \
    rna_harness/cognitive/skills/pk_decompose.py \
    rna_harness/experiments/track_b_pk.py \
    rna_harness/oracles_pk.py \
    rna_harness/benchmarks/rnainvbench.py \
    -x '*__pycache__*' '*.pyc' '*/_archive_runs/*' )
SZ=$(du -m $SUB/aaai/code_and_data.zip | cut -f1)
echo "   code_and_data.zip = ${SZ} MB (limit 50)"
[ "$SZ" -le 50 ] || { echo "  ABORT: supplement over the 50 MB limit"; exit 1; }

echo "== 3. main PDF =="
cp chomsky_climb.pdf $SUB/aaai/main.pdf

echo "== 4. claim index (rebuttal artifact) =="
python3 $SUB/make_claim_index.py > $SUB/rebuttal/claim_index.md
echo "   $(grep -c '^| ' $SUB/rebuttal/claim_index.md) rows"

echo "== done. artifacts in $SUB/ =="
ls -la $SUB/arxiv $SUB/aaai $SUB/rebuttal

# The checklist is prose, not generated; build.sh only checks it has not gone missing.
[ -f submission/aaai/reproducibility_checklist.md ] || echo "  WARNING: reproducibility checklist absent"
