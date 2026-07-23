#!/usr/bin/env bash
# Build all chomsky_climb figures: 5 publication-grade TikZ diagrams + 2 matplotlib data plots.
#   diagrams  -> tikz/{architecture,certificate,hierarchy,language,worked}.tex  (Computer-Modern vector)
#   data figs -> make_figs.py fig_oracle() + fig_collapse()
set -euo pipefail
cd "$(dirname "$0")"
for f in architecture certificate hierarchy language worked invention ladder; do
  ( cd tikz && tectonic "$f.tex" >/dev/null 2>&1 && cp "$f.pdf" "../figs/fig_cc_$f.pdf" \
       && rm -f "$f.pdf" "$f.aux" "$f.log" )   # keep tikz/ to sources only; no intermediate duplicates
  echo "  built figs/fig_cc_$f.pdf   (TikZ)"
done
/home/ubuntu/miniconda3/envs/rna/bin/python make_figs.py
echo "all figures rebuilt."
