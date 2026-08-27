#!/usr/bin/env bash
# One command from artifacts to PDF. Numbers and figures are regenerated first so the manuscript
# can never print a value that disagrees with the run that produced it.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
OMP_NUM_THREADS=1 "$PY" make_figs.py
"$PY" make_numbers.py
"$PY" make_check.py            # fails if a measured number was typed in by hand
tectonic -X compile R1_placement.tex
