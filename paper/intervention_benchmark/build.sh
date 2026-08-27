#!/usr/bin/env bash
# One command from artifacts to PDF. Numbers are regenerated first so the manuscript can never
# print a value that disagrees with the run that produced it. No figures yet -- prose first.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/ubuntu/miniconda3/envs/rna/bin/python
"$PY" make_numbers.py
"$PY" make_check.py            # fails if a measured number was typed in by hand
tectonic -X compile NM_intervention.tex
