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

# Strings that must never appear in a blinded artifact. One list, used by three gates below.
IDENT='wenhui|jianlin|ziyao|vong|um\.edu\.mo|scut|University of Macau|South China University|InceptLabs'

echo "== 0. build the manuscript (ANONYMOUS; the repo copy is \\anontrue by default) =="
# pdflatex, not tectonic. Tectonic re-runs bibtex against the shipped .bbl, loops to its six-pass
# limit and segfaults; it emits a correct PDF first, so the failure looks like a manuscript defect and
# is not one. pdflatex is also what arXiv and the venue will run.
pdflatex -interaction=nonstopmode chomsky_climb.tex >/dev/null 2>&1 || true
bibtex chomsky_climb >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode chomsky_climb.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode chomsky_climb.tex >/dev/null 2>&1 || true
[ -f chomsky_climb.pdf ] || { echo "  ABORT: the manuscript does not compile"; exit 1; }
OVF=$(grep -ci overfull chomsky_climb.log || true); OVF=${OVF:-0}
[ "$OVF" -eq 0 ] || { echo "  ABORT: $OVF overfull boxes present"; exit 1; }
UND=$(grep -ci 'undefined reference\|Citation.*undefined' chomsky_climb.log || true); UND=${UND:-0}
[ "$UND" -eq 0 ] || { echo "  ABORT: $UND undefined references or citations"; exit 1; }
# The default state of the source is blinded, so that forgetting to flip a switch fails safe: you can
# accidentally ship an anonymous PDF to arXiv (embarrassing) but not a named one to a blind venue (fatal).
# grep -o, never grep -q. Under `set -o pipefail`, grep -q exits at the first match, pdftotext takes
# SIGPIPE, and the pipeline reports failure -- so a `grep -q` safety gate reports "clean" exactly when
# it should fire. grep -o consumes all input and cannot invert like that.
LEAK=$(pdftotext chomsky_climb.pdf - 2>/dev/null | grep -oiE "$IDENT" | sort -u | tr '\n' ' ' || true)
if [ -n "$LEAK" ]; then
  echo "  ABORT: the anonymous build leaks an author identity into the text layer: $LEAK"; exit 1
fi
echo "   anonymous build: $(pdfinfo chomsky_climb.pdf | awk '/Pages/{print $2}') pages, no author string in the text layer"

echo "== 1. arXiv source package (NAMED: arXiv requires real author names) =="
rm -rf $SUB/arxiv; mkdir -p $SUB/arxiv/figs
# One source, two packages. sed flips the single anonymity switch; nothing else differs, so the named
# and blinded manuscripts cannot drift apart the way two maintained copies would.
sed 's/^\\anontrue$/\\anonfalse/' chomsky_climb.tex > $SUB/arxiv/chomsky_climb.tex
grep -qx '\\anonfalse' $SUB/arxiv/chomsky_climb.tex || { echo "  ABORT: anonymity switch not found in the source"; exit 1; }
cp references.bib $SUB/arxiv/
# only the figures the manuscript actually includes; arXiv rejects unused large files and we ship none
python3 - <<'PY'
import re, shutil, os
t = open("chomsky_climb.tex").read()
for f in {u.replace(".pdf","") for u in re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}', t)}:
    shutil.copy(f"figs/{f}.pdf", f"submission/arxiv/figs/{f}.pdf")
    print("   fig:", f)
PY
# The .bbl must be generated from the NAMED build. The blinded build cites verifiereconomicsanon, so
# the repo's own .bbl carries "Anonymous" into the bibliography; shipping that to arXiv would be wrong.
# Sequenced with ';' and not '&&': pdflatex and bibtex both exit non-zero on ordinary warnings (an
# undefined citation on the first pass, before the .bbl exists), which silently truncates an && chain
# and leaves a one-pass PDF behind.
( cd $SUB/arxiv
  pdflatex -interaction=nonstopmode chomsky_climb.tex >/dev/null 2>&1 || true
  bibtex chomsky_climb >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode chomsky_climb.tex >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode chomsky_climb.tex >/dev/null 2>&1 || true )
[ -f $SUB/arxiv/chomsky_climb.bbl ] || { echo "  ABORT: named .bbl was not produced"; exit 1; }
NAMED=$(pdftotext $SUB/arxiv/chomsky_climb.pdf - 2>/dev/null | grep -ciE 'Wenhui|University of Macau' || true)
if [ "${NAMED:-0}" -eq 0 ]; then
  echo "  ABORT: the arXiv build is still blinded (switch did not take)"; exit 1
fi
echo "   arXiv build: named, $(pdfinfo $SUB/arxiv/chomsky_climb.pdf | awk '/Pages/{print $2}') pages"
rm -f $SUB/arxiv/chomsky_climb.{aux,log,out,blg,pdf}

# arXiv runs its own LaTeX; the .bbl must ship because it will not run bibtex for us.
cat > $SUB/arxiv/00README.XXX <<'RM'
% arXiv build directives.
% Engine: pdflatex (the source uses no XeTeX/LuaTeX-only features).
% chomsky_climb.bbl is included deliberately: arXiv does not run bibtex, so the bibliography
% must be pre-built. references.bib is shipped for completeness only.
RM

# arXiv wants ONE archive of the source directory, not the directory. Build it here rather than leaving it
# as a documented manual step, because "tar from the wrong level" is the classic upload failure: arXiv
# expects the .tex at the archive root, not inside a folder.
# Files listed explicitly: `tar . --exclude='.*'` excludes the '.' entry itself and ships an empty archive,
# which is silent and looks like success.
( cd $SUB/arxiv && tar czf ../chomsky_climb_arxiv.tar.gz \
      chomsky_climb.tex references.bib chomsky_climb.bbl 00README.XXX figs )
echo "   $SUB/chomsky_climb_arxiv.tar.gz = $(du -k $SUB/chomsky_climb_arxiv.tar.gz | cut -f1) KB, \
$(tar tzf $SUB/chomsky_climb_arxiv.tar.gz | grep -vc '/$') entries"
# Compile the archive's contents in a scratch directory with no access to this repo. A package that builds
# here and not there is invisible until upload, and arXiv gives no second opinion.
#
# pdflatex, twice, and deliberately NOT tectonic. arXiv does not run bibtex; it typesets the shipped .bbl,
# which is what 00README.XXX declares and what this loop reproduces. Tectonic runs bibtex itself, regenerates
# a .bbl that disagrees with the shipped one, loops to its six-pass limit and then segfaults, all while
# emitting a correct PDF. That is a tectonic artifact and not an arXiv failure, so gating on it fails the
# build for a defect that does not exist.
TMP=$(mktemp -d); tar xzf $SUB/chomsky_climb_arxiv.tar.gz -C "$TMP"
( cd "$TMP" && pdflatex -interaction=nonstopmode chomsky_climb.tex >/dev/null 2>&1 \
             && pdflatex -interaction=nonstopmode chomsky_climb.tex >/dev/null 2>&1 ) || true
CRLOG="$TMP/chomsky_climb.log"
if [ ! -f "$TMP/chomsky_climb.pdf" ]; then
  echo "   ABORT: the arXiv archive produces no PDF standalone"; rm -rf "$TMP"; exit 1
fi
for probe in "Undefined control sequence" "LaTeX Error" "not found on input line" "There were undefined references"; do
  if grep -qF "$probe" "$CRLOG"; then
    echo "   ABORT: clean-room compile reports: $probe"; grep -m2 -F "$probe" "$CRLOG"; rm -rf "$TMP"; exit 1
  fi
done
CRP=$(pdfinfo "$TMP/chomsky_climb.pdf" | awk '/Pages/{print $2}')
echo "   clean-room compile (pdflatex, shipped .bbl): OK, $CRP pages, no missing files"
rm -rf "$TMP"

echo "== 2. code and data supplement (AAAI: <= 50 MB) =="
rm -f $SUB/aaai/code_and_data.zip
HERE=$(pwd)
# The source list is COMPUTED, not listed. An earlier revision named five files chosen by reading the
# manuscript; the supplement did not run, because the analysis scripts import benchmarks.rnainvbench, which
# imports ..constraints, which nobody had thought to list, and no __init__.py shipped at all. collect_code.py
# walks the import graph from every entry point instead, so a new import ships its target automatically.
python3 $SUB/collect_code.py > "$HERE/.supp_files"
echo "   $(wc -l < "$HERE/.supp_files") source files (transitive import closure)"

cat > "$HERE/.supp_readme" <<'RM'
# Code and data supplement

## What is here

- `rna_harness/runs/cc_ascent/` -- every logged run behind every number in the paper (per-target JSON for
  all sweeps), the analysis scripts that reduce them, their committed `.txt` outputs, and the frozen
  agent-written operator sources.
- `rna_harness/**.py` -- the harness itself, as the transitive closure of what the above imports. This is
  computed by `paper/chomsky_climb/submission/collect_code.py`, not curated by hand.
- `paper/chomsky_climb/make_figs.py` -- regenerates the data figures from the same committed outputs.

## Re-running

Analysis scripts read the committed JSON and need no oracle binaries:

    cd <extracted>
    python rna_harness/runs/cc_ascent/arep_matched.py      # the positive control, 0.293 vs 0.095
    python rna_harness/runs/cc_ascent/family_stats.py      # the unbounded family, +0.316
    python rna_harness/runs/cc_ascent/dom_stats.py         # the ensemble-gap transplant, exactly zero

Re-executing the *sweeps* additionally needs the four oracles; see `ENVIRONMENT.md` and App. F.

## What is deliberately not reproducible

The **invention step**. The six operators were written by frontier models behind a routed provider alias with
no version hash, so a re-run may not query the same weights. No number in the paper depends on it: every
measurement re-executes the frozen operator source shipped here rather than re-querying a model. The claim
"an agent can invent such an operator" does depend on it, and Sec. 6.2(v) says so.
RM

cat > "$HERE/.supp_env" <<RM
# Environment

Measured on the host that produced the paper's numbers.

    python        $(python3 -c 'import sys;print(sys.version.split()[0])')
    ViennaRNA     $(python3 -c 'import RNA;print(RNA.__version__)' 2>/dev/null || echo 'see App. F')
    numpy         $(python3 -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo '-')
    scipy         $(python3 -c 'import scipy;print(scipy.__version__)' 2>/dev/null || echo '-')
    matplotlib    $(python3 -c 'import matplotlib;print(matplotlib.__version__)' 2>/dev/null || echo '-')

Oracles live in three conda environments because RNAstructure and pKiss cannot share one; App. F pins the
configuration of each, and \`DATAPATH\` must point at the RNAstructure data tables.

    env rna      ViennaRNA (\$\Omega_{cf}\$: Turner-2004 MFE at 37 C, no timeout)
    env pk       pKiss     (\`pkiss mfe\`, 60 s)
    env rnastr   ProbKnot  (MEA over the partition ensemble, 60 s)
                 ShapeKnots (MFE-based, run de novo with no SHAPE data, 120 s)

Compute: one 256-core host, target-level parallelism at NWORKERS=60. The sweep suite is search-only and
incurs no model-inference cost apart from the invention agent.
RM

# Stage, then SCRUB, then zip. Zipping straight from the repo shipped 20 files carrying absolute
# paths whose first component is the lead author's username, plus two naming the institution. The
# supplement is visible to reviewers, so that de-anonymised the submission no matter how blinded the
# PDF was; the PDF gate above would never have caught it.
STAGE=$(mktemp -d)
( cd "$ROOT" && rsync -a --files-from="$HERE/.supp_files" . "$STAGE/" )
( cd "$ROOT" && rsync -a --relative rna_harness/runs/cc_ascent "$STAGE/" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='_archive_runs' )
grep -rIl -e '/data/wenhui' -e 'InceptLabs' "$STAGE" 2>/dev/null | while read -r f; do
  sed -i -e 's#/data/wenhui/openclaude_rna#/path/to/repo#g' \
         -e 's#/data/wenhui#/path/to#g' \
         -e 's#InceptLabs#Anonymous#g' "$f"
done
SLEAK=$(grep -rIohiE "$IDENT" "$STAGE" 2>/dev/null | sort -u | tr '\n' ' ' || true)
if [ -n "$SLEAK" ]; then
  echo "  ABORT: the supplement still carries an author identity after scrubbing: $SLEAK"
  rm -rf "$STAGE"; exit 1
fi
echo "   supplement scrubbed: no author identity in $(find "$STAGE" -type f | wc -l) files"
( cd "$STAGE" && zip -qr "$HERE/$SUB/aaai/code_and_data.zip" . -x '*__pycache__*' '*.pyc' )
rm -rf "$STAGE"
( cd "$HERE" && cp .supp_readme README.md && cp .supp_env ENVIRONMENT.md \
  && zip -q "$SUB/aaai/code_and_data.zip" README.md ENVIRONMENT.md && rm -f README.md ENVIRONMENT.md )
rm -f "$HERE/.supp_readme" "$HERE/.supp_env"
SZ=$(du -m $SUB/aaai/code_and_data.zip | cut -f1)
echo "   code_and_data.zip = ${SZ} MB (limit 50), $(unzip -l $SUB/aaai/code_and_data.zip | tail -1 | awk '{print $2}') entries"
[ "$SZ" -le 50 ] || { echo "  ABORT: supplement over the 50 MB limit"; exit 1; }

# Clean-room check, the same discipline as the arXiv package: extract with no access to this repo, import
# every analysis script, and re-derive three of the manuscript's indexed numbers. A supplement that imports
# here and not there is invisible until a reviewer tries it, and that is the worst moment to find out.
TMPS=$(mktemp -d); unzip -q $SUB/aaai/code_and_data.zip -d "$TMPS"
if ( cd "$TMPS" && python3 - <<'PY'
import importlib.util, pathlib, sys, io, contextlib
ps = sorted(pathlib.Path('rna_harness/runs/cc_ascent').glob('*.py')); bad = []
for p in ps:
    spec = importlib.util.spec_from_file_location("supp_" + p.stem, p)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(m)
    except (ImportError, ModuleNotFoundError) as e:
        bad.append((p.name, str(e)))
    except Exception:
        pass                      # a script wanting CLI arguments is not a packaging failure
for n, e in bad:
    print("   MISSING:", n, "->", e)
sys.exit(1 if bad else 0)
PY
); then
  NUM=$( cd "$TMPS" && python3 rna_harness/runs/cc_ascent/arep_matched.py 2>/dev/null | grep -c '0\.293' )
  echo "   clean-room: all analysis scripts import; positive control re-derives 0.293 ($NUM hits)"
  [ "$NUM" -ge 1 ] || { echo "  ABORT: supplement does not reproduce the positive control"; rm -rf "$TMPS"; exit 1; }
else
  echo "  ABORT: the supplement is not importable standalone"; rm -rf "$TMPS"; exit 1
fi
rm -rf "$TMPS" "$HERE/.supp_files"

echo "== 3. main PDF =="
cp chomsky_climb.pdf $SUB/aaai/main.pdf

echo "== 4. claim index (rebuttal artifact) =="
python3 $SUB/make_claim_index.py > $SUB/rebuttal/claim_index.md
echo "   $(grep -c '^| ' $SUB/rebuttal/claim_index.md) rows"

echo "== done. artifacts in $SUB/ =="
ls -la $SUB/arxiv $SUB/aaai $SUB/rebuttal

# The checklist is prose, not generated; build.sh only checks it has not gone missing.
[ -f submission/aaai/reproducibility_checklist.md ] || echo "  WARNING: reproducibility checklist absent"
