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
# openclaude is the private agent project this harness was built inside. It is not an author name, and
# while the project stays private it identifies nobody -- but the TMLR supplement is PUBLIC FOREVER and
# the project may not stay private, so the artifact is made independent of that decision. Scrubbed in
# stage 2 and enforced here. It appears nowhere in the manuscript, so the PDF gates are unaffected.
IDENT='wenhui|jianlin|ziyao|vong|um\.edu\.mo|scut|University of Macau|South China University|InceptLabs|openclaude'

# ---------------------------------------------------------------------------------------------------
# WHICH STAGES, AND ON WHICH HOST. This build genuinely does not fit on one machine: the supplement
# needs the harness and the logged runs (the compute host), the PDFs need a LaTeX toolchain (wherever
# the paper is written). Running the whole thing on a host missing half the tools is what produced the
# failure this switch exists to prevent.
#
#   ./submission/build.sh                    everything (needs LaTeX + poppler + the harness)
#   BUILD_ONLY=supplement ./submission/...   stage 2 only  -- run this on the compute host
#   BUILD_ONLY=pdf ./submission/...          stages 0,1,3,4,5 -- run this where LaTeX lives
BUILD_ONLY=${BUILD_ONLY:-all}
case "$BUILD_ONLY" in all|pdf|supplement) ;; *)
  echo "ABORT: BUILD_ONLY must be all, pdf, or supplement (got '$BUILD_ONLY')"; exit 1 ;;
esac
want() { [ "$BUILD_ONLY" = all ] || [ "$BUILD_ONLY" = "$1" ]; }

# ---------------------------------------------------------------------------------------------------
# TOOL PREFLIGHT. Non-negotiable, because a missing binary does not make the gates below fail -- it
# makes them PASS. The anonymity gate is `pdftotext ... 2>/dev/null | grep -oiE "$IDENT" ... || true`:
# with pdftotext absent, stderr is discarded, the output is empty, `|| true` absorbs the exit code, and
# the gate reports "no author string in the text layer" precisely because it could not look. That is the
# same class of silent-pass the grep -q comment below warns about, reached through a different door, and
# it already happened once on a host with no poppler-utils.
#
# Stage 0 is worse than it looks on such a host: `[ -f chomsky_climb.pdf ]` is satisfied by the PDF
# COMMITTED IN THE REPO, and the overfull/undefined checks grep the committed .log, so stage 0 passes
# entirely on stale artifacts and announces success. Fail here instead, by name.
need() {
  local missing=""
  for t in "$@"; do command -v "$t" >/dev/null 2>&1 || missing="$missing $t"; done
  if [ -n "$missing" ]; then
    echo "ABORT: required tools not on PATH:$missing"
    echo "  These are not optional: the anonymity and page-count gates SILENTLY PASS without them."
    case "$missing" in
      *pdftotext*|*pdfinfo*) echo "  pdftotext/pdfinfo come from poppler-utils (apt install poppler-utils).";;
    esac
    case "$missing" in
      *pdflatex*|*bibtex*) echo "  pdflatex/bibtex come from a TeX distribution (apt install texlive-latex-recommended texlive-latex-extra).";;
    esac
    echo "  If this host has no LaTeX, build the supplement here and the PDFs elsewhere:"
    echo "      BUILD_ONLY=supplement ./submission/build.sh     # this host"
    echo "      BUILD_ONLY=pdf        ./submission/build.sh     # the host with LaTeX"
    exit 1
  fi
}
want pdf        && need pdflatex bibtex pdftotext pdfinfo python3
want supplement && need python3 rsync zip unzip
echo "== preflight: BUILD_ONLY=$BUILD_ONLY, all required tools present =="


if want pdf; then
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
fi   # stage 0 (pdf)

if want pdf; then
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
fi   # stage 1 (pdf)

if want supplement; then
echo "== 2. code and data supplement (AAAI: <= 50 MB) =="
# Build to a temporary name and move into place only on success. Deleting the existing zip up front
# means any failure downstream -- a missing rna_harness, an aborted gate -- leaves you with no supplement
# at all rather than the previous good one. That happened while these stages were being split.
ZIPTMP=$SUB/aaai/.code_and_data.zip.new
rm -f "$ZIPTMP"
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

## The environments that produced the paper's numbers

Every oracle and every design operator ran under **Python 3.10.20**, each oracle in its own pinned conda
environment because RNAstructure and pKiss cannot share one. App. F pins the configuration of each, and
\`DATAPATH\` must point at the RNAstructure data tables.

    env rna      Python 3.10.20   ViennaRNA 2.7.2   (\$\Omega_{cf}\$: Turner-2004 MFE at 37 C, no timeout)
    env pk       Python 3.10.20   pKiss 2.3.0       (\`pkiss mfe\`, 60 s; bellmans-gapc 2024.01.12, bioconda)
    env rnastr   Python 3.10.20   RNAstructure 6.6  (ProbKnot: MEA over the partition ensemble, 60 s)
                                                   (ShapeKnots: MFE-based, de novo with no SHAPE data, 120 s)

## The comparison baseline

DesiRNA publishes no release tag, so it is pinned three ways:

    repository    https://github.com/fryzjergda/DesiRNA
    commit        bdb4908  ("Update test_DesiRNA.yml", 2024-06-06, upstream main)
    version       in-file string "Version: 11.2023" (DesiRNA.py:20)
    DOI           10.1093/nar/gkae1306   (preprint 10.1101/2023.06.04.543636)

Invoked by \`rna_harness/generators/desirna.py\` as, with DESIRNA_ARGS unset:

    DesiRNA.py -f input.txt -p 2004 -R 10 -t 120 -r 20 -seed <seed>          # hard-killed at 240 s

\`-p 2004\` is not a default and is load-bearing: DesiRNA optimizes Turner 1999 out of the box, our oracle is
Turner 2004, and sequences optimal under the former come back 1-2 base pairs off under the latter. Every
sequence it returns is re-folded under our own oracle before it counts, and re-checked under pKiss on crossing
targets. The solver never certifies itself.

## The host that assembled this supplement

Reported for completeness only. This is whichever machine ran \`build.sh\`, which need not be -- and generally
is not -- a machine that can run the oracles. Do not read these as the run environment; the block above is the
run environment.

    python        $(python3 -c 'import sys;print(sys.version.split()[0])')
    numpy         $(python3 -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo '-')
    scipy         $(python3 -c 'import scipy;print(scipy.__version__)' 2>/dev/null || echo '-')
    matplotlib    $(python3 -c 'import matplotlib;print(matplotlib.__version__)' 2>/dev/null || echo '-')
    ViennaRNA     $(python3 -c 'import RNA;print(RNA.__version__)' 2>/dev/null || echo 'not importable here')

## Compute

One 256-core host, target-level parallelism at NWORKERS=60. The sweep suite is search-only and incurs no
model-inference cost apart from the invention agent (App. E), a few dollars of GLM-5.2.
RM

# Stage, then SCRUB, then zip. Zipping straight from the repo shipped 20 files carrying absolute
# paths whose first component is the lead author's username, plus two naming the institution. The
# supplement is visible to reviewers, so that de-anonymised the submission no matter how blinded the
# PDF was; the PDF gate above would never have caught it.
STAGE=$(mktemp -d)
( cd "$ROOT" && rsync -a --files-from="$HERE/.supp_files" . "$STAGE/" )
( cd "$ROOT" && rsync -a --relative rna_harness/runs/cc_ascent "$STAGE/" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='_archive_runs' )
grep -rIl -e '/data/wenhui' -e 'InceptLabs' -e 'openclaude' "$STAGE" 2>/dev/null | while read -r f; do
  sed -i -e 's#/data/wenhui/openclaude_rna#/path/to/repo#g' \
         -e 's#/data/wenhui#/path/to#g' \
         -e 's#InceptLabs#Anonymous#g' \
         -e 's#openclaude_rna#AgentFramework_rna#g' \
         -e 's#openclaude#AgentFramework#g' "$f"
done
SLEAK=$(grep -rIohiE "$IDENT" "$STAGE" 2>/dev/null | sort -u | tr '\n' ' ' || true)
if [ -n "$SLEAK" ]; then
  echo "  ABORT: the supplement still carries an author identity after scrubbing: $SLEAK"
  rm -rf "$STAGE"; exit 1
fi
echo "   supplement scrubbed: no author identity in $(find "$STAGE" -type f | wc -l) files"

# DATASET GATE. The clean-room check below imports every analysis script and executes exactly one of
# them. That is a blind spot: 21 shipped scripts call load_pseudobase(), which opens
# $RNAINVBENCH_DIR/data/Pseudobase.csv only when *called*, so they import cleanly in the clean room and
# then fail on first use -- including contrast_paired.py (the 2/43 vs 26/43 McNemar), pk_margin.py (the
# 84% probe), d3_structural.py, d3_distributional.py, split_audit.py, positive_stringency.py and
# verify_positive_control.py. arep_matched.py, the one script the gate runs, happens not to need it,
# which is why three revisions shipped without anyone noticing.
#
# App. B promises "the harness, target lists, intrinsic grades, and per-target JSON are released". The
# manifests are what make that true, and sweep.py already reads them via TARGETS_JSON.
NEEDCSV=$(grep -rl 'load_pseudobase' "$STAGE" 2>/dev/null | grep -c '\.py$' || true)
if [ "${NEEDCSV:-0}" -gt 0 ] && [ ! -f "$STAGE/rna_harness/runs/cc_ascent/targets/pseudobase_primary.json" ]; then
  echo "  ABORT: ${NEEDCSV} shipped scripts call load_pseudobase(), but no target manifest ships."
  echo "         The supplement cannot re-execute them: RNAINVBENCH_DIR points outside the package."
  echo "         Fix: RNAINVBENCH_DIR=... python3 $SUB/make_datasets.py   (from the repo root), then rebuild."
  rm -rf "$STAGE"; exit 1
fi
echo "   dataset gate: ${NEEDCSV} scripts need Pseudobase++; target manifests present"
# The six frozen operator sources live inside result.json as `best_code`. Nothing is missing, but the two
# operators carrying the positive result should be readable without writing a JSON extractor first.
NOPS=$(find "$STAGE/rna_harness/runs/cc_ascent/invent_operator" -name operator.py 2>/dev/null | wc -l | tr -d ' ')
[ "${NOPS:-0}" -eq 6 ] || echo "  WARNING: ${NOPS:-0}/6 operator sources dumped as .py (run $SUB/dump_operators.py)"
( cd "$STAGE" && zip -qr "$HERE/$ZIPTMP" . -x '*__pycache__*' '*.pyc' )
rm -rf "$STAGE"
( cd "$HERE" && cp .supp_readme README.md && cp .supp_env ENVIRONMENT.md \
  && zip -q "$ZIPTMP" README.md ENVIRONMENT.md && rm -f README.md ENVIRONMENT.md )
rm -f "$HERE/.supp_readme" "$HERE/.supp_env"
SZ=$(du -m "$ZIPTMP" | cut -f1)
echo "   code_and_data.zip = ${SZ} MB (limit 50), $(unzip -l "$ZIPTMP" | tail -1 | awk '{print $2}') entries"
[ "$SZ" -le 50 ] || { echo "  ABORT: supplement over the 50 MB limit"; exit 1; }

# Clean-room check, the same discipline as the arXiv package: extract with no access to this repo, import
# every analysis script, and re-derive three of the manuscript's indexed numbers. A supplement that imports
# here and not there is invisible until a reviewer tries it, and that is the worst moment to find out.
TMPS=$(mktemp -d); unzip -q "$ZIPTMP" -d "$TMPS"
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
# Every gate above passed: only now replace the good zip.
mv "$ZIPTMP" $SUB/aaai/code_and_data.zip
fi   # stage 2 (supplement)

if want pdf; then
echo "== 3. main PDF =="
cp chomsky_climb.pdf $SUB/aaai/main.pdf
fi   # stage 3 (pdf)

if want pdf; then
echo "== 4. TMLR submission package (BLINDED; TMLR reviewing is double-blind) =="
# The venue copy is DERIVED, never maintained. make_tmlr.py applies the TMLR-specific differences
# (10pt + tmlr.sty, author-year natbib, no geometry/authblk, one-paragraph abstract, references before
# appendices, Broader Impact Statement) to chomsky_climb.tex, and every one of its edits is anchored on
# an exact string that must still be present -- so an edit to the manuscript that breaks an anchor fails
# this build loudly instead of shipping a silently-wrong PDF.
# Clear only what this stage generates. NOT `rm -rf $SUB/tmlr`: SUBMISSION_FORM.md lives there and is
# prose, not generated -- it is the field-by-field answer sheet for the OpenReview form, and wiping the
# directory would delete it on the next build with no warning.
mkdir -p $SUB/tmlr/figs
rm -rf $SUB/tmlr/figs
mkdir -p $SUB/tmlr/figs
rm -f $SUB/tmlr/main.tex $SUB/tmlr/main.pdf $SUB/tmlr/main.bbl $SUB/tmlr/references.bib \
      $SUB/tmlr/tmlr.sty $SUB/tmlr/tmlr.bst $SUB/tmlr/supplementary.zip $SUB/tmlr/BUILD_FACTS.txt
python3 $SUB/make_tmlr.py
# The style files are vendored and byte-identical to upstream; TMLR calls tweaking them grounds for
# rejection, so the build re-copies them every time rather than trusting whatever is in the directory.
cp $SUB/tmlr_style/tmlr.sty $SUB/tmlr_style/tmlr.bst $SUB/tmlr/
cp references.bib $SUB/tmlr/
python3 - <<'PY'
import re, shutil
t = open("submission/tmlr/main.tex").read()
figs = {u.replace(".pdf", "") for u in re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}', t)}
for f in sorted(figs):
    shutil.copy(f"figs/{f}.pdf", f"submission/tmlr/figs/{f}.pdf")
print("   %d figures resolved from \\includegraphics" % len(figs))
PY
# bibtex must run here: the bibliography switches from numeric plainnat to author-year tmlr.bst, so the
# repo's .bbl is the wrong one and cannot be reused. Sequenced with ';' for the same reason as above.
( cd $SUB/tmlr
  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
  bibtex main >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true )
[ -f $SUB/tmlr/main.pdf ] || { echo "  ABORT: the TMLR manuscript does not compile"; exit 1; }
TOVF=$(grep -ci overfull $SUB/tmlr/main.log || true); TOVF=${TOVF:-0}
[ "$TOVF" -eq 0 ] || { echo "  ABORT: $TOVF overfull boxes in the TMLR build"; exit 1; }
TUND=$(grep -ci 'undefined reference\|Citation.*undefined\|There were undefined references' $SUB/tmlr/main.log || true)
[ "${TUND:-0}" -eq 0 ] || { echo "  ABORT: ${TUND} undefined references or citations in the TMLR build"; exit 1; }
# Same gate as stage 0, and it matters more here: a non-anonymous TMLR submission is rejected without
# review, and the derived source is one sed away from the named arXiv source.
TLEAK=$(pdftotext $SUB/tmlr/main.pdf - 2>/dev/null | grep -oiE "$IDENT" | sort -u | tr '\n' ' ' || true)
if [ -n "$TLEAK" ]; then
  echo "  ABORT: the TMLR build leaks an author identity into the text layer: $TLEAK"; exit 1
fi
pdftotext -f 1 -l 1 $SUB/tmlr/main.pdf - 2>/dev/null | grep -q 'Paper under double-blind review' \
  || { echo "  ABORT: the TMLR title page is missing the anonymous author block"; exit 1; }
# TMLR's submission-type field is decided by ONE number: "the main content (all pages before references
# and appendices)". Compute it rather than eyeballing it, because the answer selects Regular (<=12) vs
# Long (>12) on the OpenReview form and the form is not something build.sh can check.
TPAGES=$(pdfinfo $SUB/tmlr/main.pdf | awk '/Pages/{print $2}')
TMAIN=$(python3 - "$SUB/tmlr/main.pdf" "$TPAGES" <<'PY'
import subprocess, sys
pdf, n = sys.argv[1], int(sys.argv[2])
for p in range(1, n + 1):
    txt = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), pdf, "-"],
                         capture_output=True, text=True).stdout
    if any(l.strip() == "References" for l in txt.splitlines()):
        print(p - 1); break
else:
    print("UNKNOWN")
PY
)
[ "$TMAIN" != "UNKNOWN" ] || { echo "  ABORT: could not locate the References page to size the main content"; exit 1; }
if [ "$TMAIN" -le 12 ]; then TTYPE="Regular (<= 12 pages of main content)"; else TTYPE="Long (> 12 pages of main content)"; fi
echo "   TMLR build: blinded, $TPAGES pages total, $TMAIN pages of main content"
echo "   OpenReview 'Submission Type' must be set to: $TTYPE"
# The supplement is the same artifact AAAI gets: computed source list, scrubbed of author identity, and
# clean-room verified in stage 2. TMLR's cap is 100 MB, and TMLR's supplement is PUBLIC during and after
# review, which is why stage 2 scrubs rather than merely checks.
if [ ! -f $SUB/aaai/code_and_data.zip ]; then
  echo "  ABORT: $SUB/aaai/code_and_data.zip is absent, so there is no supplement to attach."
  echo "         Build it on the compute host first: BUILD_ONLY=supplement ./submission/build.sh"
  exit 1
fi
cp $SUB/aaai/code_and_data.zip $SUB/tmlr/supplementary.zip
TSZ=$(du -m $SUB/tmlr/supplementary.zip | cut -f1)
echo "   supplementary.zip = ${TSZ} MB (limit 100)"
[ "$TSZ" -le 100 ] || { echo "  ABORT: supplement over TMLR's 100 MB limit"; exit 1; }
# Record the two computed form values where the person filling the form will look for them.
cat > $SUB/tmlr/BUILD_FACTS.txt <<FACTS
Generated by submission/build.sh -- do not edit.
total pages          $TPAGES
main content pages   $TMAIN   (all pages before References; appendices follow the references)
submission type      $TTYPE
supplementary.zip    ${TSZ} MB of 100 MB
anonymity gate       passed (no author string in the PDF text layer)
overfull boxes       0
undefined refs       0
FACTS
# main.bbl is KEPT on purpose. Without it this directory is a trap: a single pdflatex run there emits a
# PDF whose every cross-reference is "??", and it looks like a finished build. With the .bbl present,
# `pdflatex main` twice reproduces the shipped PDF exactly.
rm -f $SUB/tmlr/main.{aux,log,out,blg}
[ -f $SUB/tmlr/SUBMISSION_FORM.md ] || echo "  WARNING: submission/tmlr/SUBMISSION_FORM.md absent (the OpenReview answer sheet)"
fi   # stage 4 (pdf)

if want supplement; then
# NOT a pdf stage. make_claim_index.py checks that every indexed number still appears in the
# manuscript AND that the run output behind it exists under rna_harness/runs -- and it writes the
# table before exiting non-zero on failures. Run on a host without the runs, it overwrites a good
# claim_index.md with 28 FAIL rows and then aborts the build. It did exactly that once.
echo "== 5. claim index (rebuttal artifact) =="
python3 $SUB/make_claim_index.py > $SUB/rebuttal/claim_index.md
echo "   $(grep -c '^| ' $SUB/rebuttal/claim_index.md) rows"
fi   # stage 5 (supplement)

echo "== done. artifacts in $SUB/ =="
for d in arxiv aaai tmlr rebuttal; do [ -d "$SUB/$d" ] && ls -la "$SUB/$d"; done

# The checklist is prose, not generated; build.sh only checks it has not gone missing.
[ -f submission/aaai/reproducibility_checklist.md ] || echo "  WARNING: reproducibility checklist absent"
