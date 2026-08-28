#!/usr/bin/env python3
"""Assemble an arXiv submission for one paper, and refuse to assemble one that is not postable.

    python make_arxiv.py intervention_benchmark [outdir]

arXiv does not run tectonic. It runs pdfTeX from TeX Live over whatever is in the tarball, so the
only assembly worth trusting is one that has itself been compiled by pdflatex+bibtex from the
tarball's own contents, in a directory containing nothing else. That is what this does:

  1. refuse if audit_manuscript.py reports a blocking finding -- in particular [I], which fires on
     PLACEHOLDER/TODO/FIXME. A manuscript that still names something it does not have is not
     postable, and this is the gate rather than someone's memory.
  2. copy ONLY what the manuscript \\input or \\cite: main .tex, figs/*.tex actually included,
     refs.bib. Retired figure directories, PDFs, logs and generators stay behind.
  3. build it there with latexmk -pdf (pdflatex + bibtex), from scratch, twice over, and fail on
     any error or any unresolved citation or reference.
  4. keep the .bbl, since arXiv prefers it to running bibtex itself, and drop the aux files.
  5. tar it, then report page count and byte size.

Nothing here is specific to either paper.
"""
from __future__ import annotations

import os, re, shutil, subprocess, sys, tarfile

PAPERS = os.path.dirname(os.path.abspath(__file__))
KEEP_EXT = (".tex", ".bib", ".bbl", ".cls", ".sty", ".bst")


def run(cmd, cwd, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def main(pdir: str, outdir: str | None = None) -> int:
    pdir = os.path.join(PAPERS, pdir) if not os.path.isabs(pdir) else pdir
    name = os.path.basename(pdir.rstrip("/"))
    mains = [f for f in os.listdir(pdir)
             if f.endswith(".tex") and "\\documentclass" in open(os.path.join(pdir, f)).read()]
    assert len(mains) == 1, f"expected one main .tex in {pdir}, found {mains}"
    main_tex = mains[0]
    stem = main_tex[:-4]

    # ---- 1. the postability gate
    a = run([sys.executable, os.path.join(PAPERS, "audit_manuscript.py"), pdir], PAPERS)
    print(a.stdout[-1500:])
    if a.returncode != 0:
        print("REFUSING to assemble: the audit reports a blocking finding (see above).")
        return 1

    # ---- 2. copy only what is included
    out = outdir or os.path.join(pdir, "arxiv")
    if os.path.isdir(out):
        shutil.rmtree(out)          # a stale file in the tarball is worse than no tarball
    os.makedirs(os.path.join(out, "figs"))
    src = open(os.path.join(pdir, main_tex)).read()
    shutil.copy2(os.path.join(pdir, main_tex), out)
    for b in ("refs.bib",):
        if os.path.exists(os.path.join(pdir, b)):
            shutil.copy2(os.path.join(pdir, b), out)
    inputs = set(re.findall(r"\\(?:input|include)\{([^}]+)\}", src))
    copied = []
    for rel in sorted(inputs):
        rel = rel if rel.endswith(".tex") else rel + ".tex"
        s = os.path.join(pdir, rel)
        assert os.path.exists(s), f"\\input{{{rel}}} has no file"
        d = os.path.join(out, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d); copied.append(rel)
    print(f"copied {main_tex}, refs.bib and {len(copied)} included fragment(s)")

    # ---- 3. build it where it will be judged, from nothing
    r = run(["latexmk", "-pdf", "-interaction=nonstopmode", main_tex], out)
    log = open(os.path.join(out, stem + ".log"), errors="replace").read()
    errs = [l for l in log.splitlines() if l.startswith("!")]
    undef = re.findall(r"LaTeX Warning: (Citation|Reference) `([^']+)' .*undefined", log)
    if r.returncode != 0 or errs or undef:
        print(f"BUILD FAILED in the assembled directory: {len(errs)} error(s), "
              f"{len(undef)} undefined citation/reference")
        for l in errs[:5]:
            print("   ", l)
        for k, v in undef[:5]:
            print(f"    undefined {k.lower()}: {v}")
        return 1

    # ---- 4. arXiv wants the .bbl and none of the rest
    import pymupdf
    pages = pymupdf.open(os.path.join(out, stem + ".pdf")).page_count
    dropped = [f for f in sorted(os.listdir(out))
               if os.path.isfile(os.path.join(out, f)) and not f.endswith(KEEP_EXT)]
    for f in dropped:                 # every one of these was generated in this directory
        os.remove(os.path.join(out, f))
    print(f"dropped {len(dropped)} build product(s): {' '.join(dropped)}")

    # ---- 5. tar
    tarpath = os.path.join(pdir, f"{stem}_arxiv.tar.gz")
    with tarfile.open(tarpath, "w:gz") as t:
        for root, _d, files in os.walk(out):
            for f in sorted(files):
                full = os.path.join(root, f)
                t.add(full, arcname=os.path.relpath(full, out))
    size = os.path.getsize(tarpath)
    with tarfile.open(tarpath) as t:
        members = sorted(m.name for m in t.getmembers() if m.isfile())

    # ---- 6. the check that matters: extract the tarball somewhere empty and build it the way
    # arXiv will -- pdflatex alone, no bibtex, no latexmk, nothing left over from step 3.
    ver = os.path.join(out, os.pardir, f".{stem}_verify")
    ver = os.path.abspath(ver)
    if os.path.isdir(ver):
        shutil.rmtree(ver)
    os.makedirs(ver)
    with tarfile.open(tarpath) as t:
        t.extractall(ver)
    for _ in range(3):
        run(["pdflatex", "-interaction=nonstopmode", main_tex], ver)
    vlog = open(os.path.join(ver, stem + ".log"), errors="replace").read()
    verrs = [l for l in vlog.splitlines() if l.startswith("!")]
    vundef = re.findall(r"LaTeX Warning: (Citation|Reference) `([^']+)' .*undefined", vlog)
    vpages = pymupdf.open(os.path.join(ver, stem + ".pdf")).page_count if not verrs else -1
    shutil.rmtree(ver)
    if verrs or vundef or vpages != pages:
        print(f"TARBALL DOES NOT BUILD STANDALONE: {len(verrs)} error(s), {len(vundef)} undefined, "
              f"{vpages} pages against {pages} in the assembled directory")
        for l in verrs[:5]:
            print("   ", l)
        os.remove(tarpath)
        return 1
    print(f"verified: the extracted tarball builds to {vpages} pages under pdflatex alone")
    print(f"\n{name}: {pages} pages, {len(members)} files, {size/1024:.0f} KiB")
    print("  " + "\n  ".join(members))
    print(f"\nwrote {tarpath}")
    print("arXiv note: .bbl is included, so bibtex need not run there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
