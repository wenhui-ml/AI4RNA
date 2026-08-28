#!/usr/bin/env python3
"""Audit a manuscript against the artifacts it is generated from.

make_check.py already refuses hand-typed *decimals*. This is the wider pass, and it exists
because the three worst defects found in these two papers were none of them decimals:

  - a sentence in R1 asserting a preregistration row that did not exist,
  - two whole subsections in NM referenced by \\ref{} but never written,
  - two figures whose *data* were wrong while the build was clean.

So this checks provenance and reachability rather than formatting:

  A  every \\jdg{} used is defined                      (the build enforces it; we report it)
  B  every macro DEFINED is USED  -- a dead macro means a number was computed and then
     never reported, which is where a missing claim hides
  C  a >=3-digit integer typed in the body that equals a measured value is a hand-typed number
     that slipped past make_check.py's decimal-only rule
  D  every artifact the generator reads exists AND is tracked by git (branch-per-paper
     archival: the paper must travel with its data)
  E  every generated figure fragment carries its provenance header
  F  phrasings the run audits explicitly forbid do not appear
  G  every \\label is reachable by some \\ref, and no \\ref is dangling

Usage:  python audit_manuscript.py <paper-dir>
COST: $0, offline.
"""
from __future__ import annotations

import json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CEIL = os.path.join(ROOT, "rna_harness", "runs", "ceiling")

# claims the run audits refuse to license.  m31_audit.md: "May not: that a predictor beats,
# or outperforms, a depth baseline."  The interval on the difference contains zero.
FORBIDDEN = [
    (r"\b(beats?|outperform\w*|better than|superior to)\b[^.]{0,80}\bdepth\b",
     "m31_audit.md forbids claiming a predictor beats the depth baseline"),
    (r"\bdepth\b[^.]{0,60}\b(is beaten|is outperformed)\b",
     "same, in the passive"),
    (r"\bp\s*=\s*0\.0\b(?!\d)",
     "a permutation floor printed as p = 0.0 rather than as p < 1/nperm"),
]

# The manuscript is *required* to discuss the claim it refuses to make, and the
# preregistration table is required to record it as staked-and-not-met.  A forbidden
# phrasing inside a refusal, a stake, or a kill condition is the paper working correctly.
NEGATED = re.compile(
    r"\b(do(es)? not|don't|cannot|can't|no|never|refus\w*|declin\w*|withdraw\w*|"
    r"staked?|stake|kill condition|prediction|predicted|would|failed?|not\b)\b", re.I)


def tracked(path: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", path],
                       cwd=ROOT, capture_output=True)
    return r.returncode == 0


def main(pdir: str) -> int:
    pdir = os.path.abspath(pdir)
    tex = [f for f in os.listdir(pdir) if f.endswith(".tex")]
    main_tex = max((f for f in tex), key=lambda f: os.path.getsize(os.path.join(pdir, f)))
    src = open(os.path.join(pdir, main_tex)).read()
    numbers = [os.path.join(pdir, "figs", f) for f in os.listdir(os.path.join(pdir, "figs"))
               if f.endswith("_numbers.tex")]
    defs: dict[str, str] = {}
    for nf in numbers:
        for m in re.finditer(r"\\csname\s+jdg@(\w+)\\endcsname\{(.*?)\}\s*$",
                             open(nf).read(), re.M):
            defs[m.group(1)] = m.group(2)

    body = src[src.index("\\begin{document}"):]
    # Strip comments, but an ESCAPED percent is text, not a comment. Splitting on a bare "%"
    # truncated every line containing "5\%" -- and a paper of percentages has many, so macros that
    # WERE reported were being listed under [B] as never shown. Erring toward keeping text is the
    # safe direction for [B]; it can only hide a defect in [C]/[F], never invent one.
    body_nc = "\n".join(re.split(r"(?<!\\)%", ln)[0] if not ln.lstrip().startswith("%") else ""
                        for ln in body.splitlines())
    used = set(re.findall(r"\\jdg\{(\w+)\}", body_nc))

    print(f"=== {os.path.basename(pdir)}/{main_tex}")
    print(f"    {len(defs)} macros defined, {len(used)} used")
    fail = 0

    # A -- used but undefined
    undef = sorted(used - set(defs))
    if undef:
        fail += 1
        print(f"\n[A] {len(undef)} \\jdg{{}} used but NOT defined: {undef}")

    # B -- defined but never used
    figtext = ""
    for f in sorted(os.listdir(os.path.join(pdir, "figs"))):
        if f.endswith(".tex") and not f.endswith("_numbers.tex"):
            figtext += open(os.path.join(pdir, "figs", f)).read()

    def in_a_figure(v: str) -> bool:
        """the value is drawn or annotated in a generated figure"""
        s = re.sub(r"[{}\\,]|ensuremath|texttt|%", "", v).strip()
        s = s.lstrip("+")
        if not s or len(s) < 3:
            return False
        core = s.lstrip("-")
        return core in figtext or (core.rstrip("0").rstrip(".") or core) in figtext

    dead_all = sorted(set(defs) - used)
    shown = [d for d in dead_all if in_a_figure(defs[d])]
    dead = [d for d in dead_all if d not in shown]
    if shown:
        print(f"\n[B] {len(shown)} macros unused in text but printed in a figure (fine): "
              f"{', '.join(shown)}")
    if dead:
        print(f"\n[B] {len(dead)} macros computed and never shown to the reader at all:")
        for d in dead:
            print(f"      {d:22s} = {defs[d]}")

    # C -- hand-typed integers that equal a measured value
    vals = {}
    for k, v in defs.items():
        s = re.sub(r"[{},\\%]", "", v)
        if re.fullmatch(r"\d{3,}", s):
            vals.setdefault(s, []).append(k)
    hits = []
    for i, ln in enumerate(body_nc.splitlines(), 1):
        stripped = re.sub(r"\\jdg\{\w+\}", "", ln)
        stripped = re.sub(r"\\(label|ref|cite\w*|input|include\w*)\{[^}]*\}", "", stripped)
        for m in re.finditer(r"(?<![\w.,])(\d{3,})(?![\w.,%])", stripped):
            if m.group(1) in vals:
                hits.append((i, m.group(1), vals[m.group(1)], ln.strip()[:88]))
    if hits:
        fail += 1
        print(f"\n[C] {len(hits)} hand-typed integer(s) equal to a measured value:")
        for ln, v, ks, ctx in hits:
            print(f"      line {ln}: {v} (= \\jdg{{{ks[0]}}})  |  {ctx}")

    # D -- artifacts read by the generators exist and are tracked
    missing, untracked = [], []
    for gen in ("make_numbers.py", "make_figs.py"):
        p = os.path.join(pdir, gen)
        if not os.path.exists(p):
            continue
        for m in re.finditer(r"[\"']([\w.\-]+\.json)[\"']", open(p).read()):
            if os.path.exists(os.path.join(pdir, m.group(1))):
                continue                      # a paper-local file (provenance.json), not an artifact
            a = os.path.join(CEIL, m.group(1))
            if not os.path.exists(a):
                missing.append(m.group(1))
            elif not tracked(a):
                untracked.append(m.group(1))
    if missing:
        fail += 1
        print(f"\n[D] {len(set(missing))} artifact(s) read by a generator do NOT exist: "
              f"{sorted(set(missing))}")
    if untracked:
        fail += 1
        print(f"\n[D] {len(set(untracked))} artifact(s) exist but are NOT tracked by git: "
              f"{sorted(set(untracked))}")

    # E -- figure fragments carry provenance
    figdir = os.path.join(pdir, "figs")
    noprov = [f for f in sorted(os.listdir(figdir))
              if f.endswith(".tex") and not f.endswith("_numbers.tex")
              and "GENERATED by" not in open(os.path.join(figdir, f)).read()[:400]]
    if noprov:
        fail += 1
        print(f"\n[E] {len(noprov)} figure fragment(s) without a provenance header: {noprov}")

    # H -- the manuscript's claim about its own process, made checkable.
    # The preregistration appendix says which artifacts were declared before their code existed.
    # provenance.json is that statement in machine-readable form; this refuses it if it drifts.
    prov_path = os.path.join(pdir, "provenance.json")
    if os.path.exists(prov_path):
        prov = json.load(open(prov_path))
        classes = {k: v for k, v in prov.items() if not k.startswith("_")}
        listed: dict[str, list[str]] = {}
        for cls, items in classes.items():
            for a in items:
                if a.startswith("_"):
                    continue
                listed.setdefault(a, []).append(cls)
        read = set()
        for gen in ("make_numbers.py", "make_figs.py"):
            gp = os.path.join(pdir, gen)
            if os.path.exists(gp):
                read |= set(re.findall(r"[\"']([\w.\-]+\.json)[\"']", open(gp).read()))
        read.discard("provenance.json")

        unclassified = sorted(read - set(listed))
        if unclassified:
            fail += 1
            print(f"\n[H] {len(unclassified)} artifact(s) read by a generator but not classified "
                  f"in provenance.json: {unclassified}")
        stale = sorted(set(listed) - read)
        if stale:
            fail += 1
            print(f"\n[H] {len(stale)} artifact(s) classified in provenance.json but no longer "
                  f"read: {stale}")
        twice = sorted(a for a, cs in listed.items() if len(cs) > 1)
        if twice:
            fail += 1
            print(f"\n[H] {len(twice)} artifact(s) in more than one class: {twice}")

        # the claim itself: preregistered means a frozen declaration really exists and is tracked
        for a, cover in classes.get("preregistered", {}).items():
            if a.startswith("_"):
                continue
            decl = cover or (a[:-5] + "_declaration.json")
            dp = os.path.join(CEIL, decl)
            if not os.path.exists(dp):
                fail += 1
                print(f"\n[H] {a} is claimed preregistered but {decl} does not exist")
            elif not tracked(dp):
                fail += 1
                print(f"\n[H] {a} is claimed preregistered but {decl} is not tracked by git")
        # and the converse: a run with a frozen declaration must not be filed as unplanned.
        # "corpus" is exempt: it names released or derived DATA the analysis reads, not a run, so
        # there is nothing for it to have declared. It was added when R1's ledger turned up six
        # population files that no experiment class fitted, and shoehorning data into "support"
        # would have made the taxonomy lie about what those files are.
        for cls in ("survey", "review", "support"):
            for a in classes.get(cls, {}):
                if a.startswith("_"):
                    continue
                if os.path.exists(os.path.join(CEIL, a[:-5] + "_declaration.json")):
                    fail += 1
                    print(f"\n[H] {a} is filed as {cls} but a frozen declaration exists for it")

    # E2 -- a generated fragment the manuscript never inputs is a figure that was built,
    # compiled and then orphaned; it is the [B] defect in graphical form.
    inputs = {m.group(1) for m in
              re.finditer(r"\\input\{figs/([A-Za-z0-9_]+)(?:\.tex)?\}", body_nc)}
    frags = {f[:-4] for f in os.listdir(figdir)
             if f.endswith(".tex") and not f.endswith("_numbers.tex")}
    orphan_figs = sorted(frags - inputs)
    if orphan_figs:
        fail += 1
        print(f"\n[E] {len(orphan_figs)} generated figure fragment(s) never \\input by the "
              f"manuscript: {orphan_figs}")

    # F -- forbidden phrasings
    flat = re.sub(r"\s+", " ", body_nc)
    for pat, why in FORBIDDEN:
        for m in re.finditer(pat, flat, re.I):
            # the sentence the hit sits in, back to the previous full stop
            lo = flat.rfind(". ", 0, m.start()) + 1
            sentence = flat[lo:m.end() + 90]
            if NEGATED.search(sentence[:m.start() - lo + 10]):
                continue                      # a refusal or a recorded stake, not a claim
            fail += 1
            print(f"\n[F] forbidden phrasing -- {why}\n      ...{sentence.strip()[:210]}...")

    # G -- label/ref reachability
    labels = set(re.findall(r"\\label\{([^}]*)\}", body_nc))
    # A paper may define its own label-planting macro. NM's \claimhead{} does \refstepcounter
    # plus \label, so its labels never appear as a literal \label{} in the body and every \ref
    # to one read as dangling. Collect them too rather than making the manuscript work around
    # the auditor.
    labels |= set(re.findall(r"\\claimhead\{([^}]*)\}", body_nc))
    # \eqref must be in this alternation or every labelled equation reads as an orphan
    refs = set(re.findall(r"\\(?:eq|auto|c|C|page|name)?ref\{([^}]*)\}", body_nc))
    dangling = sorted(refs - labels)
    orphan = sorted(l for l in labels - refs if not l.startswith(("fig:", "tab:")))
    if dangling:
        fail += 1
        print(f"\n[G] {len(dangling)} \\ref to a label that does not exist: {dangling}")
    if orphan:
        print(f"\n[G] {len(orphan)} \\label never referenced: {orphan}")

    # I -- nothing unfinished may reach a reader. A manuscript is postable when this is empty;
    # until then the audit says so out loud rather than leaving it to whoever remembers.
    unfinished = sorted(set(re.findall(r"PLACEHOLDER[-\w]*|\bTODO\b|\bFIXME\b|\bXXX\b", body_nc)))
    if unfinished:
        fail += 1
        n = sum(body_nc.count(u) for u in unfinished)
        print(f"\n[I] {n} unfinished marker(s) in the manuscript body: {unfinished}")
        print("    NOT POSTABLE until each is replaced by the thing it stands for.")

    print(f"\n--- {'FAIL' if fail else 'clean'} ({fail} blocking finding(s))")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
