#!/usr/bin/env python3
"""Audit the audit packages. Offline, $0.

Written because "I built the package" is not evidence the package is usable. Every check below
either passes with a number or names the file that fails it.
"""
from __future__ import annotations
import hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
PKGS = [("NM", "intervention_benchmark_audit", "intervention_benchmark", "NM_intervention.tex"),
        ("R1", "rna_placement_audit", "rna_placement", "R1_placement.tex")]
FAIL = []


def bad(pkg, code, msg):
    FAIL.append((pkg, code, msg)); print(f"  !! [{code}] {msg}")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for tag, folder, pdir, texname in PKGS:
    out = os.path.join(HERE, folder); P = os.path.join(HERE, pdir)
    print(f"\n=== {tag}  {folder}")
    J = lambda n: json.load(open(os.path.join(out, n)))
    man, sub, cl, art, pc = J("MANIFEST.json"), J("submission.json"), J("claims.json"), \
        J("artifacts.json"), J("process_claims.json")

    # A. the manifest must describe what is actually on disk
    onpkg = {f for f in os.listdir(out) if f != "MANIFEST.json"}
    listed = {f["name"] for f in man["files"]}
    if onpkg != listed:
        bad(tag, "A1", f"manifest/disk mismatch: only-disk {onpkg-listed}, only-manifest {listed-onpkg}")
    drift = [f["name"] for f in man["files"] if sha(os.path.join(out, f["name"])) != f["sha256"]]
    print(f"  A  manifest: {len(listed)} files listed, {len(drift)} checksum drift")
    if drift:
        bad(tag, "A2", f"checksum drift since build: {drift}")

    # B. every artifact path must resolve, and its checksum must still be current
    miss, stale = [], []
    for grp in ("input_data", "manuscript", "our_run_artifacts", "companion_paper_artifacts"):
        for a in art.get(grp, []):
            p = a["path"] if os.path.isabs(a["path"]) else os.path.join(REPO, a["path"])
            if not os.path.exists(p):
                miss.append(a["path"]); continue
            if a.get("sha256") and sha(p) != a["sha256"]:
                stale.append(a["path"])
    print(f"  B  artifacts: {sum(len(art.get(g,[])) for g in ('input_data','manuscript','our_run_artifacts','companion_paper_artifacts'))} listed, "
          f"{len(miss)} missing, {len(stale)} stale checksum")
    if miss:
        bad(tag, "B1", f"artifact paths do not resolve: {miss[:4]}")
    if stale:
        bad(tag, "B2", f"artifact checksums stale: {stale[:4]}")

    # C. can the auditor actually OBTAIN the substrate? absolute paths on our machine are not
    #    an answer; a package that only works on the machine that built it is not a package.
    for a in art["input_data"]:
        h = a.get("how_to_obtain")
        if not isinstance(h, dict):
            bad(tag, "C1", f"{os.path.basename(a['path'])}: no acquisition route -- a package that "
                           f"only works on the machine that built it is not a package")
        elif not re.search(r"https?://", str(h.get("release", ""))):
            bad(tag, "C2", f"{os.path.basename(a['path'])}: acquisition route names no release channel")
        elif not a.get("sha256"):
            bad(tag, "C3", f"{os.path.basename(a['path'])}: no checksum, so 'the same file' is uncheckable")
    print(f"  C  input substrate: {len(art['input_data'])} files, acquisition route + checksum checked")

    # D. every backing_artifact named by a claim must be inventoried
    inv = {os.path.basename(a["path"]) for a in art["our_run_artifacts"]}
    for r in cl["claims"]:
        for a in r["backing_artifacts"]:
            if a not in inv:
                bad(tag, "D1", f"{r['id']} cites {a}, not in artifacts.json")
    ncites = sum(len(r["backing_artifacts"]) for r in cl["claims"])
    print(f"  D  claim->artifact cross refs: {ncites} checked")

    # E. no unresolved macro, and every printed macro must exist in the generated numbers file
    numsrc = open(os.path.join(P, "figs",
                  "nm_numbers.tex" if tag == "NM" else "rna_numbers.tex")).read()
    defined = set(re.findall(r"\\csname\s+jdg@(\w+)\\endcsname", numsrc))
    unres = [(r["id"], k) for r in cl["claims"] for k, v in r["printed_value"]["macros"].items()
             if v is None or k not in defined]
    print(f"  E  printed macros: {sum(len(r['printed_value']['macros']) for r in cl['claims'])} "
          f"quoted, {len(unres)} unresolved")
    if unres:
        bad(tag, "E1", f"macros missing from the generated numbers: {unres[:5]}")

    # F. every claim's label must exist in the manuscript, and its quote must have been extracted
    tex = open(os.path.join(P, texname)).read()
    quoted = {q["claim_id"] for q in sub["claim_quotes_as_printed"]}
    for r in cl["claims"]:
        if r["id"] not in quoted:
            bad(tag, "F1", f"{r['id']} has no quoted sentence in submission.json")
    leaks = [q["claim_id"] for q in sub["claim_quotes_as_printed"] if "??" in q["verbatim_as_printed"]]
    if leaks:
        bad(tag, "F2", f"unexpanded macro left in a quote: {leaks}")
    print(f"  F  quotes: {len(quoted)}/{len(cl['claims'])} claims quoted")

    # F3. verify_gate.json must cover every claim and must carry NO verdicts. A gate file with a
    # filled-in ok field is an answer sheet wearing a checklist's clothes.
    vg_path = os.path.join(out, "verify_gate.json")
    if os.path.exists(vg_path):
        vg = json.load(open(vg_path))
        seen = {r["claim_id"] for r in vg["per_claim"]}
        missing = [r["id"] for r in cl["claims"] if r["id"] not in seen]
        if missing:
            bad(tag, "F3", f"verify_gate.json does not cover {missing}")
        filled = [r["claim_id"] for r in vg["per_claim"]
                  if r.get("evidence_sufficient") is not None
                  or any(i.get("ok") is not None for i in r["items"])]
        if filled:
            bad(tag, "F4", f"verify_gate.json carries pre-filled verdicts for {filled[:5]} -- "
                            "those fields belong to the auditor")
        nart = sum(len(r["checked_artifacts"]) for r in vg["per_claim"])
        print(f"  F3 verify_gate: {len(seen)} claims, {nart} artifact refs, 0 pre-filled verdicts")

    # G. ANSWER LEAKAGE. The specification files must not carry our results.
    for fn in ("criterion.md", "estimator.md", "sanity_checks.md", "README.md"):
        txt = open(os.path.join(out, fn)).read()
        nums = [n for n in re.findall(r"(?<![\w.])[-+]?0\.\d{3,}", txt)]
        if nums:
            bad(tag, "G1", f"{fn} carries result-shaped decimals {nums[:5]} -- that is an answer sheet")
    print("  G  answer leakage in the specification files: checked")

    # H. the independence rules must actually be stated where the auditor will read them
    need = ["Do not import", "printed_value", "public"]
    txt = json.dumps(sub, ensure_ascii=False)
    for n in need:
        if n not in txt:
            bad(tag, "H1", f"independence condition missing the phrase {n!r}")
    print("  H  independence conditions: present")

print("\n" + "=" * 70)
if FAIL:
    print(f"{len(FAIL)} finding(s):")
    for f in FAIL:
        print(f"  {f[0]} [{f[1]}] {f[2]}")
else:
    print("clean")
sys.exit(1 if FAIL else 0)
