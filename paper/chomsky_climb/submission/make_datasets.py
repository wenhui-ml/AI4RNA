#!/usr/bin/env python3
"""Emit the target-list manifests the manuscript promises but the supplement does not ship.

RUN THIS ON THE PROJECT HOST, not here: it needs RNAINVBENCH_DIR to point at a real RNAInvBench
checkout and the eterna100 TSV to be reachable. It writes into rna_harness/runs/cc_ascent/targets/,
which `build.sh` stage 2 already rsyncs wholesale -- so once these files exist, the supplement picks
them up with no change to collect_code.py.

    cd <repo root>                       # the directory holding rna_harness/ and paper/
    export RNAINVBENCH_DIR=/path/to/RNAInvBench
    python3 paper/chomsky_climb/submission/make_datasets.py

WHY. App. B states twice that "the harness, target lists, intrinsic grades, and per-target JSON are
released for replay". The per-target JSON ships; the target lists do not. Every pool except the
structurally disjoint one is defined in App. D as an *index range into an external dataset*
(Pseudobase++ idx 0-59 and 60-109) that the supplement does not contain, and 21 shipped scripts call
load_pseudobase() and therefore open $RNAINVBENCH_DIR/data/Pseudobase.csv at run time. Those scripts
import cleanly in the clean room -- which is why the existing gate passes -- and then fail on the
first call.

WHAT MAKES THIS CHEAP. sweep.py already accepts a TARGETS_JSON manifest and, when it is set, never
touches load_pseudobase. Both code paths reduce each target to (id, n_pk_pairs, n, target) and neither
passes the IUPAC restraint to the operators, so a manifest of {id, target} is *bit-for-bit equivalent*
to the Pseudobase path for re-execution. No dataset needs redistributing. `seq_restr` is emitted
anyway, because split_audit.py reads it for the duplication caveat in App. B (median 0.329 normalized
edit distance), and that claim is unverifiable from the supplement without it.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())

OUT = os.path.join("rna_harness", "runs", "cc_ascent", "targets")

if not os.path.isdir("rna_harness"):
    sys.exit("ABORT: run this from the repo root (the directory containing rna_harness/).")
if not os.environ.get("RNAINVBENCH_DIR"):
    sys.exit("ABORT: set RNAINVBENCH_DIR to your RNAInvBench checkout first.")

from rna_harness.benchmarks.rnainvbench import load_pseudobase, n_pk_pairs, pk_pairs  # noqa: E402

os.makedirs(OUT, exist_ok=True)
written = []


def emit(name, pool, note, rows):
    """Write one manifest in exactly the shape sweep.py's TARGETS_JSON reader expects.

    The file is content-addressed. For the Pseudobase++ pools that is belt-and-braces, since the upstream
    CSV is versioned; for EteRNA100-V2 it is the only pin there is, because the loader reads an unpinned
    `master` branch. A SHA-256 over the emitted bytes turns "whatever master said that day" into an
    artifact a reviewer can check, without claiming a commit we cannot verify was the one fetched.
    """
    p = os.path.join(OUT, name)
    payload = json.dumps({"note": note, "n": len(rows), "rows": rows}, indent=1).encode()
    open(p, "wb").write(payload)
    grades = {}
    for r in rows:
        grades[r["g"]] = grades.get(r["g"], 0) + 1
    written.append((name, pool, len(rows),
                    " ".join(f"{g}:{grades[g]}" for g in sorted(grades)),
                    hashlib.sha256(payload).hexdigest()))
    print(f"   {name}: {len(rows)} targets")


def row(t, idx):
    return {
        "id": t.task_id,
        "target": t.target,
        "orig_index": idx,
        "n": len(t.target),
        "n_pk_pairs": n_pk_pairs(t.target),
        "n_pairs": len(pk_pairs(t.target)),
        # the intrinsic grade of App. D: crossing-pair count, a property of the target fixed before
        # design. Emitted rather than left to be recomputed, because App. B promises "intrinsic grades".
        "g": n_pk_pairs(t.target),
        # not consumed by sweep.py (neither code path passes it to the operators); shipped because
        # split_audit.py needs it for the sequence-homology bound in App. B.
        "seq_restr": (t.meta or {}).get("seq_restr"),
        "source": "Pseudobase++ (RNAInvBench, MIT)",
    }


# --- the three Pseudobase++ pools of Table 'Datasets' -------------------------------------------
tasks = load_pseudobase(limit=251)
print(f"   loaded {len(tasks)} Pseudobase++ tasks from $RNAINVBENCH_DIR")
if len(tasks) < 251:
    print(f"   WARNING: expected 251, got {len(tasks)}. App. D's index ranges assume the full table.")

emit("pseudobase_primary.json", "held-out (primary)",
     "held-out primary pool: Pseudobase++ idx 0-59. The pool behind 43/60, 1/60 and the 0/60 CF floor.",
     [row(t, i) for i, t in enumerate(tasks[:60])])
emit("pseudobase_pool2.json", "second pool",
     "second pool: Pseudobase++ idx 60-109, disjoint from primary.",
     [row(t, i + 60) for i, t in enumerate(tasks[60:110])])
emit("pseudobase_all251.json", "full Pseudobase++ table",
     "full Pseudobase++ table. Needed by split_audit.py (nearest-neighbour structural duplication) "
     "and build_struct_split.py (which derives struct_split.json from it).",
     [row(t, i) for i, t in enumerate(tasks)])

# --- EteRNA100-V2, the nested competence baseline ------------------------------------------------
# benchmarks/eterna100.py fetches a TSV from raw.githubusercontent.com on the master BRANCH with no commit
# pin, so the nested baseline can change under the paper with nothing failing. Snapshot it.
#
# load_eterna100 returns {id, name, target, sample} DICTS, not objects. An earlier version of this script
# read `t.target` and lost the whole manifest to an AttributeError that a broad `except Exception` then
# swallowed -- leaving a stale snapshot on disk that PROVENANCE.md no longer listed, while the build
# reported success. Hence: dict access, and no bare except. If this cannot be emitted, the build stops.
from rna_harness.benchmarks.eterna100 import load_eterna100  # noqa: E402

et = load_eterna100()
if len(et) != 100:
    sys.exit(f"ABORT: EteRNA100-V2 yielded {len(et)} puzzles, expected 100. App. D's table assumes 100.")
emit("eterna100_v2.json", "EteRNA100-V2 (nested CF baseline)",
     "EteRNA100-V2 nested (CF) competence baseline, snapshotted from "
     "eternagame/eterna100-benchmarking. The loader reads the master branch, which is not a pin, "
     "so this snapshot and its sha256 in PROVENANCE.md are the pin.",
     [{"id": f"eterna100:{t['id']}", "target": t["target"], "orig_index": i,
       "n": len(t["target"]), "n_pk_pairs": n_pk_pairs(t["target"]),
       "n_pairs": len(pk_pairs(t["target"])), "g": n_pk_pairs(t["target"]),
       "name": t["name"], "seq_restr": None, "source": "EteRNA100-V2"}
      for i, t in enumerate(et)])
grades_et = {r for r in (n_pk_pairs(t["target"]) for t in et)}
if grades_et != {0}:
    sys.exit(f"ABORT: EteRNA100-V2 is the NESTED baseline but carries crossing pairs: g in {sorted(grades_et)}")

# --- cross-source pool: already shipped, verified here rather than re-emitted --------------------
xsrc = os.path.join("rna_harness", "runs", "cc_ascent", "bprna_pk", "targets.json")
if os.path.exists(xsrc):
    n = len(json.load(open(xsrc))["rows"])
    print(f"   cross-source pool already ships as bprna_pk/targets.json ({n} targets) -- not re-emitted")
else:
    print("   WARNING: bprna_pk/targets.json is missing; regenerate it with bprna_pk.py "
          "(it reads RNAInvBench's inverse_rna_folding_train_dotbracket.pkl.gz, which does not ship)")

# --- provenance ---------------------------------------------------------------------------------
with open(os.path.join(OUT, "PROVENANCE.md"), "w") as f:
    f.write("""# Target lists

These are the pools of App. D, emitted as manifests so the supplement is self-contained. Each file is in
the format `sweep.py` reads from `TARGETS_JSON`, so any sweep re-executes without an RNAInvBench checkout:

    TARGETS_JSON=rna_harness/runs/cc_ascent/targets/pseudobase_primary.json \\
      python3 rna_harness/runs/cc_ascent/sweep.py mytag 0 60 0 always_cheap,autonomous

`g` is the intrinsic grade of App. D: the crossing-pair count, computed from the dot-bracket in O(|P|^2)
and fixed before design. `seq_restr` is the IUPAC restraint string. Neither `sweep.py` code path passes
the restraint to the design operators, so these manifests reproduce the runs exactly; `seq_restr` is here
only because `split_audit.py` measures edit distance over it.

| file | pool | n | grade distribution | sha256 |
|---|---|---|---|---|
""")
    for name, pool, n, grades, sha in written:
        f.write(f"| `{name}` | {pool} | {n} | `{grades}` | `{sha[:16]}` |\n")
    f.write("\nFull digests:\n\n")
    for name, pool, n, grades, sha in written:
        f.write(f"    {sha}  {name}\n")
    f.write("""
## Upstream sources and licences

- **Pseudobase++** via RNAInvBench (COLA-Laboratory), MIT licence. `data/Pseudobase.csv`, 251 tasks,
  21-137 nt. MIT permits redistribution; these manifests are derived from it. **Record the commit SHA
  of the checkout used.**
- **bpRNA / ArchiveII** via RNAInvBench `inverse_rna_folding_train_dotbracket.pkl.gz`, same licence.
  Reconstructed to crossing dot-bracket by `bprna_pk.py`; multiplets excluded.
- **EteRNA100-V2**, eternagame/eterna100-benchmarking. The loader reads the `master` branch at run time,
  which is not a pin, and the commit that was actually fetched during the paper's runs was never recorded
  -- so it cannot be reconstructed now, and claiming today's `master` SHA would assert something we did not
  verify. What *is* verifiable is the snapshot: `eterna100_v2.json` above is content-addressed, and every
  number in the paper that touches the nested baseline is reproducible against that file. Anyone re-running
  from upstream should compare their own digest to it before comparing results.

## Grade distributions to check against App. D

The manifests must reproduce Table `tab:appdata`. Primary: g:n = 3:4, 4:6, 5:22, 6:18, 7:7, 8:1, 9:2
(60 targets). Cross-source: 1:4, 4:13, 5:6, 6:16, 7:8, 8:5, 9:3, 10:3, 12:1, 13:1 (60 targets). If a
distribution disagrees, the index ranges or the upstream table have moved and the manuscript's pool
description is stale -- fix that before shipping.
""")

print()
print("   grade distributions emitted (check these against App. D's table):")
for name, pool, n, grades, sha in written:
    print(f"     {name:28} n={n:<4} g:n = {grades}")
    print(f"     {'':28} sha256 {sha}")
# The failure above was not that a manifest was missing -- it was that PROVENANCE.md stopped listing a
# file that was still on disk, and nothing noticed. Assert the two agree, every time.
on_disk = {f for f in os.listdir(OUT) if f.endswith(".json")}
listed = {w[0] for w in written}
if on_disk != listed:
    sys.exit(f"ABORT: targets/ and PROVENANCE.md disagree.\n"
             f"  on disk but unlisted: {sorted(on_disk - listed) or 'none'}\n"
             f"  listed but absent:    {sorted(listed - on_disk) or 'none'}\n"
             f"  A stale manifest that no longer appears in PROVENANCE.md is worse than a missing one.")
print(f"\n   wrote {len(written) + 1} files to {OUT}/ ({len(written)} manifests + PROVENANCE.md)")
print("   Now re-run: paper/chomsky_climb/submission/build.sh")
