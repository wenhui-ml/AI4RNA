"""R3.1 oracle-validity panel: independent pseudoknot-aware folders for cross-oracle agreement.

The Axis-1 "false accept" verdict in the paper rests on ONE reference predictor (pKiss under a fixed
configuration). A reviewer's core objection is that pKiss is a HEURISTIC, not ground truth, so a verdict of
"the design does/does not form the pseudoknotted target" is only defined *relative to that reference*. This
module adds two genuinely INDEPENDENT pseudoknot folders (different energy model + algorithm) so we can
report the oracle-DEPENDENCE of the verdict and adopt an abstention-on-disagreement rule:

  * pKiss   (Janssen & Giegerich; gapc/RNA-shape energy model)          -- the paper's reference C0
  * ProbKnot(RNAstructure; maximum-expected-accuracy over a PF ensemble) -- INDEPENDENT of pKiss
  * ShapeKnots (RNAstructure; heuristic MFE-based pseudoknot search)      -- INDEPENDENT of pKiss

Each oracle folds a sequence and returns its predicted base-PAIR SET (pseudoknot pairs included). We compare
pair sets to the target's pair set; "forms T" := symmetric-difference == 0. RNAstructure predictors need
DATAPATH set to their data_tables directory."""
from __future__ import annotations

import os
import subprocess
import tempfile
from functools import lru_cache
from shutil import which
from typing import Optional, Set, Tuple

from .benchmarks.rnainvbench import pk_pairs, pkiss_fold  # reuse the pKiss reference + pair parser

RNASTR_BIN = os.environ.get("RNASTRUCTURE_BIN", "/home/ubuntu/miniconda3/envs/rnastr/bin")
RNASTR_DATA = os.environ.get("DATAPATH", "/home/ubuntu/miniconda3/envs/rnastr/share/rnastructure/data_tables/")

PairSet = Set[Tuple[int, int]]


def _rnastr_env() -> dict:
    env = dict(os.environ)
    env["DATAPATH"] = RNASTR_DATA
    return env


def _bin(name: str) -> Optional[str]:
    cand = os.path.join(RNASTR_BIN, name)
    if os.path.exists(cand):
        return cand
    return which(name)


def rnastructure_available() -> bool:
    return _bin("ProbKnot") is not None and os.path.isdir(RNASTR_DATA)


def _ct_to_pairs(ct_path: str) -> Optional[PairSet]:
    """Parse an RNAstructure CT file (structure 1 only) into a 0-indexed base-pair set (pseudoknots kept)."""
    try:
        with open(ct_path) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    if not lines:
        return None
    # header: "<n>  <title>"; body rows: idx base i-1 i+1 pair natural
    try:
        n = int(lines[0].split()[0])
    except (ValueError, IndexError):
        return None
    pairs: PairSet = set()
    for ln in lines[1:1 + n]:
        p = ln.split()
        if len(p) < 5:
            continue
        i = int(p[0]); j = int(p[4])
        if j > 0 and i < j:
            pairs.add((i - 1, j - 1))  # CT is 1-indexed; our pair sets are 0-indexed
    return pairs


def _run_rnastr(tool: str, seq: str, timeout_s: float, extra: Optional[list] = None) -> Optional[PairSet]:
    b = _bin(tool)
    if b is None:
        return None
    s = seq.upper().replace("T", "U")
    with tempfile.TemporaryDirectory() as d:
        fa = os.path.join(d, "in.fasta"); ct = os.path.join(d, "out.ct")
        with open(fa, "w") as f:
            f.write(">s\n" + s + "\n")
        try:
            r = subprocess.run([b, fa, ct] + (extra or []), capture_output=True, text=True,
                               timeout=timeout_s, env=_rnastr_env())
        except Exception:
            return None
        if r.returncode != 0 or not os.path.exists(ct):
            return None
        return _ct_to_pairs(ct)


def probknot_pairs(seq: str, timeout_s: float = 60.0) -> Optional[PairSet]:
    """ProbKnot (RNAstructure): maximum-expected-accuracy pseudoknot prediction. Independent of pKiss.
    A raw sequence/fasta input requires the --sequence flag (else ProbKnot expects a PF save file)."""
    return _run_rnastr("ProbKnot", seq, timeout_s, extra=["--sequence"])


def shapeknots_pairs(seq: str, timeout_s: float = 120.0) -> Optional[PairSet]:
    """ShapeKnots (RNAstructure): heuristic MFE-based pseudoknot search (no SHAPE data). Independent of pKiss."""
    return _run_rnastr("ShapeKnots", seq, timeout_s)


def pkiss_pairs(seq: str, timeout_s: float = 60.0) -> Optional[PairSet]:
    """pKiss reference (config C0), as a pair set (reuses the harness's pkiss_fold)."""
    folded = pkiss_fold(seq, timeout_s=timeout_s)
    if folded is None:
        return None
    return pk_pairs(folded)


# oracle registry: name -> callable(seq)->PairSet|None
ORACLES = {"pkiss": pkiss_pairs, "probknot": probknot_pairs, "shapeknots": shapeknots_pairs}


# --------------------------------------------------------------------------- one fold per sequence
# EVERY call below spawns a subprocess, and the certify loop asked for the same three folds of the same
# candidate TWICE: once through _panel_verdict (is the whole structure reproduced?) and again through
# _knot_recall (are the crossing stems reproduced?). Those are two questions about ONE prediction, so the
# prediction is computed once here and both questions are answered from it.
#
# The counter is not decoration. The change that motivated this cache also makes certify() grade every
# candidate instead of stopping at the first one that clears the floor, and "we got better certificates"
# would be worthless if it quietly cost more oracle time. The count is what turns that into a checkable
# claim rather than an assurance.
_ORACLE_CALLS = {"pkiss": 0, "probknot": 0, "shapeknots": 0}


def oracle_call_counts() -> dict:
    """Subprocess folds actually performed, per oracle. Cache hits do not count -- that is the point."""
    return dict(_ORACLE_CALLS)


def reset_oracle_call_counts() -> None:
    for k in _ORACLE_CALLS:
        _ORACLE_CALLS[k] = 0


@lru_cache(maxsize=8192)
def _oracle_pairs_cached(seq: str):
    out = {}
    for name, fn in ORACLES.items():
        _ORACLE_CALLS[name] += 1
        out[name] = fn(seq)
    return out


def oracle_pairs(seq: str) -> dict:
    """The three predictors' pair sets for one sequence, computed once. `None` means that oracle failed
    to run, which is never a rejection -- the distinction every caller here depends on."""
    return dict(_oracle_pairs_cached(seq))


def forms_target(pairs: Optional[PairSet], target: str) -> Optional[bool]:
    """Does a predicted pair set equal the target's pair set exactly? None if the oracle failed."""
    if pairs is None:
        return None
    return pairs == pk_pairs(target)


if __name__ == "__main__":
    # smoke test: fold a couple of sequences under all three oracles and show pair-set sizes
    print("rnastructure_available:", rnastructure_available())
    for s in ["GGGAAACCCAGGGAAAUUUCCCUUU", "GGGCGCAAAGCGCCCAAAGGGCAAACCC"]:
        row = {nm: (len(f(s)) if f(s) is not None else None) for nm, f in ORACLES.items()}
        print(f"  {s}  n_pairs -> {row}")
