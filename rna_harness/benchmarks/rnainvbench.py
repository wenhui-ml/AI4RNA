"""RNAInvBench pseudoknot benchmark: loader + pseudoknot-AWARE verifier + a guard against the
silent-pseudoknot-drop false positive.

WHY THIS EXISTS. Our primary oracle is ViennaRNA MFE (Turner 2004), which is pseudoknot-FREE:
`RNA.ptable("..[[..]]..")` parses ONLY the round-bracket pairs and silently drops '[]{}' as unpaired.
So feeding a pseudoknot target (extended dot-bracket) to the MFE harness makes it verify a
PSEUDOKNOT-FREE CORRUPTION of the target -> a bogus "solved". DesiRNA optimises the same MFE model, so
on pseudoknot design it is categorically OUT OF BOX (it can only ever build the nested skeleton).

To evaluate pseudoknot design CORRECTLY we need a pseudoknot-aware folder. We use pKiss (Janssen &
Giegerich; the tool antaRNA uses). Set PKISS_CMD or install `conda create -n pk -c bioconda pkiss`.

Dataset: RNAInvBench (COLA-Laboratory, MIT license) `data/Pseudobase.csv` -- Pseudobase++, 251
constrained pseudoknot design tasks (len 21-137). Columns: index, seq (IUPAC restraint, N=any),
str (extended dot-bracket: () nested + [] {} pseudoknot levels).
"""
from __future__ import annotations

import csv
import os
import subprocess
from typing import Dict, List, Optional, Set, Tuple

from ..constraints import ConstraintSpec
from ..structure_utils import has_pseudoknot as is_pseudoknot  # canonical guard (dedup)
from ..task import DesignTask

RNAINVBENCH_DIR = os.environ.get("RNAINVBENCH_DIR", "/data/wenhui/RNAInvBench")
PSEUDOBASE_CSV = os.path.join(RNAINVBENCH_DIR, "data", "Pseudobase.csv")

# extended dot-bracket: each bracket family is an independent pairing "layer" (pseudoknot level)
_OPEN2CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
_CLOSE2OPEN = {v: k for k, v in _OPEN2CLOSE.items()}
_PK_BRACKETS = set("[]{}<>")


# ---- structure algebra (pseudoknot-aware) --------------------------------------------------------
# is_pseudoknot is imported from structure_utils (has_pseudoknot) -- the canonical guard used by
# the core verifier. It flags any target with '[]{}<>' as unverifiable under the MFE oracle.
def pk_pairs(struct: str) -> Set[Tuple[int, int]]:
    """Parse extended dot-bracket into a SET of (i, j) base pairs, i<j (all bracket families)."""
    stacks: Dict[str, List[int]] = {o: [] for o in _OPEN2CLOSE}
    pairs: Set[Tuple[int, int]] = set()
    for i, c in enumerate(struct):
        if c in _OPEN2CLOSE:
            stacks[c].append(i)
        elif c in _CLOSE2OPEN:
            st = stacks[_CLOSE2OPEN[c]]
            if st:
                pairs.add((st.pop(), i))
    return pairs


def pk_bp_distance(a: str, b: str) -> int:
    """Base-pair distance = size of the symmetric difference of the two pair SETS (ViennaRNA's
    definition, but pseudoknot-aware because pk_pairs reads every bracket family)."""
    return len(pk_pairs(a) ^ pk_pairs(b))


# ---- pseudoknot-aware verifier (pKiss) -----------------------------------------------------------
def _pkiss_cmd() -> Optional[List[str]]:
    env = os.environ.get("PKISS_CMD")
    if env:
        import shlex
        return shlex.split(env)
    for cand in ("/home/ubuntu/miniconda3/envs/pk/bin/pKiss", "pKiss"):
        from shutil import which
        if os.path.exists(cand) or which(cand):
            return [cand]
    return None


def pkiss_available() -> bool:
    return _pkiss_cmd() is not None


def pkiss_fold(seq: str, timeout_s: float = 120.0) -> Optional[str]:
    """MFE structure (may contain pseudoknots) of `seq`, as extended dot-bracket, or None on failure."""
    cmd = _pkiss_cmd()
    if cmd is None:
        return None
    s = seq.upper().replace("T", "U")
    try:
        out = subprocess.run(cmd + ["--mode=mfe", s], capture_output=True, text=True, timeout=timeout_s)
    except Exception:
        return None
    struct = None
    valid = set(".()[]{}<>")
    for line in out.stdout.splitlines():
        parts = line.split()
        # structure line looks like: "-5.80  (((...[[[...)))...]]]"; take the last dot-bracket token
        if parts and set(parts[-1]) <= valid and len(parts[-1]) == len(s):
            struct = parts[-1]
    return struct


def verify_pk(seq: str, target: str, timeout_s: float = 120.0) -> Dict:
    """Pseudoknot-aware verification: fold `seq` with pKiss, compare (pair-set) to `target`.
    solved := pk-fold(seq) has EXACTLY the target's pairs. Never trusts the MFE oracle here."""
    if not seq or len(seq) != len(target):
        return {"solved": False, "bp": len(pk_pairs(target)), "folded": None, "error": "bad seq"}
    folded = pkiss_fold(seq, timeout_s=timeout_s)
    if folded is None:
        return {"solved": False, "bp": None, "folded": None, "error": "pkiss unavailable/failed"}
    bp = pk_bp_distance(folded, target)
    return {"solved": bp == 0, "bp": bp, "folded": folded, "n_target_pairs": len(pk_pairs(target))}


# ---- loader --------------------------------------------------------------------------------------
def _round_only(struct: str) -> str:
    """The pseudoknot-free reduction: keep () only, turn [] {} <> into '.' (what the MFE model sees)."""
    return "".join(c if c in "()" else "." for c in struct)


def n_pk_pairs(struct: str) -> int:
    """How many base pairs are LOST when the MFE model drops the pseudoknot brackets."""
    return len(pk_pairs(struct)) - len(pk_pairs(_round_only(struct)))


def load_pseudobase(path: Optional[str] = None, limit: Optional[int] = None,
                    with_constraints: bool = True) -> List[DesignTask]:
    """Load Pseudobase++ (RNAInvBench) pseudoknot design tasks. The IUPAC restraint becomes
    ConstraintSpec.motifs (fixed non-N positions); the target keeps its extended dot-bracket."""
    p = path or PSEUDOBASE_CSV
    tasks: List[DesignTask] = []
    with open(p) as f:
        for k, row in enumerate(csv.DictReader(f)):
            seq_restr = row["seq"].strip().upper()
            target = row["str"].strip()
            if not target or len(seq_restr) != len(target):
                continue
            motifs = [(i, c) for i, c in enumerate(seq_restr) if c != "N"] if with_constraints else []
            tasks.append(DesignTask(
                task_id=f"pseudobase:{k}", target=target, family="pseudoknot",
                constraints=ConstraintSpec(motifs=motifs),
                meta={"pk": True, "seq_restr": seq_restr, "source": "Pseudobase++",
                      "n_pairs": len(pk_pairs(target)), "n_pk_pairs": n_pk_pairs(target)}))
            if limit and len(tasks) >= limit:
                break
    return tasks


if __name__ == "__main__":
    ts = load_pseudobase(limit=5)
    print(f"loaded {len(ts)} sample tasks; pKiss available: {pkiss_available()}")
    for t in ts:
        red = _round_only(t.target)
        print(f"  {t.task_id} n={t.n} pairs={len(pk_pairs(t.target))} "
              f"pk_pairs={len(pk_pairs(t.target)) - len(pk_pairs(red))} restr_fixed={len(t.constraints.motifs)}")
        print(f"    target : {t.target}")
        print(f"    MFE sees: {red}   <- [] {{}} dropped => MFE 'solved' would be bogus")
