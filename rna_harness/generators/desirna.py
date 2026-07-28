"""Generator: DesiRNA -- call the external SOTA solver as a tool (PLAN A: accessibility).

The openclaude agent should not re-implement a solver; it CALLS the mature specialized tool, exactly
as Claude Code calls a compiler instead of re-writing one. DesiRNA (Wirecki et al., Nucleic Acids
Research 2025) is a replica-exchange Monte Carlo RNA inverse-design program (85 @<1min / 97 @24h on
Eterna100-V2). This adapter runs it as a subprocess and VERIFIES every returned sequence with our OWN
primary oracle -- never trust the tool's self-report (same discipline as RNAinverseGenerator).

Interface matches the real tool (github.com/fryzjergda/DesiRNA, Wirecki 2025):
  CLI   `DesiRNA.py -f <input> -R <replicas> -t <timelimit_s> -r <n_results> [-seed <n>]`
  input 3 labelled sections:  ">name\\nDesign\\n>seq_restr\\n<IUPAC, N=any>\\n>sec_struct\\n<dot-bracket>\\n"
  output written to a subdir (trajectory_files/*_best_fasta.fas, *_results.csv) under the run cwd.
We recurse the working dir and extract every length-n ACGU string (format-AGNOSTIC), then re-verify.

GRACEFUL BY DESIGN: if DesiRNA is not installed, design() returns a GenerationResult with
meta["available"]=False and does NOT crash, so a portfolio/agent degrades cleanly to the built-in
solvers. ACTIVATE by installing DesiRNA (pip/conda per its README) and setting:

    DESIRNA_CMD   full launch command, e.g.  "python /opt/DesiRNA/DesiRNA.py"
                  (if unset, a binary named DesiRNA / DesiRNA.py / desirna on PATH is used)
    DESIRNA_ARGS  optional extra CLI args, space-separated (e.g. "-nd" negative design, "-acgu")

If a DesiRNA version differs, only `_argv`/`_build_input` here change; the Generator interface +
oracle verification are stable.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
from shutil import which
from typing import List, Optional, Tuple

from ..task import DesignTask
from .base import GenerationResult, Generator


def _desirna_cmd() -> Optional[List[str]]:
    """Resolve the DesiRNA launch command, or None if it cannot be found.

    DISCOVERY MATTERS MORE THAN IT LOOKS. Before this searched the filesystem, resolution was
    DESIRNA_CMD-or-PATH only. DesiRNA is a checked-out script, not a pip console-script, so it is on neither
    unless the caller happens to export the variable -- and when it is missing, `default_toolbox()` silently
    returns the REMC fallback ladder, which measures 62/100 on Eterna100-V2 against DesiRNA's 84/100
    (runs/ceiling/ladder.py). A forgotten environment variable was therefore worth 22 solves, with no warning
    anywhere. Searching conventional locations makes an existing install usable by default; DESIRNA_CMD and
    DESIRNA_HOME still override.
    """
    env = os.environ.get("DESIRNA_CMD")
    if env:
        return shlex.split(env)
    for cand in ("DesiRNA", "DesiRNA.py", "desirna"):
        if which(cand):
            return [cand]
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    roots = [os.environ.get("DESIRNA_HOME"),
             os.path.join(repo, "vendor", "DesiRNA"),
             os.path.join(os.path.dirname(repo), "DesiRNA"),
             os.path.expanduser("~/DesiRNA")]
    for r in roots:
        if not r:
            continue
        script = os.path.join(r, "DesiRNA.py")
        if os.path.exists(script):
            # sys.executable, not "python": DesiRNA needs ViennaRNA + its own deps, which live in the same
            # interpreter this harness is running under.
            return [sys.executable, script]
    return None


def desirna_available() -> bool:
    return _desirna_cmd() is not None


class DesiRNAGenerator(Generator):
    name = "desirna"

    def __init__(self, replicas: int = 10, timelimit_s: int = 120, results_number: int = 20,
                 timeout_s: Optional[float] = None):
        self.replicas = replicas
        self.timelimit_s = int(timelimit_s)              # DesiRNA's own -t (it stops itself)
        self.results_number = int(results_number)        # DesiRNA's -r (how many best seqs it reports)
        # subprocess hard-kill: allow DesiRNA to finish its -t run + startup/IO margin
        self.timeout_s = float(timeout_s) if timeout_s is not None else float(timelimit_s) + 120.0

    # -- helpers (match DesiRNA's real interface: github.com/fryzjergda/DesiRNA) --------------
    def _build_input(self, task: DesignTask) -> str:
        # 3 labelled sections: name, seq restraints (IUPAC; N = any), target dot-bracket.
        n = len(task.target)
        return (">name\nDesign\n"
                f">seq_restr\n{'N' * n}\n"
                f">sec_struct\n{task.target}\n")

    def _argv(self, cmd: List[str], input_name: str, seed: Optional[int], param: str = "2004") -> List[str]:
        # -p aligns DesiRNA's OPTIMIZATION energy model to OUR verification oracle. DesiRNA defaults to
        # Turner 1999; our oracle is turner2004/dangles=2, so sequences optimal under 1999 come back
        # bp=1-2 off under 2004 (the cross-oracle near-misses we diagnosed). Pass -p 2004 to match.
        argv = list(cmd) + ["-f", input_name,
                            "-p", param,
                            "-R", str(self.replicas),
                            "-t", str(self.timelimit_s),
                            "-r", str(self.results_number)]
        extra = os.environ.get("DESIRNA_ARGS")
        if extra:
            argv += shlex.split(extra)      # user extra args can override (argparse: last wins)
        if seed is not None:
            argv += ["-seed", str(int(seed))]
        return argv

    def _run(self, cmd: List[str], task: DesignTask, seed: Optional[int]) -> Tuple[List[str], Optional[str]]:
        param = {"turner2004": "2004", "turner1999": "1999"}.get(
            getattr(task.oracle, "param_set", "turner2004"), "2004")
        n = len(task.target)
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "input.txt"), "w") as f:
                f.write(self._build_input(task))
            try:                                             # relative -f; DesiRNA writes output under cwd
                proc = subprocess.run(self._argv(cmd, "input.txt", seed, param), cwd=td,
                                      capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return [], "timeout"
            except Exception as e:  # launch failure (bad DESIRNA_CMD etc.)
                return [], f"launch-failed: {e}"
            blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
            for root, _dirs, files in os.walk(td):           # DesiRNA writes results to a SUBDIR
                for fn in files:                             # (trajectory_files/*_best_fasta.fas, *_results.csv)
                    if fn == "input.txt":
                        continue
                    try:
                        blob += "\n" + open(os.path.join(root, fn), errors="ignore").read()
                    except Exception:
                        pass
            seqs, seen = [], set()
            for m in re.findall(r"[ACGUacgu]{%d}" % n, blob):   # any length-n ACGU string = candidate
                s = m.upper().replace("T", "U")
                if s not in seen:
                    seen.add(s)
                    seqs.append(s)
            return seqs[:200], (None if seqs else f"no length-{n} ACGU sequence in DesiRNA output")

    def _verify_best(self, task: DesignTask, seqs: List[str]) -> Tuple[Optional[str], int]:
        """Re-verify with OUR primary oracle; never trust DesiRNA's self-report."""
        from ..metrics import bp_distance
        from ..oracles.registry import primary_oracle
        orc = primary_oracle()
        n = len(task.target)
        best_seq, best_d = None, 10 ** 9
        for s in seqs:
            if len(s) != n or (set(s) - set("ACGU")):
                continue
            mfe, _ = orc.fold_mfe(s)
            d = bp_distance(mfe, task.target)
            if d < best_d:
                best_seq, best_d = s, d
                if d == 0:
                    break
        return best_seq, best_d

    # -- Generator interface ------------------------------------------------
    def design(self, task: DesignTask, budget: int = 1, seed: Optional[int] = None) -> GenerationResult:
        cmd = _desirna_cmd()
        if cmd is None:
            return GenerationResult(
                task_id=task.task_id, best_sequence="", solved=False,
                best_bp_distance=len(task.target), n_evals=0, method=self.name,
                meta={"available": False,
                      "hint": "DesiRNA not found. Install it and set DESIRNA_CMD to activate "
                              "(see rna_harness/generators/desirna.py). Falling back to built-in solvers."})
        seqs, err = self._run(cmd, task, seed)
        if not seqs:
            return GenerationResult(
                task_id=task.task_id, best_sequence="", solved=False,
                best_bp_distance=len(task.target), n_evals=0, method=self.name,
                meta={"available": True, "error": err or "no candidates"})
        best_seq, best_d = self._verify_best(task, seqs)
        # pseudoknot guard: our primary (ViennaRNA MFE) oracle cannot verify a pseudoknot target
        # (it drops '[]{}'), so best_d here is scored against the pseudoknot-FREE reduction and a
        # best_d==0 would be a FALSE 'solved'. Refuse the MFE verdict; candidates are still returned
        # so a pseudoknot-aware verifier (rnainvbench.verify_pk / pKiss) can score them.
        from ..structure_utils import has_pseudoknot
        if has_pseudoknot(task.target):
            return GenerationResult(
                task_id=task.task_id, best_sequence=best_seq or "", solved=False,
                best_bp_distance=best_d, n_evals=len(seqs), candidates=seqs, method=self.name,
                meta={"available": True, "n_returned": len(seqs),
                      "verified_by": "NONE (pseudoknot target; MFE oracle cannot verify)",
                      "mfe_bp_vs_pkfree_reduction": best_d,
                      "hint": "pseudoknot target -- verify with rnainvbench.verify_pk (pKiss); "
                              "MFE 'solved' would be a false positive."})
        return GenerationResult(
            task_id=task.task_id, best_sequence=best_seq or "", solved=(best_d == 0),
            best_bp_distance=best_d, n_evals=len(seqs), candidates=seqs, method=self.name,
            meta={"available": True, "n_returned": len(seqs), "verified_by": "primary_oracle"})
