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
                 timeout_s: Optional[float] = None, extra_args: Optional[List[str]] = None,
                 steps: Optional[int] = None):
        self.replicas = replicas
        self.timelimit_s = int(timelimit_s)              # DesiRNA's own -t (it stops itself)
        self.results_number = int(results_number)        # DesiRNA's -r (how many best seqs it reports)
        # DesiRNA's -s: stop after this many REMC steps instead of when the clock runs out. Its main loop is
        # `while time.time() - start < timlim`, with `if global_step == RE_steps: break` inside, so a step cap
        # only binds when -t is generous -- callers using steps should pass a timelimit_s they will not hit.
        #
        # WHY THIS EXISTS. A wall-clock budget makes the search IRREPRODUCIBLE: the number of REMC steps
        # completed depends on machine load, so two identical invocations diverge. Measured 2026-08-08 on a
        # 67 nt target, ten replicas: two runs at the same seed shared 1 candidate out of 121, Jaccard 0.004,
        # and returned different best sequences. That is fine for a product whose user asked for "an answer
        # in 60 seconds", and fatal for an A/B test of a code change, because the effect being measured is
        # smaller than the noise between two runs of identical code. Time budgets stay the product default;
        # experiments that need a controlled comparison pass steps instead.
        self.steps = int(steps) if steps else None
        # Per-instance extra CLI args. DESIRNA_ARGS already existed for this, but it is process-global, so it
        # cannot express "this call wants a GC target and that one does not" -- which is exactly what a
        # parallel benchmark does.
        self.extra_args = list(extra_args or [])
        # subprocess hard-kill: allow DesiRNA to finish its -t run + startup/IO margin
        self.timeout_s = float(timeout_s) if timeout_s is not None else float(timelimit_s) + 120.0

    # -- helpers (match DesiRNA's real interface: github.com/fryzjergda/DesiRNA) --------------
    def _build_input(self, task: DesignTask) -> str:
        """3 labelled sections: name, seq restraints (IUPAC; N = any), target dot-bracket.

        `meta["seed_seq"]` adds DesiRNA's own `>seed_seq` section, which sets the STARTING sequence for the
        search. It is plumbed here so that any comparison against a "stay near this sequence" requirement
        gives the tool its own mechanism rather than nothing -- a benchmark that withholds the feature under
        test measures our plumbing, not the tool.
        """
        n = len(task.target)
        meta = task.meta or {}
        out = (">name\nDesign\n"
               f">seq_restr\n{meta.get('seq_restr') or ('N' * n)}\n"
               f">sec_struct\n{task.target}\n")
        seed_seq = meta.get("seed_seq")
        if seed_seq and len(seed_seq) == n:
            out += f">seed_seq\n{seed_seq}\n"
        alts = [a for a in (meta.get("alt_sec_struct") or []) if a and len(a) == n]
        if alts:
            # DesiRNA's own negative-design channel: named alternative structures, one per line. Plumbed so
            # a comparison on negative design gives the tool its mechanism instead of nothing.
            out += ">alt_sec_struct\n" + "\n".join(alts) + "\n"
        return out

    def _argv(self, cmd: List[str], input_name: str, seed: Optional[int], param: str = "2004") -> List[str]:
        # -p aligns DesiRNA's OPTIMIZATION energy model to OUR verification oracle. DesiRNA defaults to
        # Turner 1999; our oracle is turner2004/dangles=2, so sequences optimal under 1999 come back
        # bp=1-2 off under 2004 (the cross-oracle near-misses we diagnosed). Pass -p 2004 to match.
        argv = list(cmd) + ["-f", input_name,
                            "-p", param,
                            "-R", str(self.replicas),
                            "-t", str(self.timelimit_s),
                            "-r", str(self.results_number)]
        if self.steps:
            argv += ["-s", str(self.steps)]
        argv += list(self.extra_args)
        extra = os.environ.get("DESIRNA_ARGS")
        if extra:
            argv += shlex.split(extra)      # user extra args can override (argparse: last wins)
        if seed is not None:
            # OFFSET BY ONE, DELIBERATELY. DesiRNA reads `-seed 0` as "choose a seed at random"
            # (`original_seed = random.random()` when in_seed == 0), not as "use seed zero". Zero is this
            # project's default everywhere -- every benchmark, every ladder run, every "3 seeds" comparison
            # -- so all of them have been asking the solver to randomise while believing they were pinning
            # it. Offsetting keeps distinct seeds distinct and maps our 0 onto a real one.
            #
            # This alone does NOT make a run reproducible; the wall-clock main loop dominates (see `steps`
            # above). It removes one of the two causes, and it is the cause that was silently defeating the
            # intent of every seeded experiment in this repository.
            argv += ["-seed", str(int(seed) + 1)]
        return argv

    def _run(self, cmd: List[str], task: DesignTask, seed: Optional[int]) -> Tuple[List[str], Optional[str]]:
        param = {"turner2004": "2004", "turner1999": "1999"}.get(
            getattr(task.oracle, "param_set", "turner2004"), "2004")
        n = len(task.target)
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "input.txt"), "w") as f:
                f.write(self._build_input(task))
            # PIN THE HASH SEED. This is what actually makes the solver reproducible, and it took three
            # wrong hypotheses to find. DesiRNA's design loop walks dicts and sets of string keys
            # (utils/sequence_utils.py: `for start_node in graph.keys()`, `current_graph = set()`, and the
            # coloring/acgu-state passes), and Python randomises string hashing per process by default, so
            # iteration order -- and therefore which mutation is tried -- differs on every invocation.
            #
            # Measured 2026-08-08 on a 67 nt target, ten replicas, three runs each:
            #     PYTHONHASHSEED=0 + wall clock -t 20, no step cap  -> all three identical, Jaccard 1.000
            #     step cap -s 40, hash seed left free               -> Jaccard 0.000
            # So the hash seed is the cause; the step cap is neither necessary nor sufficient. I had first
            # blamed the seed argument, then the wall-clock main loop, and both were wrong.
            #
            # This keeps the product's SECONDS-based budget intact -- the ladder, the cost model and the
            # "answer in 60 seconds" promise are unchanged -- while making a certificate reproducible, which
            # it previously was not: the same question asked twice returned different sequences. Override
            # with DESIRNA_HASHSEED if some caller ever needs the old randomised behaviour.
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = os.environ.get("DESIRNA_HASHSEED", "0")
            try:                                             # relative -f; DesiRNA writes output under cwd
                proc = subprocess.run(self._argv(cmd, "input.txt", seed, param), cwd=td, env=env,
                                      capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return [], "timeout"
            except Exception as e:  # launch failure (bad DESIRNA_CMD etc.)
                return [], f"launch-failed: {e}"
            # DesiRNA writes its RANKED ANSWERS to *_results.csv, *_mid_results.csv, *_best_fasta.fas and
            # *_best_str -- `-r` of them -- and separately writes the raw optimisation trace to
            # trajectory_files/*_traj.csv, *_replicas.csv and *_random.csv. Until 2026-08-08 this scrape
            # read every file and kept the first 200 length-n strings it found, so intermediate states the
            # solver had already rejected were handed back as candidates and could be certified and
            # returned to a user. Measured on 57 OpenKnot targets at the 60 s tier
            # (runs/ceiling/trajectory_contamination*.py): the trace states are worse than the reported
            # answers by 0.107 crossed-pair F1, on 46 of 57 targets, p = 2.8e-06. At 300 s the three trace
            # files held 1,450 sequences each against 20 real answers, which is why longer runs came back
            # WORSE on our own panel as well as on RibonanzaNet.
            # Collected as an ORDERED list rather than two blobs, because certify_design selects the first
            # candidate in pool order that the panel accepts -- so which file a sequence came from and where
            # it sits in the walk both decide what gets delivered. Keeping walk order lets the experiment
            # hatch below reproduce the old behaviour exactly instead of approximating it with
            # "all answers, then all trace", which would be a different ordering and a different A/B.
            TRACE = ("_traj.csv", "_replicas.csv", "_random.csv")
            chunks = [((proc.stdout or "") + "\n" + (proc.stderr or ""), False)]
            for root, _dirs, files in os.walk(td):
                for fn in sorted(files):                     # sorted: reproducible across filesystems
                    if fn == "input.txt":
                        continue
                    try:
                        text = open(os.path.join(root, fn), errors="ignore").read()
                    except Exception:
                        continue
                    chunks.append((text, any(fn.endswith(s) for s in TRACE)))
            reported = "\n".join(t for t, is_trace in chunks if not is_trace)
            trace = "\n".join(t for t, is_trace in chunks if is_trace)
            walk_order = "\n".join(t for t, _ in chunks)

            def scrape(blob, seen):
                out = []
                for m in re.findall(r"[ACGUacgu]{%d}" % n, blob):
                    s = m.upper().replace("T", "U")
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
                return out

            seen = set()
            if os.environ.get("DESIRNA_SCRAPE_TRACE") == "1":
                # EXPERIMENT-ONLY ESCAPE HATCH, not a supported mode. Restores the pre-2026-08-08 scrape,
                # which read the trace files in walk order alongside the answers, so an A/B can hold the
                # solver fixed and vary ONLY this. Selection in certify_design is "first candidate in pool
                # order that the panel accepts", so the scrape changes delivery through ordering as well as
                # membership, and reimplementing that selection to compare arms would risk diverging from
                # the thing being measured. Toggling here keeps both arms on the real code path, and reads
                # the recorded WALK ORDER so the restored arm is the old behaviour rather than a re-ordering
                # of it ("all answers, then all trace" would be a third thing, and a different A/B).
                seqs = scrape(walk_order, seen)
                return seqs[:200], (None if seqs else f"no length-{n} ACGU sequence in DesiRNA output")
            seqs = scrape(reported, seen)
            if not seqs:
                # Never regress to zero candidates: if a DesiRNA version names its outputs differently, the
                # trace is still better than nothing, and the caller re-adjudicates everything anyway.
                seqs = scrape(trace, seen)
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
