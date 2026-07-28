#!/usr/bin/env python3
"""Launcher that gives DesiRNA a MUTABLE starting sequence, without editing DesiRNA.

WHY A LAUNCHER AND NOT A PATCH. Today's retrieval experiment established that the library's transferred
sub-sequences carry real information (at identical pinned positions they broke 25 targets where the canonical
prior broke 30, and solved 4 where the prior solved 2) but that the only available injection channel --
DesiRNA's `seq_restr` IUPAC field -- is a HARD constraint, and pinning is destructive on its own: both arms
broke essentially the same 24 of the 72 targets the unconstrained cheap tier already solved. Net -21.
Transferred experience has to enter as a STARTING POINT that the search may still move away from.

WHY IT IS SAFE. In DesiRNA the two things are already independent:
  * utils/sequence_utils.get_nt_list() derives the mutation alphabet from input_file.pairs and
    input_file.seq_restr only.
  * utils/sequence_utils.initial_sequence_generator() is a pure function that only decides WHERE the walk
    starts (all-A unpaired, loop-boosted G/U, GC-or-AU pairs).
So replacing the second changes the start and nothing about what the search is allowed to do. That is exactly
the channel the retrieval result says is missing.

WHY runpy AND NOT AN EDIT. DesiRNA.py keeps its whole driver inside `if __name__ == "__main__":` with no
callable main(), so it cannot be imported and invoked. Importing utils.sequence_utils FIRST, patching it in
sys.modules, then runpy-ing DesiRNA.py under run_name="__main__" gets the patched function into the run,
because generate_initial_list calls `initial_sequence_generator(...)` unqualified and therefore resolves it in
its own module globals at call time. The upstream checkout stays byte-identical, and the patch is versioned
here, in our repo, where it can be reviewed alongside the experiment that motivated it.

USE: set DESIRNA_INIT_SEQ to the warm start, then invoke this file exactly as you would DesiRNA.py.
A position whose warm-start base is not in that position's allowed letters falls back to what the original
generator would have put there, so a restraint and a warm start can be combined without conflict.
"""
from __future__ import annotations

import os
import runpy
import sys

DESIRNA_HOME = os.environ.get("DESIRNA_HOME", "/data/wenhui/DesiRNA")


def _install_patch():
    sys.path.insert(0, DESIRNA_HOME)
    import utils.sequence_utils as seq_utils

    original = seq_utils.initial_sequence_generator
    warm = (os.environ.get("DESIRNA_INIT_SEQ") or "").strip().upper().replace("T", "U")
    if not warm:
        return None

    def patched(nt_list, input_file, sim_options):
        base = original(nt_list, input_file, sim_options)          # also our per-position fallback
        if len(warm) != len(base):
            print(f"DESIRNA_INIT_SEQ length {len(warm)} != sequence length {len(base)}; ignoring",
                  file=sys.stderr)
            return base
        out, n_kept = [], 0
        for i, ch in enumerate(warm):
            allowed = getattr(nt_list[i], "letters_allowed", None) or getattr(nt_list[i], "letters", None)
            if allowed and ch in allowed:
                out.append(ch)
                n_kept += 1
            else:
                out.append(base[i])                                # restraint wins over the warm start
        print(f"DesiRNA warm start: {n_kept}/{len(warm)} positions taken from DESIRNA_INIT_SEQ "
              f"(the rest fall back to the built-in generator); the sequence remains fully MUTABLE "
              f"because get_nt_list derives the mutation alphabet from seq_restr, not from the start.")
        return "".join(out)

    seq_utils.initial_sequence_generator = patched
    return patched


if __name__ == "__main__":
    _install_patch()
    script = os.path.join(DESIRNA_HOME, "DesiRNA.py")
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")
